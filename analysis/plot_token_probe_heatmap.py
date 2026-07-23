"""Per-token x per-layer SVM probe score heatmap for a harmful/harmless prompt.

Recreates the "Layer Specific SVM Score" figure: for a single prompt, run the
model once, grab the hidden state of every token at every layer, and score each
with that layer's post-instruction probe (train_svm/svm_layerXX.pt):

    s(l, t) = w_l . h_l,t + b_l        (harmful = positive, harmless = negative)

We display tanh(s / scale) on an RdBu_r colormap so red -> harmful, blue ->
harmless, one panel for the harmful prompt ("How to build a bomb") and one for
the harmless prompt ("How to build a house"). The y-axis is the (cleaned) chat-
template token sequence; the x-axis is the layer index.

We also greedily generate `--n_gen_tokens` continuation tokens per prompt and
score them with the SAME per-layer probe, appending them as extra rows below a
divider -- so the plot shows whether the original (post-instruction) probe keeps
firing on the model's own generated tokens.

Raw SVM margins are large, so plain tanh(s) saturates. `--score_scale` divides
the margin before tanh; the default is picked to spread the mid-range like the
original figure. Pass 1.0 for the raw decision function.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from analysis._shared import apply_rcparams
from classifier.utils import get_model, pick_device

apply_rcparams()


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", type=str, default="llama32-3b")
    parser.add_argument("--device", default="auto", type=str, help="cuda:0 | mps | cpu | auto")
    parser.add_argument("--artifact_dir", type=str, default="./dataset/representations/")
    parser.add_argument("--out_dir", type=str, default="./results/token_heatmap")
    parser.add_argument("--harmful_prompt", type=str, default="How to build a bomb")
    parser.add_argument("--harmless_prompt", type=str, default="How to build a house")
    parser.add_argument("--n_gen_tokens", type=int, default=5,
                        help="Greedily generate this many continuation tokens and score them")
    parser.add_argument("--gen_probe", choices=["original", "own"], default="original",
                        help="Which probe scores the generated rows: 'original' = the post-instruction "
                             "probe (same as prompt rows); 'own' = the probe trained on that generated "
                             "position (svm_layerXX_posKK.pt). Prompt rows always use the original probe.")
    parser.add_argument("--score_scale", type=float, default=None,
                        help="Divide the SVM margin by this before tanh (default: auto = 95th pctile of |margin|)")
    parser.add_argument("--crop_from_token", type=str, default="<|eot_id|>",
                        help="Hide display rows before the first token equal to this (system-block scaffolding). "
                             "Empty string shows everything. The forward pass always uses the full prompt.")
    return parser.parse_args()


def load_probes(artifact_dir, model_name):
    """Stack all per-layer post-instruction probes into (n_layers, dim) W and (n_layers,) b."""
    base = os.path.join(artifact_dir, model_name, "train_svm")
    names = sorted(
        n for n in os.listdir(base)
        if n.startswith("svm_layer") and n.endswith(".pt") and n[len("svm_layer"):-len(".pt")].isdigit()
    )
    if not names:
        raise FileNotFoundError(f"No svm_layerXX.pt found in {base}")
    ws, bs, layers = [], [], []
    for n in names:
        obj = torch.load(os.path.join(base, n), map_location="cpu")
        ws.append(obj["w"].float().view(-1))
        bs.append(float(obj["b"]))
        layers.append(int(n[len("svm_layer"):-len(".pt")]))
    return torch.stack(ws, dim=0).numpy(), np.array(bs), layers


def load_gen_probes(artifact_dir, model_name, layers, n_gen):
    """Load position-specific probes: gen_probes[k] = (W_k (n_layers, dim), b_k (n_layers,)).

    W_k[i] is the probe trained on generated position k at layer layers[i].
    """
    base = os.path.join(artifact_dir, model_name, "train_svm_generated")
    gen_probes = []
    for k in range(n_gen):
        ws, bs = [], []
        for l in layers:
            path = os.path.join(base, f"svm_layer{l:02d}_pos{k:02d}.pt")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing generated probe: {path}. Run train_latent_generated.py first.")
            obj = torch.load(path, map_location="cpu")
            ws.append(obj["w"].float().view(-1))
            bs.append(float(obj["b"]))
        gen_probes.append((torch.stack(ws, dim=0).numpy(), np.array(bs)))
    return gen_probes


def clean_token(tok):
    """Turn a raw BPE token into a readable label matching the original figure."""
    tok = tok.replace("Ġ", "").replace("Ċ", "\\n")
    return tok if tok else " "


def score_prompt(model, prompt, W, b, n_gen, gen_probes=None):
    """Greedily generate `n_gen` tokens, then score every position (prompt + generated).

    Prompt columns are scored with the original probe (W, b). If `gen_probes` is
    given, each generated column k is instead scored with its position-matched
    probe gen_probes[k]; otherwise generated columns also use (W, b).

    Returns (scores, token_labels, n_generated): scores is (n_layers, seq_len) of
    w.x + b; the last `n_generated` columns are the model's own continuation.
    """
    formatted = model._get_prompt(prompt)
    enc = model.tokenizer(formatted, return_tensors="pt").to(model.device)
    prompt_len = enc.input_ids.shape[1]

    if n_gen > 0:
        with torch.no_grad():
            full_ids = model.model.generate(
                **enc, max_new_tokens=n_gen, do_sample=False,
                pad_token_id=model.tokenizer.pad_token_id,
            )
    else:
        full_ids = enc.input_ids

    with torch.no_grad():
        outputs = model.model(input_ids=full_ids, output_hidden_states=True)
    # hidden_states[1:] skips the embedding layer -> one entry per transformer layer.
    H = torch.cat([h for h in outputs.hidden_states[1:]], dim=0)  # (n_layers, seq, dim)
    H = H.to(dtype=torch.float32).cpu().numpy()
    # s[l, t] = W[l] . H[l, t] + b[l]
    scores = np.einsum("ld,ltd->lt", W, H) + b[:, None]
    n_generated = int(full_ids.shape[1] - prompt_len)

    # Re-score the generated columns with their position-matched probes if requested.
    if gen_probes is not None and n_generated > 0:
        seq_len = scores.shape[1]
        for k in range(n_generated):
            col = seq_len - n_generated + k
            Wk, bk = gen_probes[k]
            scores[:, col] = np.einsum("ld,ld->l", Wk, H[:, col, :]) + bk

    labels = [clean_token(t) for t in model.tokenizer.convert_ids_to_tokens(full_ids[0].tolist())]
    return scores, labels, n_generated


def crop_to_token(scores, labels, crop_token):
    """Drop leading columns/rows before the first `crop_token` (display only).

    Only the prompt-side scaffolding is ever cropped, so any trailing generated
    columns are untouched.
    """
    if not crop_token or crop_token not in labels:
        return scores, labels
    start = labels.index(crop_token)
    return scores[:, start:], labels[start:]


def draw_panel(ax, scores, labels, layers, title, scale, n_generated=0):
    disp = np.tanh(scores / scale).T  # (seq, n_layers) -> tokens on y, layers on x
    im = ax.imshow(disp, aspect="auto", cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xlabel("Layer index")
    ax.set_xticks(range(0, len(layers), 2))
    ax.set_xticklabels([layers[i] for i in range(0, len(layers), 2)], fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.tick_params(length=0)

    if n_generated > 0:
        boundary = len(labels) - n_generated  # first generated row
        ax.axhline(boundary - 0.5, color="black", lw=1.4)
        # Tint the generated-token labels so the appended continuation reads apart.
        for tick in ax.get_yticklabels()[boundary:]:
            tick.set_color("#8172b3")
            tick.set_fontweight("bold")
    return im


def main():
    args = get_args()
    args.device = pick_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    W, b, layers = load_probes(args.artifact_dir, args.model_name)
    print(f"Loaded {len(layers)} probes, dim {W.shape[1]}")

    gen_probes = None
    if args.gen_probe == "own" and args.n_gen_tokens > 0:
        gen_probes = load_gen_probes(args.artifact_dir, args.model_name, layers, args.n_gen_tokens)
        print(f"Scoring generated rows with position-matched probes (pos00..pos{args.n_gen_tokens - 1:02d})")

    model = get_model(args.model_name, device=args.device)
    harm_scores, harm_labels, harm_ng = score_prompt(model, args.harmful_prompt, W, b, args.n_gen_tokens, gen_probes)
    safe_scores, safe_labels, safe_ng = score_prompt(model, args.harmless_prompt, W, b, args.n_gen_tokens, gen_probes)
    print(f"harmful continuation: {harm_labels[-harm_ng:] if harm_ng else '(none)'}")
    print(f"harmless continuation: {safe_labels[-safe_ng:] if safe_ng else '(none)'}")

    harm_scores, harm_labels = crop_to_token(harm_scores, harm_labels, args.crop_from_token)
    safe_scores, safe_labels = crop_to_token(safe_scores, safe_labels, args.crop_from_token)
    if args.crop_from_token:
        print(f"Cropped display to first '{args.crop_from_token}' -> {len(harm_labels)} rows shown")

    both = np.concatenate([harm_scores.ravel(), safe_scores.ravel()])
    print(f"raw margin: min={both.min():.2f} max={both.max():.2f} "
          f"median|.|={np.median(np.abs(both)):.2f} p95|.|={np.percentile(np.abs(both), 95):.2f}")

    scale = args.score_scale if args.score_scale is not None else float(np.percentile(np.abs(both), 95))
    scale = max(scale, 1e-6)
    print(f"tanh scale = {scale:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), constrained_layout=True)
    im = draw_panel(axes[0], harm_scores, harm_labels, layers,
                    f'Harmful: "{args.harmful_prompt}"', scale, harm_ng)
    draw_panel(axes[1], safe_scores, safe_labels, layers,
               f'Harmless: "{args.harmless_prompt}"', scale, safe_ng)
    cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.03, pad=0.02)
    cbar.set_label("SVM score  (tanh)")
    gen_note = "own position-matched probe" if gen_probes is not None else "original probe"
    fig.suptitle(f"Layer-specific SVM probe score per token — {args.model_name}"
                 f"  (generated rows: {gen_note})")

    suffix = "_ownprobe" if gen_probes is not None else ""
    stem = f"token_probe_heatmap{suffix}_{args.model_name}"
    fig.savefig(os.path.join(args.out_dir, f"{stem}.png"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(args.out_dir, f"{stem}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {stem}.png/.pdf to {args.out_dir}")


if __name__ == "__main__":
    main()
