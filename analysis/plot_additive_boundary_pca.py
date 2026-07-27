"""Visualize CLE-A's fixed additive replay against harmful/harmless activations.

CLE-A has two prompt forwards. The first uses the independently-trained probe at
every selected layer to compute one last-position delta per layer and prompt. The
second, generation forward replays those fixed deltas with add hooks. In a prefill,
each stored delta is added to every token position, exactly as in cle-a.py.

For each requested layer in the CLE-A range:
  - harmful (unsteered): train_svm/HFx_train.pt, no steering applied,
  - harmless: train_svm/HLx_train.pt,
  - harmful (steered, pre-L): the target-layer output during CLE-A's additive replay,
    after all earlier fixed-delta hooks but before layer L's own fixed delta,
  - harmful (steered, post-L): the same output after layer L's fixed delta is added.

A faint grey line connects each prompt's pre-L and post-L points. The shared
CLE-A/P probe is drawn on a PCA plane fit across all four clouds.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from analysis._shared import (
    HARMFUL_COLOR,
    HARMLESS_COLOR,
    ORIG_BOUNDARY_COLOR,
    STEERED_COLOR,
    STEERED_POST_COLOR,
    apply_rcparams,
    draw_single_probe_cascade_panel,
    load_probe,
)
from classifier.sequential_utils import encode_prompt, parse_layer_range
from classifier.utils import get_model, load_arditi, pick_device
from utils.hooks import (
    hidden_from_output,
    pipeline_delta_hook,
    remove_hooks,
    replace_hidden,
)
from utils.models_utils import get_transformer_layers
from utils.probes import load_svms

apply_rcparams()

# Same configuration as the CLE-P comparison so the two plots are directly
# comparable. Override these from the CLI for a particular tuned CLE-A run.
DEFAULT_BETA = 1.0
DEFAULT_MARGIN = 1.2


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", type=str, default="llama32-3b")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--artifact_dir", type=str, default="./dataset/representations/")
    parser.add_argument("--out_dir", type=str, default="./results/probe_boundary")
    parser.add_argument(
        "--chain_layers",
        type=str,
        default="11-23",
        help="Contiguous end-exclusive layer range CLE-A intervenes on.",
    )
    parser.add_argument(
        "--plot_layers",
        type=int,
        nargs="+",
        default=None,
        help="Subset of --chain_layers to draw (default: 4 evenly spaced).",
    )
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA)
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    parser.add_argument(
        "--n_samples",
        type=int,
        default=-1,
        help="Harmful prompts to replay (-1 = all).",
    )
    return parser.parse_args()


def compute_prompt_deltas(
    model,
    layer_modules,
    chain_layers,
    probes,
    input_ids,
    beta,
    margin,
):
    """Run CLE-A's projection pass and return its fixed per-layer deltas."""
    deltas = {}
    handles = [
        layer_modules[layer].register_forward_hook(
            pipeline_delta_hook(
                probes[layer]["w"],
                probes[layer]["b"],
                beta,
                margin,
                layer,
                deltas,
            )
        )
        for layer in chain_layers
    ]
    try:
        with torch.no_grad():
            model.model(input_ids=input_ids, use_cache=False)
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        missing = [layer for layer in chain_layers if layer not in deltas]
        if missing:
            raise RuntimeError(f"Failed to capture CLE-A deltas for layers: {missing}")
        return {layer: delta.detach().clone() for layer, delta in deltas.items()}
    finally:
        remove_hooks(handles)


def replay_and_capture(
    model,
    layer_modules,
    chain_layers,
    plot_layers,
    deltas,
    input_ids,
):
    """Run CLE-A's additive prefill and capture target outputs before/after addition."""
    captured = {}

    def capture_add_hook(layer):
        def hook(module, inputs, output):
            h = hidden_from_output(output)
            delta = deltas[layer].to(device=h.device, dtype=h.dtype)
            if delta.ndim == 1:
                delta = delta.view(1, 1, -1)
            elif delta.ndim == 2:
                delta = delta.unsqueeze(1)
            else:
                raise ValueError(f"Unexpected CLE-A delta shape: {tuple(delta.shape)}")
            if layer in plot_layers:
                pre = h[:, -1, :].squeeze(0).detach().clone()
            h_mod = h + delta
            if layer in plot_layers:
                post = h_mod[:, -1, :].squeeze(0).detach().clone()
                captured[layer] = {"pre": pre, "post": post}
            return replace_hidden(output, h_mod)

        return hook

    handles = [
        layer_modules[layer].register_forward_hook(capture_add_hook(layer))
        for layer in chain_layers
    ]
    try:
        with torch.no_grad():
            model.model(input_ids=input_ids, use_cache=False)
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        missing = [layer for layer in plot_layers if layer not in captured]
        if missing:
            raise RuntimeError(f"Failed to capture CLE-A replay layers: {missing}")
        return {
            layer: {
                name: tensor.to(dtype=torch.float32, device="cpu")
                for name, tensor in pair.items()
            }
            for layer, pair in captured.items()
        }
    finally:
        remove_hooks(handles)


def compute_steered_harmful(
    model,
    layer_modules,
    chain_layers,
    probes,
    harmful_ids_list,
    plot_layers,
    beta,
    margin,
):
    """Return per-layer pre/post clouds from CLE-A's actual two-pass replay."""
    steered = {
        layer: {"pre": [], "post": []}
        for layer in plot_layers
    }
    for index, input_ids in enumerate(harmful_ids_list, start=1):
        deltas = compute_prompt_deltas(
            model,
            layer_modules,
            chain_layers,
            probes,
            input_ids,
            beta,
            margin,
        )
        captured = replay_and_capture(
            model,
            layer_modules,
            chain_layers,
            set(plot_layers),
            deltas,
            input_ids,
        )
        for layer in plot_layers:
            steered[layer]["pre"].append(captured[layer]["pre"])
            steered[layer]["post"].append(captured[layer]["post"])
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if index % 25 == 0 or index == len(harmful_ids_list):
            print(f"  replayed {index}/{len(harmful_ids_list)} prompts")

    return {
        layer: {
            name: torch.stack(tensors).numpy()
            for name, tensors in pair.items()
        }
        for layer, pair in steered.items()
    }


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    chain_layers = parse_layer_range(args.chain_layers)
    plot_layers = args.plot_layers or [
        chain_layers[i]
        for i in np.linspace(0, len(chain_layers) - 1, 4).round().astype(int)
    ]
    plot_layers = sorted(set(plot_layers))
    for layer in plot_layers:
        if layer not in chain_layers:
            raise ValueError(
                f"--plot_layers {layer} is outside --chain_layers {chain_layers}"
            )

    base = os.path.join(args.artifact_dir, args.model_name, "train_svm")
    hf_post = torch.load(
        os.path.join(base, "HFx_train.pt"), map_location="cpu"
    ).float().numpy()
    hl_post = torch.load(
        os.path.join(base, "HLx_train.pt"), map_location="cpu"
    ).float().numpy()

    args.device = pick_device(args.device)
    model = get_model(args.model_name, device=args.device)
    layer_modules = get_transformer_layers(model)

    harmful_prompts, _ = load_arditi(args.model_name)
    if args.n_samples > 0:
        harmful_prompts = harmful_prompts[: args.n_samples]
    hf_post = hf_post[: len(harmful_prompts)]
    harmful_ids = [encode_prompt(model, prompt) for prompt in harmful_prompts]

    probes = load_svms(base, chain_layers, torch.device(args.device))
    print(
        f"Replaying {len(harmful_ids)} harmful prompts through CLE-A's two passes "
        f"(beta={args.beta}, margin={args.margin})..."
    )
    steered = compute_steered_harmful(
        model,
        layer_modules,
        chain_layers,
        probes,
        harmful_ids,
        plot_layers,
        args.beta,
        args.margin,
    )

    n = len(plot_layers)
    fig, axes = plt.subplots(
        1, n, figsize=(4.9 * n, 4.6), constrained_layout=True
    )
    axes = np.atleast_1d(axes).ravel()

    mesh = None
    for ax, layer in zip(axes, plot_layers):
        w, b, acc = load_probe(os.path.join(base, f"svm_layer{layer:02d}.pt"))
        mean_pre_score = float(np.mean(steered[layer]["pre"] @ w + b))
        mean_post_score = float(np.mean(steered[layer]["post"] @ w + b))
        mesh, var = draw_single_probe_cascade_panel(
            ax,
            hf_post[:, layer, :],
            hl_post[:, layer, :],
            steered[layer]["pre"],
            steered[layer]["post"],
            w,
            b,
            layer,
            acc,
        )
        print(
            f"Layer {layer:2d} | PC1+PC2 var = {var*100:.1f}% "
            f"| probe acc = {acc:.3f} "
            f"| mean pre-L score = {mean_pre_score:.6f} "
            f"| mean post-L score = {mean_post_score:.6f}"
        )

    handles = [
        plt.Line2D(
            [], [], marker="o", linestyle="none", color=HARMFUL_COLOR,
            label="harmful (unsteered)",
        ),
        plt.Line2D(
            [], [], marker="o", linestyle="none", color=HARMLESS_COLOR,
            label="harmless",
        ),
        plt.Line2D(
            [], [], marker="o", linestyle="none", color=STEERED_COLOR,
            label="harmful (CLE-A replay, pre-L)",
        ),
        plt.Line2D(
            [], [], marker="o", linestyle="none", color=STEERED_POST_COLOR,
            label="harmful (CLE-A replay, post-L)",
        ),
        plt.Line2D(
            [], [], color=ORIG_BOUNDARY_COLOR, lw=2.0,
            label="CLE-A/P probe",
        ),
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        ncol=len(handles),
        loc="lower center",
        bbox_to_anchor=(0.5, 1.05),
    )
    cbar = fig.colorbar(mesh, ax=axes.tolist(), fraction=0.02, pad=0.01)
    cbar.set_label("Compliance confidence")

    stem = f"probe_boundary_additive_pca_{args.model_name}"
    fig.savefig(
        os.path.join(args.out_dir, f"{stem}.png"),
        dpi=150,
        bbox_inches="tight",
    )
    fig.savefig(os.path.join(args.out_dir, f"{stem}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {stem}.png/.pdf to {args.out_dir}")


if __name__ == "__main__":
    main()
