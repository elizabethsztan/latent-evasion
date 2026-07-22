"""Train per-position refusal probes on generated-token activations.

Mirrors classifier/train_latent.py, but instead of a single post-instruction
representation per prompt it greedily generates `n_gen_tokens` continuation
tokens (un-intervened) and extracts the hidden state at each generated token
position. It then trains one LinearSVC per (layer, generated-position), using
the exact same SVM procedure as train_latent.py.

Motivation: the CLE projection hook (utils/hooks.py) applies the post-instruction
probe direction `w` to every generated token as it is decoded. These probes let
us measure how much the harmful/harmless separating direction rotates across
generated positions relative to the original post-instruction probe.

Activations are trained on the fly and discarded; only the generated tokens
(generated_tokens.json) and the trained probes are written to disk.
"""

import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from classifier.utils import (
    generate_and_extract,
    get_model,
    load_arditi,
    pick_device,
    train_layer_svm,
)


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", type=str, default="llama32-3b")
    parser.add_argument("--device", default="auto", type=str, help="cuda:0 | mps | cpu | auto")
    parser.add_argument("--artifact_dir", type=str, default="./dataset/representations/")
    parser.add_argument("--n_samples", type=int, default=-1, help="Prompts per class (-1 = all)")
    parser.add_argument("--n_gen_tokens", type=int, default=5, help="Greedy tokens to generate per prompt")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = get_args()
    args.device = pick_device(args.device)

    model = get_model(args.model_name, device=args.device)
    harmful_prompts, harmless_prompts = load_arditi(args.model_name)

    if args.n_samples > 0:
        harmful_prompts = harmful_prompts[: args.n_samples]
        harmless_prompts = harmless_prompts[: args.n_samples]

    all_prompts = harmful_prompts + harmless_prompts
    all_labels = [1] * len(harmful_prompts) + [0] * len(harmless_prompts)  # harmful=1, harmless=0

    print(f"Generating {args.n_gen_tokens} tokens for {len(all_prompts)} prompts...")
    reps_list = []          # each (num_gen, num_layers, dim) for prompts with full length
    kept_labels = []
    gen_records = []
    num_short = 0

    for idx, (prompt, label) in enumerate(tqdm(list(zip(all_prompts, all_labels)), total=len(all_prompts))):
        reps, num_gen, gen_ids, gen_text = generate_and_extract(model, prompt, args.n_gen_tokens)
        gen_records.append({
            "id": idx,
            "label": label,
            "prompt": prompt,
            "generated_text": gen_text,
            "generated_token_ids": gen_ids,
            "num_generated": num_gen,
        })
        if reps is None or num_gen < args.n_gen_tokens:
            num_short += 1
            continue
        reps_list.append(reps.unsqueeze(0))  # (1, n_gen, num_layers, dim)
        kept_labels.append(label)

    if num_short:
        print(f"WARNING: {num_short}/{len(all_prompts)} prompts produced < {args.n_gen_tokens} tokens; excluded from probes.")

    # X_full: (N, n_gen, num_layers, dim)
    X_full = torch.cat(reps_list, dim=0).numpy()
    Y = np.array(kept_labels)
    n_gen = X_full.shape[1]
    num_layers = X_full.shape[2]
    print(f"Kept {X_full.shape[0]} prompts | positions={n_gen} | layers={num_layers} | dim={X_full.shape[3]}")
    print(f"Class balance kept: harmful={int((Y == 1).sum())} harmless={int((Y == 0).sum())}")

    out_dir = os.path.join(args.artifact_dir, args.model_name, "train_svm_generated")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "generated_tokens.json"), "w", encoding="utf-8") as f:
        json.dump(gen_records, f, indent=2)
    print(f"Saved generations to {os.path.join(out_dir, 'generated_tokens.json')}")

    print(f"\nTraining {n_gen} x {num_layers} SVMs...")
    for k in range(n_gen):
        accs = []
        for l in range(num_layers):
            X_layer = X_full[:, k, l, :]  # (N, dim)
            clf, acc, report = train_layer_svm(X_layer, Y, args.seed)
            accs.append(acc)
            save_obj = {
                "w": torch.from_numpy(clf.coef_[0]).float(),
                "b": torch.tensor(clf.intercept_[0]).float(),
                "layer_idx": l,
                "gen_pos": k,
                "model_name": args.model_name,
                "hidden_dim": X_layer.shape[-1],
                "accuracy": acc,
                "n_train_prompts": int(X_full.shape[0]),
                "report": report,
            }
            torch.save(save_obj, os.path.join(out_dir, f"svm_layer{l:02d}_pos{k:02d}.pt"))
        print(f"pos {k:02d} | mean SVM test acc over layers: {np.mean(accs):.4f}")
    print(f"\nDone. Probes saved under {out_dir}")


if __name__ == "__main__":
    main()
