"""Cosine similarity of CLE-S's sequentially-trained prompt probes against the
original, independently-trained probes shared by CLE-A/CLE-P/CLE-G, layer by
layer.

For each layer, base CLE (classifier/train_latent.py) fits a probe on clean,
unsteered activations (train_svm/svm_layerXX.pt). CLE-S instead fits each
layer's probe on harmful activations already steered by every probe trained
earlier in the chain (train_svm_sequential/svm_layerXX.pt) -- see
docs/superpowers/specs/2026-07-24-cle-s-design.md. This figure checks how much
that changes the learned direction, one layer at a time: for each layer l,
cosine(w_old[l], w_new[l]). (Cross-layer comparisons aren't meaningful here --
the independently-trained probes are already highly self-similar across
nearby layers, which would just leak through and confound an old-vs-new
matrix; the same-layer comparison is the one that isolates what sequential
training actually changed.)

Saves results/probe_boundary/sequential_probe_similarity_<model>.{png,pdf} and
the raw per-layer values as .json.
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from analysis._shared import apply_rcparams
from utils.models_utils import parse_layers

apply_rcparams()


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", type=str, default="llama32-3b")
    parser.add_argument("--artifact_dir", type=str, default="./dataset/representations/")
    parser.add_argument("--out_dir", type=str, default="./results/probe_boundary")
    parser.add_argument(
        "--layers",
        type=str,
        default="11-23",
        help="Contiguous end-exclusive layer range shared by both probe sets, e.g. '11-23'.",
    )
    return parser.parse_args()


def load_weights(base_dir: str, layers: list[int]) -> np.ndarray:
    """Stack each layer's unit-normalized probe weight into (n_layers, dim)."""
    ws = []
    for layer_idx in layers:
        path = os.path.join(base_dir, f"svm_layer{layer_idx:02d}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing probe checkpoint: {path}")
        obj = torch.load(path, map_location="cpu")
        w = obj["w"].float().view(-1).numpy()
        ws.append(w / (np.linalg.norm(w) + 1e-12))
    return np.stack(ws, axis=0)


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    layers = parse_layers(args.layers)

    old_dir = os.path.join(args.artifact_dir, args.model_name, "train_svm")
    new_dir = os.path.join(args.artifact_dir, args.model_name, "train_svm_sequential")

    W_old = load_weights(old_dir, layers)  # (n_layers, dim), unit-normalized
    W_new = load_weights(new_dir, layers)
    print(f"Loaded {len(layers)} old probes from {old_dir}")
    print(f"Loaded {len(layers)} new (CLE-S) probes from {new_dir}")

    cos_sim = np.sum(W_old * W_new, axis=1)  # (n_layers,); same-layer only
    print(f"Same-layer cosine similarity: min={cos_sim.min():.4f} max={cos_sim.max():.4f} "
          f"mean={cos_sim.mean():.4f}")
    for layer_idx, sim in zip(layers, cos_sim):
        print(f"  layer {layer_idx:02d}: {sim:.4f}")

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.bar([str(l) for l in layers], cos_sim, color="#4c72b0")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlabel("Layer")
    ax.set_ylabel("cosine similarity (old vs CLE-S)")
    ax.set_title(f"Same-layer probe direction similarity: old vs CLE-S — {args.model_name}")

    stem = f"sequential_probe_similarity_{args.model_name}"
    fig.savefig(os.path.join(args.out_dir, f"{stem}.png"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(args.out_dir, f"{stem}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {stem}.png/.pdf to {args.out_dir}")

    out_json = os.path.join(args.out_dir, f"{stem}.json")
    with open(out_json, "w") as f:
        json.dump(
            {
                "model_name": args.model_name,
                "layers": layers,
                "cosine_similarity": cos_sim.tolist(),
                "mean": float(cos_sim.mean()),
            },
            f,
            indent=2,
        )
    print(f"Saved {out_json}")


if __name__ == "__main__":
    main()
