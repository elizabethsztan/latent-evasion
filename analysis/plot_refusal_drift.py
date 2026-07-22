"""Heatmaps of refusal-direction drift across generated token positions.

For each layer, builds an (n_gen + 1) x (n_gen + 1) cosine-similarity matrix over
probe weight vectors:
    [ orig post-instruction w , gen-pos-0 w , ... , gen-pos-(n_gen-1) w ]
where `orig` is the post-instruction SVM (train_svm/svm_layerXX.pt) and the
gen-pos probes come from train_svm_generated/svm_layerXX_posKK.pt.

Produces a small-multiples grid (one heatmap per layer) and dumps the raw
matrices to a JSONL for replotting individual layers.
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from analysis._shared import apply_rcparams

apply_rcparams()


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", type=str, default="llama32-3b")
    parser.add_argument("--artifact_dir", type=str, default="./dataset/representations/")
    parser.add_argument("--out_dir", type=str, default="./results/heatmaps")
    parser.add_argument("--n_gen_tokens", type=int, default=5)
    parser.add_argument("--ncols", type=int, default=7, help="Columns in the small-multiples grid")
    return parser.parse_args()


def load_w(path, key="w"):
    obj = torch.load(path, map_location="cpu")
    return obj[key].float().view(-1).numpy()


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def cosine_matrix(vectors):
    """Cosine similarity matrix over a list of vectors."""
    U = np.stack([unit(v) for v in vectors], axis=0)
    return U @ U.T


def build_layer_matrices(args):
    orig_dir = os.path.join(args.artifact_dir, args.model_name, "train_svm")
    gen_dir = os.path.join(args.artifact_dir, args.model_name, "train_svm_generated")

    # Discover layers from the original probes.
    layer_ids = sorted(
        int(n[len("svm_layer"):-len(".pt")])
        for n in os.listdir(orig_dir)
        if n.startswith("svm_layer") and n.endswith(".pt") and n[len("svm_layer"):-len(".pt")].isdigit()
    )
    if not layer_ids:
        raise FileNotFoundError(f"No svm_layerXX.pt found in {orig_dir}")

    matrices = {}
    for l in layer_ids:
        vectors = [load_w(os.path.join(orig_dir, f"svm_layer{l:02d}.pt"))]
        for k in range(args.n_gen_tokens):
            gen_path = os.path.join(gen_dir, f"svm_layer{l:02d}_pos{k:02d}.pt")
            if not os.path.exists(gen_path):
                raise FileNotFoundError(f"Missing generated probe: {gen_path}. Run train_latent_generated.py first.")
            vectors.append(load_w(gen_path))
        matrices[l] = cosine_matrix(vectors)
    return layer_ids, matrices


def make_labels(n_gen):
    return ["orig"] + [f"g{k}" for k in range(n_gen)]


def plot_grid(layer_ids, matrices, labels, out_dir, model_name, ncols):
    n = len(layer_ids)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.0 * nrows))
    axes = np.atleast_1d(axes).ravel()

    im = None
    for ax_idx, l in enumerate(layer_ids):
        ax = axes[ax_idx]
        m = matrices[l]
        im = ax.imshow(m, vmin=-1.0, vmax=1.0, cmap="RdBu_r", aspect="equal")
        ax.set_title(f"L{l}", fontsize=10, pad=3)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)
        ax.tick_params(length=0)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center",
                        color="white" if abs(m[i, j]) > 0.5 else "black", fontsize=5.5)
    for ax_idx in range(len(layer_ids), len(axes)):
        axes[ax_idx].axis("off")

    fig.suptitle(f"Refusal-direction cosine drift across generated positions — {model_name}", y=1.0)
    cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.015, pad=0.01)
    cbar.set_label("cosine similarity")

    stem = f"refusal_drift_grid_{model_name}"
    fig.savefig(os.path.join(out_dir, f"{stem}.png"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, f"{stem}.pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_single(l, matrix, labels, out_dir, model_name):
    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="RdBu_r", aspect="equal")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    color="white" if abs(matrix[i, j]) > 0.5 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="cosine similarity")
    ax.set_title(f"{model_name} — layer {l}")
    stem = f"refusal_drift_layer{l:02d}_{model_name}"
    fig.savefig(os.path.join(out_dir, f"{stem}.png"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, f"{stem}.pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    layer_ids, matrices = build_layer_matrices(args)
    labels = make_labels(args.n_gen_tokens)

    # Dump raw matrices (rounded) to JSONL, one line per layer.
    jsonl_path = os.path.join(args.out_dir, f"refusal_drift_{args.model_name}.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for l in layer_ids:
            m = matrices[l]
            rec = {
                "layer": l,
                "labels": labels,
                "orig_vs_gen": [round(float(x), 4) for x in m[0, 1:]],  # orig vs each gen pos
                "matrix": [[round(float(x), 4) for x in row] for row in m],
            }
            f.write(json.dumps(rec) + "\n")
    print(f"Saved matrices to {jsonl_path}")

    plot_grid(layer_ids, matrices, labels, args.out_dir, args.model_name, args.ncols)
    print(f"Saved grid heatmap to {args.out_dir}")

    # A couple of full annotated single-layer heatmaps for readability.
    for l in (layer_ids[len(layer_ids) // 2], layer_ids[-1]):
        plot_single(l, matrices[l], labels, args.out_dir, args.model_name)

    # Quick summary: mean orig-vs-gen cosine per layer.
    print("\nlayer | orig-vs-gen0 | orig-vs-gen(last) | mean(orig-vs-all-gen)")
    for l in layer_ids:
        row = matrices[l][0, 1:]
        print(f"{l:5d} | {row[0]:+.3f}       | {row[-1]:+.3f}            | {row.mean():+.3f}")


if __name__ == "__main__":
    main()
