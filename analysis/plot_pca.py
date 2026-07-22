"""Recreate the paper's Figure 2: harmful/harmless clusters + probe boundary in PCA space.

For each requested layer:
  - stack the post-instruction harmful/harmless activations (train_svm/HFx_train.pt,
    HLx_train.pt) and fit a 2-component PCA on them,
  - scatter the projected points (harmful vs harmless),
  - overlay the post-instruction SVM probe (train_svm/svm_layerXX.pt) as a decision
    boundary with a "compliance-confidence" background gradient.

The probe is a hyperplane w.x + b = 0 in the full hidden space; on the PCA plane it
becomes a straight line (see _shared.project_probe). We shade sigmoid of its score so
the boundary sits at 0.5. PCA axis signs are pinned (see _shared.oriented_pca) so the
layers read consistently: harmful/red left, harmless/blue right.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from analysis._shared import (
    HARMFUL_COLOR, HARMLESS_COLOR, apply_rcparams, load_probe, oriented_pca,
    project_probe, sigmoid,
)

apply_rcparams()


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", type=str, default="llama32-3b")
    parser.add_argument("--artifact_dir", type=str, default="./dataset/representations/")
    parser.add_argument("--out_dir", type=str, default="./results/probe_boundary")
    parser.add_argument("--layers", type=int, nargs="+", default=[4, 14, 24],
                        help="Early / middle / late layer indices to plot")
    return parser.parse_args()


def load_layer(artifact_dir, model_name, layer):
    """Return (X_harm, X_harmless, w, b, acc) for one layer."""
    base = os.path.join(artifact_dir, model_name, "train_svm")
    hf = torch.load(os.path.join(base, "HFx_train.pt"), map_location="cpu").float().numpy()
    hl = torch.load(os.path.join(base, "HLx_train.pt"), map_location="cpu").float().numpy()
    w, b, acc = load_probe(os.path.join(base, f"svm_layer{layer:02d}.pt"))
    return hf[:, layer, :], hl[:, layer, :], w, b, acc


def draw_panel(ax, X_harm, X_harmless, w, b, layer, acc, grid=300):
    X = np.concatenate([X_harm, X_harmless], axis=0)
    nH = len(X_harm)
    pca, V, Z = oriented_pca(X, w, nH)
    Z_harm, Z_harmless = Z[:nH], Z[nH:]

    # Probe as a linear function on the (oriented) PCA plane.
    grad, offset = project_probe(V, pca.mean_, w, b)

    # Background gradient of compliance confidence = sigmoid(probe score).
    pad_x = 0.08 * (Z[:, 0].max() - Z[:, 0].min())
    pad_y = 0.08 * (Z[:, 1].max() - Z[:, 1].min())
    xs = np.linspace(Z[:, 0].min() - pad_x, Z[:, 0].max() + pad_x, grid)
    ys = np.linspace(Z[:, 1].min() - pad_y, Z[:, 1].max() + pad_y, grid)
    gx, gy = np.meshgrid(xs, ys)
    score = grad[0] * gx + grad[1] * gy + offset
    conf = sigmoid(score)

    # Harmful should read as low compliance (red), harmless as high (blue); the SVM
    # labels harmful=1, so a positive score means harmful -> flip for "compliance".
    mesh = ax.pcolormesh(gx, gy, 1.0 - conf, cmap="RdBu", vmin=0.0, vmax=1.0,
                         shading="auto", zorder=0, rasterized=True, alpha=0.45)
    ax.contour(gx, gy, 1.0 - conf, levels=[0.1, 0.25, 0.75, 0.9],
               colors="0.4", linewidths=0.6, zorder=1)
    ax.contour(gx, gy, 1.0 - conf, levels=[0.5], colors="0.15", linewidths=1.3, zorder=1)

    ax.scatter(Z_harm[:, 0], Z_harm[:, 1], s=18, alpha=0.75, color=HARMFUL_COLOR,
               edgecolors="none", label="harmful", zorder=2)
    ax.scatter(Z_harmless[:, 0], Z_harmless[:, 1], s=18, alpha=0.75, color=HARMLESS_COLOR,
               edgecolors="none", label="harmless", zorder=2)

    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({ev[0]*100:.0f}%)")
    ax.set_ylabel(f"PC2 ({ev[1]*100:.0f}%)")
    ax.set_title(f"Layer {layer}  (probe acc {acc:.2f})", fontsize=12)
    return mesh, ev.sum()


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    n = len(args.layers)
    fig, axes = plt.subplots(1, n, figsize=(4.9 * n, 4.6), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    mesh = None
    for ax, layer in zip(axes, args.layers):
        X_harm, X_harmless, w, b, acc = load_layer(args.artifact_dir, args.model_name, layer)
        mesh, var = draw_panel(ax, X_harm, X_harmless, w, b, layer, acc)
        print(f"Layer {layer:2d} | PC1+PC2 var = {var*100:.1f}% | probe acc = {acc:.3f}")

    handles = [
        plt.Line2D([], [], marker="o", linestyle="none", color=HARMFUL_COLOR, label="harmful"),
        plt.Line2D([], [], marker="o", linestyle="none", color=HARMLESS_COLOR, label="harmless"),
    ]
    fig.legend(handles=handles, frameon=False, ncol=2, loc="lower center",
               bbox_to_anchor=(0.5, 1.02))
    cbar = fig.colorbar(mesh, ax=axes.tolist(), fraction=0.02, pad=0.01)
    cbar.set_label("Compliance confidence")

    stem = f"probe_boundary_pca_{args.model_name}"
    fig.savefig(os.path.join(args.out_dir, f"{stem}.png"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(args.out_dir, f"{stem}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {stem}.png/.pdf to {args.out_dir}")


if __name__ == "__main__":
    main()
