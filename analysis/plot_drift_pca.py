"""Overlay the generated-token probe boundaries on the post-instruction PCA scatter.

Companion to plot_pca.py. The PCA plane and the scatter are the SAME as there: a
2-component PCA fit on the post-instruction harmful/harmless activations
(train_svm/HFx_train.pt, HLx_train.pt). On top of that fixed view we draw the
decision boundary of every probe as a line (see _shared.project_probe). Lines:
  - orig  : the post-instruction probe (train_svm/svm_layerXX.pt), thick black,
  - g0..g4: the generated-position probes (train_svm_generated/svm_layerXX_posKK.pt),
            coloured dark -> bright by generated position.

This shows how much the harmful/harmless separating boundary rotates across generated
tokens, relative to the original post-instruction boundary, in a single fixed frame.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from analysis._shared import (
    HARMFUL_COLOR, HARMLESS_COLOR, apply_rcparams, closest_anchor, gen_colors,
    load_probe, oriented_pca, project_probe, sigmoid,
)

apply_rcparams()


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", type=str, default="llama32-3b")
    parser.add_argument("--artifact_dir", type=str, default="./dataset/representations/")
    parser.add_argument("--out_dir", type=str, default="./results/probe_boundary")
    parser.add_argument("--layers", type=int, nargs="+", default=[4, 14, 24],
                        help="Early / middle / late layer indices to plot")
    parser.add_argument("--n_gen_tokens", type=int, default=5)
    return parser.parse_args()


def draw_panel(ax, args, layer, grid=400):
    base = os.path.join(args.artifact_dir, args.model_name, "train_svm")
    gen_base = os.path.join(args.artifact_dir, args.model_name, "train_svm_generated")

    hf = torch.load(os.path.join(base, "HFx_train.pt"), map_location="cpu").float().numpy()[:, layer, :]
    hl = torch.load(os.path.join(base, "HLx_train.pt"), map_location="cpu").float().numpy()[:, layer, :]

    X = np.concatenate([hf, hl], axis=0)
    nH = len(hf)
    w0, b0, _ = load_probe(os.path.join(base, f"svm_layer{layer:02d}.pt"))
    pca, V, Z = oriented_pca(X, w0, nH)
    mu = pca.mean_
    Z_harm, Z_harmless = Z[:nH], Z[nH:]
    centroid = Z.mean(axis=0)

    # Collect every probe boundary as (grad, offset, color, lw, zorder): the line on
    # the plane is grad . z + offset = 0. Generated first (dark->bright), orig last.
    colors = gen_colors(args.n_gen_tokens)
    boundaries = []
    for k in range(args.n_gen_tokens):
        w, b, _ = load_probe(os.path.join(gen_base, f"svm_layer{layer:02d}_pos{k:02d}.pt"))
        grad, offset = project_probe(V, mu, w, b)
        boundaries.append((grad, offset, colors[k], 1.6, 3))
    grad0, offset0 = project_probe(V, mu, w0, b0)
    boundaries.append((grad0, offset0, "black", 2.4, 4))

    # Expand the frame so every boundary is visible: include each line's closest point
    # to the cloud centroid (the scatter may shrink when a boundary sits far off-cloud).
    anchors = [closest_anchor(centroid, grad, offset) for grad, offset, *_ in boundaries]
    pts = np.vstack([Z] + [a[None, :] for a in anchors])
    pad_x = 0.06 * (pts[:, 0].max() - pts[:, 0].min())
    pad_y = 0.06 * (pts[:, 1].max() - pts[:, 1].min())
    xlim = (pts[:, 0].min() - pad_x, pts[:, 0].max() + pad_x)
    ylim = (pts[:, 1].min() - pad_y, pts[:, 1].max() + pad_y)
    gx, gy = np.meshgrid(np.linspace(*xlim, grid), np.linspace(*ylim, grid))

    # Background = compliance-confidence field of the ORIGINAL probe only (sigmoid of
    # its score; harmful=1 -> flip so red=harmful/low-compliance, blue=harmless/high).
    conf = sigmoid(grad0[0] * gx + grad0[1] * gy + offset0)
    mesh = ax.pcolormesh(gx, gy, 1.0 - conf, cmap="RdBu", vmin=0.0, vmax=1.0,
                         shading="auto", zorder=0, rasterized=True, alpha=0.45)

    ax.scatter(Z_harm[:, 0], Z_harm[:, 1], s=16, alpha=0.55, color=HARMFUL_COLOR,
               edgecolors="none", zorder=1)
    ax.scatter(Z_harmless[:, 0], Z_harmless[:, 1], s=16, alpha=0.55, color=HARMLESS_COLOR,
               edgecolors="none", zorder=1)

    for grad, offset, color, lw, zorder in boundaries:
        score = grad[0] * gx + grad[1] * gy + offset
        ax.contour(gx, gy, score, levels=[0.0], colors=[color], linewidths=lw, zorder=zorder)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({ev[0]*100:.0f}%)")
    ax.set_ylabel(f"PC2 ({ev[1]*100:.0f}%)")
    ax.set_title(f"Layer {layer}", fontsize=12)
    return mesh


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    n = len(args.layers)
    fig, axes = plt.subplots(1, n, figsize=(4.9 * n, 4.6), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    mesh = None
    for ax, layer in zip(axes, args.layers):
        mesh = draw_panel(ax, args, layer)
        print(f"Layer {layer:2d} drawn")

    cbar = fig.colorbar(mesh, ax=axes.tolist(), fraction=0.02, pad=0.01)
    cbar.set_label("Compliance confidence (orig probe)")

    colors = gen_colors(args.n_gen_tokens)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="none", color=HARMFUL_COLOR, label="harmful"),
        plt.Line2D([], [], marker="o", linestyle="none", color=HARMLESS_COLOR, label="harmless"),
        plt.Line2D([], [], color="black", lw=2.4, label="orig boundary"),
    ]
    handles += [plt.Line2D([], [], color=colors[k], lw=1.6, label=f"g{k}")
                for k in range(args.n_gen_tokens)]
    fig.legend(handles=handles, frameon=False, ncol=len(handles), loc="lower center",
               bbox_to_anchor=(0.5, 1.02))

    stem = f"probe_boundary_drift_pca_{args.model_name}"
    fig.savefig(os.path.join(args.out_dir, f"{stem}.png"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(args.out_dir, f"{stem}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {stem}.png/.pdf to {args.out_dir}")


if __name__ == "__main__":
    main()
