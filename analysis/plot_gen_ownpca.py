"""Per-token Figure-2: each generated position in ITS OWN PCA space + its own probe.

A (G+1) x L grid. Column = layer (early / middle / late). Unlike
plot_gen_scatter_pca.py (which forces every column onto the fixed post-instruction
plane and so only shows the g-probe's off-plane shadow), here EACH panel gets a fresh
2-component PCA fit on that panel's own activations, and we shade that panel's own
probe as the compliance field -- exactly the recipe of plot_pca.py, repeated per
generated position.

Rows:
  - row 0            : post-instruction activations + post-instruction probe
                       (identical to plot_pca.py),
  - row k+1 (k=0..G-1): the k-th generated-token activations, their own PCA plane,
                       shaded by that position's probe (train_svm_generated/
                       svm_layerXX_posKK.pt), with the original post-instruction
                       probe w_orig overlaid (dashed) as a reference.

This shows how well each token's OWN probe separates its OWN activations, and how the
fixed original boundary compares once dropped into that same plane.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from analysis._shared import (
    HARMFUL_COLOR, HARMLESS_COLOR, ORIG_BOUNDARY_COLOR, SELF_BOUNDARY_COLOR,
    apply_rcparams, closest_anchor, load_probe, oriented_pca, project_probe, sigmoid,
)

apply_rcparams(tick_labelsize=10)


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", type=str, default="llama32-3b")
    parser.add_argument("--artifact_dir", type=str, default="./dataset/representations/")
    parser.add_argument("--out_dir", type=str, default="./results/probe_boundary")
    parser.add_argument("--layers", type=int, nargs="+", default=[4, 14, 24])
    parser.add_argument("--n_gen_tokens", type=int, default=5)
    return parser.parse_args()


def draw_panel(ax, HF, HL, self_probe, orig_probe, is_gen, grid=400):
    """HF/HL: (N, dim) activations for this panel. *_probe: (w, b)."""
    w_self, b_self = self_probe
    w_orig, b_orig = orig_probe

    X = np.concatenate([HF, HL], axis=0)
    nH = len(HF)
    # Orient on this panel's own probe so red/harmful reads left in every panel.
    pca, V, Z = oriented_pca(X, w_self, nH)
    Z_harm, Z_harmless = Z[:nH], Z[nH:]
    centroid = Z.mean(axis=0)

    # Each probe as a line grad.z + offset = 0 on THIS panel's (oriented) plane.
    grad_s, off_s = project_probe(V, pca.mean_, w_self, b_self)
    grad_o, off_o = project_probe(V, pca.mean_, w_orig, b_orig)

    # Frame: cloud + the orig line's closest point, so the black line stays visible.
    orig_anchor = closest_anchor(centroid, grad_o, off_o)
    pts = np.vstack([Z, orig_anchor[None, :]])
    pad_x = 0.07 * (pts[:, 0].max() - pts[:, 0].min())
    pad_y = 0.07 * (pts[:, 1].max() - pts[:, 1].min())
    xlim = (pts[:, 0].min() - pad_x, pts[:, 0].max() + pad_x)
    ylim = (pts[:, 1].min() - pad_y, pts[:, 1].max() + pad_y)
    gx, gy = np.meshgrid(np.linspace(*xlim, grid), np.linspace(*ylim, grid))

    # Background = compliance field of THIS panel's probe (flip so red = harmful).
    conf = sigmoid(grad_s[0] * gx + grad_s[1] * gy + off_s)
    mesh = ax.pcolormesh(gx, gy, 1.0 - conf, cmap="RdBu", vmin=0.0, vmax=1.0,
                         shading="auto", zorder=0, rasterized=True, alpha=0.45)
    ax.contour(gx, gy, 1.0 - conf, levels=[0.1, 0.25, 0.75, 0.9],
               colors="0.4", linewidths=0.6, zorder=1)

    ax.scatter(Z_harm[:, 0], Z_harm[:, 1], s=15, alpha=0.7, color=HARMFUL_COLOR,
               edgecolors="none", zorder=2)
    ax.scatter(Z_harmless[:, 0], Z_harmless[:, 1], s=15, alpha=0.7, color=HARMLESS_COLOR,
               edgecolors="none", zorder=2)

    # Original probe: solid black, always. On the first row the panel's own probe IS
    # the original, so black is the only line; on gen rows we add the panel's own probe
    # as dashed purple to show how far the fixed original has drifted.
    score_o = grad_o[0] * gx + grad_o[1] * gy + off_o
    ax.contour(gx, gy, score_o, levels=[0.0], colors=[ORIG_BOUNDARY_COLOR],
               linewidths=2.0, zorder=4)
    if is_gen:
        score_s = grad_s[0] * gx + grad_s[1] * gy + off_s
        ax.contour(gx, gy, score_s, levels=[0.0], colors=[SELF_BOUNDARY_COLOR],
                   linewidths=2.0, linestyles="--", zorder=5)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(length=0)
    return mesh, pca.explained_variance_ratio_.sum()


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    base = os.path.join(args.artifact_dir, args.model_name, "train_svm")
    gen_base = os.path.join(args.artifact_dir, args.model_name, "train_svm_generated")

    hf_post = torch.load(os.path.join(base, "HFx_train.pt"), map_location="cpu").float().numpy()
    hl_post = torch.load(os.path.join(base, "HLx_train.pt"), map_location="cpu").float().numpy()
    gen = torch.load(os.path.join(gen_base, "gen_activations.pt"), map_location="cpu")
    gen_layers = list(gen["layers"])
    n_gen = min(args.n_gen_tokens, int(gen["n_gen"]))

    # Columns = layers (as in the other plots); rows = post-instruction then g0..gG-1.
    n_rows, n_cols = n_gen + 1, len(args.layers)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.3 * n_rows),
                             constrained_layout=True)
    axes = np.atleast_2d(axes)

    mesh = None
    for c, layer in enumerate(args.layers):
        li = gen_layers.index(layer)
        w0, b0, acc0 = load_probe(os.path.join(base, f"svm_layer{layer:02d}.pt"))
        hf_gen = gen["HF"][:, :n_gen, li, :].float().numpy()
        hl_gen = gen["HL"][:, :n_gen, li, :].float().numpy()

        for r in range(n_rows):
            ax = axes[r, c]
            if r == 0:
                HF, HL = hf_post[:, layer, :], hl_post[:, layer, :]
                mesh, var = draw_panel(ax, HF, HL, (w0, b0), (w0, b0), is_gen=False)
                acc = acc0
            else:
                k = r - 1
                wk, bk, acck = load_probe(os.path.join(gen_base, f"svm_layer{layer:02d}_pos{k:02d}.pt"))
                HF, HL = hf_gen[:, k, :], hl_gen[:, k, :]
                mesh, var = draw_panel(ax, HF, HL, (wk, bk), (w0, b0), is_gen=True)
                acc = acck

            ax.annotate(f"var {var * 100:.0f}%  ·  acc {acc:.2f}", xy=(0.03, 0.97),
                        xycoords="axes fraction", ha="left", va="top", fontsize=8, color="0.25")
            if r == 0:
                ax.set_title(f"Layer {layer}", fontsize=12)
            if c == 0:
                row_label = "post-instruction" if r == 0 else f"gen token g{r - 1}"
                ax.set_ylabel(f"{row_label}\nPC2")
            if r == n_rows - 1:
                ax.set_xlabel("PC1")
        print(f"Layer {layer:2d} drawn")

    cbar = fig.colorbar(mesh, ax=axes.ravel().tolist(), fraction=0.012, pad=0.01)
    cbar.set_label("Compliance confidence (this-panel probe)")

    handles = [
        plt.Line2D([], [], marker="o", linestyle="none", color=HARMFUL_COLOR, label="harmful"),
        plt.Line2D([], [], marker="o", linestyle="none", color=HARMLESS_COLOR, label="harmless"),
        plt.Line2D([], [], color=ORIG_BOUNDARY_COLOR, lw=2.0, label="orig probe (w$_0$)"),
        plt.Line2D([], [], color=SELF_BOUNDARY_COLOR, lw=2.0, ls="--", label="this-panel probe"),
    ]
    fig.legend(handles=handles, frameon=False, ncol=len(handles), loc="lower center",
               bbox_to_anchor=(0.5, 1.005))

    stem = f"probe_boundary_gen_ownpca_{args.model_name}"
    fig.savefig(os.path.join(args.out_dir, f"{stem}.png"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(args.out_dir, f"{stem}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {stem}.png/.pdf to {args.out_dir}")


if __name__ == "__main__":
    main()
