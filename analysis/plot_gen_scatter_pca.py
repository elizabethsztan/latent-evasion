"""Generated-token activations drifting across a FIXED post-instruction PCA plane.

Companion to plot_gen_ownpca.py and formatted to match it, but the key difference is
the plane: here every panel in a column shares ONE PCA fit on that layer's
post-instruction activations (train_svm/HFx_train.pt, HLx_train.pt). Because the axes
and limits are held fixed down a column, you can watch the harmful/harmless clusters
move as tokens are generated.

Grid: columns = layers (early / middle / late), rows = post-instruction then g0..gG-1.
Lines:
  - orig probe (post-instruction, train_svm/svm_layerXX.pt): solid black, always,
  - the generated position's probe (train_svm_generated/svm_layerXX_posKK.pt):
    dashed purple, on the gen rows only.
The gen-probe line is the *projection* of a 3072-D hyperplane onto the fixed
post-instruction plane, so it can look like it does not separate the points -- that
off-plane shadow is exactly the point of this view (compare plot_gen_ownpca.py, which
instead fits each panel's own plane so every probe separates cleanly).

Background = the original probe's compliance-confidence field, the fixed reference
every panel is read against. Generated activations come from gen_activations.pt
(extract_generated_activations.py).
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from analysis._shared import (
    GEN_BOUNDARY_COLOR, HARMFUL_COLOR, HARMLESS_COLOR, ORIG_BOUNDARY_COLOR,
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
        nH = len(hf_post)

        # ONE PCA per column, fit on this layer's post-instruction activations.
        Xpost = np.concatenate([hf_post[:, layer, :], hl_post[:, layer, :]], axis=0)
        w0, b0, acc0 = load_probe(os.path.join(base, f"svm_layer{layer:02d}.pt"))
        pca, V, Zpost = oriented_pca(Xpost, w0, nH)
        mu = pca.mean_
        Zp_h, Zp_l = Zpost[:nH], Zpost[nH:]
        centroid = Zpost.mean(axis=0)
        ev = pca.explained_variance_ratio_.sum()

        grad_o, off_o = project_probe(V, mu, w0, b0)

        # Project each generated position onto the SAME fixed plane; load its probe.
        hf_gen = gen["HF"][:, :n_gen, li, :].float().numpy()
        hl_gen = gen["HL"][:, :n_gen, li, :].float().numpy()
        Zg_h = [(hf_gen[:, k, :] - mu) @ V.T for k in range(n_gen)]
        Zg_l = [(hl_gen[:, k, :] - mu) @ V.T for k in range(n_gen)]
        gen_probes = []
        for k in range(n_gen):
            wk, bk, acck = load_probe(os.path.join(gen_base, f"svm_layer{layer:02d}_pos{k:02d}.pt"))
            gen_probes.append((*project_probe(V, mu, wk, bk), acck))

        # Shared column limits: all points + every boundary's closest point, so the
        # (possibly far-off-cloud) gen shadows stay visible even if points shrink.
        anchors = [closest_anchor(centroid, grad_o, off_o)]
        anchors += [closest_anchor(centroid, g, o) for g, o, _ in gen_probes]
        pts = np.vstack([Zpost] + Zg_h + Zg_l + [a[None, :] for a in anchors])
        pad_x = 0.06 * (pts[:, 0].max() - pts[:, 0].min())
        pad_y = 0.06 * (pts[:, 1].max() - pts[:, 1].min())
        xlim = (pts[:, 0].min() - pad_x, pts[:, 0].max() + pad_x)
        ylim = (pts[:, 1].min() - pad_y, pts[:, 1].max() + pad_y)
        gx, gy = np.meshgrid(np.linspace(*xlim, 400), np.linspace(*ylim, 400))

        # Background = original probe compliance field (flip so red = harmful).
        conf = sigmoid(grad_o[0] * gx + grad_o[1] * gy + off_o)

        for r in range(n_rows):
            ax = axes[r, c]
            mesh = ax.pcolormesh(gx, gy, 1.0 - conf, cmap="RdBu", vmin=0.0, vmax=1.0,
                                 shading="auto", zorder=0, rasterized=True, alpha=0.45)
            ax.contour(gx, gy, 1.0 - conf, levels=[0.1, 0.25, 0.75, 0.9],
                       colors="0.4", linewidths=0.6, zorder=1)

            if r == 0:
                Zh, Zl, acc = Zp_h, Zp_l, acc0
            else:
                Zh, Zl, acc = Zg_h[r - 1], Zg_l[r - 1], gen_probes[r - 1][2]
            ax.scatter(Zh[:, 0], Zh[:, 1], s=15, alpha=0.7, color=HARMFUL_COLOR,
                       edgecolors="none", zorder=2)
            ax.scatter(Zl[:, 0], Zl[:, 1], s=15, alpha=0.7, color=HARMLESS_COLOR,
                       edgecolors="none", zorder=2)

            # Orig probe solid black always; the gen-position probe dashed purple.
            ax.contour(gx, gy, grad_o[0] * gx + grad_o[1] * gy + off_o, levels=[0.0],
                       colors=[ORIG_BOUNDARY_COLOR], linewidths=2.0, zorder=4)
            if r >= 1:
                g, o, _ = gen_probes[r - 1]
                ax.contour(gx, gy, g[0] * gx + g[1] * gy + o, levels=[0.0],
                           colors=[GEN_BOUNDARY_COLOR], linewidths=2.0, linestyles="--", zorder=5)

            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(length=0)
            ax.annotate(f"var {ev * 100:.0f}%  ·  acc {acc:.2f}", xy=(0.03, 0.97),
                        xycoords="axes fraction", ha="left", va="top", fontsize=8, color="0.25")
            if r == 0:
                ax.set_title(f"Layer {layer}", fontsize=12)
            if c == 0:
                row_label = "post-instruction" if r == 0 else f"gen token g{r - 1}"
                ax.set_ylabel(f"{row_label}\nPC2")
            if r == n_rows - 1:
                ax.set_xlabel("PC1")
        print(f"Layer {layer:2d} drawn | PC1+PC2 var = {ev * 100:.1f}%")

    cbar = fig.colorbar(mesh, ax=axes.ravel().tolist(), fraction=0.012, pad=0.01)
    cbar.set_label("Compliance confidence (orig probe)")

    handles = [
        plt.Line2D([], [], marker="o", linestyle="none", color=HARMFUL_COLOR, label="harmful"),
        plt.Line2D([], [], marker="o", linestyle="none", color=HARMLESS_COLOR, label="harmless"),
        plt.Line2D([], [], color=ORIG_BOUNDARY_COLOR, lw=2.0, label="orig probe (w$_0$)"),
        plt.Line2D([], [], color=GEN_BOUNDARY_COLOR, lw=2.0, ls="--", label="gen-token probe"),
    ]
    fig.legend(handles=handles, frameon=False, ncol=len(handles), loc="lower center",
               bbox_to_anchor=(0.5, 1.005))

    stem = f"probe_boundary_gen_scatter_pca_{args.model_name}"
    fig.savefig(os.path.join(args.out_dir, f"{stem}.png"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(args.out_dir, f"{stem}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {stem}.png/.pdf to {args.out_dir}")


if __name__ == "__main__":
    main()
