"""Per-layer PCA of refusal-direction drift across generated token positions.

For each layer, takes the (n_gen + 1) probe weight vectors
    [ orig post-instruction w , gen-pos-0 w , ... , gen-pos-(n_gen-1) w ],
unit-normalizes each (so PCA describes orientation, not LinearSVC magnitude),
fits a 2-component PCA on just those vectors, and scatters them in PC1-PC2 with
the ordered trajectory orig -> g0 -> ... -> g4 overlaid.

Complements plot_refusal_drift.py: the heatmap shows pairwise similarity, this
shows *how* the direction moves. Each layer has its own PCA basis, so panels are
only comparable within themselves; explained-variance ratios are annotated so
the fraction of drift captured by the 2D view is explicit.
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from analysis._shared import apply_rcparams

apply_rcparams()


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", type=str, default="llama32-3b")
    parser.add_argument("--artifact_dir", type=str, default="./dataset/representations/")
    parser.add_argument("--out_dir", type=str, default="./results/pca")
    parser.add_argument("--n_gen_tokens", type=int, default=5)
    parser.add_argument("--ncols", type=int, default=7, help="Columns in the by-layer grid")
    parser.add_argument(
        "--mode", type=str, default="both", choices=["by_layer", "by_position", "both"],
        help="by_layer: one PCA per layer over the G+1 positions. "
             "by_position: one PCA per position over the L layers.",
    )
    return parser.parse_args()


def load_w(path, key="w"):
    obj = torch.load(path, map_location="cpu")
    return obj[key].float().view(-1).numpy()


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def discover_layers(orig_dir):
    layer_ids = sorted(
        int(n[len("svm_layer"):-len(".pt")])
        for n in os.listdir(orig_dir)
        if n.startswith("svm_layer") and n.endswith(".pt") and n[len("svm_layer"):-len(".pt")].isdigit()
    )
    if not layer_ids:
        raise FileNotFoundError(f"No svm_layerXX.pt found in {orig_dir}")
    return layer_ids


def build_layer_pcas(args):
    """Return layer_ids, {layer: (coords (P+1, 2), explained_var (2,))}."""
    orig_dir = os.path.join(args.artifact_dir, args.model_name, "train_svm")
    gen_dir = os.path.join(args.artifact_dir, args.model_name, "train_svm_generated")
    layer_ids = discover_layers(orig_dir)

    results = {}
    for l in layer_ids:
        vectors = [load_w(os.path.join(orig_dir, f"svm_layer{l:02d}.pt"))]
        for k in range(args.n_gen_tokens):
            gen_path = os.path.join(gen_dir, f"svm_layer{l:02d}_pos{k:02d}.pt")
            if not os.path.exists(gen_path):
                raise FileNotFoundError(f"Missing generated probe: {gen_path}. Run train_latent_generated.py first.")
            vectors.append(load_w(gen_path))

        # Unit-normalize so PCA captures orientation, not LinearSVC magnitude.
        U = np.stack([unit(v) for v in vectors], axis=0)  # (P+1, dim)
        pca = PCA(n_components=2)
        coords = pca.fit_transform(U)                     # (P+1, 2)
        results[l] = (coords, pca.explained_variance_ratio_)
    return layer_ids, results


def build_position_pcas(args):
    """Complementary grouping to build_layer_pcas over the L x (G+1) probe grid.

    build_layer_pcas holds a layer fixed and runs PCA over its G+1 positions;
    here we hold a generated position fixed and run PCA over its L layers.
    Returns pos_labels, {pos_label: (coords (L, 2), explained_var (2,))}, layer_ids.
    """
    orig_dir = os.path.join(args.artifact_dir, args.model_name, "train_svm")
    gen_dir = os.path.join(args.artifact_dir, args.model_name, "train_svm_generated")
    layer_ids = discover_layers(orig_dir)
    pos_labels = make_labels(args.n_gen_tokens)

    results = {}
    for p, pos_label in enumerate(pos_labels):
        vectors = []
        for l in layer_ids:
            if pos_label == "orig":
                path = os.path.join(orig_dir, f"svm_layer{l:02d}.pt")
            else:
                path = os.path.join(gen_dir, f"svm_layer{l:02d}_pos{p - 1:02d}.pt")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing probe: {path}. Run train_latent_generated.py first.")
            vectors.append(load_w(path))

        # Unit-normalize so PCA captures orientation, not LinearSVC magnitude.
        U = np.stack([unit(v) for v in vectors], axis=0)  # (L, dim)
        pca = PCA(n_components=2)
        coords = pca.fit_transform(U)                     # (L, 2)
        results[pos_label] = (coords, pca.explained_variance_ratio_)
    return pos_labels, results, layer_ids


def make_labels(n_gen):
    return ["orig"] + [f"g{k}" for k in range(n_gen)]


def draw_trajectory(ax, coords, labels, annotate=True, markersize=60):
    """Scatter the P+1 points, color by position, connect in order with arrows."""
    n = coords.shape[0]
    # orig = black star; g0..g_last = viridis gradient
    gen_colors = plt.cm.viridis(np.linspace(0, 0.9, n - 1))

    # ordered path (thin grey) with directional arrows
    ax.plot(coords[:, 0], coords[:, 1], "-", color="0.6", lw=1.0, zorder=1)
    for i in range(n - 1):
        ax.annotate(
            "", xy=coords[i + 1], xytext=coords[i],
            arrowprops=dict(arrowstyle="->", color="0.6", lw=1.0), zorder=1,
        )

    ax.scatter(coords[0, 0], coords[0, 1], s=markersize * 1.4, marker="*",
               color="black", zorder=3, label="orig")
    ax.scatter(coords[1:, 0], coords[1:, 1], s=markersize, c=gen_colors, zorder=3)

    if annotate:
        for i, lab in enumerate(labels):
            ax.annotate(lab, coords[i], textcoords="offset points", xytext=(4, 4), fontsize=7)


def plot_grid(layer_ids, results, labels, out_dir, model_name, ncols):
    n = len(layer_ids)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.0 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax_idx, l in enumerate(layer_ids):
        ax = axes[ax_idx]
        coords, evr = results[l]
        draw_trajectory(ax, coords, labels, annotate=True, markersize=35)
        ax.set_title(f"L{l}  (PC1+2: {evr.sum():.0%})", fontsize=9, pad=3)
        ax.set_xlabel(f"PC1 {evr[0]:.0%}", fontsize=7, labelpad=1)
        ax.set_ylabel(f"PC2 {evr[1]:.0%}", fontsize=7, labelpad=1)
        ax.tick_params(labelsize=6, length=2, pad=1)
        ax.axhline(0, color="0.9", lw=0.6, zorder=0)
        ax.axvline(0, color="0.9", lw=0.6, zorder=0)
    for ax_idx in range(len(layer_ids), len(axes)):
        axes[ax_idx].axis("off")

    fig.suptitle(f"Refusal-direction PCA trajectory across generated positions — {model_name}", y=1.0)
    fig.tight_layout()
    stem = f"refusal_pca_grid_{model_name}"
    fig.savefig(os.path.join(out_dir, f"{stem}.png"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, f"{stem}.pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_single(l, coords, evr, labels, out_dir, model_name):
    fig, ax = plt.subplots(figsize=(5, 4.4))
    draw_trajectory(ax, coords, labels, annotate=True, markersize=90)
    ax.set_xlabel(f"PC1 ({evr[0]:.0%} var)")
    ax.set_ylabel(f"PC2 ({evr[1]:.0%} var)")
    ax.set_title(f"{model_name} — layer {l}")
    ax.axhline(0, color="0.9", lw=0.6, zorder=0)
    ax.axvline(0, color="0.9", lw=0.6, zorder=0)
    fig.tight_layout()
    stem = f"refusal_pca_layer{l:02d}_{model_name}"
    fig.savefig(os.path.join(out_dir, f"{stem}.png"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, f"{stem}.pdf"), bbox_inches="tight")
    plt.close(fig)


def draw_layer_path(ax, coords, layer_ids, markersize=40):
    """Scatter the L points colored by layer index, connect in depth order.

    Returns the scatter handle (a ScalarMappable) for building a colorbar.
    """
    ax.plot(coords[:, 0], coords[:, 1], "-", color="0.75", lw=0.8, zorder=1)
    sc = ax.scatter(coords[:, 0], coords[:, 1], s=markersize, c=layer_ids,
                    cmap="viridis", vmin=min(layer_ids), vmax=max(layer_ids), zorder=3)
    # anchor direction with endpoint labels only (28 labels would be unreadable)
    ax.annotate(f"L{layer_ids[0]}", coords[0], textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.annotate(f"L{layer_ids[-1]}", coords[-1], textcoords="offset points", xytext=(4, 4), fontsize=7)
    return sc


def plot_position_grid(pos_labels, results, layer_ids, out_dir, model_name):
    n = len(pos_labels)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows))
    axes = np.atleast_1d(axes).ravel()

    sc = None
    for i, pl in enumerate(pos_labels):
        ax = axes[i]
        coords, evr = results[pl]
        sc = draw_layer_path(ax, coords, layer_ids, markersize=40)
        ax.set_title(f"{pl}  (PC1+2: {evr.sum():.0%})", fontsize=10, pad=3)
        ax.set_xlabel(f"PC1 {evr[0]:.0%}", fontsize=8, labelpad=1)
        ax.set_ylabel(f"PC2 {evr[1]:.0%}", fontsize=8, labelpad=1)
        ax.tick_params(labelsize=7, length=2, pad=1)
        ax.axhline(0, color="0.9", lw=0.6, zorder=0)
        ax.axvline(0, color="0.9", lw=0.6, zorder=0)
    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Refusal-direction PCA across layers, per generated position — {model_name}", y=1.0)
    cbar = fig.colorbar(sc, ax=axes.tolist(), fraction=0.02, pad=0.01)
    cbar.set_label("layer")
    stem = f"refusal_pca_byposition_grid_{model_name}"
    fig.savefig(os.path.join(out_dir, f"{stem}.png"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, f"{stem}.pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_position_single(pos_label, coords, evr, layer_ids, out_dir, model_name):
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    sc = draw_layer_path(ax, coords, layer_ids, markersize=70)
    ax.set_xlabel(f"PC1 ({evr[0]:.0%} var)")
    ax.set_ylabel(f"PC2 ({evr[1]:.0%} var)")
    ax.set_title(f"{model_name} — position {pos_label}")
    ax.axhline(0, color="0.9", lw=0.6, zorder=0)
    ax.axvline(0, color="0.9", lw=0.6, zorder=0)
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="layer")
    fig.tight_layout()
    stem = f"refusal_pca_byposition_{pos_label}_{model_name}"
    fig.savefig(os.path.join(out_dir, f"{stem}.png"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, f"{stem}.pdf"), bbox_inches="tight")
    plt.close(fig)


def run_by_layer(args):
    layer_ids, results = build_layer_pcas(args)
    labels = make_labels(args.n_gen_tokens)

    jsonl_path = os.path.join(args.out_dir, f"refusal_pca_{args.model_name}.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for l in layer_ids:
            coords, evr = results[l]
            rec = {
                "layer": l,
                "labels": labels,
                "coords": [[round(float(x), 4) for x in row] for row in coords],
                "explained_variance_ratio": [round(float(x), 4) for x in evr],
            }
            f.write(json.dumps(rec) + "\n")
    print(f"[by_layer] saved coords to {jsonl_path}")

    plot_grid(layer_ids, results, labels, args.out_dir, args.model_name, args.ncols)
    for l in (layer_ids[len(layer_ids) // 2], layer_ids[-1]):
        coords, evr = results[l]
        plot_single(l, coords, evr, labels, args.out_dir, args.model_name)
    print(f"[by_layer] saved {len(layer_ids)} layer PCAs to {args.out_dir}")

    print("\n[by_layer] layer | PC1 var | PC2 var | PC1+2")
    for l in layer_ids:
        _, evr = results[l]
        print(f"        {l:5d} | {evr[0]:.3f}   | {evr[1]:.3f}   | {evr.sum():.3f}")


def run_by_position(args):
    pos_labels, results, layer_ids = build_position_pcas(args)

    jsonl_path = os.path.join(args.out_dir, f"refusal_pca_byposition_{args.model_name}.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for pl in pos_labels:
            coords, evr = results[pl]
            rec = {
                "position": pl,
                "layers": layer_ids,
                "coords": [[round(float(x), 4) for x in row] for row in coords],
                "explained_variance_ratio": [round(float(x), 4) for x in evr],
            }
            f.write(json.dumps(rec) + "\n")
    print(f"[by_position] saved coords to {jsonl_path}")

    plot_position_grid(pos_labels, results, layer_ids, args.out_dir, args.model_name)
    for pl in pos_labels:
        coords, evr = results[pl]
        plot_position_single(pl, coords, evr, layer_ids, args.out_dir, args.model_name)
    print(f"[by_position] saved {len(pos_labels)} position PCAs to {args.out_dir}")

    print("\n[by_position] position | PC1 var | PC2 var | PC1+2")
    for pl in pos_labels:
        _, evr = results[pl]
        print(f"           {pl:>8} | {evr[0]:.3f}   | {evr[1]:.3f}   | {evr.sum():.3f}")


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.mode in ("by_layer", "both"):
        run_by_layer(args)
    if args.mode in ("by_position", "both"):
        run_by_position(args)


if __name__ == "__main__":
    main()
