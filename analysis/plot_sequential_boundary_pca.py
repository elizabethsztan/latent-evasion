"""Compare old (CLE-A/P/G) vs new (CLE-S) probe boundaries against harmful/harmless
activations, per layer -- the CLE-S analogue of plot_pca.py.

For each requested layer in the CLE-S prompt chain:
  - harmful (unsteered): train_svm/HFx_train.pt, the same prompts CLE-S trains on,
    with no steering applied,
  - harmless: train_svm/HLx_train.pt,
  - harmful (steered, pre-L): the same harmful prompts re-encoded and pushed through
    the CLE-S hook cascade for every chain layer strictly below this one -- i.e.
    exactly the input distribution this layer's probe was trained on (see
    classifier.sequential_utils.train_layer_chain),
  - harmful (steered, post-L): the same, but with layer L's own hook also applied --
    i.e. what this layer's probe actually does to its own training input,
  - old probe: train_svm/svm_layerXX.pt (CLE-A/P/G, trained independently per layer),
  - new probe: train_svm_sequential/svm_layerXX.pt (CLE-S, trained on the cascaded
    input).

A faint grey line connects each prompt's pre-L and post-L points. Both probes are
drawn as decision-boundary lines on a single PCA plane per layer (fit across all
four point clouds, oriented by the new probe), following the recipe in plot_pca.py
/ analysis._shared.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from analysis._shared import (
    HARMFUL_COLOR, HARMLESS_COLOR, ORIG_BOUNDARY_COLOR, SELF_BOUNDARY_COLOR,
    STEERED_COLOR, STEERED_POST_COLOR, apply_rcparams, cascade_pre_post, load_probe,
    oriented_pca, project_probe, sigmoid,
)
from classifier.sequential_utils import encode_prompt, parse_layer_range
from classifier.utils import get_model, load_arditi, pick_device
from utils.models_utils import get_transformer_layers
from utils.probes import load_svms

apply_rcparams()


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", type=str, default="llama32-3b")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--artifact_dir", type=str, default="./dataset/representations/")
    parser.add_argument("--out_dir", type=str, default="./results/probe_boundary")
    parser.add_argument(
        "--chain_layers", type=str, default="11-23",
        help="Contiguous end-exclusive range the CLE-S chain was trained over.",
    )
    parser.add_argument(
        "--plot_layers", type=int, nargs="+", default=None,
        help="Subset of --chain_layers to draw a panel for (default: 4 evenly spaced).",
    )
    parser.add_argument(
        "--n_samples", type=int, default=-1,
        help="Harmful prompts to cascade through the chain (-1 = all).",
    )
    parser.add_argument(
        "--hide_post", action="store_true",
        help="Drop the post-L cloud from both the PCA fit and the plot (debug: "
             "compare against the pre-L-only 3-cloud framing).",
    )
    return parser.parse_args()


def compute_steered_harmful(model, layer_modules, chain_layers, probes, harmful_ids_list, plot_layers, beta):
    """Return {layer: {"pre": ..., "post": ...}} (N, dim) np.ndarray pairs: harmful
    reps cascaded through every chain layer strictly below `layer` ("pre", matching
    train_layer_chain's training input) and through `layer` itself too ("post", what
    that layer's own probe actually does to its training input)."""
    steered = {}
    for target_layer in plot_layers:
        lower_layers = [l for l in chain_layers if l < target_layer]
        pre, post = cascade_pre_post(
            model, layer_modules, lower_layers, target_layer, probes, beta,
            probes[target_layer]["margin"], harmful_ids_list,
        )
        steered[target_layer] = {"pre": pre, "post": post}
        print(f"  layer {target_layer:2d}: cascaded through {len(lower_layers)} prior hook(s)")
    return steered


def draw_panel(ax, X_harm, X_harmless, X_pre, X_post, w_old, b_old, w_new, b_new,
                layer, acc_old, acc_new, grid=300, show_post=True):
    clouds = [X_harm, X_pre, X_post, X_harmless] if show_post else [X_harm, X_pre, X_harmless]
    X = np.concatenate(clouds, axis=0)
    n_harm_total = len(X_harm) + len(X_pre) + (len(X_post) if show_post else 0)
    pca, V, Z = oriented_pca(X, w_new, n_harm_total)

    n0 = len(X_harm)
    n1 = n0 + len(X_pre)
    Z_harm = Z[:n0]
    Z_pre = Z[n0:n1]
    Z_post = Z[n1:n_harm_total] if show_post else None
    Z_harmless = Z[n_harm_total:]

    grad_new, off_new = project_probe(V, pca.mean_, w_new, b_new)
    grad_old, off_old = project_probe(V, pca.mean_, w_old, b_old)

    pad_x = 0.08 * (Z[:, 0].max() - Z[:, 0].min())
    pad_y = 0.08 * (Z[:, 1].max() - Z[:, 1].min())
    xs = np.linspace(Z[:, 0].min() - pad_x, Z[:, 0].max() + pad_x, grid)
    ys = np.linspace(Z[:, 1].min() - pad_y, Z[:, 1].max() + pad_y, grid)
    gx, gy = np.meshgrid(xs, ys)

    # Background = compliance field of the NEW (CLE-S) probe, since that's what the
    # steered points are optimized against.
    conf = sigmoid(grad_new[0] * gx + grad_new[1] * gy + off_new)
    mesh = ax.pcolormesh(gx, gy, 1.0 - conf, cmap="RdBu", vmin=0.0, vmax=1.0,
                         shading="auto", zorder=0, rasterized=True, alpha=0.4)
    ax.contour(gx, gy, 1.0 - conf, levels=[0.1, 0.25, 0.75, 0.9],
               colors="0.4", linewidths=0.5, zorder=1)

    if show_post:
        # Faint displacement lines: pre-L steered -> post-L steered, same prompt.
        for (x0, y0), (x1, y1) in zip(Z_pre, Z_post):
            ax.plot([x0, x1], [y0, y1], color="0.6", linewidth=0.5, alpha=0.5, zorder=1)

    ax.scatter(Z_harm[:, 0], Z_harm[:, 1], s=16, alpha=0.55, color=HARMFUL_COLOR,
               edgecolors="none", zorder=2)
    ax.scatter(Z_harmless[:, 0], Z_harmless[:, 1], s=16, alpha=0.7, color=HARMLESS_COLOR,
               edgecolors="none", zorder=2)
    ax.scatter(Z_pre[:, 0], Z_pre[:, 1], s=16, alpha=0.85, color=STEERED_COLOR,
               edgecolors="none", zorder=3)
    if show_post:
        ax.scatter(Z_post[:, 0], Z_post[:, 1], s=16, alpha=0.85, color=STEERED_POST_COLOR,
                   edgecolors="none", zorder=3)

    score_old = grad_old[0] * gx + grad_old[1] * gy + off_old
    score_new = grad_new[0] * gx + grad_new[1] * gy + off_new
    ax.contour(gx, gy, score_old, levels=[0.0], colors=[ORIG_BOUNDARY_COLOR],
               linewidths=2.0, zorder=4)
    ax.contour(gx, gy, score_new, levels=[0.0], colors=[SELF_BOUNDARY_COLOR],
               linewidths=2.0, linestyles="--", zorder=5)

    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({ev[0]*100:.0f}%)")
    ax.set_ylabel(f"PC2 ({ev[1]*100:.0f}%)")
    ax.set_title(f"Layer {layer}  (old acc {acc_old:.2f} · new acc {acc_new:.2f})", fontsize=11)
    return mesh, ev.sum()


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    chain_layers = parse_layer_range(args.chain_layers)

    plot_layers = args.plot_layers or [
        chain_layers[i] for i in np.linspace(0, len(chain_layers) - 1, 4).round().astype(int)
    ]
    plot_layers = sorted(set(plot_layers))
    for layer in plot_layers:
        if layer not in chain_layers:
            raise ValueError(f"--plot_layers {layer} is outside --chain_layers {chain_layers}")

    old_base = os.path.join(args.artifact_dir, args.model_name, "train_svm")
    new_base = os.path.join(args.artifact_dir, args.model_name, "train_svm_sequential")
    hf_post = torch.load(os.path.join(old_base, "HFx_train.pt"), map_location="cpu").float().numpy()
    hl_post = torch.load(os.path.join(old_base, "HLx_train.pt"), map_location="cpu").float().numpy()

    args.device = pick_device(args.device)
    model = get_model(args.model_name, device=args.device)
    layer_modules = get_transformer_layers(model)

    harmful_prompts, _ = load_arditi(args.model_name)
    if args.n_samples > 0:
        harmful_prompts = harmful_prompts[: args.n_samples]
    hf_post = hf_post[: len(harmful_prompts)]
    harmful_ids = [encode_prompt(model, prompt) for prompt in harmful_prompts]

    new_probes = load_svms(new_base, chain_layers, torch.device(args.device))
    beta = 1.0  # SEQUENTIAL_BETA at training time
    print(f"Cascading {len(harmful_ids)} harmful prompts through the CLE-S chain...")
    steered = compute_steered_harmful(
        model, layer_modules, chain_layers, new_probes, harmful_ids, plot_layers, beta
    )

    n = len(plot_layers)
    fig, axes = plt.subplots(1, n, figsize=(4.9 * n, 4.6), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    mesh = None
    for ax, layer in zip(axes, plot_layers):
        w_old, b_old, acc_old = load_probe(os.path.join(old_base, f"svm_layer{layer:02d}.pt"))
        w_new, b_new, acc_new = load_probe(os.path.join(new_base, f"svm_layer{layer:02d}.pt"))
        mesh, var = draw_panel(
            ax, hf_post[:, layer, :], hl_post[:, layer, :],
            steered[layer]["pre"], steered[layer]["post"],
            w_old, b_old, w_new, b_new, layer, acc_old, acc_new,
            show_post=not args.hide_post,
        )
        print(f"Layer {layer:2d} | PC1+PC2 var = {var*100:.1f}% | old acc {acc_old:.3f} | new acc {acc_new:.3f}")

    handles = [
        plt.Line2D([], [], marker="o", linestyle="none", color=HARMFUL_COLOR, label="harmful (unsteered)"),
        plt.Line2D([], [], marker="o", linestyle="none", color=HARMLESS_COLOR, label="harmless"),
        plt.Line2D([], [], marker="o", linestyle="none", color=STEERED_COLOR, label="harmful (steered, pre-L)"),
    ]
    if not args.hide_post:
        handles.append(
            plt.Line2D([], [], marker="o", linestyle="none", color=STEERED_POST_COLOR, label="harmful (steered, post-L)")
        )
    handles += [
        plt.Line2D([], [], color=ORIG_BOUNDARY_COLOR, lw=2.0, label="old probe (CLE-A/P/G)"),
        plt.Line2D([], [], color=SELF_BOUNDARY_COLOR, lw=2.0, ls="--", label="new probe (CLE-S)"),
    ]
    fig.legend(handles=handles, frameon=False, ncol=len(handles), loc="lower center",
               bbox_to_anchor=(0.5, 1.05))
    cbar = fig.colorbar(mesh, ax=axes.tolist(), fraction=0.02, pad=0.01)
    cbar.set_label("Compliance confidence (new probe)")

    stem = f"probe_boundary_sequential_pca_{args.model_name}"
    stem += "_nopost" if args.hide_post else ""
    fig.savefig(os.path.join(args.out_dir, f"{stem}.png"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(args.out_dir, f"{stem}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {stem}.png/.pdf to {args.out_dir}")


if __name__ == "__main__":
    main()
