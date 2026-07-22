"""Shared helpers for the probe-analysis figures/tables in this subpackage.

Collects the scaffolding that plot_*.py and transfer_table.py all reuse: the
matplotlib style, the colour palette, the probe loader, and the PCA-projection
maths (orientation pinning, probe-as-a-line, frame anchor). Each script keeps its
own draw logic; only the copy-pasted pieces live here.

Projecting a probe onto a PCA plane: PCA is the linear map z = (x - mu) V^T with
orthonormal rows V. A point z reconstructs to x = mu + z V, so the probe score
w.x + b = (V w).z + (w.mu + b) is a straight line grad.z + offset = 0 in 2D
(grad = V w, offset = w.mu + b). The residual off-plane component is dropped -- the
standard way to draw a high-D hyperplane in 2D.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from sklearn.decomposition import PCA

HARMFUL_COLOR = "#c44e52"
HARMLESS_COLOR = "#4c72b0"
ORIG_BOUNDARY_COLOR = "black"      # original post-instruction probe (solid black)
GEN_BOUNDARY_COLOR = "#8172b3"     # generated-position probe (dashed purple)
SELF_BOUNDARY_COLOR = "#8172b3"    # this-panel probe (dashed purple); alias of GEN


def apply_rcparams(tick_labelsize=11):
    """Paper style shared by every figure here (dense grids pass tick_labelsize=10)."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": tick_labelsize,
        "ytick.labelsize": tick_labelsize,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def load_probe(path):
    """Return (w, b, accuracy) for a saved SVM probe dict (harmful=1, harmless=0)."""
    obj = torch.load(path, map_location="cpu")
    return obj["w"].float().view(-1).numpy(), float(obj["b"]), float(obj.get("accuracy", np.nan))


def oriented_pca(X, w_ref, n_harm, n_components=2):
    """Fit a PCA on X and pin the axis signs so every panel reads the same.

    Each principal component's sign is arbitrary, which makes the harmful/harmless
    clusters jump around from panel to panel. We flip them (purely cosmetic -- the
    probe field s(z) is sign-invariant) so that, using w_ref as the reference probe:
      - PC1: harmful/red on the left (w_ref score increases left -> right),
      - PC2: the harmful cluster toward the bottom.
    X must stack the n_harm harmful rows first, then the harmless rows. Returns the
    fitted pca (for explained_variance_ratio_ / mean_), the oriented components V,
    and the oriented projection Z = (X - mu) V^T.
    """
    pca = PCA(n_components=n_components).fit(X)
    V, mu = pca.components_.copy(), pca.mean_
    if (V[0] @ w_ref) > 0:
        V[0] = -V[0]
    Z = (X - mu) @ V.T
    if Z[:n_harm, 1].mean() > Z[n_harm:, 1].mean():
        V[1] = -V[1]
        Z = (X - mu) @ V.T
    return pca, V, Z


def project_probe(V, mu, w, b):
    """Probe hyperplane w.x + b = 0 as a line grad.z + offset = 0 on the PCA plane."""
    return V @ w, float(w @ mu) + b


def closest_anchor(centroid, grad, offset):
    """Point on the line grad.z + offset = 0 nearest the centroid (keeps it in frame)."""
    return centroid - ((grad @ centroid + offset) / (grad @ grad + 1e-12)) * grad


def gen_colors(n_gen):
    """Sequential colours for g0..g(n-1); truncate viridis to avoid pale yellow on white."""
    return cm.viridis(np.linspace(0.0, 0.82, n_gen))
