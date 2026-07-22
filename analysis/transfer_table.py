"""Accuracy table: does the fixed post-instruction probe still separate the
generated-token activations, and how does a per-token probe compare?

Companion to the plot_*.py figures here. For each layer we build a small table

    columns : post-instruction | g0 | g1 | ... | g(G-1)   (token positions)
    row 1   : orig probe  -- the FIXED post-instruction probe (svm_layerXX.pt),
              evaluated on each column's activations,
    row 2   : own probe   -- the probe trained on THAT token position
              (svm_layerXX_posKK.pt; for the post-instruction column this IS the
              orig probe), evaluated on its own activations.

The post-instruction column is identical in both rows (same probe, same data). The
story is row 1 degrading across generated tokens while row 2 -- a probe retrained per
position -- stays high: the harmful/harmless split stays linearly decodable, but the
*direction* drifts away from the original probe.

Accuracies are in-sample (each probe is scored on the points it was trained on, which
greedy determinism makes exactly reproducible; see extract_generated_activations.py).
This is a separability/drift diagnostic, not a held-out generalisation estimate.

Saves results/probe_boundary/probe_transfer_table_<model>.json and prints the table
to stdout.
"""

import argparse
import json
import os

import torch

from analysis._shared import load_probe


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", type=str, default="llama32-3b")
    parser.add_argument("--artifact_dir", type=str, default="./dataset/representations/")
    parser.add_argument("--out_dir", type=str, default="./results/probe_boundary")
    parser.add_argument("--layers", type=int, nargs="+", default=[4, 14, 24])
    parser.add_argument("--n_gen_tokens", type=int, default=5)
    return parser.parse_args()


def accuracy(w, b, HF, HL):
    """SVM labels harmful=1, harmless=0; predict harmful when w.x + b > 0."""
    s_h = HF @ w + b
    s_l = HL @ w + b
    correct = int((s_h > 0).sum()) + int((s_l <= 0).sum())
    return correct / (len(HF) + len(HL))


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

    col_labels = ["post-instr"] + [f"g{k}" for k in range(n_gen)]
    results = {}

    for layer in args.layers:
        li = gen_layers.index(layer)
        w0, b0, _ = load_probe(os.path.join(base, f"svm_layer{layer:02d}.pt"))

        # Activations per column: post-instruction, then each generated position.
        acts = [(hf_post[:, layer, :], hl_post[:, layer, :])]
        for k in range(n_gen):
            acts.append((gen["HF"][:, k, li, :].float().numpy(),
                         gen["HL"][:, k, li, :].float().numpy()))

        # Own probe per column: orig for post-instruction, gen-position probe otherwise.
        own_probes = [(w0, b0)]
        for k in range(n_gen):
            wk, bk, _ = load_probe(os.path.join(gen_base, f"svm_layer{layer:02d}_pos{k:02d}.pt"))
            own_probes.append((wk, bk))

        orig_row = [accuracy(w0, b0, HF, HL) for HF, HL in acts]
        own_row = [accuracy(w, b, HF, HL) for (w, b), (HF, HL) in zip(own_probes, acts)]

        results[str(layer)] = {
            "columns": col_labels,
            "orig_probe": [round(a, 4) for a in orig_row],
            "own_probe": [round(a, 4) for a in own_row],
        }

        header = "  ".join(f"{c:>10s}" for c in col_labels)
        print(f"\nLayer {layer}")
        print(f"{'':>18s}{header}")
        print(f"{'orig probe':>18s}" + "  ".join(f"{a:10.3f}" for a in orig_row))
        print(f"{'own probe':>18s}" + "  ".join(f"{a:10.3f}" for a in own_row))

    out_json = os.path.join(args.out_dir, f"probe_transfer_table_{args.model_name}.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_json}")


if __name__ == "__main__":
    main()
