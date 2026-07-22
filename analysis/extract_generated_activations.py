"""Re-extract and SAVE the generated-token activations for a few layers.

train_latent_generated.py generates continuation tokens, trains one probe per
(layer, generated-position), and then throws the activations away. To scatter
those generated-token activations (e.g. on the post-instruction PCA plane) we
need the points themselves, so this script repeats the exact same greedy
generation + hidden-state extraction and writes the activations to disk.

Generation is greedy (do_sample=False) and causal, so position k's hidden state
is identical regardless of how many tokens are generated in total -- the points
saved here are exactly the ones the matching svm_layerXX_posKK.pt probe was
trained on.

Saves dataset/representations/<model>/train_svm_generated/gen_activations.pt:
    {"layers": [...], "n_gen": K,
     "HF": (N_harm, n_gen, n_layers_saved, dim),
     "HL": (N_harmless, n_gen, n_layers_saved, dim)}
Only the requested layers are stored to keep the file small.
"""

import argparse
import os

import torch
from tqdm import tqdm

from classifier.utils import generate_and_extract, get_model, load_arditi, pick_device


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", type=str, default="llama32-3b")
    parser.add_argument("--device", default="auto", type=str, help="cuda:0 | mps | cpu | auto")
    parser.add_argument("--artifact_dir", type=str, default="./dataset/representations/")
    parser.add_argument("--n_samples", type=int, default=-1, help="Prompts per class (-1 = all)")
    parser.add_argument("--n_gen_tokens", type=int, default=5, help="Greedy tokens to generate per prompt")
    parser.add_argument("--layers", type=int, nargs="+", default=[4, 14, 24],
                        help="Layer indices to store activations for")
    return parser.parse_args()


def main():
    args = get_args()
    args.device = pick_device(args.device)

    model = get_model(args.model_name, device=args.device)
    harmful_prompts, harmless_prompts = load_arditi(args.model_name)
    if args.n_samples > 0:
        harmful_prompts = harmful_prompts[: args.n_samples]
        harmless_prompts = harmless_prompts[: args.n_samples]

    all_prompts = harmful_prompts + harmless_prompts
    all_labels = [1] * len(harmful_prompts) + [0] * len(harmless_prompts)  # harmful=1, harmless=0
    layers = torch.tensor(args.layers)

    print(f"Generating {args.n_gen_tokens} tokens for {len(all_prompts)} prompts; storing layers {args.layers}")
    hf_reps, hl_reps = [], []
    num_short = 0
    for prompt, label in tqdm(list(zip(all_prompts, all_labels)), total=len(all_prompts)):
        reps, num_gen, _, _ = generate_and_extract(model, prompt, args.n_gen_tokens)
        if reps is None or num_gen < args.n_gen_tokens:
            num_short += 1
            continue
        sub = reps[:, layers, :]  # (n_gen, n_layers_saved, dim)
        (hf_reps if label == 1 else hl_reps).append(sub.unsqueeze(0))

    if num_short:
        print(f"WARNING: {num_short}/{len(all_prompts)} prompts produced < {args.n_gen_tokens} tokens; excluded.")

    HF = torch.cat(hf_reps, dim=0)  # (N_harm, n_gen, n_layers_saved, dim)
    HL = torch.cat(hl_reps, dim=0)
    print(f"HF {tuple(HF.shape)} | HL {tuple(HL.shape)}")

    out_dir = os.path.join(args.artifact_dir, args.model_name, "train_svm_generated")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "gen_activations.pt")
    torch.save({"layers": args.layers, "n_gen": args.n_gen_tokens, "HF": HF, "HL": HL}, out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
