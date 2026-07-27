"""Train CLE-S probe chains across layers and generated-token positions."""

import argparse
import os

from classifier.sequential_utils import (
    SEQUENTIAL_BETA,
    append_greedy_token,
    encode_prompt,
    parse_layer_range,
    save_chain,
    steer_and_append_token,
    train_layer_chain,
)
from classifier.utils import get_model, load_arditi, pick_device
from utils.models_utils import get_transformer_layers


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", type=str, default="llama32-3b")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="cuda:0 | mps | cpu | auto",
    )
    parser.add_argument(
        "--artifact_dir",
        type=str,
        default="./dataset/representations/",
    )
    parser.add_argument(
        "--layers",
        type=str,
        required=True,
        help="Contiguous end-exclusive layer range, for example '11-23'.",
    )
    parser.add_argument(
        "--n_gen_tokens",
        type=int,
        default=5,
        help="Number of generated-position chains to train.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=-1,
        help="Prompts per class (-1 = all).",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = get_args()
    selected_layers = parse_layer_range(args.layers)
    if args.n_gen_tokens < 0:
        raise ValueError("--n_gen_tokens must be >= 0")

    args.device = pick_device(args.device)
    model = get_model(args.model_name, device=args.device)
    harmful_prompts, harmless_prompts = load_arditi(args.model_name)
    if args.n_samples > 0:
        harmful_prompts = harmful_prompts[: args.n_samples]
        harmless_prompts = harmless_prompts[: args.n_samples]

    layer_modules = get_transformer_layers(model)
    for layer_idx in selected_layers:
        if layer_idx >= len(layer_modules):
            raise ValueError(
                f"Layer index {layer_idx} out of bounds [0, {len(layer_modules) - 1}]"
            )

    out_dir = os.path.join(
        args.artifact_dir,
        args.model_name,
        "train_svm_sequential",
    )
    print(
        f"Encoding {len(harmful_prompts)} harmful + "
        f"{len(harmless_prompts)} harmless prompts..."
    )
    harmful_ids = [encode_prompt(model, prompt) for prompt in harmful_prompts]
    harmless_ids = [encode_prompt(model, prompt) for prompt in harmless_prompts]

    print(f"Training prompt-level chain over layers {selected_layers}...")
    probes = train_layer_chain(
        model,
        layer_modules,
        selected_layers,
        harmful_ids,
        harmless_ids,
        beta=SEQUENTIAL_BETA,
        seed=args.seed,
    )
    save_chain(probes, out_dir, args.model_name, pos=None)
    mean_accuracy = sum(probe["accuracy"] for probe in probes.values()) / len(probes)
    print(
        f"Prompt chain done. Mean test accuracy over {len(probes)} layers: "
        f"{mean_accuracy:.4f}"
    )

    for position in range(args.n_gen_tokens):
        print(f"Advancing to generated position {position}...")
        harmful_ids = [
            steer_and_append_token(
                model,
                layer_modules,
                selected_layers,
                probes,
                input_ids,
                beta=SEQUENTIAL_BETA,
            )
            for input_ids in harmful_ids
        ]
        harmless_ids = [
            append_greedy_token(model, input_ids) for input_ids in harmless_ids
        ]
        probes = train_layer_chain(
            model,
            layer_modules,
            selected_layers,
            harmful_ids,
            harmless_ids,
            beta=SEQUENTIAL_BETA,
            seed=args.seed,
        )
        save_chain(probes, out_dir, args.model_name, pos=position)
        mean_accuracy = (
            sum(probe["accuracy"] for probe in probes.values()) / len(probes)
        )
        print(
            f"Position {position} chain done. Mean test accuracy over "
            f"{len(probes)} layers: {mean_accuracy:.4f}"
        )

    print(f"\nDone. Probes saved under {out_dir}")


if __name__ == "__main__":
    main()
