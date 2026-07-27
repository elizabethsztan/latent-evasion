"""Run CLE-S with sequentially trained prompt and generated-token probes."""

import argparse
import json
import os
from typing import Dict, List

import torch
from tqdm import tqdm

from classifier.sequential_utils import (
    SEQUENTIAL_BETA,
    parse_layer_range,
)
from utils.hooks import generated_projection_hook, remove_hooks
from utils.models_utils import get_transformer_layers
from utils.probes import ProbeDict, load_generated_svms, load_svms
from utils.runtime import (
    chunked,
    evaluate,
    load_model,
    load_prompts,
    load_prompts_from_file,
    set_seed,
    validate_probe_dims,
)


GeneratedProbeDict = Dict[int, Dict[int, Dict[str, torch.Tensor]]]


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", type=str, default="llama32-3b")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--svm_dir",
        type=str,
        default=None,
        help="Directory containing the sequential prompt-chain probes.",
    )
    parser.add_argument(
        "--gen_svm_dir",
        type=str,
        default=None,
        help="Generated-chain directory. Defaults to --svm_dir.",
    )
    parser.add_argument(
        "--layers",
        type=str,
        required=True,
        help="Contiguous end-exclusive range matching training, e.g. '11-23'.",
    )
    parser.add_argument("--dataset", type=str, default="harmbench_test")
    parser.add_argument(
        "--prompts_file",
        type=str,
        default=None,
        help="Optional JSON prompt list overriding --dataset and --limit.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def register_sequential_hooks(
    layers,
    selected_layers: List[int],
    prompt_probes: ProbeDict,
    generated_probes: GeneratedProbeDict,
    max_pos: int,
    step_state: Dict[str, int],
):
    step_state["step"] = -1
    first_layer = min(selected_layers)
    return [
        layers[layer_idx].register_forward_hook(
            generated_projection_hook(
                prompt_probes[layer_idx],
                generated_probes[layer_idx],
                SEQUENTIAL_BETA,
                None,  # use each probe's own data-derived margin
                None,  # use each probe's own data-derived margin
                layer_idx,
                first_layer,
                max_pos,
                step_state,
            )
        )
        for layer_idx in selected_layers
    ]


def generate_with_sequential_probes(
    *,
    batch_prompts: List[str],
    layers,
    selected_layers: List[int],
    model,
    prompt_probes: ProbeDict,
    generated_probes: GeneratedProbeDict,
    max_pos: int,
    args,
) -> List[str]:
    step_state = {"step": -1}
    handles = register_sequential_hooks(
        layers,
        selected_layers,
        prompt_probes,
        generated_probes,
        max_pos,
        step_state,
    )
    try:
        return model.batch_generate(
            batch_prompts,
            max_new_tokens=args.max_new_tokens,
        )
    except Exception as error:
        print(
            "Batch generation error. Falling back to single-prompt generation: "
            f"{error}"
        )
        remove_hooks(handles)
        handles = []
        responses = []
        for prompt in batch_prompts:
            single_handles = register_sequential_hooks(
                layers,
                selected_layers,
                prompt_probes,
                generated_probes,
                max_pos,
                step_state,
            )
            try:
                responses.append(
                    model.generate(prompt, max_new_tokens=args.max_new_tokens)
                )
            except Exception as inner_error:
                print(f"Gen Error: {inner_error}")
                responses.append("")
            finally:
                remove_hooks(single_handles)
        return responses
    finally:
        remove_hooks(handles)


def main():
    args = get_args()
    selected_layers = parse_layer_range(args.layers)
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    set_seed(args.seed)

    device = torch.device(args.device)
    model = load_model(args)
    layers = get_transformer_layers(model)
    for layer_idx in selected_layers:
        if layer_idx >= len(layers):
            raise ValueError(
                f"Layer index {layer_idx} out of bounds [0, {len(layers) - 1}]"
            )

    if args.svm_dir is None:
        args.svm_dir = os.path.join(
            "./dataset/representations",
            args.model_name,
            "train_svm_sequential",
        )
    if not os.path.isdir(args.svm_dir):
        raise FileNotFoundError(f"SVM directory not found: {args.svm_dir}")
    if args.gen_svm_dir is None:
        args.gen_svm_dir = args.svm_dir
    if not os.path.isdir(args.gen_svm_dir):
        raise FileNotFoundError(
            f"Generated SVM directory not found: {args.gen_svm_dir}"
        )

    hidden_dim = model.model.config.hidden_size
    prompt_probes = load_svms(args.svm_dir, selected_layers, device)
    validate_probe_dims(prompt_probes, selected_layers, hidden_dim)
    generated_probes, max_pos = load_generated_svms(
        args.gen_svm_dir,
        selected_layers,
        device,
    )
    for layer_idx in selected_layers:
        for position, probe in generated_probes[layer_idx].items():
            if probe["w"].numel() != hidden_dim:
                raise ValueError(
                    "Generated-probe hidden dim mismatch in layer "
                    f"{layer_idx}, position {position}: got "
                    f"{probe['w'].numel()}, expected {hidden_dim}"
                )

    if args.prompts_file is not None:
        prompts, categories = load_prompts_from_file(args.prompts_file)
    else:
        prompts, categories = load_prompts(args)

    args.out_dir = args.out_dir or os.path.join(
        "./completions",
        args.model_name,
        "sequential",
    )
    os.makedirs(args.out_dir, exist_ok=True)
    layers_tag = args.layers.replace("-", "to")
    limit_tag = f"limit{args.limit}" if args.limit else "FULL"
    run_tag = (
        f"{args.dataset}_{limit_tag}_layers{layers_tag}_"
        f"seqbeta{SEQUENTIAL_BETA}_maxpos{max_pos}_seed{args.seed}"
    )
    output_path = os.path.join(args.out_dir, f"completions_{run_tag}.json")

    print("--- Configuration ---")
    print(f"Model: {args.model_name}")
    print(f"Layers: {selected_layers}")
    print(
        f"Beta (fixed): {SEQUENTIAL_BETA} | "
        f"Margin: per-probe, derived from each probe's harmless-class mean score"
    )
    print(f"Prompt-chain dir: {args.svm_dir}")
    print(f"Generated-chain dir: {args.gen_svm_dir}")
    print(
        f"Generated-token positions: 0..{max_pos} "
        f"(token >{max_pos} reuses pos{max_pos:02d})"
    )
    print(f"Dataset: {args.dataset} | Prompts: {len(prompts)}")
    print(f"Output: {output_path}")

    results = []
    progress = tqdm(total=len(prompts), desc="CLE-S projection + Generate")
    for batch_start, batch_prompts in chunked(prompts, args.batch_size):
        batch_categories = categories[
            batch_start : batch_start + len(batch_prompts)
        ]
        responses = generate_with_sequential_probes(
            batch_prompts=batch_prompts,
            layers=layers,
            selected_layers=selected_layers,
            model=model,
            prompt_probes=prompt_probes,
            generated_probes=generated_probes,
            max_pos=max_pos,
            args=args,
        )
        results.extend(
            {
                "category": category,
                "prompt": prompt,
                "response": response,
            }
            for prompt, response, category in zip(
                batch_prompts,
                responses,
                batch_categories,
            )
        )
        progress.update(len(batch_prompts))
    progress.close()

    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(results, output_file, indent=4)
    print(f"\nSaved completions to {output_path}")

    if args.evaluate:
        del model
        evaluate(
            results=results,
            eval_path=os.path.join(
                args.out_dir,
                "evaluation",
                f"evaluation_{run_tag}.json",
            ),
        )


if __name__ == "__main__":
    main()
