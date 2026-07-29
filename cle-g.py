import argparse
import json
import os
from typing import Dict, List

import torch
from tqdm import tqdm

from utils.args import build_run_tag, parse_layer_margins, parse_layers_arg
from utils.hooks import (
    additive_prompt_generated_hook,
    generated_projection_hook,
    pipeline_delta_hook,
    remove_hooks,
)
from utils.models_utils import get_transformer_layers
from utils.probes import ProbeDict, load_generated_svms, load_probes
from utils.runtime import chunked, evaluate, load_model, load_prompts, load_prompts_from_file
from utils.runtime import print_configuration, set_seed, validate_probe_dims


def get_args():
    parser = argparse.ArgumentParser(
        description=(
            "Live latent projection where the probe direction depends on token position: "
            "prompt tokens use the post-instruction probe, generated token k uses the "
            "generated-token probe at position min(k, max_pos)."
        )
    )

    # Model config
    parser.add_argument("--model_name", type=str, default="llama3-8b")
    parser.add_argument("--device", type=str, default="cuda:0")

    # Probe / layer config
    parser.add_argument(
        "--svm_dir",
        type=str,
        default=None,
        help="Directory with the post-instruction probes (svm_layerXX.pt) applied to prompt tokens.",
    )
    parser.add_argument(
        "--gen_svm_dir",
        type=str,
        default=None,
        help=(
            "Directory with the generated-token probes (svm_layerXX_posPP.pt) applied to "
            "generated tokens. Defaults to ./dataset/representations/<model>/train_svm_generated."
        ),
    )
    parser.add_argument(
        "--max_pos",
        type=int,
        default=None,
        help=(
            "Cap on generated-token probe position. Generated token k uses probe "
            "pos min(k, max_pos). Defaults to the last available position on disk."
        ),
    )
    parser.add_argument(
        "--prompt_mode",
        type=str,
        default="project",
        choices=["project", "additive"],
        help=(
            "How the post-instruction probe steers the PROMPT tokens (generated tokens are "
            "always live-projected with the gen probes). 'project' = CLE-P live projection "
            "on prompt tokens (default). 'additive' = CLE-A: compute a fixed per-layer delta "
            "from a clean prompt pass and replay it during generation."
        ),
    )
    parser.add_argument(
        "--probe_type",
        type=str,
        default="svm",
        choices=["svm", "single_direction"],
        help=(
            "Type of the post-instruction (prompt) probe. 'svm' loads svm_layerXX.pt with "
            "learned intercepts; 'single_direction' loads sd_layerXX.pt and derives the "
            "midpoint bias from class means. Generated-token probes are always svm-format."
        ),
    )
    parser.add_argument(
        "--probe_reps_dir",
        type=str,
        default=None,
        help=(
            "Directory containing HFx_train.pt and HLx_train.pt for deriving "
            "single-direction prompt-probe biases. Defaults to --svm_dir when available."
        ),
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="all",
        help="Layers to intervene: 'all', '5-25' (end-exclusive), or '10,14,20'.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Scale in h* = h - beta * (w^T h + b + m) / ||w||^2 * w.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.0,
        help="Margin m in h* = h - beta * (w^T h + b + m) / ||w||^2 * w.",
    )
    parser.add_argument(
        "--layer_margin",
        "--layer_margins",
        dest="layer_margin",
        nargs="+",
        type=str,
        default=None,
        help=(
            "Optional per-layer margins aligned with --layers. Accepts either "
            "space-separated values or comma-separated values."
        ),
    )

    parser.add_argument(
        "--gen_margin",
        type=float,
        default=None,
        help=(
            "Margin m applied only to the generated-token (decode) projection. "
            "Defaults to --margin, so by default prompt and generated tokens share a margin. "
            "Set this to tune the generated probes independently of the post-instruction probe."
        ),
    )
    parser.add_argument(
        "--gen_layer_margin",
        "--gen_layer_margins",
        dest="gen_layer_margin",
        nargs="+",
        type=str,
        default=None,
        help="Optional per-layer margins for the generated-token projection, aligned with --layers.",
    )

    # Generation / dataset
    parser.add_argument("--dataset", type=str, default="harmbench_test")
    parser.add_argument(
        "--prompts_file",
        type=str,
        default=None,
        help=(
            "Optional JSON file with an explicit prompt list to run instead of --dataset "
            "(e.g. results/cle/steering_failures_*.json). Overrides --dataset/--limit."
        ),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for projection and generation.")

    # Output / eval
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def register_generated_hooks(
    layers,
    selected_layers: List[int],
    prompt_probes: ProbeDict,
    gen_probes: Dict[int, Dict[int, Dict[str, torch.Tensor]]],
    prompt_margin_map: Dict[int, float],
    gen_margin_map: Dict[int, float],
    max_pos: int,
    step_state: Dict[str, int],
    args,
):
    step_state["step"] = -1
    first_layer = min(selected_layers)
    handles = []
    for layer_idx in selected_layers:
        handles.append(
            layers[layer_idx].register_forward_hook(
                generated_projection_hook(
                    prompt_probes[layer_idx],
                    gen_probes[layer_idx],
                    args.beta,
                    prompt_margin_map[layer_idx],
                    gen_margin_map[layer_idx],
                    layer_idx,
                    first_layer,
                    max_pos,
                    step_state,
                )
            )
        )
    return handles


def generate_with_generated_probes(
    *,
    batch_prompts: List[str],
    layers,
    selected_layers: List[int],
    model,
    prompt_probes: ProbeDict,
    gen_probes: Dict[int, Dict[int, Dict[str, torch.Tensor]]],
    prompt_margin_map: Dict[int, float],
    gen_margin_map: Dict[int, float],
    max_pos: int,
    args,
) -> List[str]:
    step_state: Dict[str, int] = {"step": -1}
    generation_handles = register_generated_hooks(
        layers, selected_layers, prompt_probes, gen_probes, prompt_margin_map, gen_margin_map, max_pos, step_state, args
    )
    try:
        return model.batch_generate(batch_prompts, max_new_tokens=args.max_new_tokens)
    except Exception as e:
        print(f"Batch generation error. Falling back to single-prompt generation: {e}")
        remove_hooks(generation_handles)
        generation_handles = []
        responses = []
        for prompt in batch_prompts:
            single_handles = register_generated_hooks(
                layers, selected_layers, prompt_probes, gen_probes, prompt_margin_map, gen_margin_map, max_pos, step_state, args
            )
            try:
                responses.append(model.generate(prompt, max_new_tokens=args.max_new_tokens))
            except Exception as inner_e:
                print(f"Gen Error: {inner_e}")
                responses.append("")
            finally:
                remove_hooks(single_handles)
        return responses
    finally:
        remove_hooks(generation_handles)


def compute_prompt_deltas(
    *,
    inputs,
    layers,
    selected_layers: List[int],
    model,
    prompt_probes: ProbeDict,
    prompt_margin_map: Dict[int, float],
    args,
) -> Dict[int, torch.Tensor]:
    """Clean-pass per-layer CLE-A deltas from the post-instruction probe (see cle-a.py)."""
    deltas: Dict[int, torch.Tensor] = {}
    delta_handles = []
    for layer_idx in selected_layers:
        delta_handles.append(
            layers[layer_idx].register_forward_hook(
                pipeline_delta_hook(
                    prompt_probes[layer_idx]["w"],
                    prompt_probes[layer_idx]["b"],
                    args.beta,
                    prompt_margin_map[layer_idx],
                    layer_idx,
                    deltas,
                )
            )
        )
    try:
        with torch.no_grad():
            _ = model.model(input_ids=inputs.input_ids, attention_mask=inputs.get("attention_mask", None))
    finally:
        remove_hooks(delta_handles)

    missing = [layer_idx for layer_idx in selected_layers if layer_idx not in deltas]
    if missing:
        raise RuntimeError(f"Failed to capture prompt deltas for layers: {missing}")
    return deltas


def register_additive_generated_hooks(
    layers,
    selected_layers: List[int],
    deltas: Dict[int, torch.Tensor],
    gen_probes: Dict[int, Dict[int, Dict[str, torch.Tensor]]],
    gen_margin_map: Dict[int, float],
    max_pos: int,
    step_state: Dict[str, int],
    args,
    local_idx: int = None,
):
    step_state["step"] = -1
    first_layer = min(selected_layers)
    handles = []
    for layer_idx in selected_layers:
        delta = deltas[layer_idx]
        if local_idx is not None:
            delta = delta[local_idx]
        handles.append(
            layers[layer_idx].register_forward_hook(
                additive_prompt_generated_hook(
                    delta,
                    gen_probes[layer_idx],
                    args.beta,
                    gen_margin_map[layer_idx],
                    layer_idx,
                    first_layer,
                    max_pos,
                    step_state,
                )
            )
        )
    return handles


def generate_with_additive_prompt(
    *,
    batch_prompts: List[str],
    layers,
    selected_layers: List[int],
    model,
    deltas: Dict[int, torch.Tensor],
    gen_probes: Dict[int, Dict[int, Dict[str, torch.Tensor]]],
    gen_margin_map: Dict[int, float],
    max_pos: int,
    args,
) -> List[str]:
    step_state: Dict[str, int] = {"step": -1}
    generation_handles = register_additive_generated_hooks(
        layers, selected_layers, deltas, gen_probes, gen_margin_map, max_pos, step_state, args
    )
    try:
        return model.batch_generate(batch_prompts, max_new_tokens=args.max_new_tokens)
    except Exception as e:
        print(f"Batch generation error. Falling back to single-prompt generation: {e}")
        remove_hooks(generation_handles)
        generation_handles = []
        responses = []
        for local_idx, prompt in enumerate(batch_prompts):
            single_handles = register_additive_generated_hooks(
                layers, selected_layers, deltas, gen_probes, gen_margin_map, max_pos, step_state, args, local_idx=local_idx
            )
            try:
                responses.append(model.generate(prompt, max_new_tokens=args.max_new_tokens))
            except Exception as inner_e:
                print(f"Gen Error: {inner_e}")
                responses.append("")
            finally:
                remove_hooks(single_handles)
        return responses
    finally:
        remove_hooks(generation_handles)


def main():
    args = get_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    model = load_model(args)
    layers = get_transformer_layers(model)
    n_layers = len(layers)
    hidden_dim = model.model.config.hidden_size

    selected_layers = parse_layers_arg(args.layers, n_layers)
    layer_margin_map = parse_layer_margins(args.layer_margin, selected_layers, args.margin)
    # Generated-token margin: inherit the prompt margins unless a gen-specific
    # override (--gen_margin and/or --gen_layer_margin) is given.
    if args.gen_margin is None and args.gen_layer_margin is None:
        gen_margin_map = dict(layer_margin_map)
    else:
        gen_margin_default = args.margin if args.gen_margin is None else args.gen_margin
        gen_margin_map = parse_layer_margins(args.gen_layer_margin, selected_layers, gen_margin_default)

    if args.svm_dir is None:
        args.svm_dir = os.path.join("./dataset/representations", args.model_name, "train_svm")
    if not os.path.isdir(args.svm_dir):
        raise FileNotFoundError(f"SVM directory not found: {args.svm_dir}")

    if args.gen_svm_dir is None:
        args.gen_svm_dir = os.path.join("./dataset/representations", args.model_name, "train_svm_generated")
    if not os.path.isdir(args.gen_svm_dir):
        raise FileNotFoundError(f"Generated SVM directory not found: {args.gen_svm_dir}")

    # Post-instruction probes for the prompt tokens (prefill).
    prompt_probes = load_probes(
        probe_type=args.probe_type,
        svm_dir=args.svm_dir,
        layer_indices=selected_layers,
        device=device,
        explicit_reps_dir=args.probe_reps_dir,
    )
    validate_probe_dims(prompt_probes, selected_layers, hidden_dim)

    # Generated-token probes for the output tokens (decode).
    gen_probes, max_pos = load_generated_svms(
        gen_svm_dir=args.gen_svm_dir,
        layer_indices=selected_layers,
        device=device,
        max_pos=args.max_pos,
    )
    for layer_idx in selected_layers:
        if gen_probes[layer_idx][0]["w"].numel() != hidden_dim:
            raise ValueError(
                f"Generated-probe hidden dim mismatch in layer {layer_idx}: "
                f"got {gen_probes[layer_idx][0]['w'].numel()}, expected {hidden_dim}"
            )

    if args.prompts_file is not None:
        prompts, categories = load_prompts_from_file(args.prompts_file)
    else:
        prompts, categories = load_prompts(args)
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")

    base_out_dir = args.out_dir if args.out_dir is not None else os.path.join("./completions", args.model_name, "generated")
    args.out_dir = base_out_dir
    os.makedirs(args.out_dir, exist_ok=True)

    run_tag = f"{build_run_tag(args, selected_layers, layer_margin_map)}_maxpos{max_pos}"
    if args.prompt_mode == "additive":
        run_tag = f"{run_tag}_promptadd"
    gen_margins_differ = any(gen_margin_map[l] != layer_margin_map[l] for l in selected_layers)
    if gen_margins_differ:
        if args.gen_layer_margin is not None:
            run_tag = f"{run_tag}_genmargin-vec"
        else:
            run_tag = f"{run_tag}_genmargin{gen_margin_map[selected_layers[0]]}"
    if args.prompts_file is not None:
        src_stem = os.path.splitext(os.path.basename(args.prompts_file))[0]
        run_tag = f"{run_tag}_src-{src_stem}"
    output_path = os.path.join(args.out_dir, f"completions_{run_tag}.json")
    print_configuration(
        args=args,
        selected_layers=selected_layers,
        layer_margin_map=layer_margin_map,
        probe_reps_dir=args.probe_reps_dir if args.probe_type == "single_direction" else None,
        output_path=output_path,
        prompt_count=len(prompts),
    )
    print(f"Generated probe dir: {args.gen_svm_dir}")
    print(f"Generated-token positions: 0..{max_pos} (token >{max_pos} reuses pos{max_pos:02d})")
    print(f"Prompt-token margin: {layer_margin_map[selected_layers[0]] if len(set(layer_margin_map.values())) == 1 else layer_margin_map}")
    print(f"Generated-token margin: {gen_margin_map[selected_layers[0]] if len(set(gen_margin_map.values())) == 1 else gen_margin_map}")
    prompt_mode_desc = "CLE-A additive delta" if args.prompt_mode == "additive" else "CLE-P live projection"
    print(f"Prompt-token mode: {args.prompt_mode} ({prompt_mode_desc}); generated tokens: CLE-G live projection")

    results = []
    pbar = tqdm(total=len(prompts), desc="Generated-probe projection + Generate")
    for batch_start, batch_prompts in chunked(prompts, args.batch_size):
        batch_categories = categories[batch_start:batch_start + len(batch_prompts)]

        if args.prompt_mode == "additive":
            inputs = model.prepare_batch_inputs(batch_prompts)
            deltas = compute_prompt_deltas(
                inputs=inputs,
                layers=layers,
                selected_layers=selected_layers,
                model=model,
                prompt_probes=prompt_probes,
                prompt_margin_map=layer_margin_map,
                args=args,
            )
            responses = generate_with_additive_prompt(
                batch_prompts=batch_prompts,
                layers=layers,
                selected_layers=selected_layers,
                model=model,
                deltas=deltas,
                gen_probes=gen_probes,
                gen_margin_map=gen_margin_map,
                max_pos=max_pos,
                args=args,
            )
        else:
            responses = generate_with_generated_probes(
                batch_prompts=batch_prompts,
                layers=layers,
                selected_layers=selected_layers,
                model=model,
                prompt_probes=prompt_probes,
                gen_probes=gen_probes,
                prompt_margin_map=layer_margin_map,
                gen_margin_map=gen_margin_map,
                max_pos=max_pos,
                args=args,
            )

        for prompt, response, category in zip(batch_prompts, responses, batch_categories):
            results.append({"category": category, "prompt": prompt, "response": response})
        pbar.update(len(batch_prompts))
    pbar.close()

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved completions to {output_path}")

    if args.evaluate:
        del model
        evaluate(
            results=results,
            eval_path=os.path.join(args.out_dir, "evaluation", f"evaluation_{run_tag}.json"),
        )


if __name__ == "__main__":
    main()
