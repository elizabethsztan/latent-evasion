"""Shared training utilities for CLE-S sequential probe chains."""

import os
import re
from typing import Dict, List, Optional

import numpy as np
import torch

from classifier.utils import train_layer_svm
from utils.hooks import projection_hook, remove_hooks
from utils.models_utils import parse_layers


SEQUENTIAL_BETA = 1.0


def parse_layer_range(value: str) -> List[int]:
    """Parse the single contiguous, end-exclusive range required by CLE-S."""
    normalized = value.strip()
    if re.fullmatch(r"\d+-\d+", normalized) is None:
        raise ValueError(
            f"--layers must be a contiguous 's-e' range for CLE-S, got: {value!r}"
        )
    selected_layers = parse_layers(normalized)
    if not selected_layers:
        raise ValueError(f"--layers selects no layers: {value!r}")
    return selected_layers


def encode_prompt(model, prompt: str) -> torch.Tensor:
    """Return a chat-templated prompt as input IDs on the model device."""
    formatted_prompt = model._get_prompt(prompt)
    return model.tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(model.device)


def hidden_states_at_last_pos(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Return last-position hidden states as a (layers, hidden_dim) CPU tensor."""
    with torch.no_grad():
        outputs = model.model(
            input_ids=input_ids,
            output_hidden_states=True,
            use_cache=False,
        )
    return torch.cat(
        [hidden[:, -1, :] for hidden in outputs.hidden_states[1:]],
        dim=0,
    ).to(dtype=torch.float32, device="cpu")


def append_greedy_token(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Append one greedily decoded token without steering."""
    with torch.no_grad():
        outputs = model.model(input_ids=input_ids, use_cache=False)
    next_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    return torch.cat((input_ids, next_id.to(input_ids.device)), dim=1)


def steer_and_append_token(
    model,
    layer_modules,
    selected_layers: List[int],
    probes: Dict[int, Dict[str, torch.Tensor]],
    input_ids: torch.Tensor,
    beta: float = SEQUENTIAL_BETA,
) -> torch.Tensor:
    """Append one greedy token while applying a completed probe chain."""
    handles = [
        layer_modules[layer_idx].register_forward_hook(
            projection_hook(
                probes[layer_idx]["w"].to(model.device),
                probes[layer_idx]["b"].to(model.device),
                beta,
                probes[layer_idx]["margin"],
            )
        )
        for layer_idx in selected_layers
    ]
    try:
        with torch.no_grad():
            outputs = model.model(input_ids=input_ids, use_cache=False)
        next_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    finally:
        remove_hooks(handles)

    return torch.cat((input_ids, next_id.to(input_ids.device)), dim=1)


def train_layer_chain(
    model,
    layer_modules,
    selected_layers: List[int],
    harmful_ids_list: List[torch.Tensor],
    harmless_ids_list: List[torch.Tensor],
    beta: float = SEQUENTIAL_BETA,
    seed: int = 42,
) -> Dict[int, Dict[str, object]]:
    """Train one cumulative layer cascade for a single token position."""
    harmless_representations = torch.stack(
        [hidden_states_at_last_pos(model, input_ids) for input_ids in harmless_ids_list]
    )
    labels = np.array(
        [1] * len(harmful_ids_list) + [0] * len(harmless_ids_list)
    )

    probes: Dict[int, Dict[str, object]] = {}
    handles = []
    try:
        for layer_idx in selected_layers:
            harmful_representations = torch.stack(
                [
                    hidden_states_at_last_pos(model, input_ids)[layer_idx]
                    for input_ids in harmful_ids_list
                ]
            )
            layer_representations = torch.cat(
                (
                    harmful_representations,
                    harmless_representations[:, layer_idx, :],
                )
            ).numpy()
            classifier, accuracy, report = train_layer_svm(
                layer_representations,
                labels,
                seed=seed,
            )

            w = torch.from_numpy(classifier.coef_[0]).float()
            b = torch.tensor(classifier.intercept_[0]).float()

            # Steer every harmful activation to land at exactly the harmless
            # class's own mean score (target_score), not just past the raw
            # decision boundary -- and never past it, since beta=1 here means
            # the correction always lands precisely at target_score regardless
            # of the starting score (see docs/superpowers/specs/2026-07-24-cle-s-design.md).
            harmless_mean_l = harmless_representations[:, layer_idx, :].mean(dim=0)
            target_score = float(torch.dot(w, harmless_mean_l) + b)
            margin = -target_score

            probes[layer_idx] = {
                "w": w,
                "b": b,
                "margin": margin,
                "target_score": target_score,
                "accuracy": accuracy,
                "report": report,
            }
            handles.append(
                layer_modules[layer_idx].register_forward_hook(
                    projection_hook(
                        w.to(model.device),
                        b.to(model.device),
                        beta,
                        margin,
                    )
                )
            )
    finally:
        remove_hooks(handles)

    return probes


def save_chain(
    probes: Dict[int, Dict[str, object]],
    out_dir: str,
    model_name: str,
    pos: Optional[int],
) -> None:
    """Save a prompt or generated-position chain in the existing probe format."""
    os.makedirs(out_dir, exist_ok=True)
    suffix = "" if pos is None else f"_pos{pos:02d}"
    for layer_idx, probe in probes.items():
        torch.save(
            {
                "w": probe["w"],
                "b": probe["b"],
                "margin": probe["margin"],
                "target_score": probe["target_score"],
                "layer_idx": layer_idx,
                "gen_pos": pos,
                "model_name": model_name,
                "accuracy": probe["accuracy"],
                "report": probe["report"],
            },
            os.path.join(out_dir, f"svm_layer{layer_idx:02d}{suffix}.pt"),
        )
