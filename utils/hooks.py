from typing import Dict, List

import torch


def hidden_from_output(output):
    return output[0] if isinstance(output, tuple) else output


def replace_hidden(output, hidden):
    if isinstance(output, tuple):
        return (hidden,) + output[1:]
    return hidden


def projection_hook(
    w: torch.Tensor,
    b: torch.Tensor,
    beta: float,
    margin: float,
    eps: float = 1e-12,
):
    def hook(module, inputs, output):
        h = hidden_from_output(output)
        w_local = w.to(device=h.device, dtype=h.dtype)
        b_local = b.to(device=h.device, dtype=h.dtype)
        m_local = torch.as_tensor(margin, device=h.device, dtype=h.dtype)
        w_norm_sq = torch.sum(w_local * w_local).clamp_min(eps)

        raw_score = (h * w_local.view(1, 1, -1)).sum(dim=-1, keepdim=True) + b_local.view(1, 1, 1)
        score = raw_score + m_local.view(1, 1, 1)
        h_mod = h - (beta * (score / w_norm_sq) * w_local.view(1, 1, -1))

        return replace_hidden(output, h_mod)

    return hook


def pipeline_delta_hook(
    w: torch.Tensor,
    b: torch.Tensor,
    beta: float,
    margin: float,
    layer_idx: int,
    delta_store: Dict[int, torch.Tensor],
    eps: float = 1e-12,
):
    def hook(module, inputs, output):
        h = hidden_from_output(output)
        w_local = w.to(device=h.device, dtype=h.dtype)
        b_local = b.to(device=h.device, dtype=h.dtype)
        m_local = torch.as_tensor(margin, device=h.device, dtype=h.dtype)
        w_norm_sq = torch.sum(w_local * w_local).clamp_min(eps)

        raw_score = (h * w_local.view(1, 1, -1)).sum(dim=-1, keepdim=True) + b_local.view(1, 1, 1)
        score = raw_score + m_local.view(1, 1, 1)
        h_mod = h - (beta * (score / w_norm_sq) * w_local.view(1, 1, -1))
        delta_store[layer_idx] = (h_mod[:, -1, :] - h[:, -1, :]).detach().to(dtype=torch.float32)

        return replace_hidden(output, h_mod)

    return hook


def generated_projection_hook(
    prompt_probe: Dict[str, torch.Tensor],
    gen_probes: Dict[int, Dict[str, torch.Tensor]],
    beta: float,
    prompt_margin: float,
    gen_margin: float,
    layer_idx: int,
    first_layer: int,
    max_pos: int,
    step_state: Dict[str, int],
    eps: float = 1e-12,
):
    """Live projection whose direction depends on token position.

    Prefill forwards (seq_len > 1, the prompt) project with prompt_probe (the
    post-instruction probe) using prompt_margin. Decode forwards (seq_len == 1,
    one generated token) project with gen_probes[min(step, max_pos)] using
    gen_margin. The lowest-index hooked layer advances the shared step_state
    (forward hooks fire in layer order), so every hooked layer within a forward
    reads the same generated position; step resets to -1 on every prefill, so
    each new generation call re-initialises itself.
    """

    def hook(module, inputs, output):
        h = hidden_from_output(output)
        if h.shape[1] > 1:  # prefill (prompt tokens)
            if layer_idx == first_layer:
                step_state["step"] = -1
            probe = prompt_probe
            margin = prompt_margin
        else:  # decode step (one generated token)
            if layer_idx == first_layer:
                step_state["step"] += 1
            probe = gen_probes[min(step_state["step"], max_pos)]
            margin = gen_margin

        w_local = probe["w"].to(device=h.device, dtype=h.dtype)
        b_local = probe["b"].to(device=h.device, dtype=h.dtype)
        m_local = torch.as_tensor(margin, device=h.device, dtype=h.dtype)
        w_norm_sq = torch.sum(w_local * w_local).clamp_min(eps)

        raw_score = (h * w_local.view(1, 1, -1)).sum(dim=-1, keepdim=True) + b_local.view(1, 1, 1)
        score = raw_score + m_local.view(1, 1, 1)
        h_mod = h - (beta * (score / w_norm_sq) * w_local.view(1, 1, -1))

        return replace_hidden(output, h_mod)

    return hook


def add_hook(delta: torch.Tensor):
    def hook(module, inputs, output):
        h = hidden_from_output(output)
        if delta.ndim == 1:
            v = delta.view(1, 1, -1)
        elif delta.ndim == 2:
            if delta.shape[0] != h.shape[0]:
                raise ValueError(f"Batch delta size mismatch: delta batch={delta.shape[0]} hidden batch={h.shape[0]}")
            v = delta.unsqueeze(1)
        else:
            raise ValueError(f"Unexpected delta shape: {tuple(delta.shape)}")

        h_mod = h + v.to(device=h.device, dtype=h.dtype)
        return replace_hidden(output, h_mod)

    return hook


def remove_hooks(handles: List[object]) -> None:
    for handle in handles:
        handle.remove()
