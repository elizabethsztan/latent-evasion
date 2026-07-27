# CLE-S (Sequential) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new attack variant, CLE-S, whose probes are trained layer-by-layer and position-by-position on activations already steered by every probe trained earlier in the chain, closing the train/attack distribution mismatch present in CLE-A/P/G's independently-trained probes.

**Architecture:** One reusable "layer cascade" core function (`train_layer_chain`) is called once for the prompt (post-instruction token) and once per generated position (0..4), each generated position seeded by steering-and-decoding one real token with the previous position's completed chain. A new training script (`classifier/train_latent_sequential.py`) drives this; a new attack script (`cle-s.py`) reuses the existing, unmodified `projection_hook`/`generated_projection_hook` hook implementations to apply the trained chain at inference with a fixed steering strength.

**Tech Stack:** Python, PyTorch, scikit-learn (`LinearSVC`), HuggingFace `transformers` (via this repo's `LanguageModel` wrapper classes).

**Reference spec:** `docs/superpowers/specs/2026-07-24-cle-s-design.md`

## Global Constraints

- Steering formula is fixed everywhere (training-data construction, cross-position steered decoding, and attack time): `beta=2.0`, `margin=0.0`. No CLI flag exposes either value — they are module-level constants (`SEQUENTIAL_BETA = 2.0`, `SEQUENTIAL_MARGIN = 0.0` in `classifier/sequential_utils.py`).
- Probe type: SVM only (`LinearSVC(C=0.1, dual="auto", max_iter=1_000_000, random_state=seed)`, matching `classifier/train_latent.py`/`classifier/utils.py::train_layer_svm`). No single-direction sequential chain in this version.
- Exactly 5 generated-position chains by default (`--n_gen_tokens 5`), matching CLE-G's convention.
- `--layers` must be a single contiguous range `"s-e"` (end-exclusive, via `utils.models_utils.parse_layers`) — `"all"` and comma-separated lists are rejected with a `ValueError`, since the cascade requires a strict layer order.
- Target configuration for this work: model `llama32-3b` (28 transformer layers), `--layers 11-23` → selected layers `11..22` (12 layers), matching this repo's prior CLE-P/CLE-G experiments (`completions/llama32-3b/*/completions_*_layers11to23_*.json`).
- No new functions are added to `utils/hooks.py` or `utils/probes.py` — CLE-S reuses `projection_hook`, `generated_projection_hook`, `load_svms`, and `load_generated_svms` exactly as they exist today (verified in Task 1/Task 2/Task 3 below).
- No automated test suite exists in this repo (no pytest, no `tests/` directory) and none is being added — verification is via small smoke runs against the real, already-cached `llama32-3b` model, matching how CLE-A/P/G were validated.
- Do not `git commit` `docs/superpowers/specs/*.md` or `docs/superpowers/plans/*.md` (user preference) — but DO commit the actual code changes in each task as normal.
- Harmless-side prompts/sequences are never steered, at any layer, for any position — always plain greedy decoding / clean forward passes.

---

### Task 1: Sequential layer-cascade core (`classifier/sequential_utils.py`)

**Files:**
- Create: `classifier/sequential_utils.py`

**Interfaces:**
- Consumes: `classifier.utils.train_layer_svm(X_layer, Y, seed=42, test_size=0.1) -> (clf, acc, report)` (existing, `classifier/utils.py:64-79`); `utils.hooks.projection_hook(w, b, beta, margin, eps=1e-12) -> hook_fn` and `utils.hooks.remove_hooks(handles) -> None` (existing, `utils/hooks.py:16-36,133-135`).
- Produces (used by Task 2):
  - `SEQUENTIAL_BETA: float = 2.0`, `SEQUENTIAL_MARGIN: float = 0.0`
  - `encode_prompt(model, prompt: str) -> torch.Tensor` — `(1, seq_len)` LongTensor on `model.device`.
  - `hidden_states_at_last_pos(model, input_ids: torch.Tensor) -> torch.Tensor` — `(num_layers, hidden_dim)` float32 CPU tensor, respects currently-registered forward hooks.
  - `append_greedy_token(model, input_ids: torch.Tensor) -> torch.Tensor` — `(1, seq_len+1)`, no hooks.
  - `steer_and_append_token(model, layers_modules, selected_layers: List[int], probes: Dict[int, Dict[str, torch.Tensor]], input_ids: torch.Tensor, beta: float = SEQUENTIAL_BETA) -> torch.Tensor` — `(1, seq_len+1)`.
  - `train_layer_chain(model, layers_modules, selected_layers: List[int], harmful_ids_list: List[torch.Tensor], harmless_ids_list: List[torch.Tensor], beta: float = SEQUENTIAL_BETA, seed: int = 42) -> Dict[int, Dict[str, object]]` — each value has keys `"w"` (`torch.Tensor(dim,)`), `"b"` (`torch.Tensor(())`), `"accuracy"` (`float`), `"report"` (`str`).
  - `save_chain(probes: Dict[int, Dict[str, object]], out_dir: str, model_name: str, pos: Optional[int]) -> None` — writes `svm_layerXX.pt` (pos=None) or `svm_layerXX_posPP.pt` (pos=int), in the exact `{"w", "b", ...}` dict format `utils.probes.load_svms`/`load_generated_svms` already expect.

- [ ] **Step 1: Write `classifier/sequential_utils.py`**

```python
"""Reusable core for CLE-S: train per-layer probe chains where each probe is
fit on harmful activations already steered by every probe trained earlier in
the chain. See docs/superpowers/specs/2026-07-24-cle-s-design.md.

Shared by the prompt-level chain and each of the 5 generated-position chains
(classifier/train_latent_sequential.py) so the cascade logic exists in exactly
one place, and reused identically at attack time in cle-s.py.
"""

import os
from typing import Dict, List, Optional

import numpy as np
import torch

from classifier.utils import train_layer_svm
from utils.hooks import projection_hook, remove_hooks

SEQUENTIAL_BETA = 2.0
SEQUENTIAL_MARGIN = 0.0


def encode_prompt(model, prompt: str) -> torch.Tensor:
    """Tokenize a prompt through the model's chat template. Returns (1, seq_len)."""
    formatted = model._get_prompt(prompt)
    enc = model.tokenizer(formatted, return_tensors="pt").to(model.device)
    return enc.input_ids


def hidden_states_at_last_pos(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Run one forward pass (any currently-registered forward hooks apply) and
    return the last-token activation for every layer.

    Returns (num_layers, hidden_dim) float32 CPU tensor, embedding layer skipped
    (same layer indexing as LanguageModel.get_representations).
    """
    with torch.no_grad():
        outputs = model.model(input_ids=input_ids, output_hidden_states=True)
    return torch.cat([h[:, -1, :] for h in outputs.hidden_states[1:]], dim=0).to(
        dtype=torch.float32
    ).cpu()


def append_greedy_token(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Decode one token with no steering hooks active (used for harmless sequences)."""
    with torch.no_grad():
        outputs = model.model(input_ids=input_ids)
    next_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    return torch.cat([input_ids, next_id.to(input_ids.device)], dim=1)


def steer_and_append_token(
    model,
    layers_modules,
    selected_layers: List[int],
    probes: Dict[int, Dict[str, torch.Tensor]],
    input_ids: torch.Tensor,
    beta: float = SEQUENTIAL_BETA,
) -> torch.Tensor:
    """Decode one token with `probes` cascaded (projection_hook, fixed beta,
    margin=0.0) across every layer in `selected_layers`, then append the argmax
    token id. Used to produce the real steered continuation between positions.
    """
    handles = [
        layers_modules[layer_idx].register_forward_hook(
            projection_hook(
                probes[layer_idx]["w"].to(device=model.device),
                probes[layer_idx]["b"].to(device=model.device),
                beta,
                SEQUENTIAL_MARGIN,
            )
        )
        for layer_idx in selected_layers
    ]
    try:
        with torch.no_grad():
            outputs = model.model(input_ids=input_ids)
        next_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    finally:
        remove_hooks(handles)
    return torch.cat([input_ids, next_id.to(input_ids.device)], dim=1)


def train_layer_chain(
    model,
    layers_modules,
    selected_layers: List[int],
    harmful_ids_list: List[torch.Tensor],
    harmless_ids_list: List[torch.Tensor],
    beta: float = SEQUENTIAL_BETA,
    seed: int = 42,
) -> Dict[int, Dict[str, object]]:
    """Loop 1: cumulative layer cascade for one token position.

    harmful_ids_list / harmless_ids_list are the (1, seq_len) input_ids for the
    CURRENT token position (prompt-only for the prompt chain; prompt + generated
    tokens so far for a generated-position chain). Harmless is never steered.
    """
    harmless_reps = torch.stack(
        [hidden_states_at_last_pos(model, ids) for ids in harmless_ids_list], dim=0
    )  # (N_harmless, num_layers_total, dim)

    handles = []
    probes: Dict[int, Dict[str, object]] = {}
    try:
        for layer_idx in selected_layers:
            harmful_reps_l = torch.stack(
                [hidden_states_at_last_pos(model, ids)[layer_idx] for ids in harmful_ids_list],
                dim=0,
            )  # (N_harmful, dim); cumulatively steered by every hook registered so far
            harmless_reps_l = harmless_reps[:, layer_idx, :]

            X_layer = torch.cat([harmful_reps_l, harmless_reps_l], dim=0).numpy()
            Y = np.array([1] * len(harmful_ids_list) + [0] * len(harmless_ids_list))
            clf, acc, report = train_layer_svm(X_layer, Y, seed=seed)

            w = torch.from_numpy(clf.coef_[0]).float()
            b = torch.tensor(clf.intercept_[0]).float()
            probes[layer_idx] = {"w": w, "b": b, "accuracy": acc, "report": report}

            handles.append(
                layers_modules[layer_idx].register_forward_hook(
                    projection_hook(
                        w.to(device=model.device), b.to(device=model.device), beta, SEQUENTIAL_MARGIN
                    )
                )
            )
    finally:
        remove_hooks(handles)
    return probes


def save_chain(
    probes: Dict[int, Dict[str, object]], out_dir: str, model_name: str, pos: Optional[int]
) -> None:
    """Save one chain's checkpoints, matching load_svms/load_generated_svms naming:
    svm_layerXX.pt for the prompt chain (pos=None), svm_layerXX_posPP.pt for a
    generated position.
    """
    os.makedirs(out_dir, exist_ok=True)
    for layer_idx, p in probes.items():
        suffix = "" if pos is None else f"_pos{pos:02d}"
        path = os.path.join(out_dir, f"svm_layer{layer_idx:02d}{suffix}.pt")
        torch.save(
            {
                "w": p["w"],
                "b": p["b"],
                "layer_idx": layer_idx,
                "gen_pos": pos,
                "model_name": model_name,
                "accuracy": p["accuracy"],
                "report": p["report"],
            },
            path,
        )
```

- [ ] **Step 2: Verify the module imports cleanly**

Run: `python3 -c "import classifier.sequential_utils as s; print(s.SEQUENTIAL_BETA, s.SEQUENTIAL_MARGIN)"` from the repo root.
Expected: `2.0 0.0` printed, no import errors.

- [ ] **Step 3: Smoke-verify the cascade against the real (already-cached) llama32-3b model**

This repo has no pytest/mock infrastructure for the model wrapper (confirmed: no `tests/` directory, no `conftest.py`), so verification runs the actual small model, matching how `classifier/train_latent.py`/`classifier/train_latent_generated.py` are validated elsewhere in this repo. `HF_TOKEN` must already be set in the environment (required by `models/language_models.py:24`); the model must already be downloaded/cached (this repo has prior `llama32-3b` runs, so it should be).

Run this from the repo root:

```bash
python3 - <<'EOF'
import torch
from classifier.sequential_utils import (
    encode_prompt, hidden_states_at_last_pos, train_layer_chain, save_chain,
)
from classifier.utils import get_model, load_arditi, pick_device
from utils.models_utils import get_transformer_layers

device = pick_device("auto")
model = get_model("llama32-3b", device=device)
harmful_prompts, harmless_prompts = load_arditi("llama32-3b")
harmful_prompts, harmless_prompts = harmful_prompts[:15], harmless_prompts[:15]

layers_modules = get_transformer_layers(model)
assert len(layers_modules) == 28, f"expected 28 layers, got {len(layers_modules)}"
selected_layers = [11, 12]

harmful_ids = [encode_prompt(model, p) for p in harmful_prompts]
harmless_ids = [encode_prompt(model, p) for p in harmless_prompts]

# Baseline: unsteered harmful activation at layer 12 (no hooks registered).
baseline_l12 = hidden_states_at_last_pos(model, harmful_ids[0])[12]

probes = train_layer_chain(model, layers_modules, selected_layers, harmful_ids, harmless_ids)
assert set(probes.keys()) == {11, 12}
for l, p in probes.items():
    assert p["w"].shape == (3072,), p["w"].shape
    assert p["b"].shape == (), p["b"].shape
    assert 0.0 <= p["accuracy"] <= 1.0

save_chain(probes, "/tmp/cle_s_smoke_test/train_svm_sequential", "llama32-3b", pos=None)
import os
assert os.path.exists("/tmp/cle_s_smoke_test/train_svm_sequential/svm_layer11.pt")
assert os.path.exists("/tmp/cle_s_smoke_test/train_svm_sequential/svm_layer12.pt")

# Layer 12's probe should have been trained on activations already steered by
# layer 11's probe -- confirm layer 11's hook was actually applied by re-running
# a forward pass with ONLY layer 11's trained hook active and checking the
# resulting layer-12 input differs from the fully clean baseline.
from utils.hooks import projection_hook, remove_hooks
w11, b11 = probes[11]["w"], probes[11]["b"]
handle = layers_modules[11].register_forward_hook(
    projection_hook(w11.to(device=model.device), b11.to(device=model.device), 2.0, 0.0)
)
try:
    steered_l12 = hidden_states_at_last_pos(model, harmful_ids[0])[12]
finally:
    remove_hooks([handle])
diff = (steered_l12 - baseline_l12).abs().sum().item()
assert diff > 1e-6, "layer-11 steering had no measurable effect on layer 12's activation"

print("OK: train_layer_chain + save_chain smoke test passed. diff =", diff)
EOF
```

Expected: prints `OK: train_layer_chain + save_chain smoke test passed. diff = <some positive number>` with no assertion errors.

- [ ] **Step 4: Commit**

```bash
git add classifier/sequential_utils.py
git commit -m "$(cat <<'EOF'
Add CLE-S sequential layer-cascade core

Shared training helper that fits each layer's probe on harmful activations
already steered by every probe trained earlier in the chain, closing the
train/attack distribution mismatch in the independently-trained CLE probes.
EOF
)"
```

---

### Task 2: CLE-S training script (`classifier/train_latent_sequential.py`)

**Files:**
- Create: `classifier/train_latent_sequential.py`

**Interfaces:**
- Consumes (from Task 1): `classifier.sequential_utils.{SEQUENTIAL_BETA, encode_prompt, append_greedy_token, steer_and_append_token, train_layer_chain, save_chain}`.
- Consumes (existing): `classifier.utils.{get_model, load_arditi, pick_device}` (`classifier/utils.py:34-61`); `utils.models_utils.{get_transformer_layers, parse_layers}` (`utils/models_utils.py:87-116`).
- Produces (used by Task 3): on-disk checkpoints at `<artifact_dir>/<model_name>/train_svm_sequential/svm_layerXX.pt` and `svm_layerXX_posPP.pt` for `PP` in `00..(n_gen_tokens-1)`.

- [ ] **Step 1: Write `classifier/train_latent_sequential.py`**

```python
"""CLE-S training: build per-layer probe chains for the prompt and each of the
first --n_gen_tokens generated positions, where every probe after the first in
a chain is trained on harmful activations already steered by every earlier
probe in that same chain. See docs/superpowers/specs/2026-07-24-cle-s-design.md.

Layer window must be a contiguous range (--layers "s-e", end-exclusive) since
the cascade requires a strict layer order; unlike other CLE probe trainers this
script does not support "all" or a comma-separated layer list.
"""

import argparse

from classifier.sequential_utils import (
    SEQUENTIAL_BETA,
    append_greedy_token,
    encode_prompt,
    save_chain,
    steer_and_append_token,
    train_layer_chain,
)
from classifier.utils import get_model, load_arditi, pick_device
from utils.models_utils import get_transformer_layers, parse_layers


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", type=str, default="llama32-3b")
    parser.add_argument("--device", default="auto", type=str, help="cuda:0 | mps | cpu | auto")
    parser.add_argument("--artifact_dir", type=str, default="./dataset/representations/")
    parser.add_argument(
        "--layers",
        type=str,
        required=True,
        help="Contiguous layer window 's-e' (end-exclusive), e.g. '11-23'. 'all' and comma lists are not supported.",
    )
    parser.add_argument("--n_gen_tokens", type=int, default=5, help="Number of generated-position chains to train.")
    parser.add_argument("--n_samples", type=int, default=-1, help="Prompts per class (-1 = all)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = get_args()
    if args.layers.strip().lower() == "all" or "," in args.layers:
        raise ValueError(f"--layers must be a contiguous 's-e' range for CLE-S, got: {args.layers!r}")

    args.device = pick_device(args.device)
    model = get_model(args.model_name, device=args.device)
    harmful_prompts, harmless_prompts = load_arditi(args.model_name)
    if args.n_samples > 0:
        harmful_prompts = harmful_prompts[: args.n_samples]
        harmless_prompts = harmless_prompts[: args.n_samples]

    layers_modules = get_transformer_layers(model)
    n_layers = len(layers_modules)
    selected_layers = parse_layers(args.layers)
    for layer_idx in selected_layers:
        if layer_idx < 0 or layer_idx >= n_layers:
            raise ValueError(f"Layer index {layer_idx} out of bounds [0, {n_layers - 1}]")

    out_dir = f"{args.artifact_dir.rstrip('/')}/{args.model_name}/train_svm_sequential"

    print(f"Encoding {len(harmful_prompts)} harmful + {len(harmless_prompts)} harmless prompts...")
    harmful_ids = [encode_prompt(model, p) for p in harmful_prompts]
    harmless_ids = [encode_prompt(model, p) for p in harmless_prompts]

    print(f"Training prompt-level chain over layers {selected_layers}...")
    prompt_probes = train_layer_chain(
        model, layers_modules, selected_layers, harmful_ids, harmless_ids, beta=SEQUENTIAL_BETA, seed=args.seed
    )
    save_chain(prompt_probes, out_dir, args.model_name, pos=None)
    mean_acc = sum(p["accuracy"] for p in prompt_probes.values()) / len(prompt_probes)
    print(f"Prompt chain done. Mean test accuracy over {len(prompt_probes)} layers: {mean_acc:.4f}")

    prev_probes = prompt_probes
    for k in range(args.n_gen_tokens):
        print(f"Advancing to generated position {k}...")
        harmful_ids = [
            steer_and_append_token(model, layers_modules, selected_layers, prev_probes, ids, beta=SEQUENTIAL_BETA)
            for ids in harmful_ids
        ]
        harmless_ids = [append_greedy_token(model, ids) for ids in harmless_ids]

        pos_probes = train_layer_chain(
            model, layers_modules, selected_layers, harmful_ids, harmless_ids, beta=SEQUENTIAL_BETA, seed=args.seed
        )
        save_chain(pos_probes, out_dir, args.model_name, pos=k)
        mean_acc = sum(p["accuracy"] for p in pos_probes.values()) / len(pos_probes)
        print(f"Position {k} chain done. Mean test accuracy over {len(pos_probes)} layers: {mean_acc:.4f}")
        prev_probes = pos_probes

    print(f"\nDone. Probes saved under {out_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify --layers validation rejects "all" and comma lists**

Run: `python3 classifier/train_latent_sequential.py --layers all`
Expected: exits immediately with `ValueError: --layers must be a contiguous 's-e' range for CLE-S, got: 'all'` (fails before any model load).

Run: `python3 classifier/train_latent_sequential.py --layers 10,14,20`
Expected: same error shape, `got: '10,14,20'`.

- [ ] **Step 3: Smoke-run the full script on a tiny config**

```bash
rm -rf /tmp/cle_s_train_smoke
python3 classifier/train_latent_sequential.py \
  --model_name llama32-3b \
  --artifact_dir /tmp/cle_s_train_smoke \
  --layers 11-13 \
  --n_gen_tokens 2 \
  --n_samples 15
```

Expected: prints `Prompt chain done. Mean test accuracy over 2 layers: <float>`, then `Advancing to generated position 0...` / `Position 0 chain done. ...`, then the same for position 1, then `Done. Probes saved under /tmp/cle_s_train_smoke/llama32-3b/train_svm_sequential`.

- [ ] **Step 4: Verify the exact checkpoint set was written**

```bash
python3 -c "
import os
d = '/tmp/cle_s_train_smoke/llama32-3b/train_svm_sequential'
expected = {'svm_layer11.pt', 'svm_layer12.pt'}
expected |= {f'svm_layer{l:02d}_pos{k:02d}.pt' for l in (11, 12) for k in (0, 1)}
actual = set(os.listdir(d))
missing = expected - actual
extra = actual - expected
assert not missing, f'missing: {missing}'
assert not extra, f'unexpected extra files: {extra}'
print('OK: exactly the expected 6 checkpoint files are present')
"
```

Expected: `OK: exactly the expected 6 checkpoint files are present`.

- [ ] **Step 5: Verify a loaded checkpoint's format matches what `utils.probes.load_svms`/`load_generated_svms` expect**

```bash
python3 -c "
from utils.probes import load_svms, load_generated_svms
import torch
probes = load_svms('/tmp/cle_s_train_smoke/llama32-3b/train_svm_sequential', [11, 12], torch.device('cpu'))
assert probes[11]['w'].shape == (3072,)
gen_probes, max_pos = load_generated_svms(
    gen_svm_dir='/tmp/cle_s_train_smoke/llama32-3b/train_svm_sequential',
    layer_indices=[11, 12],
    device=torch.device('cpu'),
)
assert max_pos == 1, max_pos
assert gen_probes[11][0]['w'].shape == (3072,)
assert gen_probes[12][1]['w'].shape == (3072,)
print('OK: existing load_svms/load_generated_svms load CLE-S checkpoints unchanged')
"
```

Expected: `OK: existing load_svms/load_generated_svms load CLE-S checkpoints unchanged`.

- [ ] **Step 6: Commit**

```bash
git add classifier/train_latent_sequential.py
git commit -m "$(cat <<'EOF'
Add CLE-S training script

Drives the sequential layer-cascade core across the prompt and 5 generated
positions, steering-and-decoding one real token between positions with each
completed chain so later positions train on a realistic continuation.
EOF
)"
```

---

### Task 3: CLE-S attack script (`cle-s.py`)

**Files:**
- Create: `cle-s.py`

**Interfaces:**
- Consumes (from Task 1): `classifier.sequential_utils.{SEQUENTIAL_BETA, SEQUENTIAL_MARGIN}`.
- Consumes (existing, unmodified): `utils.hooks.{generated_projection_hook, remove_hooks}` (`utils/hooks.py:65-135`); `utils.models_utils.{get_transformer_layers, parse_layers}`; `utils.probes.{ProbeDict, load_svms, load_generated_svms}` (`utils/probes.py:55-137`); `utils.runtime.{chunked, evaluate, load_model, load_prompts, load_prompts_from_file, set_seed, validate_probe_dims}`.
- Produces: `completions/<model_name>/sequential/completions_<run_tag>.json`, same `{"category", "prompt", "response"}` record shape as `cle-a.py`/`cle-p.py`/`cle-g.py`.

- [ ] **Step 1: Write `cle-s.py`**

```python
import argparse
import json
import os
from typing import Dict, List

import torch
from tqdm import tqdm

from classifier.sequential_utils import SEQUENTIAL_BETA, SEQUENTIAL_MARGIN
from utils.hooks import generated_projection_hook, remove_hooks
from utils.models_utils import get_transformer_layers, parse_layers
from utils.probes import ProbeDict, load_generated_svms, load_svms
from utils.runtime import chunked, evaluate, load_model, load_prompts, load_prompts_from_file
from utils.runtime import set_seed, validate_probe_dims


def get_args():
    parser = argparse.ArgumentParser(
        description=(
            "CLE-S attack: steer with sequentially-trained probes. Prompt tokens use "
            "the prompt-chain probe; generated token k uses the position-k probe "
            "(clamped to the last trained position). Steering strength is fixed "
            f"(beta={SEQUENTIAL_BETA}, margin={SEQUENTIAL_MARGIN}) to match training -- "
            "there are no --beta/--margin flags for this variant."
        )
    )
    parser.add_argument("--model_name", type=str, default="llama32-3b")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--svm_dir",
        type=str,
        default=None,
        help="Directory with the prompt-chain probes (svm_layerXX.pt). Defaults to train_svm_sequential.",
    )
    parser.add_argument(
        "--gen_svm_dir",
        type=str,
        default=None,
        help="Directory with the generated-position-chain probes (svm_layerXX_posPP.pt). Defaults to --svm_dir.",
    )
    parser.add_argument(
        "--layers",
        type=str,
        required=True,
        help="Contiguous layer window 's-e' (end-exclusive) matching the trained chain, e.g. '11-23'.",
    )
    parser.add_argument("--dataset", type=str, default="harmbench_test")
    parser.add_argument(
        "--prompts_file",
        type=str,
        default=None,
        help="Optional JSON file with an explicit prompt list, overrides --dataset/--limit.",
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
    gen_probes: Dict[int, Dict[int, Dict[str, torch.Tensor]]],
    max_pos: int,
    step_state: Dict[str, int],
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
                    SEQUENTIAL_BETA,
                    SEQUENTIAL_MARGIN,
                    SEQUENTIAL_MARGIN,
                    layer_idx,
                    first_layer,
                    max_pos,
                    step_state,
                )
            )
        )
    return handles


def generate_with_sequential_probes(
    *,
    batch_prompts: List[str],
    layers,
    selected_layers: List[int],
    model,
    prompt_probes: ProbeDict,
    gen_probes: Dict[int, Dict[int, Dict[str, torch.Tensor]]],
    max_pos: int,
    args,
) -> List[str]:
    step_state: Dict[str, int] = {"step": -1}
    generation_handles = register_sequential_hooks(
        layers, selected_layers, prompt_probes, gen_probes, max_pos, step_state
    )
    try:
        return model.batch_generate(batch_prompts, max_new_tokens=args.max_new_tokens)
    except Exception as e:
        print(f"Batch generation error. Falling back to single-prompt generation: {e}")
        remove_hooks(generation_handles)
        generation_handles = []
        responses = []
        for prompt in batch_prompts:
            single_handles = register_sequential_hooks(
                layers, selected_layers, prompt_probes, gen_probes, max_pos, step_state
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
    if args.layers.strip().lower() == "all" or "," in args.layers:
        raise ValueError(f"--layers must be a contiguous 's-e' range for CLE-S, got: {args.layers!r}")

    device = torch.device(args.device)
    model = load_model(args)
    layers = get_transformer_layers(model)
    n_layers = len(layers)
    hidden_dim = model.model.config.hidden_size

    selected_layers = parse_layers(args.layers)
    for layer_idx in selected_layers:
        if layer_idx < 0 or layer_idx >= n_layers:
            raise ValueError(f"Layer index {layer_idx} out of bounds [0, {n_layers - 1}]")

    if args.svm_dir is None:
        args.svm_dir = os.path.join("./dataset/representations", args.model_name, "train_svm_sequential")
    if not os.path.isdir(args.svm_dir):
        raise FileNotFoundError(f"SVM directory not found: {args.svm_dir}")
    if args.gen_svm_dir is None:
        args.gen_svm_dir = args.svm_dir
    if not os.path.isdir(args.gen_svm_dir):
        raise FileNotFoundError(f"Generated SVM directory not found: {args.gen_svm_dir}")

    prompt_probes = load_svms(args.svm_dir, selected_layers, device)
    validate_probe_dims(prompt_probes, selected_layers, hidden_dim)

    gen_probes, max_pos = load_generated_svms(
        gen_svm_dir=args.gen_svm_dir,
        layer_indices=selected_layers,
        device=device,
        max_pos=None,
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

    base_out_dir = args.out_dir if args.out_dir is not None else os.path.join("./completions", args.model_name, "sequential")
    args.out_dir = base_out_dir
    os.makedirs(args.out_dir, exist_ok=True)

    layers_str = args.layers.replace(",", "_").replace("-", "to")
    limit_str = f"limit{args.limit}" if args.limit else "FULL"
    run_tag = f"{args.dataset}_{limit_str}_layers{layers_str}_seqbeta{SEQUENTIAL_BETA}_maxpos{max_pos}_seed{args.seed}"
    output_path = os.path.join(args.out_dir, f"completions_{run_tag}.json")

    print("--- Configuration ---")
    print(f"Model: {args.model_name}")
    print(f"Layers: {selected_layers}")
    print(f"Beta (fixed): {SEQUENTIAL_BETA} | Margin (fixed): {SEQUENTIAL_MARGIN}")
    print(f"Prompt-chain dir: {args.svm_dir}")
    print(f"Generated-chain dir: {args.gen_svm_dir}")
    print(f"Generated-token positions: 0..{max_pos} (token >{max_pos} reuses pos{max_pos:02d})")
    print(f"Dataset: {args.dataset} | Prompts: {len(prompts)}")
    print(f"Output: {output_path}")

    results = []
    pbar = tqdm(total=len(prompts), desc="CLE-S projection + Generate")
    for batch_start, batch_prompts in chunked(prompts, args.batch_size):
        batch_categories = categories[batch_start:batch_start + len(batch_prompts)]
        responses = generate_with_sequential_probes(
            batch_prompts=batch_prompts,
            layers=layers,
            selected_layers=selected_layers,
            model=model,
            prompt_probes=prompt_probes,
            gen_probes=gen_probes,
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
```

- [ ] **Step 2: Verify --layers validation rejects "all" and comma lists**

Run: `python3 cle-s.py --layers all`
Expected: `ValueError: --layers must be a contiguous 's-e' range for CLE-S, got: 'all'`, fails before any model load.

- [ ] **Step 3: Smoke-run the attack against Task 2's smoke-trained checkpoints**

Reuses the tiny 2-layer, 2-generated-position checkpoint set written in Task 2 Step 3 (`/tmp/cle_s_train_smoke/llama32-3b/train_svm_sequential`).

```bash
python3 cle-s.py \
  --model_name llama32-3b \
  --svm_dir /tmp/cle_s_train_smoke/llama32-3b/train_svm_sequential \
  --layers 11-13 \
  --dataset harmbench_test \
  --limit 1 \
  --max_new_tokens 20 \
  --out_dir /tmp/cle_s_attack_smoke
```

Expected: prints the `--- Configuration ---` block including `Beta (fixed): 2.0 | Margin (fixed): 0.0` and `Generated-token positions: 0..1 (token >1 reuses pos01)`, then `Saved completions to /tmp/cle_s_attack_smoke/completions_....json`.

- [ ] **Step 4: Verify the completions file has a non-empty response**

```bash
python3 -c "
import json, glob
path = glob.glob('/tmp/cle_s_attack_smoke/completions_*.json')[0]
data = json.load(open(path))
assert len(data) == 1
assert set(data[0].keys()) == {'category', 'prompt', 'response'}
assert isinstance(data[0]['response'], str)
print('OK:', repr(data[0]['response'][:80]))
"
```

Expected: `OK: '<some generated text>'` (a non-crashing string; exact content is non-deterministic model output, not asserted).

- [ ] **Step 5: Commit**

```bash
git add cle-s.py
git commit -m "$(cat <<'EOF'
Add CLE-S attack script

Reuses the existing generated_projection_hook/load_svms/load_generated_svms
unchanged, applying the sequentially-trained probe chain at a fixed steering
strength (beta=2.0, margin=0.0) matching how the probes were trained.
EOF
)"
```

---

## Plan self-review notes

- **Spec coverage:** Loop 1 (Task 1 `train_layer_chain`), Loop 2 (Task 1 `steer_and_append_token`/`append_greedy_token` + Task 2's position loop), fixed `beta=2.0`/`margin=0.0` (Task 1 constants, reused in Task 2 and Task 3), SVM-only (Task 1 `train_layer_svm` reuse), contiguous-`--layers`-only validation (Task 2 Step 2, Task 3 Step 2), reuse of unmodified `projection_hook`/`generated_projection_hook`/`load_svms`/`load_generated_svms` (Task 1/Task 3 Interfaces + Task 2 Step 5 verification), post-position-4 clamping via `min(step, max_pos)` (already inside `generated_projection_hook`, not reimplemented), and the `llama32-3b` / `--layers 11-23` target configuration (Global Constraints, exercised at smaller scale in the smoke tests) are all covered by a task.
- **No placeholders:** every step has complete, runnable code or an exact command with a concrete expected-output description.
- **Type/name consistency checked:** `train_layer_chain`, `steer_and_append_token`, `append_greedy_token`, `save_chain`, `SEQUENTIAL_BETA`, `SEQUENTIAL_MARGIN` are named and typed identically across Task 1's Interfaces block, Task 1's code, and their call sites in Task 2/Task 3.
