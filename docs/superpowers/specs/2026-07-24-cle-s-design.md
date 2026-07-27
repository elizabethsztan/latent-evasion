# CLE-S (Sequential) — Design

## Motivation

Every existing CLE variant (CLE-A, CLE-P, CLE-G) trains one linear probe `φ_l = (w_l, b_l)` per layer `l` completely independently, on **clean** (unsteered) harmful/harmless activations (`classifier/train_latent.py:92-159`, `classifier/train_latent_generated.py`). But at attack time, hooks are registered on multiple transformer layers within a chosen window `[s,e]`, and because `register_forward_hook` fires in layer order during a single forward pass, layer `l+1`'s hook already receives whatever layer `l`'s hook modified (`utils/hooks.py`, `cle-p.py:97-116`). So every probe below the top of the window is, in practice, asked to classify activations that have already been partially steered toward the harmless side — a distribution it never saw during training.

CLE-S closes this training/attack mismatch by training each probe on activations that reflect the *actual* cascading steering effect of every probe trained before it, so that by the time an attack reaches layer `l+1`, `φ_{l+1}` already has a sensible decision boundary for the partially-evaded activations it will actually see.

## Core mechanism

Two nested procedures:

**Loop 1 — layer cascade (within one token position).** For the prompt, this operates on the **post-instruction token** specifically (the last token of the chat-templated prompt, before any generation) — matching the existing repo terminology for the prompt-level probe (`cle-g.py:35`, `--svm_dir` help text: "the post-instruction probes ... applied to prompt tokens"), not any other token within the prompt. For a generated position, it operates on that generated token's activation. Given a token position whose activation at the first layer of the window is clean (unsteered), walk `l = s..e` in increasing order:
1. Run a forward pass with `projection_hook(w_i, b_i, beta=2.0, margin=0.0)` (`utils/hooks.py:16-36`) registered at every layer `i` already trained in this position (`s..l-1`), on the harmful prompt. Extract the resulting (cumulatively steered) activation at layer `l`.
2. Fit `φ_l` via `LinearSVC(C=0.1, dual="auto", max_iter=1_000_000, random_state=42)` (same hyperparameters as `classifier/train_latent.py:101`) on `(harmless_h_l [clean], harmful_h_l [steered])`.
3. Register `projection_hook(w_l, b_l, beta=2.0, margin=0.0)` at layer `l` permanently for this position's remaining steps, and move to `l+1`.

The harmless side is never steered, anywhere, for any position — always a clean forward pass.

**Loop 2 — position cascade (across the prompt and 5 generated tokens).** The prompt (last instruction token) is the base case: Loop 1 runs on it starting from genuinely clean activations, exactly like base CLE. Once its chain `φ_s..φ_e` is fully trained:
1. Use that completed chain as real steering hooks (`projection_hook`, `beta=2.0`, `margin=0.0`, one hook per layer `s..e`, cascading naturally in one forward pass) to actually decode the next real token for the harmful prompts. Harmless prompts continue with no hooks (plain greedy decode).
2. That new token's activation at layer `s` is clean (nothing has steered it yet) — run Loop 1 again, seeded from this token instead of the prompt, producing `φ_{s,0}..φ_{e,0}`.
3. Repeat for positions 1, 2, 3, 4, each seeded by steering-and-decoding one more real token using the *previous* position's completed chain.

Total probes trained: `(e-s+1)` prompt probes + `5 × (e-s+1)` generated-position probes.

## Target configuration

This design targets the model already loaded for this work: **llama32-3b** (28 transformer layers total, per `get_transformer_layers`). Layer window: **`--layers 11-23`**, matching the exact CLI value already used in this repo's prior CLE-P/CLE-G experiments (`completions/llama32-3b/{pipeline,generated}/completions_*_layers11to23_*.json`). Under the repo's existing end-exclusive `--layers "s-e"` parsing, this selects layers `11..22` (12 layers) — so CLE-S trains `12` prompt probes and `5 × 12 = 60` generated-position probes for this configuration.

## Scope for this version

- **Layer window is fixed at training time.** `--layers` must be one contiguous range (`"s-e"`, end-exclusive, matching `utils/models_utils.py:107-116`'s existing range syntax) — not `"all"` or a comma list, since the cascade requires a strict layer order. A different window requires retraining.
- **Probe type: SVM only.** No single-direction (difference-of-means) sequential chain in this version.
- **5 generated positions**, matching CLE-G's convention of training a small fixed number of generated-token probes.
- **Steering formula is fixed, not tunable**: `beta=2.0`, `margin=0.0`, everywhere — during training-data construction, during the real steered-decoding step between positions, and at attack time. This is deliberate: since the probe chain was trained anticipating exactly this steering strength, changing it at attack time would reintroduce the same train/attack mismatch CLE-S exists to remove. No `--beta`/`--margin`/`--layer_margin`/`--gen_margin` CLI flags for this variant.
- **Past position 4**: attack-time steering keeps reapplying position 4's probes for all subsequent generated tokens (not "steering off"). This falls out of `generated_projection_hook`'s existing `gen_probes[min(step_state["step"], max_pos)]` clamping (`utils/hooks.py:98`) with no code changes needed — feeding it `max_pos=4` already gives this behavior for free.

## Files

**New:**
- `classifier/train_latent_sequential.py` — training script. CLI mirrors `classifier/train_latent.py`'s scaffolding (`--model_name`, `--device`, `--artifact_dir`) plus a required `--layers "s-e"`. Implements one reusable function (the shared "approach B" core) for Loop 1, called once for the prompt and once per generated position, with Loop 2's steer-and-decode-one-token step driving position-to-position transitions in `main()`.
- `cle-s.py` — attack script. Same overall shape as `cle-g.py` (model/device args, dataset/`--prompts_file`/`--limit`/`--max_new_tokens`/`--batch_size`, `--out_dir`/`--evaluate`/`--seed`, `--layers`), but:
  - No beta/margin flags anywhere — `beta=2.0`, `prompt_margin=0.0`, `gen_margin=0.0` hardcoded at the two `generated_projection_hook(...)` call sites.
  - Reuses `utils/hooks.py::generated_projection_hook` and `utils/probes.py::load_svms` / `load_generated_svms` **unchanged** — no new hook function is needed. `register_generated_hooks`/`generate_with_generated_probes` (`cle-g.py:148-219`) can be copied nearly verbatim, just without the margin-map plumbing.
  - `--probe_type` is not exposed (SVM only, per scope above).

**Modified:** none required in `utils/hooks.py` or `utils/probes.py` — CLE-S reuses `projection_hook` and `generated_projection_hook` as-is (called with fixed `beta=2.0`/`margin=0.0` args) for both training-data construction and attack-time inference, and reuses `load_svms`/`load_generated_svms` as-is since the checkpoint format (`{"w":..., "b":..., ...}` dict, `svm_layerXX.pt` / `svm_layerLL_posPP.pt` naming) is unchanged.

**New checkpoint artifacts** (same naming convention as base CLE / CLE-G, new directory to avoid colliding with independently-trained probes):
- `dataset/representations/<model>/train_svm_sequential/svm_layerXX.pt` — prompt-chain probes.
- `dataset/representations/<model>/train_svm_sequential/svm_layerXX_posP.pt` — generated-position-chain probes (`P` in `0..4`).

## Training-time token generation (Loop 2 mechanics)

Producing the real "next token" between positions requires stepping the model one token at a time with steering hooks active and recovering the actual token id chosen (not just decoded text), so it can be appended to a running sequence and fed back in for the next position's Loop 1. This needs a manual single-step decode (forward pass → hooked logits → argmax/sample → append token id), rather than the model wrapper's higher-level `generate`/`batch_generate` used elsewhere in the repo. The exact integration point (which model-wrapper method to extend or bypass) is left to the implementation plan, since it depends on inspecting `models/language_models.py`'s current generation loop.

## Error handling

Follows existing repo conventions rather than introducing new validation:
- `LinearSVC` fit failures are allowed to raise (no silent fallback), same as `classifier/train_latent.py`.
- `cle-s.py`'s batched generation falls back to per-prompt generation on exception, same try/except pattern as `cle-a.py`/`cle-p.py`/`cle-g.py`.
- Missing/incomplete checkpoints raise `FileNotFoundError`/`ValueError` the same way `load_svms`/`load_generated_svms` already do — no new bespoke window-validation logic.

## Validation

No new automated test suite (matches how CLE-A/P/G were validated in this repo): run the actual pipeline (train → attack → HarmBench-judge eval via `utils/eval_jailbreaks.py`) and compare attack-success-rate against existing CLE-P/CLE-G results at a comparable layer window, on a small window first as a smoke test.

## Cost note

Training cost scales roughly as `O((e-s) layers × 6 positions)` forward passes per harmful prompt (one incremental forward pass per new layer trained, per position), substantially more than base CLE's single clean forward pass per prompt. Expected and accepted given the goal.
