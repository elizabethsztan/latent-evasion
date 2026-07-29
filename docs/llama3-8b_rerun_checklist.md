# LLaMA3-8B experiment re-run checklist

Model key: `llama3-8b`  (registered in `utils/models_utils.py` → `meta-llama/Meta-Llama-3-8B-Instruct`,
and in `classifier/utils.py`). Training splits already exist:
`dataset/splits/llama3-8b/harmful_train_filtered.json` + `harmless_train_filtered.json`.

Everything below writes under `dataset/representations/llama3-8b/`, `completions/llama3-8b/`,
and `results/`.

---

## 0. Config (DECIDED)

```bash
LAYERS=11-18                          # end-exclusive → layers 11,12,13,14,15,16,17 (7 layers)
PROMPT_MARGINS="1.2 2.0 1.8 1.8 2.0 0.9 1.2"   # per-layer, aligned to layers 11..17 (--layer_margin)
TEST=harmbench_standard               # held-out test set (200 prompts), registered in step 2
```

- **PROMPT margins** come from the per-layer margin plot: 11→1.2, 12→2.0, 13→1.8,
  14→1.8, 15→2.0, 16→0.9, 17→1.2.
- **GENERATED-token margins:** *what we did before = inherit the prompt margins.* The
  main llama32-3b CLE-G run carried no `_genmargin` suffix, i.e. generated tokens reused
  the prompt margin. So **omit `--gen_layer_margin`** and the generated probes automatically
  use `[1.2, 2.0, 1.8, 1.8, 2.0, 0.9, 1.2]` per layer. (The `genmargin1.5/1.6` files were
  exploratory reruns on failure subsets, not the headline run.) If you ever want a single
  flat value instead, the mean of these is **1.557** — but that's a new choice, not precedent.
- [ ] **Held-out test set** — register it per step 2 and set `TEST`.

Sanity: 7 layers → 7 prompt margins (already matches). Pre-check the vector/tag with:
`python -m utils.margin_utils --layers 11-18 --layer_margin 1.2 2.0 1.8 1.8 2.0 0.9 1.2`

---

## 1. Train probes (both kinds are required)

### 1a. Post-instruction (prompt) probes → `train_svm/`
Used as the prompt-token probes in CLE-G/A/P, and by every plot.
```bash
python -m classifier.train_latent \
  --model_name llama3-8b \
  --device cuda \
  --artifact_dir ./dataset/representations \
  --n_samples 128
```
- [ ] Produces `dataset/representations/llama3-8b/train_svm/svm_layerXX.pt`,
      `HFx_train.pt`, `HLx_train.pt`.

### 1b. Generated-token probes (5 positions) → `train_svm_generated/`
This is the CLE-G probe set. `--n_gen_tokens 5` = positions pos00..pos04.
```bash
python -m classifier.train_latent_generated \
  --model_name llama3-8b \
  --device cuda \
  --artifact_dir ./dataset/representations \
  --n_samples -1 \
  --n_gen_tokens 5
```
- [ ] Produces `svm_layerXX_posKK.pt` (K=00..04) + `generated_tokens.json`.
- [ ] Check the printed per-position mean SVM test accuracy looks sane.

### 1c. Extract generated activations (only needed for two of the plots) → `gen_activations.pt`
Required by `plot_gen_ownpca` and `plot_gen_scatter_pca` (they scatter the actual
generated points, which 1b throws away). **The `--layers` here MUST match the layers
you pass to those plot scripts** (default `4 14 24` = early/mid/late).
```bash
python -m analysis.extract_generated_activations \
  --model_name llama3-8b \
  --device cuda \
  --n_samples -1 \
  --n_gen_tokens 5 \
  --layers 4 14 24
```
- [ ] Produces `dataset/representations/llama3-8b/train_svm_generated/gen_activations.pt`.

---

## 2. Held-out test set — DONE ✅

`harmbench_standard.json` (200 prompts, `{"instruction","category","behavior_id"}`) is
registered:
- File at `dataset/processed/harmbench_standard.json` (tracked, will push).
- Name whitelisted in `dataset/load_dataset.py` (`PROCESSED_DATASET_NAMES`).
- Verified: `load_dataset("harmbench_standard")` → 200 items, 6 categories.

Use `--dataset harmbench_standard` everywhere. Both files are committed, so the other
computer gets it on `git pull` — no extra setup.

---

## 3. Run the two CLE-G attacks on the held-out test set

Both variants share `cle-g.py`, selected by `--prompt_mode {project,additive}` (built in
this session). `--gen_layer_margin` is intentionally omitted so generated tokens inherit
the prompt margins (see step 0). `--max_pos` defaults to the last trained position (4);
tokens beyond pos04 reuse pos04.

### Variant A — CLE-G (generated) + **CLE-P** on prompt tokens  (`--prompt_mode project`, default)
```bash
python cle-g.py \
  --model_name llama3-8b \
  --device cuda:0 \
  --prompt_mode project \
  --layers 11-18 \
  --layer_margin 1.2 2.0 1.8 1.8 2.0 0.9 1.2 \
  --beta 1.0 \
  --batch_size 16 \
  --dataset $TEST \
  --out_dir ./completions/llama3-8b/cle-g_promptP \
  --evaluate
```
- `svm_dir` / `gen_svm_dir` default to the `train_svm` / `train_svm_generated` dirs from step 1.
- [ ] Completions + evaluation written under `.../cle-g_promptP/` (and `.../evaluation/`).

### Variant B — CLE-G (generated) + **CLE-A** on prompt tokens  (`--prompt_mode additive`)
Now runnable: `--prompt_mode additive` makes `cle-g.py` compute a fixed per-layer additive
delta from a clean prompt pass (CLE-A, via `pipeline_delta_hook`) and replay it on the
prompt tokens during generation, while generated tokens are still live-projected with the
CLE-G gen probes. Run tag gets a `_promptadd` suffix so files don't collide with Variant A.
```bash
python cle-g.py \
  --model_name llama3-8b \
  --device cuda:0 \
  --prompt_mode additive \
  --layers 11-18 \
  --layer_margin 1.2 2.0 1.8 1.8 2.0 0.9 1.2 \
  --beta 1.0 \
  --batch_size 16 \
  --dataset $TEST \
  --out_dir ./completions/llama3-8b/cle-g_promptA \
  --evaluate
```
- [ ] Completions + evaluation written under `.../cle-g_promptA/` (and `.../evaluation/`).

---

## 4. ASR evaluation (HarmBench Llama-2 judge)

Adding `--evaluate` (step 3) already runs the judge. It uses the default `harmbench`
methodology = **`cais/HarmBench-Llama-2-13b-cls`** (`utils/eval_jailbreaks.py`,
`HARMBENCH_LLAMA2_MODEL`), which is exactly the judge you want.

To (re)evaluate an existing completions file explicitly:
```bash
python utils/eval_jailbreaks.py \
  --completions_path ./completions/llama3-8b/cle-g_promptP/completions_<run_tag>.json \
  --methodologies harmbench \
  --evaluation_path ./completions/llama3-8b/cle-g_promptP/evaluation/evaluation_<run_tag>.json
```
- [ ] Variant A ASR recorded (`harmbench_success_rate` in the evaluation JSON).
- [ ] Variant B ASR recorded (once Variant B is buildable).
- Optional baseline: run the model with no intervention on the same `$TEST` for a clean ASR reference.

---

## 5. Plots (all default to `--model_name llama32-3b`; pass `--model_name llama3-8b`)

Each script writes both `.png` and `.pdf` with the model name in the filename.
Depends on step 1 artifacts as noted.

| Target file (want `_llama3-8b`) | Script | Command | Extra deps |
|---|---|---|---|
| `results/probe_boundary/probe_boundary_drift_pca_llama3-8b.pdf` | `analysis/plot_drift_pca.py` | `python -m analysis.plot_drift_pca --model_name llama3-8b --layers 4 14 24` | train_svm + train_svm_generated |
| `results/probe_boundary/probe_boundary_gen_ownpca_llama3-8b.png` | `analysis/plot_gen_ownpca.py` | `python -m analysis.plot_gen_ownpca --model_name llama3-8b --layers 4 14 24` | + `gen_activations.pt` (step 1c) |
| `results/probe_boundary/probe_boundary_gen_scatter_pca_llama3-8b.png` | `analysis/plot_gen_scatter_pca.py` | `python -m analysis.plot_gen_scatter_pca --model_name llama3-8b --layers 4 14 24` | + `gen_activations.pt` (step 1c) |
| `results/heatmaps/refusal_drift_grid_llama3-8b.png` | `analysis/plot_refusal_drift.py` | `python -m analysis.plot_refusal_drift --model_name llama3-8b` | train_svm + train_svm_generated |
| `results/token_heatmap/token_probe_heatmap_llama3-8b.png` | `analysis/plot_token_probe_heatmap.py` | `python -m analysis.plot_token_probe_heatmap --model_name llama3-8b --device cuda` | runs model live (+ train_svm) |
| `results/token_heatmap/token_probe_heatmap_ownprobe_llama3-8b.png` | `analysis/plot_token_probe_heatmap.py` | `python -m analysis.plot_token_probe_heatmap --model_name llama3-8b --device cuda --gen_probe own` | + train_svm_generated |

Checklist:
- [ ] `probe_boundary_drift_pca_llama3-8b.pdf`
- [ ] `probe_boundary_gen_ownpca_llama3-8b.png`
- [ ] `probe_boundary_gen_scatter_pca_llama3-8b.png`
- [ ] `refusal_drift_grid_llama3-8b.png`
- [ ] `token_probe_heatmap_llama3-8b.png`
- [ ] `token_probe_heatmap_ownprobe_llama3-8b.png`

> ⚠️ Layer consistency: the three `probe_boundary` plots default to layers `4 14 24`.
> Whatever layers you pass to `plot_gen_ownpca` / `plot_gen_scatter_pca` must match the
> `--layers` you gave `extract_generated_activations` in step 1c, or the plots won't find
> the stored activations.

---

## Quick dependency order
1c needs the model but not 1a/1b. 1a → needed by 1b? No (independent). But CLE-G attacks
need BOTH 1a and 1b. Plots need 1a (+1b, +1c for the two ownpca/scatter plots).
Recommended order: **1a → 1b → 1c → 2 → 3(A) → 3(B) → 5**.
