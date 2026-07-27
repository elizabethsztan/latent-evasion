# Llama 3.2 3B probe scores for CLE-P and CLE-A

Mean raw linear-SVM decision scores, \(w^\top h + b\), over the 124 harmful
prompts in the model-specific Arditi training split. Positive values are on the
harmful side of the probe boundary; negative values are on the harmless side.

Configuration:

- Model: `llama32-3b`
- Prompt split: `dataset/splits/llama32-3b/harmful_train_filtered.json`
- Layers intervened on: `11-23` (end-exclusive)
- Beta: `1.0`
- Margin: `1.2`
- CLE-P values use the live projection cascade.
- CLE-A values use the second, fixed-delta additive replay.

| Layer | Mean score for harmful unsteered | Mean score for CLE-P pre-L (orange) | Mean score for CLE-P post-L (green) | Mean score for CLE-A pre-L (orange) | Mean score for CLE-A post-L (green) |
|---:|---:|---:|---:|---:|---:|
| 11 | 1.64 | 1.64 | -1.20 | 1.64 | -1.20 |
| 15 | 1.40 | -1.13 | -1.20 | -1.27 | -1.34 |
| 18 | 1.28 | -1.13 | -1.20 | -1.37 | -1.44 |
| 22 | 1.24 | -1.20 | -1.20 | -1.54 | -1.53 |

Layer 11 is the first selected intervention layer, so there are no earlier hooks:

- CLE-P pre-L and CLE-A pre-L are the same live re-encoded activations.
- CLE-P post-L and CLE-A post-L apply the same layer-11 delta and agree to the
  reported precision.
- The harmful-unsteered mean is calculated from the saved `HFx_train.pt` red
  cloud, which was extracted in an earlier model run. The live re-encoding used
  for the orange clouds differs from that saved extraction by `0.002498` in mean
  probe score because of floating-point/MPS run-to-run numerical variation. This
  small discrepancy is not an effect of steering.

The values are emitted by:

```bash
python -m analysis.plot_pipeline_boundary_pca --model_name llama32-3b
python -m analysis.plot_additive_boundary_pca --model_name llama32-3b
```
