# Reproducibility and compatibility

## Origin

The architecture in `owl_gps/backbone.py`, `owl_gps/heads.py`, and
`RelevanceEncoder2` is extracted from the author's original ~4,300-line
2019 research script stored in the private `OWL-GPS_NeurIPS` repository.

The direct Adam batch-gradient update, acquisition-score structure, KL term,
and core/reservoir memory logic follow that script conceptually.

## Important differences from the historical experiment

- The original code includes a training-time acquisition quota that examines
  class labels of candidates *before* acquisition. It was intentionally
  excluded here to honor the paper's hidden-label protocol.
- The original inference selector uses label-derived valid-pixel masks. The
  clean scorer receives a scalar probability computed over an input-derived
  region of interest instead; it must not access hidden labels.
- The memory selector here uses a transparent greedy-diversity representative
  selection, not the original greedy-intersection algorithm with PCA.
- The original model's per-concept pretrained encoders need the original
  checkpoint files and legacy-compatible timm, mmcv, and mmseg dependencies.
- The old script applies per-pixel noise masks to its focal loss and uses
  two segmentation heads. The clean loss accepts explicit noise weights, and
  will not silently derive their paths.
- The old script uses warmup/poly decay and sometimes takes five Adam steps.
  `OnlineAdapter` implements the direct update but not the same LR schedule.
- The light synthetic example is just a smoke demonstration; its architecture
  and results are not the paper's experiments.

**Consequently, do not claim bitwise reproduction or use this package alone
to reproduce the published scores.** Original pretrained checkpoints, data
splits, normalization statistics, full training procedure, and a parity
check against the historical scripts are needed.

## Labels and evaluation

Observed pixel labels are `0/1`; `2` denotes ignore. Pseudo/expanded
masks may train the predictor. Evaluate only on measured masks; no
pre-acquisition code should use the target label map or class counts.

## Verification checklist

1. Obtain the exact 33 pretrained concept checkpoints for 2019, the
   normalization means/stds, and paired imagery and sparse ground-truth masks.
2. Validate `TemporalViTEncoder` and `DualSegmentationHead` checkpoint
   loading in the authors' compatible Torch/timm/MMCV/MMSeg environment.
3. Verify acquisition scores, memory batches, and gradients against saved
   reference traces from the original implementation.
4. Check no hidden labels or masks derived from targets are used for scoring.
5. Re-run each published experiment with identical splits and seeds.
