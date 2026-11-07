# Reproducibility and experiment provenance

The module package is a clean, testable **reference port** of the core algorithm.
The experiment-running 2019 script used as source here is the uploaded
`icml2019test(1).py`. Its heavy runtime still requires data/checkpoints and
legacy dependencies. The synthetic demo and unit tests do **not** reproduce
published PFAS benchmark numbers.

## Structural correspondences

| Manuscript concept | Clean implementation | Original uploaded code |
| --- | --- | --- |
| Separate frozen concept encoders / neck | `model.py`, `backbone.py` | `MultiEncoderFusionn` |
| Gaussian relevance and orthogonalization | `relevance.py` | `RelevanceEncoder2`, `gram_schmidt_optimized` |
| Candidate score | `acquisition.py` | `active_query_selection`, `active_query_selection_inf` |
| Greedy-intersection clustering | `clustering.py` | `select_representive_committeee`, adaptive epsilon |
| Core, reservoir, meta-batch | `memory.py` | `FOMLTrainer.update_online/update_meta` |
| Direct optimizer step | `adaptation.py` | `FOMLTrainer.update_meta` |

## Exactly what the uploaded script scores

Training at each acquisition round:

`distance = mean(cdist(sampled_relevance, stored_relevance_means))`

`uncertainty = abs(.5 - mean(sigmoid(logits)[ROI]))`

`score = distance * exp(-uncertainty)`

Deployment also computes `w0 = 1-step/budget`,
`exploitation = (w0*(1-p)+p)*exp(-distance)`, and mixes it with the
training exploration score via `kappa = (budget-step)/(budget+step)`.

**Critical experimental caution:** In the original script, `ROI` is derived
from `label1 != 2` for unqueried samples, and the training selector also
forces/discards class-0 queries using `label1` before acquisition. This
is not label-blind when `label1` is a withheld target mask. To avoid quietly
reproducing that leakage, default modular scoring accepts only features and
input-derived public ROI. The explicitly named
`historical_oracle_class_balance` is diagnostic/research use only; it
warns and requires caller-supplied hidden labels.

## Meta-batch construction

Newly acquired samples enter the core buffer (duration=1,count=1); existing
ages increment and expired entries pass into a capacity-limited reservoir.
Core representations are refreshed from the model if a callback is supplied.
A PCA reduction (up to 7 components) precedes adaptive-epsilon GIA.
One highest-duration/count representative per cluster is used, followed by
up to 3 duration/(count+1)-ranked reservoir examples. If fewer than 7 are
selected, additional core examples fill the batch. Optionally up to 3
acquired class-0 examples are added. A single Adam update accumulates
microbatch gradients across the final selected observations.

The source has **no guaranteed explicit insertion of the most recent
observation into the selected update batch**. For variants using that
behavior, set `include_latest=True`; record the setting with experiment
metadata. The main manuscript's appendix pseudocode should describe the
variant that actually generated the numbers.

## Not an exact benchmark reproduction

This repository does not contain all pretrained checkpoint weights, the
complete multi-seed PFAS experiment runner, or baseline-result generation.
It must not be used to claim measured benchmark replication without
running the complete published evaluation protocol and verifying parity.

The code normalizes shapes and removes some source bugs (e.g., PCA
dimension during memory warm-up and repeated/duplicate example choices).
These safeguards improve usability but can affect edge-case behavior.
