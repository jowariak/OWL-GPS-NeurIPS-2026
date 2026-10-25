# OWL-GPS

**Adapting Actively on the Fly: Relevance-Guided Sequential Learning with Latent Concepts for Geospatial Discovery**

**NeurIPS 2026** · Jowaria Khan, Anindya Sarkar, Yevgeniy Vorobeychik, Elizabeth Bondi-Kelly

[Project page](https://jowariak.github.io/OWL-GPS/) · [Paper](https://jowariak.github.io/OWL-GPS/paper.pdf)

OWL-GPS studies budgeted sequential geospatial discovery under sparse
supervision, distribution shift, and bounded replay memory. Its shared
concept-relevance representation guides both *where to acquire labels* and
*which acquired observations to use for online adaptation*.

## Repository structure

| Path | Purpose |
| --- | --- |
| `owl_gps/backbone.py` | Original single-band Temporal ViT and feature neck |
| `owl_gps/heads.py` | Original dual-head MMSeg segmentation decoder |
| `owl_gps/relevance.py` | Original relevance encoder, relevance sampling and Gram–Schmidt |
| `owl_gps/model.py` | Concept fusion and original 2019-model builder |
| `owl_gps/data.py` | Aligned band TIFFs, pseudo-labels and measured labels |
| `owl_gps/acquisition.py` | Label-blind concept-guided query scoring |
| `owl_gps/memory.py` | Core/reservoir replay memory and relevance-structured meta-batches |
| `owl_gps/adaptation.py` | Direct Adam updates on acquired relevance-structured batches |
| `owl_gps/losses.py` | Binary focal loss and relevance CVAE KL loss |
| `owl_gps/metrics.py` | Evaluation at measured pixels only |
| `configs/2019.example.yaml` | Portable path/budget configuration template |
| `examples/synthetic_demo.py` | Small CPU end-to-end smoke demonstration |
| `tests/test_components.py` | Deterministic unit tests |
| `docs/REPRODUCIBILITY.md` | Reproduction requirements and implementation differences |

## Install

Requires Python 3.10+ and PyTorch.

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python examples/synthetic_demo.py
```

The synthetic example uses a **toy model and synthetic labels**. Its losses
are not PFAS measurements or evidence of the published experimental scores.

For GeoTIFF loading:

```bash
python -m pip install -e ".[geo]"
```

The original heavy encoder and decoder use **legacy timm/MMCV/MMSeg APIs**.
Install versions compatible with your model checkpoints separately. See
`docs/REPRODUCIBILITY.md` before attempting the original 2019 backbone.
For example, the model constructor needs 33 original pretrained concept
checkpoints, in concept order:

```python
from owl_gps.model import build_original_2019_model

with open("/path/to/concept_checkpoints.txt", encoding="utf-8") as f:
    paths = [line.strip() for line in f if line.strip()]
model = build_original_2019_model(paths)
```

Provide local dataset paths and normalization statistics through configuration,
not hardcoded `/scratch` or `/nfs` strings. No raw data or model weights are
committed to this repository.

## Method notes

- The implemented online update optimizes directly over acquired
  relevance-structured **meta-batches**, rather than computing bilevel
  meta-gradients or a separate newest-sample gradient.
- Newly acquired observations enter bounded memory before batch construction.
- Labels must remain hidden from acquisition until the query is made.
- Pseudo-labels may train the model, but reported evaluations should use
  independent measured ground-truth pixels.


## Citation

Jowaria Khan, Anindya Sarkar, Yevgeniy Vorobeychik, and Elizabeth Bondi-Kelly.
*Adapting Actively on the Fly: Relevance-Guided Sequential Learning with Latent
Concepts for Geospatial Discovery.* NeurIPS 2026.


