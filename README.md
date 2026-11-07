# OWL-GPS

**Adapting Actively on the Fly: Relevance-Guided Sequential Learning with Latent Concepts for Geospatial Discovery**

**NeurIPS 2026** · Jowaria Khan, Anindya Sarkar, Yevgeniy Vorobeychik, Elizabeth Bondi-Kelly


[Project page](https://jowariak.github.io/OWL-GPS/) · [Paper](https://jowariak.github.io/OWL-GPS/paper.pdf) · [Dataset](https://doi.org/10.7302/zzpc-2s61)


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
| `owl_gps/clustering.py` | PCA and adaptive-epsilon greedy-intersection clustering (GIA)  |\n| `owl_gps/memory.py` | Core/reservoir policy, GIA representative selection, historical class-0 replay augmentation |
| `owl_gps/adaptation.py` | Direct Adam updates on acquired relevance-structured batches |
| `owl_gps/losses.py` | Binary focal loss and relevance CVAE KL loss |
| `owl_gps/metrics.py` | Evaluation at measured pixels only |
| `configs/2019.example.yaml` | Portable path/budget configuration template |
| `examples/synthetic_demo.py` | Small CPU end-to-end smoke demonstration |
| `tests/test_components.py` | Deterministic unit tests |


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

## Dataset

The PFAS geospatial dataset used in this study is available through the University of Michigan's Deep Blue Data repository.

**Dataset:** [OWL-GPS PFAS Dataset: Multisource Geospatial Imagery and Sparse Ground-Truth Labels for Contamination Discovery](https://doi.org/10.7302/zzpc-2s61)

**DOI:** [10.7302/zzpc-2s61](https://doi.org/10.7302/zzpc-2s61)


## Citation

Jowaria Khan, Anindya Sarkar, Yevgeniy Vorobeychik, and Elizabeth Bondi-Kelly.
*Adapting Actively on the Fly: Relevance-Guided Sequential Learning with Latent
Concepts for Geospatial Discovery.* NeurIPS 2026.


