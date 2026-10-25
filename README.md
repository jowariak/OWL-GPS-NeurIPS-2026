# OWL-GPS

**Adapting Actively on the Fly: Relevance-Guided Sequential Learning with Latent Concepts for Geospatial Discovery**

**NeurIPS 2026**

Jowaria Khan · Anindya Sarkar · Yevgeniy Vorobeychik · Elizabeth Bondi-Kelly

[Project page](https://jowariak.github.io/OWL-GPS/) · [Paper](https://jowariak.github.io/OWL-GPS/paper.pdf)

## Overview

OWL-GPS is a framework for **sequential geospatial discovery** when labels are expensive, acquisition budgets are limited, and previously observed examples cannot be retained indefinitely. Candidate regions are visible to the acquisition policy, but their labels are revealed only after querying.

The framework uses a shared, concept-based relevance representation for two coupled decisions:

- **Relevance-guided acquisition:** selecting regions to query using learned concept relevance together with prediction uncertainty and exploration–exploitation criteria.
- **Relevance-aware bounded-memory meta-batches:** selecting diverse, informative observations from core and reservoir memory to support online model adaptation.

In the evaluated implementation, online adaptation optimizes the model directly over the relevance-structured batch, including newly acquired supervision, rather than performing a separate bilevel meta-gradient update.

## Evaluation

The primary evaluation studies PFAS contamination discovery using sparsely observed measurements. An additional sparsified land-cover prediction task tests cross-domain applicability.

## Code and reproducibility

This repository is being prepared for the official code release. Installation instructions, dataset preparation, training and inference commands, and configuration details will be added together with the verified release files.

Large datasets and model checkpoints will be hosted separately where appropriate; avoid committing data, credentials, and generated checkpoints to this repository.

## Citation

Jowaria Khan, Anindya Sarkar, Yevgeniy Vorobeychik, and Elizabeth Bondi-Kelly. *Adapting Actively on the Fly: Relevance-Guided Sequential Learning with Latent Concepts for Geospatial Discovery*. NeurIPS 2026.
