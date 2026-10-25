"""Label-blind candidate scoring based on OWL-GPS training/inference loops."""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class Candidate:
    index: int
    probability: float          # mean predicted P(target) over publicly known ROI
    relevance: np.ndarray       # sampled / normalized relevance vector


def novelty_score(relevance, references) -> float:
    """Mean Euclidean distance in relevance space, as in research code."""
    v = np.asarray(relevance, dtype=float).reshape(-1)
    refs = np.asarray(references, dtype=float)
    if refs.size == 0:
        return 1.0
    refs = refs.reshape(-1, v.size)
    return float(np.linalg.norm(refs - v, axis=1).mean())


def score_candidate(candidate: Candidate, references, *, step: int,
                    budget: int, phase: str = "train") -> float:
    """Scoring as observed in active_query_selection[ _inf ].

    Unlike the old script, this function never reads held-out target labels.
    Pass an ROI mask derived from input-side information when computing
    candidate.probability; a ground-truth-derived mask would leak labels.
    """
    if budget < 1:
        raise ValueError("budget must be positive")
    if phase not in {"train", "inference"}:
        raise ValueError("phase must be train or inference")
    p = float(candidate.probability)
    if not 0 <= p <= 1:
        raise ValueError("probability must be in [0,1]")
    distance = novelty_score(candidate.relevance, references)
    uncertainty = abs(.5 - p)
    explore = distance * math.exp(-uncertainty)
    if phase == "train":
        return explore
    kappa = (budget - step) / (budget + step)
    kappa = min(1.0, max(0.0, kappa))
    # Original inference loop combines avg P(class1) with a weighted P(class0).
    w0 = 1 - step / budget
    exploit = (w0 * (1-p) + p) * math.exp(-distance)
    return kappa * explore + (1 - kappa) * exploit


def choose_candidates(candidates, references, *, step: int,
                      budget: int, phase: str, batch_size: int):
    ranked = sorted(
        candidates,
        key=lambda c: score_candidate(c, references, step=step,
                                      budget=budget, phase=phase),
        reverse=True,
    )
    # Do not access labels while scoring or rerank by hidden target class.
    return ranked[:min(batch_size, len(ranked), budget - step)]
