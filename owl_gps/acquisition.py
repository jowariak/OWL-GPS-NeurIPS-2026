"""Acquisition scores corresponding to icml2019test(1).py.

The historical code ranks mean Euclidean novelty times exponential uncertainty,
not the sum-of-squared-distances expression printed in the paper. Here, target
labels are NEVER needed by the default scorer. The original source additionally
used withheld sparse labels to mask and force class-balanced queries; that
oracle-assisted intervention is deliberately not presented as label-blind.
"""
from dataclasses import dataclass
import math
import numpy as np

@dataclass(frozen=True)
class Candidate:
    index: int
    probability: float  # mean model probability on an observable ROI
    relevance: np.ndarray  # sampled, softmax-normalized relevance
    probability_map: np.ndarray | None = None

def novelty_score(relevance, references) -> float:
    """Original: torch.cdist(sampled_r, stored_means).mean(), Euclidean."""
    v = np.asarray(relevance, dtype=float).reshape(-1)
    refs = np.asarray(references, dtype=float)
    if not refs.size:
        return 1.0  # initial-query convention; normally a seed is provided
    refs = refs.reshape(-1, v.size)
    return float(np.linalg.norm(refs - v, axis=1).mean())

def mean_probability(probabilities, *, observable_roi=None) -> float:
    """Aggregate sigmoid probabilities using input-derived ROI only.

    Do not pass withheld sparse target masks; these were accessed by the
    original training/inference script before acquisition.
    """
    p = np.asarray(probabilities, dtype=float)
    if observable_roi is None:
        return float(p.mean())
    roi = np.asarray(observable_roi, dtype=bool)
    if roi.shape != p.shape or not roi.any():
        raise ValueError("ROI must match probability map and include pixels")
    return float(p[roi].mean())

def score_candidate(candidate: Candidate, references, *, step: int,
                    budget: int, phase: str = "train") -> float:
    """Faithful scoring algebra from the uploaded original script.

    train: mean_distance * exp(-abs(0.5 - mean_probability))
    inference:
        kappa * train_score
        + (1-kappa) * (w0*(1-p)+p) * exp(-mean_distance)
    where kappa=(budget-step)/(budget+step), w0=1-step/budget.
    This is NOT equations (11)--(14) verbatim.
    """
    if budget <= 0 or step < 0 or step > budget:
        raise ValueError("Invalid budget or step")
    if phase not in {"train", "inference"}:
        raise ValueError("phase must be train or inference")
    p = float(candidate.probability)
    if not 0 <= p <= 1:
        raise ValueError("probability must be in [0, 1]")
    distance = novelty_score(candidate.relevance, references)
    explore = distance * math.exp(-abs(0.5 - p))
    if phase == "train":
        return explore
    kappa = (budget-step)/(budget+step)
    w0 = 1-step/budget
    exploit = (w0*(1-p)+p)*math.exp(-distance)
    return kappa*explore + (1-kappa)*exploit

def choose_candidates(candidates, references, *, step: int, budget: int,
                      phase: str, batch_size: int):
    """Rank unqueried candidate features; never examine target labels."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    ranked = sorted(candidates, key=lambda c: score_candidate(
        c, references, step=step, budget=budget, phase=phase), reverse=True)
    return ranked[:max(0, min(batch_size, len(ranked), budget-step))]

def historical_oracle_class_balance(ranked, hidden_class_by_index, *,
                                    step: int, every: int = 20):
    """EXPERIMENTAL LEGACY/ORACLE MODE ONLY -- NOT VALID LABEL-BLIND SEARCH.

    Reproduces the intention of original training's class-0 query adjustment.
    Requires privileged labels for *unqueried* regions, so should not be used
    for fair online discovery evaluation. Keep separate from default policy.
    """
    import warnings
    warnings.warn("Oracle class balancing accesses withheld query labels; "
                  "do not use for label-blind benchmark reporting.", RuntimeWarning)
    chosen = list(ranked)
    if not chosen:
        return chosen
    if step % every:
        if not any(hidden_class_by_index[c.index] == 0 for c in chosen):
            extra = next((idx for idx,v in hidden_class_by_index.items() if v == 0), None)
            if extra is not None:
                raise ValueError("Pass a full ranked candidate pool to reproduce "
                                 "oracle class balancing; selected-only pool cannot force index")
    return chosen
