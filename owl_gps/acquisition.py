
from dataclasses import dataclass
import math
import numpy as np

@dataclass(frozen=True)
class Candidate:
    index: int
    probability: float  
    relevance: np.ndarray  
    probability_map: np.ndarray | None = None

def novelty_score(relevance, references) -> float:
    """Original: torch.cdist(sampled_r, stored_means).mean(), Euclidean."""
    v = np.asarray(relevance, dtype=float).reshape(-1)
    refs = np.asarray(references, dtype=float)
    if not refs.size:
        return 1.0  
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
                                    step: int, batch_size: int = 10,
                                    every: int = 20):
    """Reproduce *historical*, label-informed training query adjustment.

    This requires knowing class-0 presence for UNQUERIED candidates. It
    therefore violates label-blind discovery and must never be used in the
    default policy or to claim label-blind acquisition performance.
    ranked: entire pre-sorted pool (not the initial top-10).
    hidden_class_by_index: 0 when the candidate contains a sparse class-0 label,
        1 otherwise. Caller is responsible for supplying this privileged data.
    """
    import warnings
    warnings.warn(
        "Historical oracle class balancing reads labels of unqueried regions; "
        "not a valid label-blind acquisition method.", RuntimeWarning
    )
    if batch_size < 1 or every < 1:
        raise ValueError("batch_size and every must be positive")
    chosen = list(ranked[:batch_size])
    alternatives = list(ranked[batch_size:])

    def class0(c):
        return hidden_class_by_index[c.index] == 0

    if not chosen:
        return []
    if step % every != 0:
        
        if not any(class0(c) for c in chosen):
            forced = next((c for c in ranked if class0(c)), None)
            if forced is not None:
                chosen[-1] = forced
        
        have_zero = False
        for i, c in enumerate(list(chosen)):
            if not class0(c):
                continue
            if have_zero:
                replacement = next(
                    (a for a in alternatives
                     if not class0(a) and a.index not in {q.index for q in chosen}),
                    None)
                if replacement is not None:
                    chosen[i] = replacement
            else:
                have_zero = True
    else:
        
        for i, c in enumerate(list(chosen)):
            if class0(c):
                replacement = next(
                    (a for a in alternatives
                     if not class0(a) and a.index not in {q.index for q in chosen}),
                    None)
                if replacement is not None:
                    chosen[i] = replacement
    return chosen
