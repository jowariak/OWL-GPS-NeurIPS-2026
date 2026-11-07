"""PCA + adaptive-epsilon Greedy Intersection Algorithm (GIA).

Faithful port of the ACTIVE clustering path in icml2019test(1).py:
 FOMLTrainer.update_clusters -> PCA(n_components=7)
 select_representive_committeee -> greedy clusters + nearest assignment
 choose_eps_for_target_coverage -> median spacing, 80%-coverage target
 select_representatives_intersectionss_fast -> axis-wise greedy intersections

This is the fast greedy-intersection variant, NOT the unused exponential-time
Cartesian-product intersection function also present in the research script.
Identical relevance vectors retain separate buffer IDs: the source's vector
deduplication could otherwise make cluster indices point at wrong observations.
"""
from __future__ import annotations

from collections.abc import Mapping
import numpy as np
from sklearn.decomposition import PCA


def axis_coverage_sets(points: Mapping[int, np.ndarray], epsilon: float):
    """Exactly the source's per-axis, 2*epsilon coverage-set construction."""
    if not points:
        raise ValueError("points must not be empty")
    if epsilon < 0:
        raise ValueError("epsilon must be nonnegative")
    dimension = len(next(iter(points.values())))
    coverage_sets = []
    for axis in range(dimension):
        ranked = sorted(points.items(), key=lambda item: item[1][axis])
        coverage_list = [[index] for index, _ in ranked]
        axis_sets = [set(coverage_list[0])]
        for i in range(1, len(ranked)):
            index_i = coverage_list[i][0]
            for j in range(i - 1, -1, -1):
                index_j = coverage_list[j][0]
                if points[index_i][axis] <= points[index_j][axis] + 2 * epsilon:
                    coverage_list[i].append(index_j)
                else:
                    break
            candidate = set(coverage_list[i])
            if axis_sets[-1] <= candidate:
                axis_sets[-1] = candidate
            else:
                axis_sets.append(candidate)
        coverage_sets.append(axis_sets)
    return coverage_sets


def greedy_intersection_representative(points: Mapping[int, np.ndarray],
                                       epsilon: float):
    """Exact greedy-intersection scoring and fallback from the fast source.

    Each axis chooses the coverage set with maximum overlap with the
    surviving observations. If overlap is empty, reset to the chosen set.
    Representative = mean of actually covered relevance vectors.
    """
    if not points:
        raise ValueError("points must not be empty")
    coverage_sets = axis_coverage_sets(points, epsilon)
    all_indices = set(points)
    current = all_indices.copy()
    for axis_sets in coverage_sets:
        best_set = None
        best_size = -1
        for candidate in axis_sets:
            overlap = candidate & current
            if len(overlap) > best_size:
                best_size = len(overlap)
                best_set = candidate
        if best_set is None:
            best_set = all_indices
        intersection = current & best_set
        current = intersection if intersection else best_set.copy()
    if not current:
        current = max((s for axis in coverage_sets for s in axis),
                      key=len).copy()
    representative = np.stack([
        np.asarray(points[index], dtype=float)
        for index in sorted(current)
    ]).mean(axis=0)
    return representative, current


def adaptive_epsilon_from_spacings(points: Mapping[int, np.ndarray], *,
                                   alpha: float = 2.0,
                                   min_eps: float = 1e-3,
                                   max_eps: float = 0.5) -> float:
    """Source's median inter-point coordinate-spacing epsilon."""
    if not points:
        raise ValueError("points must not be empty")
    values = np.stack([np.asarray(v, dtype=float) for v in points.values()])
    spacings = []
    for axis in range(values.shape[1]):
        gaps = np.diff(np.sort(values[:, axis]))
        if len(gaps):
            spacings.append(np.median(gaps))
    if not spacings:
        return min_eps
    return float(np.clip(alpha * np.median(spacings), min_eps, max_eps))


def choose_epsilon_for_target_coverage(points: Mapping[int, np.ndarray], *,
                                       base_alpha: float = 2.0,
                                       target_fraction: float = 0.8,
                                       min_eps: float = 1e-3,
                                       max_eps: float = 0.5):
    """Original epsilon sweep: factors 0.5, 1, 2, 4, early stop at 80%."""
    base = adaptive_epsilon_from_spacings(
        points, alpha=base_alpha, min_eps=min_eps, max_eps=max_eps)
    best_eps, best_fraction = base, 0.0
    for factor in (0.5, 1.0, 2.0, 4.0):
        epsilon = float(np.clip(factor * base, min_eps, max_eps))
        _, covered = greedy_intersection_representative(points, epsilon)
        fraction = len(covered) / len(points)
        if fraction > best_fraction:
            best_fraction, best_eps = fraction, epsilon
        if fraction >= target_fraction:
            break
    return best_eps, best_fraction


def greedy_intersection_committee(vectors, *, k: int = 4,
                                  epsilon: float | None = None):
    """Original repeated greedy intersections and nearest-cluster assignment.

    Input vectors are already PCA-reduced. Every original buffer row retains
    its ID, unlike research source's unsafe deduplication step.
    """
    reduced = np.asarray(vectors, dtype=float)
    if reduced.ndim != 2 or len(reduced) == 0:
        raise ValueError("vectors must be a nonempty two-dimensional array")
    if k < 1:
        raise ValueError("k must be positive")
    if epsilon is not None and epsilon < 0:
        raise ValueError("epsilon must be nonnegative")

    remaining = {i: reduced[i] for i in range(len(reduced))}
    if epsilon is None:
        epsilon, _ = choose_epsilon_for_target_coverage(remaining)
    representatives = []
    groups = []
    for _ in range(min(k, len(reduced))):
        if not remaining:
            break
        representative, covered = greedy_intersection_representative(
            remaining, epsilon)
        representatives.append(representative)
        groups.append(set(covered))
        for idx in covered:
            remaining.pop(idx, None)

    for idx, vector in remaining.items():
        distances = [np.linalg.norm(vector - rep) for rep in representatives]
        groups[int(np.argmin(distances))].add(idx)
    return groups


def greedy_intersection_clusters(vectors, *, k: int = 4,
                                 epsilon: float | None = None,
                                 pca_dims: int = 7):
    """Core-buffer clustering: PCA(7) -> fast GIA -> original row-index sets.

    Original source crashes when retained observations are fewer than seven;
    in that case, reduce PCA dimensionality for a stable warm-up.
    """
    data = np.asarray(vectors, dtype=float)
    if data.ndim != 2 or len(data) == 0:
        raise ValueError("vectors must be a nonempty two-dimensional array")
    if k < 1 or pca_dims < 1:
        raise ValueError("k and pca_dims must be positive")
    if len(data) == 1:
        return [{0}]
    dimensions = min(pca_dims, len(data), data.shape[1])
    if len(data) < pca_dims:
        dimensions = min(dimensions, len(data)-1)
    reduced = PCA(n_components=dimensions).fit_transform(data)
    return greedy_intersection_committee(reduced, k=k, epsilon=epsilon)
