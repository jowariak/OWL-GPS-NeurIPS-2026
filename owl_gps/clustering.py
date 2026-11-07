"""Greedy-intersection clustering (GIA) from icml2019test(1).py.

Implements the uploaded fast greedy intersection, followed by the original
nearest-representative assignment. PCA is applied separately to the core
relevance vectors, never to labels. The source fixed 7 PCA dimensions, which
fails for <8 retained observations; here we use min(7,n-1,d) until memory
grows, making the startup phase well-defined.
"""
from __future__ import annotations
import numpy as np


def _greedy_intersection(points: dict[int, np.ndarray], epsilon: float):
    """Greedy axis-wise maximum intersection, as in the uploaded source."""
    if not points:
        return np.array([]), set()
    indices_all = set(points)
    d = len(next(iter(points.values())))
    current = indices_all.copy()
    for dim in range(d):
        ranked = sorted(points, key=lambda i: points[i][dim])
        coverage_sets = []
        prior = set()
        for position, index in enumerate(ranked):
            coverage = {index}
            for j in range(position - 1, -1, -1):
                previous = ranked[j]
                if points[index][dim] <= points[previous][dim] + 2 * epsilon:
                    coverage.add(previous)
                else:
                    break
            if not coverage_sets:
                coverage_sets.append(coverage)
            elif coverage_sets[-1].issubset(coverage):
                coverage_sets[-1] = coverage
            else:
                coverage_sets.append(coverage)
        best = max(coverage_sets, key=lambda s: len(s & current))
        new_current = current & best
        if not new_current:
            new_current = set(best)  # exact legacy fallback
        current = new_current
    if not current:
        current = {min(points)}
    return np.mean([points[i] for i in sorted(current)], axis=0), current


def _adaptive_epsilon(points: dict[int, np.ndarray]) -> float:
    """Source's adaptive spacing and target-first-cluster coverage (80%)."""
    values = np.stack(list(points.values()))
    spacings = [np.median(np.diff(np.sort(values[:,i])))
                for i in range(values.shape[1]) if len(values) > 1]
    typical = float(np.median(spacings)) if spacings else 0.
    base = float(np.clip(2*typical, 1e-3, .5))
    best_eps, best_coverage = base, -1
    for multiplier in (.5, 1., 2., 4.):
        eps = float(np.clip(multiplier*base, 1e-3, .5))
        _, covered = _greedy_intersection(points, eps)
        fraction = len(covered)/len(points)
        if fraction > best_coverage:
            best_coverage, best_eps = fraction, eps
        if fraction >= .8:
            break
    return best_eps


def greedy_intersection_clusters(vectors, *, k=4, epsilon=None, pca_dims=7):
    """Return clusters of ORIGINAL row indices for core-buffer selection.

    Mirrors the script: PCA reduction; k repeated greedy intersections;
    nearest-cluster assignment for uncovered observations. Preserve duplicate
    vectors as different observations, fixing source's index-loss on dedup.
    """
    data = np.asarray(vectors, dtype=float)
    if data.ndim != 2 or not len(data):
        raise ValueError("vectors must be a nonempty 2-D array")
    if k < 1:
        raise ValueError("k must be positive")
    if len(data) == 1:
        return [{0}]
    # sklearn PCA requires n_components <= min(n_samples,n_features).
    dims = min(pca_dims, len(data)-1, data.shape[1])
    if dims >= 1:
        from sklearn.decomposition import PCA
        projected = PCA(n_components=dims).fit_transform(data)
    else:
        projected = data
    point_map = {i:projected[i] for i in range(len(data))}
    eps = _adaptive_epsilon(point_map) if epsilon is None else float(epsilon)
    if eps < 0:
        raise ValueError("epsilon must be nonnegative")
    remaining = dict(point_map)
    clusters, representatives = [], []
    for _ in range(min(k,len(data))):
        if not remaining:
            break
        representative, covered = _greedy_intersection(remaining, eps)
        if not covered:
            break
        clusters.append(set(covered))
        representatives.append(representative)
        for idx in covered:
            remaining.pop(idx, None)
    for idx, v in remaining.items():
        distances = [np.linalg.norm(v-r) for r in representatives]
        clusters[int(np.argmin(distances))].add(idx)
    return clusters
