"""Bounded memory and relevance-structured meta-batch construction."""
from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class Observation:
    key: str
    relevance: np.ndarray
    image: Any
    pseudo_label: Any
    measured_label: Any
    duration: int = 1
    count: int = 1


@dataclass
class BoundedMemory:
    core_capacity: int = 40
    reservoir_capacity: int = 60
    core_lifespan: int = 40
    core: list[Observation] = field(default_factory=list)
    reservoir: list[Observation] = field(default_factory=list)

    def add(self, observation: Observation):
        """Add acquired sample before making an online update."""
        if any(s.key == observation.key for s in self.core + self.reservoir):
            raise ValueError(f"Duplicate acquired observation: {observation.key}")
        for s in self.core + self.reservoir:
            s.duration += 1
        self.core.append(observation)
        # The original code discards the first sample reaching age 40.
        expired = next((s for s in self.core if s.duration >= self.core_lifespan), None)
        if expired is not None:
            self.core.remove(expired)
            self.reservoir.append(expired)
        while len(self.core) > self.core_capacity:
            self.reservoir.append(self.core.pop(0))
        while len(self.reservoir) > self.reservoir_capacity:
            self.reservoir.pop(max(range(len(self.reservoir)),
                                   key=lambda i: self.reservoir[i].duration))

    def references(self):
        """Acquisition references: core means plus up to 3 reservoir means."""
        samples = self.core + self.reservoir[:3]
        return [np.asarray(s.relevance) for s in samples]

    def meta_batch(self, *, clusters: int = 4, reservoir_topk: int = 3,
                   include_latest: bool = True, extra_class0: int = 0):
        """Construct direct online-update batch.

        Deterministic relevance-space representative selection. The original
        script uses custom greedy-intersection clustering following PCA; this
        simplified representative allocation is NOT bitwise equivalent.
        Reservoir items are ranked by lifespan/(usage+1), as in the script.
        """
        if not self.core:
            return []
        if clusters < 1:
            raise ValueError("clusters must be positive")
        ranked = sorted(self.core, key=lambda s: s.duration / max(1, s.count),
                        reverse=True)
        # Greedy diversity in stored relevance space.
        selected = [ranked[0]]
        while len(selected) < min(clusters, len(ranked)):
            remaining = [s for s in ranked if s.key not in {t.key for t in selected}]
            candidate = max(remaining, key=lambda s: min(
                np.linalg.norm(np.asarray(s.relevance) - np.asarray(t.relevance))
                for t in selected))
            selected.append(candidate)
        reservoir = sorted(self.reservoir, key=lambda s: s.duration/(s.count+1),
                           reverse=True)[:reservoir_topk]
        for obs in reservoir:
            if obs.key not in {s.key for s in selected}:
                selected.append(obs)
        if include_latest and self.core[-1].key not in {s.key for s in selected}:
            selected.append(self.core[-1])
        # Experimental scripts sometimes inject class-0 samples using acquired
        # memory labels, not unqueried labels. No new labels are exposed.
        if extra_class0:
            candidates = [s for s in self.core + self.reservoir
                          if s.key not in {v.key for v in selected}
                          and hasattr(s.measured_label, "eq")
                          and bool((s.measured_label == 0).any())]
            selected += sorted(candidates, key=lambda s: s.count)[:extra_class0]
        for obs in selected:
            obs.count += 1
        return selected
