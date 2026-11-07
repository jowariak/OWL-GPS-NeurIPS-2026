"""Original OWL-GPS memory policy, cleaned and made testable.

Based on the uploaded icml2019test(1).py:
- core samples age one tick per acquisition and expire at duration >= 40;
- expired samples pass into a bounded reservoir;
- refresh relevance before PCA/GIA when an encoder callback is available;
- one duration/count-ranked sample per GIA cluster (up to four);
- up to three duration/(count+1)-ranked reservoir observations;
- fill to seven from additional core representatives;
- optionally inject up to three low-use acquired class-0 samples.
No hidden labels are read by acquisition. A deliberate optional latest-sample
override is available, but it is NOT present in the uploaded source.
"""
from dataclasses import dataclass, field
from typing import Any, Callable
import numpy as np
from .clustering import greedy_intersection_clusters


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
    class0_usage: dict[str, int] = field(default_factory=dict)

    def add(self, observation: Observation):
        """Insert newly queried observation, then age/evict retained memory."""
        if any(s.key == observation.key for s in self.core + self.reservoir):
            raise ValueError(f"Duplicate acquired observation: {observation.key}")
        for sample in self.core + self.reservoir:
            sample.duration += 1
        self.core.append(observation)
        # Source moved first expiring core entry on each acquisition.
        expired = next((s for s in self.core if s.duration >= self.core_lifespan), None)
        if expired is not None:
            self.core.remove(expired)
            self.reservoir.append(expired)
        while len(self.core) > self.core_capacity:
            self.reservoir.append(self.core.pop(0))
        while len(self.reservoir) > self.reservoir_capacity:
            # Original script evicts the oldest reservoir observation.
            old = max(range(len(self.reservoir)),
                      key=lambda i:self.reservoir[i].duration)
            self.reservoir.pop(old)

    def references(self, *, reservoir_sample=3, rng=None):
        """Acquisition comparisons use all core means + 3 random reservoir means."""
        refs = list(self.core)
        if self.reservoir:
            import random
            generator = rng if rng is not None else random
            refs += generator.sample(self.reservoir,
                                     min(reservoir_sample,len(self.reservoir)))
        return [np.asarray(s.relevance) for s in refs]

    def refresh_relevance(self, encoder: Callable[[Observation], Any]):
        """Recompute stored mean embeddings after model updates, as source did."""
        for s in self.core + self.reservoir:
            s.relevance = np.asarray(encoder(s)).reshape(-1)

    def meta_batch(self, *, clusters=4, reservoir_topk=3, epsilon=None,
                   include_latest=False, extra_class0=3, min_batch=7,
                   relevance_encoder=None):
        """Construct the meta-batch used for ONE aggregated Adam update.

        include_latest=True is an optional safety guarantee, not part of this
        uploaded 2019 implementation. Original script uses GIA core clustering,
        age/count ranking, up to 3 reservoir, then core fill; the source's
        class-0 injections look only at ACQUIRED masks, which is permissible.
        """
        if not self.core:
            return []
        if relevance_encoder is not None:
            self.refresh_relevance(relevance_encoder)

        matrix = np.stack([np.asarray(s.relevance).reshape(-1) for s in self.core])
        groups = greedy_intersection_clusters(matrix, k=clusters, epsilon=epsilon)
        selected = []
        used = set()

        def add_unique(obs):
            if obs.key in used:
                return False
            selected.append(obs)
            used.add(obs.key)
            return True

        # Source used duration/count (not count+1) for core representative rank.
        for group in groups:
            indices = sorted(group, key=lambda i:
                             self.core[i].duration/max(1,self.core[i].count),
                             reverse=True)
            for i in indices:
                if add_unique(self.core[i]):
                    break

        reservoir_ranked = sorted(
            self.reservoir,
            key=lambda s:s.duration/(s.count+1),
            reverse=True)
        for s in reservoir_ranked[:reservoir_topk]:
            add_unique(s)

        # Source fills toward seven when reservoir is too small.
        if len(selected)<min_batch:
            for group in groups:
                for idx in sorted(group, key=lambda i:
                    self.core[i].duration/max(1,self.core[i].count), reverse=True):
                    if len(selected)>=min_batch:
                        break
                    add_unique(self.core[idx])
                if len(selected)>=min_batch:
                    break

        # Acquired observations only; no oracle search over unqueried samples.
        if extra_class0:
            def is_class0(obs):
                mask = obs.pseudo_label
                if mask is None:
                    return False
                try:
                    return bool((mask == 0).any())
                except (TypeError, AttributeError):
                    return bool(np.asarray(mask == 0).any())
            candidates = sorted(
                (s for s in self.core+self.reservoir if is_class0(s)),
                key=lambda s:self.class0_usage.get(s.key,0))
            for obs in candidates[:extra_class0]:
                if add_unique(obs):
                    self.class0_usage[obs.key]=self.class0_usage.get(obs.key,0)+1

        if include_latest:
            # An opt-in explicit guarantee, not implemented in the uploaded file.
            add_unique(self.core[-1])
        for sample in selected:
            sample.count += 1
        return selected
