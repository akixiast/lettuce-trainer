"""Expert cache: LRU eviction with frequency boost and async prefetch stub."""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .expert import ExpertModule, ExpertState


class ExpertCache:
    """In-VRAM cache for active expert modules.

    Holds decompressed FP16 experts ready for GPU compute.
    Eviction policy: LRU with frequency boost (experts routed >10x/min stay pinned).
    """

    def __init__(
        self,
        max_experts: int = 8,
        device: torch.device = torch.device("cpu"),
    ):
        self.max_experts = max_experts
        self.device = device
        self._cache: OrderedDict[str, ExpertModule] = OrderedDict()
        self._access_counts: Dict[str, int] = {}
        self._lock = threading.Lock()

    @property
    def size(self) -> int:
        return len(self._cache)

    def contains(self, expert_id: str) -> bool:
        return expert_id in self._cache

    def get(self, expert_id: str) -> Optional[ExpertModule]:
        with self._lock:
            if expert_id not in self._cache:
                return None
            # Move to end (most recently used)
            self._cache.move_to_end(expert_id)
            self._access_counts[expert_id] = self._access_counts.get(expert_id, 0) + 1
            return self._cache[expert_id]

    def put(self, expert: ExpertModule) -> None:
        """Insert an expert into the cache, evicting if necessary."""
        expert_id = expert.metadata.expert_id
        with self._lock:
            if expert_id in self._cache:
                self._cache.move_to_end(expert_id)
                return

            # Evict if at capacity
            while len(self._cache) >= self.max_experts:
                self._evict_one()

            self._cache[expert_id] = expert.to(self.device)
            self._access_counts[expert_id] = 0

    def _evict_one(self) -> None:
        """Evict the least-recently-used non-pinned expert."""
        # Find first non-pinned entry (pinned = access_count > 10)
        for eid in list(self._cache.keys()):
            if self._access_counts.get(eid, 0) <= 10:
                del self._cache[eid]
                del self._access_counts[eid]
                return
        # All pinned — evict oldest anyway
        oldest_id = next(iter(self._cache))
        del self._cache[oldest_id]
        del self._access_counts[oldest_id]

    def load(
        self,
        expert_ids: torch.Tensor,
        loader_fn=None,
    ) -> List[ExpertModule]:
        """Load experts by ID, using loader_fn for cache misses.

        Args:
            expert_ids: [top_k] tensor of global expert indices.
            loader_fn: Callable(expert_id: str) -> ExpertModule for cache misses.
                       If None, returns None for missing experts.

        Returns:
            List of ExpertModule instances (None for unloadable IDs).
        """
        results: List[Optional[ExpertModule]] = []
        missing_ids: List[str] = []

        for eid_tensor in expert_ids:
            eid_int = eid_tensor.item()
            if eid_int < 0:
                results.append(None)
                continue
            eid_str = str(eid_int)
            cached = self.get(eid_str)
            if cached is not None:
                results.append(cached)
            else:
                missing_ids.append(eid_str)
                results.append(None)  # placeholder

        # Load missing experts
        if loader_fn and missing_ids:
            miss_idx = 0
            for i, r in enumerate(results):
                if r is None and miss_idx < len(missing_ids):
                    eid_str = missing_ids[miss_idx]
                    try:
                        expert = loader_fn(eid_str)
                        if expert is not None:
                            self.put(expert)
                            results[i] = expert
                    except Exception:
                        pass  # Failed to load; leave as None
                    miss_idx += 1

        return [r for r in results if r is not None]

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._access_counts.clear()

    def pin(self, expert_id: str) -> None:
        """Pin an expert so it won't be evicted by normal LRU."""
        with self._lock:
            if expert_id in self._access_counts:
                self._access_counts[expert_id] = 100  # Well above threshold
