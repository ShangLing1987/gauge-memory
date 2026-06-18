# -*- coding: utf-8 -*-
"""
MemoryCompressor - prototype-based memory compression.

Reduces storage by merging similar memories into prototype vectors.
Pure math, no application logic.
"""
import numpy as np
from typing import List, Dict, Optional, Tuple, Any


class MemoryCompressor:
    """Prototype-based memory compressor using online clustering.
    
    Maintains prototype vectors. New memories are either merged
    into the closest prototype (if similar enough) or become new.
    """
    
    def __init__(self, similarity_threshold: float = 0.85, max_prototypes: int = 1000):
        self.similarity_threshold = similarity_threshold
        self.max_prototypes = max_prototypes
        self.prototypes: List[np.ndarray] = []
        self.prototype_counts: List[int] = []
        self.prototype_stats: List[Dict] = []
    
    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
    
    def fit(self, memories: List[np.ndarray]) -> List[int]:
        """Batch-fit initial memories. Returns cluster assignments."""
        if not memories:
            return []
        n = len(memories)
        assignments = [-1] * n
        for i, mem in enumerate(memories):
            best_idx, best_sim = -1, -1.0
            for j, proto in enumerate(self.prototypes):
                sim = self._cosine_sim(mem, proto)
                if sim > best_sim:
                    best_sim = sim
                    best_idx = j
            if best_sim >= self.similarity_threshold:
                cnt = self.prototype_counts[best_idx]
                self.prototypes[best_idx] = (self.prototypes[best_idx] * cnt + mem) / (cnt + 1)
                self.prototype_counts[best_idx] = cnt + 1
                assignments[i] = best_idx
            else:
                if len(self.prototypes) < self.max_prototypes:
                    self.prototypes.append(mem.copy())
                    self.prototype_counts.append(1)
                    assignments[i] = len(self.prototypes) - 1
        return assignments
    
    def compress(self, memory: np.ndarray) -> Tuple[int, float]:
        """Online compress single memory. Returns (prototype_idx, similarity)."""
        best_idx, best_sim = -1, -1.0
        for j, proto in enumerate(self.prototypes):
            sim = self._cosine_sim(memory, proto)
            if sim > best_sim:
                best_sim = sim
                best_idx = j
        if best_sim >= self.similarity_threshold:
            cnt = self.prototype_counts[best_idx]
            self.prototypes[best_idx] = (self.prototypes[best_idx] * cnt + memory) / (cnt + 1)
            self.prototype_counts[best_idx] = cnt + 1
        else:
            if len(self.prototypes) < self.max_prototypes:
                self.prototypes.append(memory.copy())
                self.prototype_counts.append(1)
                best_idx = len(self.prototypes) - 1
                best_sim = 1.0
        return best_idx, best_sim
    
    def decompress(self, prototype_idx: int) -> Dict[str, Any]:
        if prototype_idx < 0 or prototype_idx >= len(self.prototypes):
            return {}
        return {
            'prototype': self.prototypes[prototype_idx].tolist(),
            'count': self.prototype_counts[prototype_idx],
            'centroid': self.prototypes[prototype_idx].tolist(),
        }
    
    def prune(self, min_count: int = 2) -> int:
        before = len(self.prototypes)
        keep = [i for i in range(before) if self.prototype_counts[i] >= min_count]
        self.prototypes = [self.prototypes[i] for i in keep]
        self.prototype_counts = [self.prototype_counts[i] for i in keep]
        return before - len(keep)
    
    def state_dict(self) -> Dict:
        return {
            'prototypes': [p.tolist() for p in self.prototypes],
            'counts': self.prototype_counts,
            'threshold': self.similarity_threshold,
            'max_protos': self.max_prototypes,
        }
    
    @classmethod
    def from_state(cls, state: Dict) -> 'MemoryCompressor':
        mc = cls(similarity_threshold=state.get('threshold', 0.85),
                 max_prototypes=state.get('max_protos', 1000))
        mc.prototypes = [np.array(p) for p in state.get('prototypes', [])]
        mc.prototype_counts = state.get('counts', [])
        return mc
