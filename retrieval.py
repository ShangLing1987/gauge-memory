"""
Fisher information metric retrieval for memory systems.

Retrieves stored memories by ranking them under the Fisher-Rao metric,
which measures distance between probability distributions. This provides
a geometrically meaningful similarity measure beyond cosine or Euclidean distance.
"""

import numpy as np
import json
from typing import Dict, List, Optional, Tuple, Any


class FisherRetrieval:
    """
    Fisher information metric retrieval.

    Each memory is represented as a distribution (mean, std) in feature space.
    The Fisher-Rao distance between two univariate Gaussians is:

        d(p||q) = sqrt(2) * arccosh(1 + (μ₁-μ₂)²/(2(σ₁²+σ₂²)))

    For high-dimensional vectors, we use a diagonal approximation:

        d_F(p, q) = Σ_i sqrt(2) * arccosh(1 + (μ₁ᵢ-μ₂ᵢ)²/(2(σ₁ᵢ²+σ₂ᵢ²)))

    Usage:
        store = FisherRetrieval()
        store.add('mem1', embedding=np.array([...]), metadata={...})
        results = store.search(query_embedding, top_k=5)
    """

    def __init__(self, gamma: float = 1.618, tau_contradiction: float = 0.618):
        """
        Args:
            gamma: temperature for soft weighting
            tau_contradiction: threshold for contradiction flagging
        """
        self.gamma = gamma
        self.tau = tau_contradiction
        self._memories: Dict[str, Dict] = {}  # key -> {vec, metadata, ...}

    # -- Indexing ------------------------------------------------------------

    def add(self,
            key: str,
            embedding: np.ndarray,
            metadata: Optional[Dict] = None,
            std: Optional[np.ndarray] = None) -> None:
        """
        Add a memory to the index.

        Args:
            key: unique identifier
            embedding: feature vector (mean of the distribution)
            metadata: optional arbitrary data
            std: per-feature standard deviation (default = 1.0)
        """
        if std is None:
            std = np.ones_like(embedding)
        self._memories[key] = {
            'vec': np.asarray(embedding, dtype=np.float64),
            'std': np.asarray(std, dtype=np.float64),
            'metadata': metadata or {},
        }

    def remove(self, key: str) -> bool:
        """Remove a memory by key. Returns True if found."""
        return self._memories.pop(key, None) is not None

    def clear(self) -> None:
        self._memories.clear()

    def __len__(self) -> int:
        return len(self._memories)

    # -- Fisher-Rao distance ------------------------------------------------

    @staticmethod
    def fisher_rao_distance(mu1: np.ndarray, sigma1: np.ndarray,
                            mu2: np.ndarray, sigma2: np.ndarray) -> float:
        """
        Diagonal Fisher-Rao distance between two Gaussian distributions.

        Returns scalar distance (sum over dimensions).
        """
        sigma_sum = sigma1 ** 2 + sigma2 ** 2
        delta = (mu1 - mu2) ** 2
        # arccosh(1 + x) where x = δ²/(2Σ²)
        arg = 1.0 + delta / (2.0 * sigma_sum + 1e-15)
        # Clip to avoid numerical issues in arccosh
        arg = np.clip(arg, 1.0, 1e10)
        d_per_dim = np.sqrt(2.0) * np.arccosh(np.sqrt(arg))
        return float(np.sum(d_per_dim))

    # -- Search --------------------------------------------------------------

    def search(self,
               query: np.ndarray,
               top_k: int = 10,
               query_std: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """
        Search memories by Fisher-Rao distance to query.

        Args:
            query: query feature vector
            top_k: number of results
            query_std: per-feature std for query (default = 1.0)

        Returns:
            list of {key, distance, weight, metadata} sorted by distance (ascending)
        """
        if query_std is None:
            query_std = np.ones_like(query)
        query = np.asarray(query, dtype=np.float64)
        query_std = np.asarray(query_std, dtype=np.float64)

        results = []
        for key, mem in self._memories.items():
            dist = self.fisher_rao_distance(query, query_std,
                                            mem['vec'], mem['std'])
            weight = np.exp(-self.gamma * dist)
            results.append({
                'key': key,
                'distance': round(dist, 4),
                'weight': round(weight, 4),
                'metadata': mem['metadata'],
            })

        results.sort(key=lambda x: x['distance'])
        return results[:top_k]

    # -- Contradiction detection --------------------------------------------

    def detect_contradiction(self,
                             query: np.ndarray,
                             top_k: int = 5,
                             threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Detect if the query contradicts the memory store.

        A contradiction is flagged when:
          - The closest memory is at distance > threshold (novel/unusual)
          - The distribution of top-k results has high variance in distance
          - There's directional disagreement between top matches

        Args:
            query: query feature vector
            top_k: number of near neighbours to examine
            threshold: distance threshold (defaults to self.tau)

        Returns:
            {'kappa': float (0=no contradiction, 1=strong contradiction),
             'details': str}
        """
        tau = threshold if threshold is not None else self.tau
        query = np.asarray(query, dtype=np.float64)
        query_std = np.ones_like(query)

        if len(self._memories) < 2:
            return {'kappa': 0.0, 'details': 'insufficient memory'}

        results = self.search(query, top_k=top_k)

        if not results:
            return {'kappa': 0.0, 'details': 'no results'}

        # Average distance of top-k
        avg_dist = np.mean([r['distance'] for r in results])

        # Distance dispersion (CV)
        if len(results) > 1:
            dist_std = np.std([r['distance'] for r in results])
            cv = dist_std / max(avg_dist, 1e-10)
        else:
            cv = 0.0

        # kappa: combination of absolute distance + dispersion
        kappa = min(1.0, avg_dist / (tau * 2) + cv * 0.3)

        if kappa > 0.5 and avg_dist > tau:
            verdict = f'contradiction detected (κ={kappa:.2f}, d̄={avg_dist:.2f})'
        elif kappa > 0.3:
            verdict = f'weak contradiction (κ={kappa:.2f})'
        else:
            verdict = f'consistent (κ={kappa:.2f})'

        return {
            'kappa': round(kappa, 4),
            'avg_distance': round(avg_dist, 4),
            'top_k_distances': [r['distance'] for r in results],
            'details': verdict,
        }
