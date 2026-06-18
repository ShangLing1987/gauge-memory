#!/usr/bin/env python3
"""
GaugeMemory 鈥?full pipeline demonstration.

Run: python example.py
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from gauge_memory.core import ContradictionDetector, LangevinForgetting
from gauge_memory.retrieval import FisherRetrieval
from gauge_memory.storage import MemoryStore


def demo_contradiction():
    print("=" * 50)
    print("1. Contradiction Detection")
    print("=" * 50)
    detector = ContradictionDetector()

    # Simulate a sequence of observations
    observations = [
        {'key': 'obs_1', 'short_term': 'excitatory', 'dimension': 3.8,
         'damping': 0.7, 'curvature': 2.0, 'geodesic': 0.8, 'position': 0.3},
        {'key': 'obs_2', 'short_term': 'stable', 'dimension': 2.5,
         'damping': 0.4, 'curvature': 4.5, 'geodesic': 0.5, 'position': 0.6},
        {'key': 'obs_3', 'short_term': 'inhibitory', 'dimension': 1.5,
         'damping': 0.2, 'curvature': 7.0, 'geodesic': 0.3, 'position': 0.9},
    ]

    for obs in observations:
        result = detector.compute(obs, long_term_state='excitatory')
        print(f"\n  {obs['key']}:")
        print(f"    C = {result['C']:.3f}"
              f"  (scale={result['scale']:.1f}"
              f"  indicator={result['indicator']:.1f}"
              f"  structure={result['structure']:.1f})")
        print(f"    verdict: {result['details']}")


def demo_langevin():
    print("\n" + "=" * 50)
    print("2. Langevin Forgetting")
    print("=" * 50)

    langevin = LangevinForgetting()

    # Simulate 10 memories over 100 time steps
    n_memories = 10
    V = np.ones(n_memories) * 0.8  # start values

    np.random.seed(42)
    for t in range(100):
        C = 0.2 + 0.3 * np.sin(t * 0.1)  # cyclical contradiction
        feedback = np.random.uniform(-0.1, 0.1, n_memories)
        V = langevin.step(V, C=C, feedback=feedback)

        # Evict dead memories and re-inject
        dead = langevin.evict_mask(V)
        if dead.any():
            V[dead] = 1.0  # refresh

    print(f"  After 100 steps:")
    print(f"    Alive: {(~langevin.evict_mask(V)).sum()} / {n_memories}")
    print(f"    Values: min={V.min():.2f}  max={V.max():.2f}  mean={V.mean():.2f}")


def demo_fisher():
    print("\n" + "=" * 50)
    print("3. Fisher-Rao Retrieval")
    print("=" * 50)

    store = FisherRetrieval(gamma=1.618)

    # Add some memories
    categories = {
        'text': ([0.1, 0.2, 0.3], [0.05, 0.05, 0.05]),
        'image': ([0.5, 0.4, 0.6], [0.08, 0.08, 0.08]),
        'audio': ([0.8, 0.9, 0.7], [0.06, 0.06, 0.06]),
        'tabular': ([0.3, 0.1, 0.5], [0.07, 0.07, 0.07]),
    }
    for name, (vec, std) in categories.items():
        store.add(name, np.array(vec), std=np.array(std),
                  metadata={'type': name})

    # Search
    query = np.array([0.15, 0.25, 0.35])
    results = store.search(query, top_k=3)
    print(f"\n  Query: {query}")
    for r in results:
        print(f"    {r['key']:8s}  d={r['distance']:.4f}  w={r['weight']:.4f}")

    # Contradiction
    outlier = np.array([5.0, 5.0, 5.0])
    kappa = store.detect_contradiction(outlier)
    print(f"\n  Outlier query:")
    print(f"    魏 = {kappa['kappa']:.4f}  鈥? {kappa['details']}")


def demo_storage():
    print("\n" + "=" * 50)
    print("4. Persistent Storage")
    print("=" * 50)

    store = MemoryStore(":memory:")
    store.save_memory("demo_1", np.array([0.1, 0.2, 0.3]),
                      metadata={'source': 'demo'})
    store.save_memory("demo_2", np.array([0.4, 0.5, 0.6]))
    store.log_contradiction("demo_1", C=0.42, scale=0.3,
                            indicator=0.1, structure=0.02)
    store.set_value("demo_1", 0.85)

    print(f"  Keys: {store.list_keys()}")
    print(f"  Value(demo_1): {store.get_value('demo_1')}")
    hist = store.get_contradiction_history("demo_1")
    print(f"  Contradiction log: {len(hist)} entries")

    store.close()


if __name__ == '__main__':
    print("GaugeMemory v0.1.0 鈥?Full Pipeline Demo")
    print()
    demo_contradiction()
    demo_langevin()
    demo_fisher()
    demo_storage()
    print("\nDone.")
