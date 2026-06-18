# GaugeMemory

**Physics-inspired hierarchical memory with contradiction detection and Langevin forgetting.**

GaugeMemory is a pure-python memory infrastructure library that brings concepts from gauge theory,
information geometry, and stochastic dynamics to AI memory systems.

## Why GaugeMemory?

Most memory systems (vector databases, key-value stores, etc.) treat memories as isolated items.
GaugeMemory adds two crucial capabilities:

1. **Contradiction detection** — Identifies when new information conflicts with prior memories,
   quantified as a hierarchical contradiction index C ∈ [0,1] across three layers:
   - *Scale conflict*: disagreement between short-term and long-term patterns
   - *Indicator conflict*: cross-metric directional disagreement  
   - *Structure conflict*: phase-space discontinuities

2. **Langevin forgetting** — Memories decay according to a stochastic differential equation:
   ```
   dV = -θ·V·dt + σ·dW + η·feedback - λ·C
   ```
   Where natural decay is balanced by noise (exploration), feedback (learning), and
   contradiction penalty (self-correction).

## Components

| Module | Class | Purpose |
|--------|-------|---------|
| `core.py` | `ContradictionDetector` | Hierarchical contradiction index C |
| `core.py` | `LangevinForgetting` | Langevin dynamics for memory value |
| `retrieval.py` | `FisherRetrieval` | Fisher-Rao metric retrieval |
| `storage.py` | `MemoryStore` | SQLite-backed persistence |

## Quick Start

```python
from gauge_memory.core import ContradictionDetector, LangevinForgetting
from gauge_memory.retrieval import FisherRetrieval
import numpy as np

# --- Contradiction detection ---
detector = ContradictionDetector()

report = {
    'key': 'observation_001',
    'short_term': 'bullish',
    'dimension': 3.5,
    'damping': 0.6,
    'curvature': 2.1,
    'geodesic': 0.75,
    'position': 0.4,
}

result = detector.compute(report, long_term_state='bearish')
print(f"Contradiction index C = {result['C']}")
print(f"  Scale: {result['scale']}")
print(f"  Indicator: {result['indicator']}")
print(f"  Structure: {result['structure']}")

# --- Langevin forgetting ---
langevin = LangevinForgetting()
values = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
updated = langevin.step(values, C=0.3, feedback=np.array([0, 0.5, 0, -0.2, 0]))
print(f"Updated values: {updated}")
print(f"Evict candidates: {langevin.evict_mask(updated)}")

# --- Fisher retrieval ---
store = FisherRetrieval()
store.add('mem_a', np.array([0.1, 0.2, 0.3]))
store.add('mem_b', np.array([0.5, 0.6, 0.7]))
store.add('mem_c', np.array([0.9, 0.8, 0.7]))

results = store.search(np.array([0.2, 0.3, 0.4]), top_k=2)
for r in results:
    print(f"  {r['key']}: distance={r['distance']:.4f} weight={r['weight']:.4f}")

# Contradiction check
kappa = store.detect_contradiction(np.array([5.0, 5.0, 5.0]))
print(f"Contradiction κ = {kappa['kappa']:.4f}: {kappa['details']}")
```

## Requirements

- Python 3.8+
- NumPy
- (optional) SQLite3 — built into Python stdlib

No GPU, no deep learning framework, no external vector database required.

## License

MIT
