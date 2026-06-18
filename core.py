"""
GaugeMemory — core contradiction detection and Langevin forgetting.

ContradictionDetector
  Multi-scale contradiction index C ∈ [0,1]:
    • scale_conflict: disagreement across time scales
    • indicator_conflict: cross-metric directional disagreement
    • structure_conflict: phase-space structural jumps

LangevinForgetting
  Memory value evolves as:
    dV = -θ·V·dt + σ·dW + η·feedback - λ·C
  where θ=forgetting rate, σ=exploration noise, λ=contradiction penalty.
"""

import numpy as np
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

logger = logging.getLogger("gauge_memory")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class LangevinConfig:
    theta: float = 0.05         # forgetting rate (1/20)
    sigma: float = 0.30         # exploration noise (std)
    lambd: float = 0.50         # contradiction penalty (mid-scale)
    eta: float = 0.10           # feedback learning rate (0.1)
    v_min: float = 0.10         # eviction threshold (0.1)
    dt: float = 1.0             # time step


# ---------------------------------------------------------------------------
# ContradictionDetector
# ---------------------------------------------------------------------------

class ContradictionDetector:
    """
    Hierarchical contradiction index C ∈ [0,1].

    Three layers:
      1. Scale conflict — disagreement between long-term and short-term states
      2. Indicator conflict — directional disagreement between metrics
      3. Structure conflict — phase-space jumps (dimension + damping)

    Usage:
        detector = ContradictionDetector()
        result = detector.compute(current_report, history=history_store)
    """

    STATE_MAP = {'excitatory': 1, 'stable': 0, 'inhibitory': -1}

    def __init__(self):
        self._prev_reports: Dict[str, Dict] = {}  # key -> last report

    def compute(self,
                report: Dict[str, Any],
                long_term_state: Optional[str] = None,
                prev_report: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Compute contradiction index for a single observation.

        Args:
            report: {
                'key': str identifier,
                'short_term': 'excitatory' | 'stable' | 'inhibitory',
                'dimension': float (effective dimensionality),
                'damping': float (damping ratio),
                'curvature': float (curvature magnitude),
                'geodesic': float (geodesic stability),
                ... other metrics
            }
            long_term_state: optional long-term regime ('excitatory'|'stable'|'inhibitory')
            prev_report: optional previous report for structural jump detection

        Returns:
            {'C': float, 'scale': float, 'indicator': float,
             'structure': float, 'details': dict}
        """
        C = 0.0
        details = {}

        key = report.get('key', '_default')
        short_state = report.get('short_term', 'stable')

        # -- 1. Scale conflict ------------------------------------------------
        scale_conflict = 0.0
        if long_term_state is not None:
            ls = self.STATE_MAP.get(long_term_state, 0)
            ss = self.STATE_MAP.get(short_state, 0)
            if ls * ss < 0:
                scale_conflict = 0.618
            elif ls == 0 and ss != 0:
                scale_conflict = 0.382
            elif ls != 0 and ss == 0:
                scale_conflict = 0.2361
        details['scale'] = {
            'long': long_term_state,
            'short': short_state,
            'score': scale_conflict,
        }
        C += scale_conflict

        # -- 2. Indicator conflict -------------------------------------------
        indicator_conflict = 0.0
        dim = report.get('dimension', 0)
        curvature = report.get('curvature', 0)
        geodesic = report.get('geodesic', 0)
        damping = report.get('damping', 0.5)
        position = report.get('position', 0.5)

        # Dimensionality vs curvature: high-dim excitatory, low-curvature excitatory
        dim_excitatory = 1 if dim >= 3 else -1
        curve_excitatory = 1 if curvature < 5 and geodesic > 0.618 else -1
        if dim_excitatory * curve_excitatory < 0:
            indicator_conflict += 0.382

        # Damping vs position: high damping + extreme position = warning
        if damping > 0.618 and position > 0.854:
            indicator_conflict += 0.2361
        if damping < 0.2361 and position < 0.1459:
            indicator_conflict += 0.2361

        details['indicator'] = {
            'dim_vs_curvature': dim_excitatory != curve_excitatory,
            'damp_vs_position': damping > 0.618 and position > 0.854,
        }
        C += indicator_conflict

        # -- 3. Structure conflict -------------------------------------------
        structure_conflict = 0.0
        prev = prev_report or self._prev_reports.get(key)
        if prev is not None:
            prev_dim = prev.get('dimension', dim)
            prev_damping = prev.get('damping', damping)
            dim_jump = abs(dim - prev_dim)
            damp_jump = abs(damping - prev_damping)
            if dim_jump >= 2.618 and damp_jump > 0.618:
                structure_conflict = 0.2361
            elif dim_jump >= 2.618 or damp_jump > 0.618:
                structure_conflict = 0.1459
        details['structure'] = {
            'dim_jump': abs(dim - prev.get('dimension', dim)) if prev else 0,
            'damp_jump': abs(damping - prev.get('damping', damping)) if prev else 0,
        }
        C += structure_conflict

        C = min(1.0, C)
        self._prev_reports[key] = report

        return {
            'C': round(C, 4),
            'scale': round(scale_conflict, 4),
            'indicator': round(indicator_conflict, 4),
            'structure': round(structure_conflict, 4),
            'details': details,
        }


# ---------------------------------------------------------------------------
# LangevinForgetting
# ---------------------------------------------------------------------------

class LangevinForgetting:
    """
    Langevin dynamics for memory value evolution.

    The value V of each memory decays naturally, is perturbed by noise,
    punished by contradiction, and reinforced by positive feedback.

        dV = -θ·V·dt + σ·dW + η·feedback - λ·C

    Memories whose value drops below v_min are candidates for eviction.
    """

    def __init__(self, config: Optional[LangevinConfig] = None):
        self.cfg = config or LangevinConfig()

    def step(self,
             V: np.ndarray,
             C: float = 0.0,
             feedback: np.ndarray = None) -> np.ndarray:
        """
        Apply one Langevin step to a batch of memory values.

        Args:
            V: 1-D array of current memory values
            C: scalar contradiction index (applied uniformly)
            feedback: 1-D array of per-memory feedback (same shape as V)

        Returns:
            Updated V array (clipped to [0, ∞))
        """
        cfg = self.cfg
        n = len(V)

        # Deterministic decay
        dv_decay = -cfg.theta * V * cfg.dt

        # Langevin noise
        dW = np.random.normal(0, np.sqrt(cfg.dt), size=n)
        dv_noise = cfg.sigma * dW

        # Contradiction penalty (uniform)
        dv_contra = -cfg.lambd * C

        # Feedback (per-memory)
        dv_feedback = np.zeros(n)
        if feedback is not None:
            dv_feedback = cfg.eta * feedback

        V_new = V + dv_decay + dv_noise + dv_contra + dv_feedback
        V_new = np.maximum(0.0, V_new)
        return V_new

    def update_single(self,
                      V: float,
                      C: float = 0.0,
                      feedback: float = 0.0) -> float:
        """Apply one Langevin step to a single memory value."""
        cfg = self.cfg
        dW = np.random.normal(0, np.sqrt(cfg.dt))
        dv = (-cfg.theta * V * cfg.dt
              + cfg.sigma * dW
              - cfg.lambd * C
              + cfg.eta * feedback)
        return max(0.0, V + dv)

    def evict_mask(self, V: np.ndarray) -> np.ndarray:
        """Return boolean mask: True where V < v_min (candidates for eviction)."""
        return V < self.cfg.v_min
