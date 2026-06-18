# -*- coding: utf-8 -*-
# GaugeMemory
# ===========
# Physics-inspired hierarchical memory infrastructure.
# No application-specific logic.
#
# Core components:
#   contradiction  — Multi-scale contradiction index C ∈ [0,1]
#   forgetting     — Langevin dynamics for memory value decay
#   retrieval      — Fisher information metric retrieval
#   compression    — Prototype-based memory compression / dedup
#   graph          — Associative graph layer with BFS/path/community
#   tiered         — Short-medium-long term memory with auto promotion

__version__ = "0.2.0"
__all__ = [
    "ContradictionDetector", "LangevinForgetting",
    "FisherRetrieval", "MemoryStore",
    "MemoryCompressor", "GraphLayer", "TieredManager",
]

# Explicit imports so from gauge_memory import X works
from .core import ContradictionDetector, LangevinForgetting
from .retrieval import FisherRetrieval
from .storage import MemoryStore
from .compressor import MemoryCompressor
from .graph import GraphLayer
from .tiered import TieredManager
