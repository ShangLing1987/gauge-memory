# -*- coding: utf-8 -*-
"""Quick test for gauge_memory new modules."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from gauge_memory import MemoryCompressor, GraphLayer, TieredManager

mc = MemoryCompressor(0.8)
mc.fit([np.random.randn(10) for _ in range(20)])
print('Compressor: %d prototypes' % len(mc.prototypes))

gl = GraphLayer()
gl.add_node('A','src')
gl.add_node('B','tgt')
gl.add_edge('A','B','follow',0.9)
s = gl.summarize()
print('Graph: nodes=%d edges=%d' % (s['nodes'], s['edges']))

tm = TieredManager()
tm.insert('k1','v1',1)
print('Tiered:', tm.stats())

print('All 3 OK.')
