# -*- coding: utf-8 -*-
"""
GraphLayer - associative memory graph.

Memories are graph nodes with typed edges:
  - cause: causal relationship
  - similar: similarity weight
  - contradict: contradiction flag
  - follow: temporal sequence

Search supports graph traversal (BFS, path finding),
not just vector distance.
"""
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import json
from collections import deque


@dataclass
class Edge:
    target_id: str
    edge_type: str
    weight: float = 1.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class Node:
    node_id: str
    vector: Optional[Any] = None
    label: str = ''
    edges: List[Edge] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


class GraphLayer:
    """In-memory directed graph for memory association."""
    
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
    
    def add_node(self, node_id: str, label: str = '',
                 vector: Optional[Any] = None,
                 metadata: Optional[Dict] = None) -> None:
        if node_id not in self.nodes:
            self.nodes[node_id] = Node(
                node_id=node_id, label=label,
                vector=vector, metadata=metadata or {}
            )
    
    def add_edge(self, source_id: str, target_id: str,
                 edge_type: str, weight: float = 1.0,
                 metadata: Optional[Dict] = None) -> None:
        if source_id not in self.nodes or target_id not in self.nodes:
            raise ValueError(f"Node not found: {source_id} or {target_id}")
        self.nodes[source_id].edges.append(Edge(
            target_id=target_id, edge_type=edge_type,
            weight=weight, metadata=metadata or {}
        ))
    
    def get_neighbors(self, node_id: str,
                      edge_type: Optional[str] = None) -> List[Tuple[str, Edge]]:
        if node_id not in self.nodes:
            return []
        results = []
        for edge in self.nodes[node_id].edges:
            if edge_type is None or edge.edge_type == edge_type:
                if edge.target_id in self.nodes:
                    results.append((edge.target_id, edge))
        return results
    
    def bfs(self, start_id: str, max_depth: int = 3,
            edge_type: Optional[str] = None) -> Dict[str, int]:
        if start_id not in self.nodes:
            return {}
        visited: Dict[str, int] = {start_id: 0}
        queue = [(start_id, 0)]
        while queue:
            curr, depth = queue.pop(0)
            if depth >= max_depth:
                continue
            for neighbor_id, edge in self.get_neighbors(curr, edge_type):
                if neighbor_id not in visited:
                    visited[neighbor_id] = depth + 1
                    queue.append((neighbor_id, depth + 1))
        return visited
    
    def find_path(self, start_id: str, end_id: str,
                  max_depth: int = 5) -> List[str]:
        if start_id not in self.nodes or end_id not in self.nodes:
            return []
        queue = deque([(start_id, [start_id])])
        visited = {start_id}
        while queue:
            curr, path = queue.popleft()
            if curr == end_id:
                return path
            if len(path) >= max_depth:
                continue
            for neighbor_id, _ in self.get_neighbors(curr):
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, path + [neighbor_id]))
        return []
    
    def detect_communities(self) -> Dict[str, List[str]]:
        groups: Dict[str, List[str]] = {}
        for nid, node in self.nodes.items():
            prefix = node.label.split('_')[0] if '_' in node.label else 'default'
            groups.setdefault(prefix, []).append(nid)
        return groups
    
    def summarize(self) -> Dict[str, Any]:
        n_nodes = len(self.nodes)
        n_edges = sum(len(n.edges) for n in self.nodes.values())
        type_counts: Dict[str, int] = {}
        for n in self.nodes.values():
            for e in n.edges:
                type_counts[e.edge_type] = type_counts.get(e.edge_type, 0) + 1
        return {
            'nodes': n_nodes,
            'edges': n_edges,
            'edge_types': type_counts,
        }
    
    def to_json(self) -> str:
        data = {'nodes': {}}
        for nid, node in self.nodes.items():
            data['nodes'][nid] = {
                'label': node.label,
                'edges': [(e.target_id, e.edge_type, e.weight) for e in node.edges],
                'metadata': node.metadata,
            }
        return json.dumps(data, ensure_ascii=False, indent=2)
    
    @classmethod
    def from_json(cls, text: str) -> 'GraphLayer':
        gl = cls()
        data = json.loads(text)
        for nid, ndata in data.get('nodes', {}).items():
            gl.add_node(nid, label=ndata.get('label', ''), metadata=ndata.get('metadata', {}))
            for target, etype, weight in ndata.get('edges', []):
                gl.add_edge(nid, target, etype, weight)
        return gl
