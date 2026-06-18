# -*- coding: utf-8 -*-
"""
TieredManager - time-scale memory tiering.

Three tiers:
  Tier1 (short): intraday, hours
  Tier2 (medium): cross-day patterns, days-weeks
  Tier3 (long): structural prototypes, months

Each tier has independent capacity, TTL, decay rate,
and promotion/demotion criteria.
"""
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import heapq


@dataclass
class MemoryItem:
    key: str
    value: Any
    tier: int
    timestamp: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    importance: float = 1.0
    
    @property
    def age_seconds(self) -> float:
        return (datetime.now() - self.timestamp).total_seconds()


@dataclass
class TierConfig:
    capacity: int = 1000
    ttl_seconds: float = 21600
    decay_rate: float = 0.1
    promote_threshold: float = 0.8
    demote_threshold: float = 0.2


class TieredManager:
    """Three-tier memory manager with auto promotion/demotion."""
    
    def __init__(self):
        self.configs = {
            1: TierConfig(capacity=1000, ttl_seconds=21600,    # 6h
                          decay_rate=0.1, promote_threshold=0.8),
            2: TierConfig(capacity=500,  ttl_seconds=604800,   # 7d
                          decay_rate=0.05, promote_threshold=0.85,
                          demote_threshold=0.3),
            3: TierConfig(capacity=200,  ttl_seconds=15724800, # 6m
                          decay_rate=0.01, demote_threshold=0.15),
        }
        self.tiers: Dict[int, Dict[str, MemoryItem]] = {1: {}, 2: {}, 3: {}}
    
    def insert(self, key: str, value: Any, tier: int = 1,
               importance: float = 1.0) -> None:
        if tier not in self.configs:
            raise ValueError(f"Unknown tier: {tier}")
        config = self.configs[tier]
        items = self.tiers[tier]
        if len(items) >= config.capacity:
            self._evict(tier)
        items[key] = MemoryItem(
            key=key, value=value, tier=tier,
            importance=importance, timestamp=datetime.now()
        )
    
    def get(self, key: str) -> Optional[Any]:
        for tier_idx in [1, 2, 3]:
            items = self.tiers[tier_idx]
            if key in items:
                items[key].access_count += 1
                items[key].importance = min(1.0, items[key].importance + 0.01)
                return items[key].value
        return None
    
    def tick(self) -> Dict[str, Any]:
        now = datetime.now()
        stats = {'expired': 0, 'promoted': 0, 'demoted': 0, 'decayed': 0}
        
        for tier_idx in [1, 2, 3]:
            config = self.configs[tier_idx]
            items = self.tiers[tier_idx]
            expired_keys = []
            for key, item in list(items.items()):
                age = (now - item.timestamp).total_seconds()
                if age > config.ttl_seconds:
                    expired_keys.append(key)
                    continue
                item.importance *= (1 - config.decay_rate)
                stats['decayed'] += 1
            for key in expired_keys:
                del items[key]
                stats['expired'] += 1
        
        # Promote
        for promote_from, promote_to in [(1, 2), (2, 3)]:
            from_items = self.tiers[promote_from]
            to_items = self.tiers[promote_to]
            promote_config = self.configs[promote_from]
            promote_keys = [
                k for k, v in from_items.items()
                if v.importance >= promote_config.promote_threshold
                and v.access_count >= 3
            ]
            promote_keys.sort(key=lambda k: from_items[k].importance, reverse=True)
            promote_keys = promote_keys[:self.configs[promote_to].capacity // 4]
            for key in promote_keys:
                item = from_items.pop(key)
                item.tier = promote_to
                item.timestamp = datetime.now()
                if len(to_items) < self.configs[promote_to].capacity:
                    to_items[key] = item
                    stats['promoted'] += 1
        
        # Demote
        for demote_from, demote_to in [(3, 2), (2, 1)]:
            from_items = self.tiers[demote_from]
            to_items = self.tiers[demote_to]
            config = self.configs[demote_from]
            demote_keys = [
                k for k, v in from_items.items()
                if v.importance < config.demote_threshold
            ]
            for key in demote_keys:
                item = from_items.pop(key)
                item.tier = demote_to
                if len(to_items) < self.configs[demote_to].capacity:
                    to_items[key] = item
                    stats['demoted'] += 1
        return stats
    
    def search(self, query: Any, top_k: int = 5,
               tier_filter: Optional[int] = None) -> List[MemoryItem]:
        candidates = []
        tiers_to_search = [tier_filter] if tier_filter else [1, 2, 3]
        for tier_idx in tiers_to_search:
            if tier_idx not in self.tiers:
                continue
            for key, item in self.tiers[tier_idx].items():
                score = item.importance * (1 + 0.1 * item.access_count)
                candidates.append((-score, item))
        heapq.heapify(candidates)
        results = []
        for _ in range(min(top_k, len(candidates))):
            neg_score, item = heapq.heappop(candidates)
            results.append(item)
        return results
    
    def _evict(self, tier: int) -> int:
        items = self.tiers[tier]
        if not items:
            return 0
        lowest_key = min(items.keys(), key=lambda k: (
            items[k].importance * 0.7 - 0.3 * items[k].access_count
        ))
        del items[lowest_key]
        return len(items)
    
    def stats(self) -> Dict[str, Any]:
        return {
            str(k): {
                'count': len(v),
                'capacity': self.configs[k].capacity,
                'utilization': f"{len(v)/self.configs[k].capacity*100:.0f}%",
            }
            for k, v in self.tiers.items()
        }
