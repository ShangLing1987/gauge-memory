"""
SQLite storage backend for GaugeMemory.

Provides persistent storage for:
  - Memory embeddings and metadata
  - Contradiction logs
  - Memory values (Langevin forgetting state)
"""

import sqlite3
import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple


class MemoryStore:
    """
    SQLite-backed persistent store for gauge memory data.

    Tables:
      memories      — feature vectors and metadata
      memory_values — Langevin value states
      contradictions — historical contradiction indices
    """

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self):
        cur = self.conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT UNIQUE NOT NULL,
                embedding BLOB,
                std BLOB,
                metadata TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS memory_values (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_key TEXT NOT NULL,
                value REAL DEFAULT 1.0,
                access_count INTEGER DEFAULT 0,
                success_count INTEGER DEFAULT 0,
                fail_count INTEGER DEFAULT 0,
                last_updated TEXT,
                UNIQUE(memory_key)
            );
            CREATE TABLE IF NOT EXISTS contradictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT NOT NULL,
                timestamp TEXT DEFAULT (datetime('now')),
                C REAL DEFAULT 0,
                scale REAL DEFAULT 0,
                indicator REAL DEFAULT 0,
                structure REAL DEFAULT 0,
                details TEXT DEFAULT '{}'
            );
        """)
        self.conn.commit()

    # -- Memories ------------------------------------------------------------

    def save_memory(self, key: str, embedding: np.ndarray,
                    std: Optional[np.ndarray] = None,
                    metadata: Optional[Dict] = None) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO memories (key, embedding, std, metadata) VALUES (?, ?, ?, ?)",
            (key, embedding.tobytes() if embedding is not None else None,
             std.tobytes() if std is not None else None,
             json.dumps(metadata or {})))
        self.conn.commit()

    def load_memory(self, key: str) -> Optional[Dict]:
        row = self.conn.execute(
            "SELECT * FROM memories WHERE key=?", (key,)).fetchone()
        if row is None:
            return None
        emb = np.frombuffer(row['embedding'], dtype=np.float64) if row['embedding'] else None
        std = np.frombuffer(row['std'], dtype=np.float64) if row['std'] else None
        return {
            'key': row['key'],
            'embedding': emb,
            'std': std,
            'metadata': json.loads(row['metadata']) if row['metadata'] else {},
        }

    def delete_memory(self, key: str) -> bool:
        self.conn.execute("DELETE FROM memories WHERE key=?", (key,))
        self.conn.commit()
        return self.conn.total_changes > 0

    def list_keys(self) -> List[str]:
        return [r[0] for r in self.conn.execute("SELECT key FROM memories").fetchall()]

    # -- Memory values (Langevin) -------------------------------------------

    def get_value(self, memory_key: str) -> float:
        row = self.conn.execute(
            "SELECT value FROM memory_values WHERE memory_key=?", (memory_key,)).fetchone()
        return float(row['value']) if row else 1.0

    def set_value(self, memory_key: str, value: float) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO memory_values (memory_key, value, last_updated) VALUES (?, ?, datetime('now'))",
            (memory_key, value))
        self.conn.commit()

    def get_all_values(self) -> Dict[str, float]:
        rows = self.conn.execute("SELECT memory_key, value FROM memory_values").fetchall()
        return {r['memory_key']: r['value'] for r in rows}

    # -- Contradiction logs -------------------------------------------------

    def log_contradiction(self, key: str, C: float,
                          scale: float = 0, indicator: float = 0,
                          structure: float = 0,
                          details: Optional[Dict] = None) -> None:
        self.conn.execute(
            "INSERT INTO contradictions (key, C, scale, indicator, structure, details) VALUES (?, ?, ?, ?, ?, ?)",
            (key, C, scale, indicator, structure, json.dumps(details or {})))
        self.conn.commit()

    def get_contradiction_history(self, key: str, limit: int = 20) -> List[Dict]:
        rows = self.conn.execute(
            "SELECT * FROM contradictions WHERE key=? ORDER BY id DESC LIMIT ?",
            (key, limit)).fetchall()
        return [dict(r) for r in rows]

    # -- Batch operations ---------------------------------------------------

    def load_all_embeddings(self) -> Tuple[List[str], np.ndarray, Optional[np.ndarray]]:
        """Load all stored embeddings into arrays."""
        rows = self.conn.execute("SELECT key, embedding, std FROM memories").fetchall()
        keys = []
        embs = []
        stds = []
        for r in rows:
        	if r['embedding']:
        		keys.append(r['key'])
        		embs.append(np.frombuffer(r['embedding'], dtype=np.float64))
        		if r['std']:
        			stds.append(np.frombuffer(r['std'], dtype=np.float64))
        if not keys:
        	return [], np.array([]), None
        return keys, np.array(embs), np.array(stds) if stds else None

    def close(self):
        self.conn.close()
