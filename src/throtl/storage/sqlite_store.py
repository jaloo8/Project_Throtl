"""
SQLite storage for metrics history. Flat schema -- one row per
snapshot, one column per metric. Good enough for now, swap to
ClickHouse later if we need to.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from src.throtl.metrics import InferenceSnapshot

DEFAULT_DB_PATH = "throtl_metrics.db"


class MetricsStore:

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path)
        # WAL mode lets us read while writing (matters for concurrent dashboards)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_table()

    def _create_table(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                requests_running INTEGER,
                requests_waiting INTEGER,
                requests_completed INTEGER,
                prompt_tokens_total INTEGER,
                generation_tokens_total INTEGER,
                tokens_per_second REAL,
                ttft_p50 REAL,
                ttft_p95 REAL,
                ttft_p99 REAL,
                tbt_p50 REAL,
                tbt_p95 REAL,
                tbt_p99 REAL,
                gpu_cache_usage_pct REAL,
                gpu_memory_used_gb REAL,
                gpu_memory_total_gb REAL,
                gpu_utilization_pct REAL,
                avg_batch_size REAL,
                max_batch_size INTEGER,
                cost_per_1k_tokens REAL
            )
        """)
        self._conn.commit()

    def save(self, snapshot: InferenceSnapshot):
        self._conn.execute(
            """
            INSERT INTO snapshots (
                timestamp, requests_running, requests_waiting, requests_completed,
                prompt_tokens_total, generation_tokens_total, tokens_per_second,
                ttft_p50, ttft_p95, ttft_p99, tbt_p50, tbt_p95, tbt_p99,
                gpu_cache_usage_pct, gpu_memory_used_gb, gpu_memory_total_gb,
                gpu_utilization_pct, avg_batch_size, max_batch_size, cost_per_1k_tokens
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot.timestamp.isoformat(),
                snapshot.requests_running,
                snapshot.requests_waiting,
                snapshot.requests_completed,
                snapshot.prompt_tokens_total,
                snapshot.generation_tokens_total,
                snapshot.tokens_per_second,
                snapshot.time_to_first_token_p50,
                snapshot.time_to_first_token_p95,
                snapshot.time_to_first_token_p99,
                snapshot.time_per_output_token_p50,
                snapshot.time_per_output_token_p95,
                snapshot.time_per_output_token_p99,
                snapshot.gpu_cache_usage_percent,
                snapshot.gpu_memory_used_gb,
                snapshot.gpu_memory_total_gb,
                snapshot.gpu_utilization_percent,
                snapshot.avg_batch_size,
                snapshot.max_batch_size,
                snapshot.estimated_cost_per_1k_tokens,
            ),
        )
        self._conn.commit()

    def get_recent(self, minutes: int = 10) -> List[InferenceSnapshot]:
        """Pull snapshots from the last N minutes.

        We compute the cutoff in Python (not SQLite's datetime('now'))
        because timestamps are stored in local time, not UTC.
        """
        from datetime import timedelta
        cutoff = (datetime.now() - timedelta(minutes=minutes)).isoformat()
        cursor = self._conn.execute(
            """
            SELECT timestamp, requests_running, requests_waiting, requests_completed,
                   prompt_tokens_total, generation_tokens_total, tokens_per_second,
                   ttft_p50, ttft_p95, ttft_p99, tbt_p50, tbt_p95, tbt_p99,
                   gpu_cache_usage_pct, gpu_memory_used_gb, gpu_memory_total_gb,
                   gpu_utilization_pct, avg_batch_size, max_batch_size, cost_per_1k_tokens
            FROM snapshots
            WHERE timestamp >= ?
            ORDER BY timestamp ASC
            """,
            (cutoff,),
        )

        results = []
        for row in cursor.fetchall():
            results.append(InferenceSnapshot(
                timestamp=datetime.fromisoformat(row[0]),
                requests_running=row[1],
                requests_waiting=row[2],
                requests_completed=row[3],
                prompt_tokens_total=row[4],
                generation_tokens_total=row[5],
                tokens_per_second=row[6],
                time_to_first_token_p50=row[7],
                time_to_first_token_p95=row[8],
                time_to_first_token_p99=row[9],
                time_per_output_token_p50=row[10],
                time_per_output_token_p95=row[11],
                time_per_output_token_p99=row[12],
                gpu_cache_usage_percent=row[13],
                gpu_memory_used_gb=row[14],
                gpu_memory_total_gb=row[15],
                gpu_utilization_percent=row[16],
                avg_batch_size=row[17],
                max_batch_size=row[18],
                estimated_cost_per_1k_tokens=row[19],
            ))
        return results

    def count(self) -> int:
        cursor = self._conn.execute("SELECT COUNT(*) FROM snapshots")
        return cursor.fetchone()[0]

    def close(self):
        self._conn.close()
