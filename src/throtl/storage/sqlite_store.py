"""
SQLite storage for metrics history. Flat schema -- one row per
snapshot, one column per metric. Good enough for now, swap to
ClickHouse later if we need to.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

from throtl.metrics import InferenceSnapshot

log = logging.getLogger(__name__)

DEFAULT_DB_PATH = "throtl_metrics.db"

SCHEMA_VERSION = 2


class MetricsStore:

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path)
        # WAL mode lets us read while writing (matters for concurrent dashboards)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._ensure_version_table()
        self._create_table()

    def _ensure_version_table(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER NOT NULL
            )
        """)
        row = self._conn.execute("SELECT version FROM schema_version").fetchone()
        if row is None:
            self._conn.execute("INSERT INTO schema_version (version) VALUES (?)", (0,))
            self._conn.commit()

    def _get_version(self) -> int:
        row = self._conn.execute("SELECT version FROM schema_version").fetchone()
        return row[0] if row else 0

    def _set_version(self, version: int):
        self._conn.execute("UPDATE schema_version SET version = ?", (version,))

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
                cost_per_1k_tokens REAL,
                gpu_cost_per_hour REAL DEFAULT 1.0,
                cumulative_cost_usd REAL DEFAULT 0.0,
                sla_target_ttft_ms REAL DEFAULT 200.0,
                sla_compliance_pct REAL DEFAULT 1.0
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS savings_ledger (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                description TEXT NOT NULL,
                cost_before REAL NOT NULL,
                cost_after REAL NOT NULL,
                estimated_daily_savings REAL NOT NULL,
                config_change TEXT
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS tuning_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                suggestion_title TEXT NOT NULL,
                config_flag TEXT,
                config_change TEXT,
                ttft_p95_before REAL,
                ttft_p95_after REAL,
                tps_before REAL,
                tps_after REAL,
                cost_per_1k_before REAL,
                cost_per_1k_after REAL,
                outcome TEXT
            )
        """)
        self._migrate()
        self._conn.commit()

    def _migrate(self):
        """Run incremental schema migrations based on version number."""
        current = self._get_version()
        if current >= SCHEMA_VERSION:
            return

        existing = {
            row[1] for row in
            self._conn.execute("PRAGMA table_info(snapshots)").fetchall()
        }

        if current < 1:
            log.info("Running migration v0 -> v1: add cost tracking columns")
            for col, typedef in [
                ("gpu_cost_per_hour", "REAL DEFAULT 1.0"),
                ("cumulative_cost_usd", "REAL DEFAULT 0.0"),
            ]:
                if col not in existing:
                    self._conn.execute(f"ALTER TABLE snapshots ADD COLUMN {col} {typedef}")

        if current < 2:
            log.info("Running migration v1 -> v2: add SLA columns")
            for col, typedef in [
                ("sla_target_ttft_ms", "REAL DEFAULT 200.0"),
                ("sla_compliance_pct", "REAL DEFAULT 1.0"),
            ]:
                if col not in existing:
                    self._conn.execute(f"ALTER TABLE snapshots ADD COLUMN {col} {typedef}")

        self._set_version(SCHEMA_VERSION)
        log.info("Schema migrated to version %d", SCHEMA_VERSION)

    def save(self, snapshot: InferenceSnapshot):
        self._conn.execute(
            """
            INSERT INTO snapshots (
                timestamp, requests_running, requests_waiting, requests_completed,
                prompt_tokens_total, generation_tokens_total, tokens_per_second,
                ttft_p50, ttft_p95, ttft_p99, tbt_p50, tbt_p95, tbt_p99,
                gpu_cache_usage_pct, gpu_memory_used_gb, gpu_memory_total_gb,
                gpu_utilization_pct, avg_batch_size, max_batch_size, cost_per_1k_tokens,
                gpu_cost_per_hour, cumulative_cost_usd, sla_target_ttft_ms, sla_compliance_pct
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                snapshot.gpu_cost_per_hour,
                snapshot.cumulative_cost_usd,
                snapshot.sla_target_ttft_ms,
                snapshot.sla_compliance_percent,
            ),
        )
        self._conn.commit()

    def get_recent(self, minutes: int = 10) -> List[InferenceSnapshot]:
        """Pull snapshots from the last N minutes.

        Timestamps are stored as UTC ISO strings. The cutoff is computed
        in Python using UTC to match.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat()
        cursor = self._conn.execute(
            """
            SELECT timestamp, requests_running, requests_waiting, requests_completed,
                   prompt_tokens_total, generation_tokens_total, tokens_per_second,
                   ttft_p50, ttft_p95, ttft_p99, tbt_p50, tbt_p95, tbt_p99,
                   gpu_cache_usage_pct, gpu_memory_used_gb, gpu_memory_total_gb,
                   gpu_utilization_pct, avg_batch_size, max_batch_size, cost_per_1k_tokens,
                   gpu_cost_per_hour, cumulative_cost_usd, sla_target_ttft_ms,
                   sla_compliance_pct
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
                gpu_cost_per_hour=row[20] or 1.0,
                cumulative_cost_usd=row[21] or 0.0,
                sla_target_ttft_ms=row[22] or 200.0,
                sla_compliance_percent=row[23] or 1.0,
            ))
        return results

    def count(self) -> int:
        cursor = self._conn.execute("SELECT COUNT(*) FROM snapshots")
        return cursor.fetchone()[0]

    # -- Savings ledger --

    def record_savings(
        self,
        description: str,
        cost_before: float,
        cost_after: float,
        config_change: Optional[str] = None,
    ):
        daily_savings = (cost_before - cost_after) * 24
        self._conn.execute(
            """
            INSERT INTO savings_ledger
                (timestamp, description, cost_before, cost_after,
                 estimated_daily_savings, config_change)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (datetime.now(timezone.utc).isoformat(), description, cost_before,
             cost_after, daily_savings, config_change),
        )
        self._conn.commit()
        log.info("Savings recorded: %s, daily savings=$%.4f", description, daily_savings)

    def get_savings_history(self) -> List[dict]:
        cursor = self._conn.execute(
            "SELECT timestamp, description, cost_before, cost_after, "
            "estimated_daily_savings, config_change FROM savings_ledger ORDER BY timestamp"
        )
        return [
            {
                "timestamp": row[0], "description": row[1],
                "cost_before": row[2], "cost_after": row[3],
                "daily_savings": row[4], "config_change": row[5],
            }
            for row in cursor.fetchall()
        ]

    def total_estimated_daily_savings(self) -> float:
        cursor = self._conn.execute(
            "SELECT COALESCE(SUM(estimated_daily_savings), 0) FROM savings_ledger"
        )
        return cursor.fetchone()[0]

    # -- Tuning events --

    def record_tuning_event(
        self,
        suggestion_title: str,
        config_flag: Optional[str] = None,
        config_change: Optional[str] = None,
        ttft_p95_before: Optional[float] = None,
        ttft_p95_after: Optional[float] = None,
        tps_before: Optional[float] = None,
        tps_after: Optional[float] = None,
        cost_per_1k_before: Optional[float] = None,
        cost_per_1k_after: Optional[float] = None,
        outcome: Optional[str] = None,
    ):
        self._conn.execute(
            """
            INSERT INTO tuning_events
                (timestamp, suggestion_title, config_flag, config_change,
                 ttft_p95_before, ttft_p95_after, tps_before, tps_after,
                 cost_per_1k_before, cost_per_1k_after, outcome)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (datetime.now(timezone.utc).isoformat(), suggestion_title, config_flag,
             config_change, ttft_p95_before, ttft_p95_after, tps_before,
             tps_after, cost_per_1k_before, cost_per_1k_after, outcome),
        )
        self._conn.commit()
        log.info("Tuning event recorded: %s", suggestion_title)

    def get_tuning_history(self) -> List[dict]:
        cursor = self._conn.execute(
            "SELECT timestamp, suggestion_title, config_flag, config_change, "
            "ttft_p95_before, ttft_p95_after, tps_before, tps_after, "
            "cost_per_1k_before, cost_per_1k_after, outcome "
            "FROM tuning_events ORDER BY timestamp"
        )
        return [
            {
                "timestamp": row[0], "suggestion_title": row[1],
                "config_flag": row[2], "config_change": row[3],
                "ttft_p95_before": row[4], "ttft_p95_after": row[5],
                "tps_before": row[6], "tps_after": row[7],
                "cost_per_1k_before": row[8], "cost_per_1k_after": row[9],
                "outcome": row[10],
            }
            for row in cursor.fetchall()
        ]

    def close(self):
        self._conn.close()
