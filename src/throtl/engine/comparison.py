"""
Before/after comparison for measuring the impact of config changes.

Copyright (c) 2026 JL -- see NOTICE and LICENSE files.

Usage: collect a baseline window, make a change, collect an after window,
then compare. Shows deltas for every key metric so you can see exactly
what improved, what regressed, and by how much.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from throtl.metrics import InferenceSnapshot


@dataclass
class MetricDelta:
    name: str
    before: float
    after: float
    delta: float
    delta_pct: float       # percentage change
    improved: Optional[bool]  # None if we can't determine direction


@dataclass
class ComparisonReport:
    before_count: int
    after_count: int
    deltas: List[MetricDelta]
    summary: str  # one-line verdict

    @property
    def improvements(self) -> List[MetricDelta]:
        return [d for d in self.deltas if d.improved is True]

    @property
    def regressions(self) -> List[MetricDelta]:
        return [d for d in self.deltas if d.improved is False]


# Metrics and whether "lower is better"
_METRICS = [
    ("tokens_per_second",          "Tokens/sec",          False),
    ("requests_waiting",           "Queue depth",         True),
    ("time_to_first_token_p50",    "TTFT p50",            True),
    ("time_to_first_token_p95",    "TTFT p95",            True),
    ("time_to_first_token_p99",    "TTFT p99",            True),
    ("time_per_output_token_p50",  "TBT p50",             True),
    ("time_per_output_token_p95",  "TBT p95",             True),
    ("gpu_cache_usage_percent",    "KV cache usage",      True),
    ("gpu_utilization_percent",    "GPU utilization",     False),
    ("avg_batch_size",             "Avg batch size",      False),
    ("estimated_cost_per_1k_tokens", "Cost per 1K tokens", True),
    ("cumulative_cost_usd",        "Cumulative cost",     True),
    ("sla_compliance_percent",     "SLA compliance",      False),
]


def _avg(snapshots: List[InferenceSnapshot], attr: str) -> float:
    values = [getattr(s, attr) for s in snapshots]
    return sum(values) / len(values) if values else 0


def compare(
    before: List[InferenceSnapshot],
    after: List[InferenceSnapshot],
) -> ComparisonReport:
    """Compare two windows of snapshots and report what changed."""
    if not before or not after:
        return ComparisonReport(
            before_count=len(before),
            after_count=len(after),
            deltas=[],
            summary="Need snapshots in both windows to compare.",
        )

    deltas = []

    for attr, display_name, lower_is_better in _METRICS:
        before_val = _avg(before, attr)
        after_val = _avg(after, attr)
        delta = after_val - before_val
        delta_pct = (delta / abs(before_val) * 100) if before_val != 0 else 0

        # Determine if this change is an improvement
        if abs(delta_pct) < 1.0:
            improved = None  # within noise
        elif lower_is_better:
            improved = delta < 0
        else:
            improved = delta > 0

        deltas.append(MetricDelta(
            name=display_name,
            before=before_val,
            after=after_val,
            delta=delta,
            delta_pct=delta_pct,
            improved=improved,
        ))

    # Build a one-line summary
    improvements = sum(1 for d in deltas if d.improved is True)
    regressions = sum(1 for d in deltas if d.improved is False)

    if regressions == 0 and improvements > 0:
        summary = f"{improvements} metrics improved, no regressions."
    elif improvements == 0 and regressions > 0:
        summary = f"{regressions} metrics regressed, no improvements."
    elif improvements > 0 and regressions > 0:
        summary = f"{improvements} improved, {regressions} regressed -- mixed results."
    else:
        summary = "No significant changes detected."

    return ComparisonReport(
        before_count=len(before),
        after_count=len(after),
        deltas=deltas,
        summary=summary,
    )
