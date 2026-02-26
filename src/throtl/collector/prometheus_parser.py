"""
Prometheus text format parser for the subset vLLM uses
(gauges, counters, histograms). No external deps.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class MetricSample:
    name: str
    labels: Dict[str, str]
    value: float


@dataclass
class MetricFamily:
    name: str
    metric_type: str  # "gauge", "counter", "histogram", "summary", "untyped"
    help_text: str
    samples: List[MetricSample] = field(default_factory=list)


# Matches key="value" pairs inside braces, e.g. {method="GET",code="200"}
_LABEL_RE = re.compile(r'(\w+)="([^"]*)"')


def parse_labels(label_str: str) -> Dict[str, str]:
    if not label_str:
        return {}
    return dict(_LABEL_RE.findall(label_str))


def parse_prometheus_text(text: str) -> Dict[str, MetricFamily]:
    """Returns a dict keyed by base metric name (strips _total, _bucket, etc)."""
    families: Dict[str, MetricFamily] = {}
    current_type: Dict[str, str] = {}
    current_help: Dict[str, str] = {}

    for line in text.strip().split("\n"):
        line = line.strip()

        if not line:
            continue

        if line.startswith("# HELP "):
            parts = line[7:].split(" ", 1)
            if len(parts) == 2:
                current_help[parts[0]] = parts[1]
            continue

        if line.startswith("# TYPE "):
            parts = line[7:].split(" ", 1)
            if len(parts) == 2:
                current_type[parts[0]] = parts[1]
            continue

        if line.startswith("#"):
            continue

        # Parse a sample line: metric_name{labels} value [timestamp]
        # or: metric_name value [timestamp]
        brace_start = line.find("{")
        if brace_start != -1:
            name = line[:brace_start]
            brace_end = line.find("}", brace_start)
            label_str = line[brace_start + 1:brace_end]
            value_str = line[brace_end + 1:].strip().split()[0]
        else:
            parts = line.split()
            if len(parts) < 2:
                continue
            name = parts[0]
            label_str = ""
            value_str = parts[1]

        try:
            value = float(value_str)
        except ValueError:
            continue

        labels = parse_labels(label_str)

        # Figure out the base family name
        base_name = name
        for suffix in ("_total", "_bucket", "_sum", "_count", "_created"):
            if name.endswith(suffix):
                base_name = name[: -len(suffix)]
                break

        if base_name not in families:
            families[base_name] = MetricFamily(
                name=base_name,
                metric_type=current_type.get(base_name, "untyped"),
                help_text=current_help.get(base_name, ""),
            )

        families[base_name].samples.append(
            MetricSample(name=name, labels=labels, value=value)
        )

    return families


def get_gauge(families: Dict[str, MetricFamily], name: str) -> Optional[float]:
    family = families.get(name)
    if family and family.samples:
        return family.samples[0].value
    return None


def get_counter(families: Dict[str, MetricFamily], name: str) -> Optional[float]:
    family = families.get(name)
    if not family:
        return None
    for sample in family.samples:
        if sample.name.endswith("_total") or sample.name == name:
            return sample.value
    return None


def get_histogram_percentile(
    families: Dict[str, MetricFamily],
    name: str,
    percentile: float,
) -> Optional[float]:
    """Estimate a percentile from histogram buckets using linear interpolation.

    This is the same approach Prometheus uses for histogram_quantile().
    Not perfect -- accuracy depends on bucket boundary choices -- but it's
    good enough for monitoring latency distributions in real time.
    """
    family = families.get(name)
    if not family:
        return None

    buckets = []
    total_count = None

    for sample in family.samples:
        if sample.name.endswith("_bucket"):
            le = sample.labels.get("le")
            if le is not None and le != "+Inf":
                try:
                    buckets.append((float(le), sample.value))
                except ValueError:
                    continue
            elif le == "+Inf":
                total_count = sample.value
        elif sample.name.endswith("_count"):
            total_count = sample.value

    if not buckets or total_count is None or total_count == 0:
        return None

    buckets.sort(key=lambda x: x[0])
    target = percentile * total_count

    prev_bound = 0.0
    prev_count = 0.0

    for bound, count in buckets:
        if count >= target:
            # Linear interpolation within this bucket
            fraction = (target - prev_count) / max(1, count - prev_count)
            return prev_bound + fraction * (bound - prev_bound)
        prev_bound = bound
        prev_count = count

    # If we're past all buckets, return the last boundary
    return buckets[-1][0] if buckets else None
