"""
Fake vLLM /metrics server for testing without a GPU.

    python -m src.throtl.mock.fake_vllm_server
    python -m src.throtl.main --url http://localhost:9100
"""

from __future__ import annotations

import math
import random
import time
from http.server import HTTPServer, BaseHTTPRequestHandler


_tick = 0
_rng = random.Random(42)
_prompt_tokens = 0
_gen_tokens = 0

# Histogram bucket boundaries matching vLLM defaults
TTFT_BUCKETS = [0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
TBT_BUCKETS = [0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 1.0, 2.5]


def _build_histogram_buckets(buckets: list, center: float, spread: float, count: int) -> str:
    """Build Prometheus histogram bucket lines.

    Uses a sigmoid to approximate a CDF around `center` -- not statistically
    precise, but produces believable bucket distributions for testing.
    """
    lines = []
    cumulative = 0
    total_sum = 0.0

    for le in buckets:
        z = (le - center) / max(0.001, spread)
        frac = 1.0 / (1.0 + math.exp(-3 * z))
        cumulative = int(frac * count)
        lines.append(f'{{le="{le}"}} {cumulative}')
        total_sum += le * max(0, cumulative - (int(frac * count) if le == buckets[0] else int((1.0 / (1.0 + math.exp(-3 * ((buckets[buckets.index(le) - 1] - center) / max(0.001, spread))))) * count)))

    lines.append(f'{{le="+Inf"}} {count}')
    return lines, count, center * count


def _generate_metrics_text() -> str:
    """Build a full Prometheus text blob mimicking vLLM's /metrics output."""
    global _tick, _prompt_tokens, _gen_tokens
    _tick += 1
    t = _tick

    # Traffic pattern
    base_load = 8 + 6 * math.sin(t * 0.1)
    spike = _rng.random() * 8 if _rng.random() > 0.9 else 0
    running = max(1, int(base_load + spike))
    waiting = max(0, int((running - 12) * _rng.uniform(0.5, 1.5)))

    # Throughput
    batch = min(running, 16)
    tps = max(10, 45 + batch * 8 + _rng.gauss(0, 5))
    gen_this_tick = int(tps * 2)
    prompt_this_tick = int(gen_this_tick * 0.6)
    _gen_tokens += gen_this_tick
    _prompt_tokens += prompt_this_tick

    # Cache
    cache = max(0.1, min(0.98, 0.3 + (running / 20) * 0.5 + _rng.gauss(0, 0.03)))

    # Latency centers
    ttft_center = 0.08 + (waiting * 0.02) + (cache * 0.05)
    tbt_center = 0.012 + (running / 50) * 0.005

    request_count = max(1, _tick * 3)

    ttft_bucket_lines, ttft_count, ttft_sum = _build_histogram_buckets(
        TTFT_BUCKETS, ttft_center, ttft_center * 0.5, request_count
    )
    tbt_bucket_lines, tbt_count, tbt_sum = _build_histogram_buckets(
        TBT_BUCKETS, tbt_center, tbt_center * 0.3, request_count * 50
    )

    lines = [
        "# HELP vllm:num_requests_running Number of requests currently running",
        "# TYPE vllm:num_requests_running gauge",
        f"vllm:num_requests_running {running}",
        "",
        "# HELP vllm:num_requests_waiting Number of requests waiting to be processed",
        "# TYPE vllm:num_requests_waiting gauge",
        f"vllm:num_requests_waiting {waiting}",
        "",
        "# HELP vllm:gpu_cache_usage_perc GPU KV-cache usage",
        "# TYPE vllm:gpu_cache_usage_perc gauge",
        f"vllm:gpu_cache_usage_perc {cache:.4f}",
        "",
        "# HELP vllm:cpu_cache_usage_perc CPU KV-cache usage",
        "# TYPE vllm:cpu_cache_usage_perc gauge",
        f"vllm:cpu_cache_usage_perc 0.0000",
        "",
        "# HELP vllm:prompt_tokens_total Number of prefill tokens processed",
        "# TYPE vllm:prompt_tokens_total counter",
        f"vllm:prompt_tokens_total {_prompt_tokens}",
        "",
        "# HELP vllm:generation_tokens_total Number of generation tokens processed",
        "# TYPE vllm:generation_tokens_total counter",
        f"vllm:generation_tokens_total {_gen_tokens}",
        "",
        "# HELP vllm:avg_prompt_throughput_toks_per_s Average prefill throughput",
        "# TYPE vllm:avg_prompt_throughput_toks_per_s gauge",
        f"vllm:avg_prompt_throughput_toks_per_s {tps * 0.4:.1f}",
        "",
        "# HELP vllm:avg_generation_throughput_toks_per_s Average generation throughput",
        "# TYPE vllm:avg_generation_throughput_toks_per_s gauge",
        f"vllm:avg_generation_throughput_toks_per_s {tps * 0.6:.1f}",
        "",
        "# HELP vllm:time_to_first_token_seconds Histogram of TTFT in seconds",
        "# TYPE vllm:time_to_first_token_seconds histogram",
    ]

    for bl in ttft_bucket_lines:
        lines.append(f"vllm:time_to_first_token_seconds_bucket{bl}")
    lines.append(f"vllm:time_to_first_token_seconds_count {ttft_count}")
    lines.append(f"vllm:time_to_first_token_seconds_sum {ttft_sum:.4f}")

    lines.append("")
    lines.append("# HELP vllm:time_per_output_token_seconds Histogram of inter-token latency")
    lines.append("# TYPE vllm:time_per_output_token_seconds histogram")

    for bl in tbt_bucket_lines:
        lines.append(f"vllm:time_per_output_token_seconds_bucket{bl}")
    lines.append(f"vllm:time_per_output_token_seconds_count {tbt_count}")
    lines.append(f"vllm:time_per_output_token_seconds_sum {tbt_sum:.4f}")

    return "\n".join(lines) + "\n"


class _MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/metrics":
            body = _generate_metrics_text().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress request logging noise


def run_fake_server(host: str = "127.0.0.1", port: int = 9100):
    server = HTTPServer((host, port), _MetricsHandler)
    print(f"Fake vLLM metrics server running at http://{host}:{port}/metrics")
    print("Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    server.server_close()
    print("\nServer stopped.")


if __name__ == "__main__":
    run_fake_server()
