from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from starter.agent import Agent
from starter.retrieval import (
    RetrievalConfig,
    e4_1_candidate_config,
    e4_fallback_config,
)

try:
    import resource
except ImportError:  # pragma: no cover - unavailable on Windows
    resource = None


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(quantile * len(ordered) + 0.999999) - 1))
    return ordered[index]


class TimedAgent:
    def __init__(
        self,
        catalog_path: str | Path,
        retrieval_config: RetrievalConfig | None = None,
    ) -> None:
        started = time.perf_counter()
        self.agent = Agent(catalog_path, retrieval_config=retrieval_config)
        self.startup_seconds = time.perf_counter() - started
        self.response_seconds: list[float] = []

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.agent.reset(session_id, user_profile)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        started = time.perf_counter()
        response = self.agent.respond(session_id, user_message, turn, top_k)
        self.response_seconds.append(time.perf_counter() - started)
        return response


def peak_rss_mb() -> float | None:
    if resource is None:
        return None
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        rss /= 1024.0
    return round(rss / 1024.0, 3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark E4 with the public harness.")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="results/e4_benchmark.json")
    parser.add_argument(
        "--disable-multi-route-ranking",
        action="store_true",
        help="Benchmark the E3 compatibility path.",
    )
    parser.add_argument(
        "--e4-1-candidate",
        action="store_true",
        help="Benchmark the complete E4.1 experiment instead of frozen E4.",
    )
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    selected = (
        e4_1_candidate_config()
        if args.e4_1_candidate
        else e4_fallback_config()
    )
    agent = TimedAgent(
        args.catalog,
        replace(selected, enabled=not args.disable_multi_route_ranking),
    )
    evaluation_started = time.perf_counter()
    result = evaluate(agent, samples, catalog_ids, categories, products)
    evaluation_seconds = time.perf_counter() - evaluation_started
    latencies_ms = [seconds * 1000.0 for seconds in agent.response_seconds]
    report = {
        "sample_count": len(samples),
        "response_count": len(latencies_ms),
        "startup_seconds": round(agent.startup_seconds, 6),
        "evaluation_seconds": round(evaluation_seconds, 6),
        "latency_ms": {
            "mean": round(statistics.fmean(latencies_ms), 6),
            "p50": round(percentile(latencies_ms, 0.50), 6),
            "p95": round(percentile(latencies_ms, 0.95), 6),
            "p99": round(percentile(latencies_ms, 0.99), 6),
            "max": round(max(latencies_ms), 6),
        },
        "peak_rss_mb": peak_rss_mb(),
        "network_calls": 0,
        "reported_token_usage": result["reported_token_usage"],
        "recommended_technical_score": result["recommended_technical_score"],
    }
    rendered = json.dumps(report, indent=2) + "\n"
    Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
