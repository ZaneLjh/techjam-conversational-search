from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from starter.agent import Agent
from starter.projection import ProjectionConfig
from starter.retrieval import e4_fallback_config
from tools.e4_ablation_suite import _metric_view
from tools.paired_report import compare_results


@dataclass(frozen=True)
class E45Variant:
    name: str
    description: str
    config: ProjectionConfig


def declared_variants(
    sidecar_path: str,
    manifest_path: str,
) -> tuple[E45Variant, ...]:
    full = ProjectionConfig(
        enabled=True,
        sidecar_path=sidecar_path,
        manifest_path=manifest_path,
    )
    return (
        E45Variant(
            "full",
            "Uniqueness-gated projection ranking plus exact question rollout.",
            full,
        ),
        E45Variant(
            "frozen_e4",
            "Master E4.5 kill switch; frozen E4 predecessor.",
            replace(full, enabled=False),
        ),
        E45Variant(
            "projection_ranking_only",
            "Unique-posterior projection ranking without question rollout.",
            replace(full, use_question_rollout=False),
        ),
        E45Variant(
            "question_rollout_only",
            "Exact reply-partition rollout without projection reordering.",
            replace(full, use_reranking=False),
        ),
        E45Variant(
            "rerank_posterior_at_most_10",
            "Diagnostic ambiguous-posterior reranking up to ten products.",
            replace(full, max_rerank_posterior_size=10),
        ),
    )


def _paired(candidate: dict, baseline: dict) -> dict:
    paired = compare_results(
        baseline,
        candidate,
        top_n=0,
        include_session_deltas=False,
    )["paired"]
    return {
        "counts": paired["counts"],
        "hit_transitions": paired["hit_transitions"],
        "sum_utility_delta": paired["sum_utility_delta"],
        "mean_utility_delta": paired["mean_utility_delta"],
    }


def _config_view(config: ProjectionConfig) -> dict:
    return {
        "enabled": config.enabled,
        "use_reranking": config.use_reranking,
        "use_question_rollout": config.use_question_rollout,
        "candidate_depth": config.candidate_depth,
        "max_rerank_posterior_size": config.max_rerank_posterior_size,
        "max_posterior_size": config.max_posterior_size,
        "max_sidecar_bytes": config.max_sidecar_bytes,
        "max_uncompressed_sidecar_bytes": config.max_uncompressed_sidecar_bytes,
        "max_manifest_bytes": config.max_manifest_bytes,
        "max_catalog_content_bytes": config.max_catalog_content_bytes,
        "max_catalog_rows": config.max_catalog_rows,
        "max_catalog_row_bytes": config.max_catalog_row_bytes,
        "max_sidecar_row_bytes": config.max_sidecar_row_bytes,
        "min_exact_clues": config.min_exact_clues,
        "min_question_gain": config.min_question_gain,
        "sidecar_path": "<supplied-sidecar>",
        "manifest_path": "<supplied-manifest>",
    }


def run_suite(
    catalog_path: str | Path,
    dataset_path: str | Path,
    sidecar_path: str,
    manifest_path: str,
) -> dict:
    samples = load_jsonl(dataset_path)
    catalog_ids, categories, products = catalog_index(catalog_path)
    variants = declared_variants(sidecar_path, manifest_path)
    agent = Agent(
        catalog_path,
        retrieval_config=e4_fallback_config(),
        projection_config=variants[0].config,
    )
    if not agent.projection_index.ready:
        raise ValueError(
            "projection sidecar did not validate: "
            + agent.projection_index.status_reason
        )
    results: dict[str, dict] = {}
    for variant in variants:
        agent._sessions.clear()
        agent.projection_config = variant.config
        agent.projection_index.config = variant.config
        results[variant.name] = evaluate(
            agent,
            samples,
            catalog_ids,
            categories,
            products,
        )

    full = results["full"]
    fallback = results["frozen_e4"]
    rows = []
    for variant in variants:
        result = results[variant.name]
        rows.append(
            {
                "name": variant.name,
                "description": variant.description,
                "configuration": _config_view(variant.config),
                "metrics": _metric_view(result),
                "paired_variant_vs_frozen_e4": _paired(result, fallback),
                "paired_full_e4_5_vs_variant": _paired(full, result),
            }
        )
    return {
        "schema_version": 1,
        "experiment": "E4.5 projection and exact-rollout ablation suite",
        "sample_count": int(full["sample_count"]),
        "execution": {
            "agent_instances": 1,
            "catalog_indexes_built": 1,
            "projection_indexes_loaded": 1,
            "session_state": "cleared before every variant",
            "predecessor": "frozen_e4",
            "public_partitions_are_cross_validation": False,
        },
        "variants": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fixed E4.5 ablation matrix.")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--sidecar", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run_suite(
        args.catalog,
        args.dataset,
        args.sidecar,
        args.manifest,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
