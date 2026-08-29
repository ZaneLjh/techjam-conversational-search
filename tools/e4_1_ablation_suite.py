from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from starter.agent import Agent
from starter.retrieval import (
    RetrievalConfig,
    e4_1_candidate_config,
    e4_1_strict_only_config,
    e4_fallback_config,
)
from tools.e4_ablation_suite import _metric_view, evaluate_variants
from tools.paired_report import compare_results


@dataclass(frozen=True)
class E41Variant:
    name: str
    description: str
    config: RetrievalConfig


def declared_variants() -> tuple[E41Variant, ...]:
    """Return the fixed E4.1 repair and route-interaction matrix."""

    full = e4_1_candidate_config()
    relaxed = full
    variants: list[E41Variant] = [
        E41Variant(
            "full",
            "Complete E4.1 strict-front, recall-backfill, and gated-fusion policy.",
            full,
        ),
        E41Variant(
            "e4_fallback",
            "Original full-E4 score-fused ranking.",
            e4_fallback_config(),
        ),
        E41Variant(
            "strict_only_diagnostic",
            "Public-only strict diagnostic; excludes UNKNOWN and is not promotable.",
            e4_1_strict_only_config(),
        ),
        E41Variant(
            "strict_front_two_slots",
            "Strict front with one first-page and two post-miss recovery slots.",
            relaxed,
        ),
        E41Variant(
            "one_relaxed_slot",
            "Reserve one rather than two lower Top-10 recovery slots.",
            replace(relaxed, relaxed_backfill_slots=1),
        ),
        E41Variant(
            "zero_relaxed_slots",
            "Strict-front ordering without a reserved recovery slot.",
            replace(relaxed, relaxed_backfill_slots=0),
        ),
        E41Variant(
            "no_strict_front",
            "Retain confidence gating but remove the strict-front cascade.",
            replace(relaxed, use_strict_front=False),
        ),
        E41Variant(
            "no_auxiliary_gate",
            "Retain strict-front ordering but restore ungated auxiliary fusion.",
            replace(relaxed, use_auxiliary_confidence_gate=False),
        ),
    ]
    for current, ledger, category in (
        (False, True, True),
        (True, False, True),
        (True, True, False),
        (False, False, True),
        (False, True, False),
        (True, False, False),
        (False, False, False),
    ):
        label = f"routes_c{int(current)}l{int(ledger)}g{int(category)}"
        variants.append(
            E41Variant(
                label,
                "Route interaction with facet retrieval retained; "
                f"current={current}, ledger={ledger}, category={category}.",
                replace(
                    full,
                    use_current_turn_route=current,
                    use_ledger_route=ledger,
                    use_category_route=category,
                ),
            )
        )
    return tuple(variants)


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


def build_report(
    variants: tuple[E41Variant, ...],
    results: dict[str, dict],
) -> dict:
    full = results["full"]
    fallback = results["e4_fallback"]
    rows = []
    for variant in variants:
        result = results[variant.name]
        rows.append(
            {
                "name": variant.name,
                "description": variant.description,
                "configuration": asdict(variant.config),
                "metrics": _metric_view(result),
                "paired_candidate_vs_e4_fallback": _paired(result, fallback),
                "paired_full_e4_1_vs_variant": _paired(full, result),
            }
        )
    return {
        "schema_version": 1,
        "experiment": "E4.1 strict-front compliance and fusion repair",
        "sample_count": int(full["sample_count"]),
        "execution": {
            "agent_instances": 1,
            "catalog_indexes_built": 1,
            "session_state": "cleared before every variant",
            "question_candidate_ranking": "e3_frozen",
            "public_partitions_are_cross_validation": False,
        },
        "variants": rows,
    }


def run_suite(catalog_path: str | Path, dataset_path: str | Path) -> dict:
    samples = load_jsonl(dataset_path)
    catalog_ids, categories, products = catalog_index(catalog_path)
    variants = declared_variants()
    agent = Agent(catalog_path, retrieval_config=variants[0].config)
    results = evaluate_variants(
        agent,
        variants,  # type: ignore[arg-type]
        samples,
        catalog_ids,
        categories,
        products,
    )
    return build_report(variants, results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fixed E4.1 ablation matrix.")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run_suite(args.catalog, args.dataset)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
