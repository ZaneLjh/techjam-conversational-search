from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from starter.agent import Agent
from starter.retrieval import RetrievalConfig
from tools.fold_report import METRIC_NAMES
from tools.paired_report import compare_results


SCENARIO_METRIC_NAMES = ("hit_rate_at_10", "mrr", "mttc")


@dataclass(frozen=True)
class AblationVariant:
    name: str
    description: str
    config: RetrievalConfig


def declared_variants() -> tuple[AblationVariant, ...]:
    """Return the fixed E4 ablation matrix in execution order."""

    # Keep this historical E4 suite stable while E4.1 is evaluated separately.
    # The explicit switches reproduce the original score-fused E4.
    full = RetrievalConfig(
        use_strict_front=False,
        use_auxiliary_confidence_gate=False,
    )
    return (
        AblationVariant("full", "All declared E4 retrieval components.", full),
        AblationVariant(
            "e3_compatibility",
            "Disable E4 retrieval and execute the byte-compatible E3 path.",
            RetrievalConfig(enabled=False),
        ),
        AblationVariant(
            "no_current_turn",
            "Remove the current-turn route only.",
            replace(full, use_current_turn_route=False),
        ),
        AblationVariant(
            "no_ledger",
            "Remove the accumulated-ledger route only.",
            replace(full, use_ledger_route=False),
        ),
        AblationVariant(
            "no_category",
            "Remove the category-field route only.",
            replace(full, use_category_route=False),
        ),
        AblationVariant(
            "no_facet",
            "Remove exact facet routes only.",
            replace(full, use_facet_route=False),
        ),
        AblationVariant(
            "no_constraint_reranking",
            "Retain routes but disable the E4 fusion/reranking order.",
            replace(full, use_constraint_reranking=False),
        ),
        AblationVariant(
            "no_soft_relaxation",
            "Keep only strict candidates when exact MUST matches exist.",
            replace(full, use_soft_relaxation=False),
        ),
    )


def _set_variant(agent: Agent, config: RetrievalConfig) -> None:
    """Switch a shared indexed agent while isolating per-variant session state."""

    agent._sessions.clear()
    agent.retrieval_config = config
    agent.retriever.config = config


def evaluate_variants(
    agent: Agent,
    variants: tuple[AblationVariant, ...],
    samples: list[dict],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for variant in variants:
        _set_variant(agent, variant.config)
        results[variant.name] = evaluate(
            agent,
            samples,
            catalog_ids,
            categories,
            products,
        )
    return results


def _metric_view(result: dict) -> dict:
    overall = {
        "sample_count": int(result["sample_count"]),
        **{metric: result[metric] for metric in METRIC_NAMES},
    }
    scenario = {
        name: {
            "sample_count": int(values["sample_count"]),
            **{metric: values[metric] for metric in SCENARIO_METRIC_NAMES},
        }
        for name, values in sorted(result["scenario_metrics"].items())
    }
    return {"overall": overall, "scenario": scenario}


def _metric_delta(full: dict, variant: dict) -> dict:
    overall = {
        metric: round(full[metric] - variant[metric], 6)
        for metric in METRIC_NAMES
    }
    scenario = {}
    if full["scenario_metrics"].keys() != variant["scenario_metrics"].keys():
        raise ValueError("full and variant scenario sets differ")
    for name in sorted(full["scenario_metrics"]):
        scenario[name] = {
            metric: round(
                full["scenario_metrics"][name][metric]
                - variant["scenario_metrics"][name][metric],
                6,
            )
            for metric in SCENARIO_METRIC_NAMES
        }
    return {"overall": overall, "scenario": scenario}


def build_report(
    variants: tuple[AblationVariant, ...],
    results: dict[str, dict],
) -> dict:
    expected_names = [variant.name for variant in variants]
    if not expected_names or expected_names[0] != "full":
        raise ValueError("the first declared variant must be full")
    if set(expected_names) != set(results):
        raise ValueError("results do not match the declared variant names")

    full = results["full"]
    variant_rows = []
    for variant in variants:
        result = results[variant.name]
        paired = compare_results(
            result,
            full,
            top_n=0,
            include_session_deltas=False,
        )["paired"]
        variant_rows.append(
            {
                "name": variant.name,
                "description": variant.description,
                "configuration": asdict(variant.config),
                "metrics": _metric_view(result),
                "delta_full_minus_variant": _metric_delta(full, result),
                "paired_full_vs_variant": {
                    "counts": {
                        "full_improved": paired["counts"]["improved"],
                        "full_regressed": paired["counts"]["regressed"],
                        "tied": paired["counts"]["tied"],
                    },
                    "hit_transitions_variant_to_full": paired["hit_transitions"],
                    "sum_utility_delta": paired["sum_utility_delta"],
                    "mean_utility_delta": paired["mean_utility_delta"],
                },
            }
        )

    return {
        "schema_version": 1,
        "experiment": "E4 controlled multi-route ablation suite",
        "sample_count": int(full["sample_count"]),
        "execution": {
            "agent_instances": 1,
            "catalog_indexes_built": 1,
            "session_state": "cleared before every variant",
            "question_candidate_ranking": "e3_frozen",
        },
        "delta_direction": (
            "full minus variant; positive HitRate/MRR/Efficiency/TechnicalScore "
            "and negative MTTC favor full"
        ),
        "variants": variant_rows,
    }


def run_suite(catalog_path: str | Path, dataset_path: str | Path) -> dict:
    samples = load_jsonl(dataset_path)
    catalog_ids, categories, products = catalog_index(catalog_path)
    variants = declared_variants()

    # Build the superset index once. Switching to E3 compatibility later simply
    # bypasses the E4 side tables, so every variant shares identical catalog data.
    agent = Agent(catalog_path, retrieval_config=variants[0].config)
    results = evaluate_variants(
        agent,
        variants,
        samples,
        catalog_ids,
        categories,
        products,
    )
    return build_report(variants, results)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the fixed E4 ablation matrix with one shared Agent/catalog index."
        )
    )
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
