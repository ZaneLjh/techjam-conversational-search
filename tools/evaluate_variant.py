from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from starter.agent import Agent
from starter.retrieval import (
    RetrievalConfig,
    e4_1_candidate_config,
    e4_1_strict_only_config,
    e4_fallback_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a declared E2/E4 variant through the public harness."
    )
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--disable-candidate-exploration",
        action="store_true",
        help=(
            "Disable unseen-result paging while retaining the selected "
            "ranking configuration."
        ),
    )
    parser.add_argument(
        "--disable-multi-route-ranking",
        action="store_true",
        help="Disable all E4 ranking changes and reproduce the E3 path.",
    )
    parser.add_argument("--disable-current-turn-route", action="store_true")
    parser.add_argument("--disable-ledger-route", action="store_true")
    parser.add_argument("--disable-category-route", action="store_true")
    parser.add_argument("--disable-facet-route", action="store_true")
    parser.add_argument(
        "--disable-constraint-reranking",
        action="store_true",
        help="Keep the route union but order it by the legacy ledger route.",
    )
    parser.add_argument(
        "--disable-soft-relaxation",
        action="store_true",
        help="When exact MUST candidates exist, omit relaxed candidates.",
    )
    parser.add_argument(
        "--e4-1-candidate",
        action="store_true",
        help=(
            "Run the complete E4.1 strict-front/recall-backfill candidate. "
            "Full E4 remains the Agent default."
        ),
    )
    parser.add_argument(
        "--e4-1-strict-only-diagnostic",
        action="store_true",
        help=(
            "Run the public-only compliance-repaired strict diagnostic. "
            "It excludes UNKNOWN candidates and is not promotable as E4.1."
        ),
    )
    parser.add_argument(
        "--disable-e4-1",
        "--e4-fallback-ranking",
        dest="e4_fallback_ranking",
        action="store_true",
        help=(
            "Disable the E4.1 strict-front and auxiliary confidence gate, "
            "reproducing full E4 ranking while retaining E4 retrieval."
        ),
    )
    parser.add_argument("--disable-strict-front", action="store_true")
    parser.add_argument("--disable-auxiliary-confidence-gate", action="store_true")
    parser.add_argument(
        "--relaxed-backfill-slots",
        type=int,
        choices=(0, 1, 2),
        default=2,
    )
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    selected = (
        e4_fallback_config()
        if args.e4_fallback_ranking
        else e4_1_strict_only_config()
        if args.e4_1_strict_only_diagnostic
        else e4_1_candidate_config()
        if args.e4_1_candidate
        else e4_fallback_config()
    )
    retrieval_config = RetrievalConfig(
        enabled=not args.disable_multi_route_ranking,
        use_current_turn_route=not args.disable_current_turn_route,
        use_ledger_route=not args.disable_ledger_route,
        use_category_route=not args.disable_category_route,
        use_facet_route=not args.disable_facet_route,
        use_constraint_reranking=not args.disable_constraint_reranking,
        use_soft_relaxation=(
            selected.use_soft_relaxation and not args.disable_soft_relaxation
        ),
        use_strict_front=(
            selected.use_strict_front and not args.disable_strict_front
        ),
        use_auxiliary_confidence_gate=(
            selected.use_auxiliary_confidence_gate
            and not args.disable_auxiliary_confidence_gate
        ),
        relaxed_backfill_slots=args.relaxed_backfill_slots,
    )
    agent = Agent(
        args.catalog,
        explore_unseen=not args.disable_candidate_exploration,
        retrieval_config=retrieval_config,
    )
    result = evaluate(agent, samples, catalog_ids, categories, products)
    rendered = json.dumps(result, indent=2) + "\n"
    Path(args.output).write_text(rendered, encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "sessions"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
