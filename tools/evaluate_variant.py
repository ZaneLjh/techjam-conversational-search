from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from starter.agent import Agent
from starter.retrieval import RetrievalConfig


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
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    retrieval_config = RetrievalConfig(
        enabled=not args.disable_multi_route_ranking,
        use_current_turn_route=not args.disable_current_turn_route,
        use_ledger_route=not args.disable_ledger_route,
        use_category_route=not args.disable_category_route,
        use_facet_route=not args.disable_facet_route,
        use_constraint_reranking=not args.disable_constraint_reranking,
        use_soft_relaxation=not args.disable_soft_relaxation,
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
