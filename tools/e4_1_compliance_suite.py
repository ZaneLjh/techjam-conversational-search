from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

from starter.agent import Agent
from starter.constraints import ConstraintLedger, Facet, parse_message
from starter.retrieval import RetrievalConfig, e4_1_strict_only_config


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    evidence: dict

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "evidence": self.evidence,
        }


def _write_catalog(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _ranking_checks() -> list[Check]:
    rows = [
        {
            "parent_asin": "STRICT",
            "title": "Walking shoe",
            "features": ["rare alpha feature"],
            "categories": ["Shoes", "Walking"],
        },
        {
            "parent_asin": "UNKNOWN",
            "title": "Walking shoe",
            "categories": ["Shoes", "Walking"],
        },
        {
            "parent_asin": "MISMATCH",
            "title": "Walking shoe",
            "features": ["different beta feature"],
            "categories": ["Shoes", "Walking"],
        },
    ]
    checks: list[Check] = []
    with tempfile.TemporaryDirectory() as temp_dir:
        catalog = Path(temp_dir) / "catalog.jsonl"
        _write_catalog(catalog, rows)
        # Exercise the three-state E4.1 cascade explicitly. Agent() remains the
        # frozen E4 deployment fallback until product-disjoint promotion.
        agent = Agent(catalog, retrieval_config=RetrievalConfig())
        agent.reset("compat", {})
        response = agent.respond(
            "compat",
            "I'm looking for walking shoes. A key requirement is: rare alpha feature.",
            1,
            3,
        )
        trace = agent.evidence_trace("compat")["retrieval_decisions"][-1]
        by_id = {item["parent_asin"]: item for item in trace["top_candidates"]}
        checks.append(
            Check(
                "three_state_compatibility",
                response["recommendations"][0]["parent_asin"] == "STRICT"
                and by_id["STRICT"]["strict"]
                and by_id["UNKNOWN"]["unknown_must_count"] == 1
                and by_id["UNKNOWN"]["mismatched_must_count"] == 0
                and by_id["MISMATCH"]["mismatched_must_count"] == 1,
                {
                    "displayed": response["recommendations"],
                    "tiers": {
                        key: {
                            "strict": value["strict"],
                            "unknown": value["unknown_must_count"],
                            "mismatch": value["mismatched_must_count"],
                        }
                        for key, value in by_id.items()
                    },
                },
            )
        )

        budget_agent = Agent(catalog, retrieval_config=RetrievalConfig())
        budget_agent.reset("budget", {})
        budget_agent.respond(
            "budget",
            "I'm looking for walking shoes. A key requirement is: budget around $25.",
            1,
            3,
        )
        budget_trace = budget_agent.evidence_trace("budget")["retrieval_decisions"][-1]
        checks.append(
            Check(
                "price_is_soft_and_missing_is_neutral",
                budget_trace["routed_must_constraint_count"] == 0
                and not budget_trace["strict_front_applied"],
                {
                    "routed_must_constraint_count": budget_trace[
                        "routed_must_constraint_count"
                    ],
                    "strict_front_applied": budget_trace["strict_front_applied"],
                },
            )
        )

        strict_agent = Agent(
            catalog,
            retrieval_config=e4_1_strict_only_config(),
        )
        strict_agent.reset("page", {})
        first = strict_agent.respond(
            "page",
            "I'm looking for walking shoes. A key requirement is: rare alpha feature.",
            1,
            3,
        )
        second = strict_agent.respond(
            "page",
            "Those options are not quite right yet. Ask me about one specific attribute.",
            2,
            3,
        )
        checks.append(
            Check(
                "partial_strict_page_advances",
                bool(first["recommendations"])
                and not {
                    item["parent_asin"] for item in first["recommendations"]
                }.intersection(
                    item["parent_asin"] for item in second["recommendations"]
                ),
                {"first": first["recommendations"], "second": second["recommendations"]},
            )
        )
    return checks


def _ledger_checks() -> list[Check]:
    checks: list[Check] = []
    ledger = ConstraintLedger()
    ledger.apply(
        parse_message(
            "I'm looking for jackets. A key requirement is: cotton.",
            1,
        )
    )
    ledger.apply(
        parse_message(
            "I don't have an additional preference for material.",
            2,
            "material",
        )
    )
    additional_query = ledger.canonical_query()
    ledger.apply(
        parse_message(
            "I don't have a preference for material; please use your judgment.",
            3,
            "material",
        )
    )
    boundary_query = ledger.canonical_query()
    checks.append(
        Check(
            "no_preference_and_no_additional_are_behaviorally_distinct",
            "cotton" in additional_query and "cotton" not in boundary_query,
            {
                "after_no_additional": additional_query,
                "after_no_preference": boundary_query,
            },
        )
    )

    correction = parse_message(
        "Actually, ignore my earlier preference. What I need is: color: blue; slim fit.",
        3,
    )
    facets = [item.facet.value for item in correction.constraints]
    checks.append(
        Check(
            "semicolon_multi_slot_override",
            correction.is_override
            and Facet.COLOR.value in facets
            and Facet.STYLE.value in facets,
            {"facets": facets, "is_override": correction.is_override},
        )
    )
    return checks


def build_report() -> dict:
    checks = [*_ranking_checks(), *_ledger_checks()]
    return {
        "schema_version": 1,
        "experiment": "E4.1 compliance suite",
        "passed": all(check.passed for check in checks),
        "checks": [check.as_dict() for check in checks],
        "frozen_components": [
            "starter/constraints.py",
            "starter/question_policy.py",
            "evaluator/local_evaluator.py",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic E4.1 compliance probes.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = build_report()
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
