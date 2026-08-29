from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from evaluator.local_evaluator import (
    MAX_TURNS,
    TOP_K,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
    normalize_recommendations,
)
from starter.agent import Agent
from starter.retrieval import (
    RetrievalConfig,
    e4_1_candidate_config,
    e4_fallback_config,
)


DEPTHS = (10, 20, 50, 100)


def _safe_ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def run_report(
    catalog_path: str | Path,
    dataset_path: str | Path,
    *,
    config: RetrievalConfig | None = None,
) -> dict:
    """Replay all ten turns and measure target inclusion diagnostics.

    Unlike the official evaluator this intentionally continues after a hit.
    It is an offline public-label diagnostic only; no target or scenario value
    is passed into the production Agent.
    """

    samples = load_jsonl(dataset_path)
    catalog_ids, categories, products = catalog_index(catalog_path)
    agent = Agent(catalog_path, retrieval_config=config or e4_1_candidate_config())
    turn_rows: dict[int, list[dict]] = defaultdict(list)
    session_rows: list[dict] = []

    for sample in samples:
        session_id = f"funnel_{sample['sample_id']}"
        agent.reset(session_id, sample["user_profile"])
        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(sample, products)
        effective = {**sample, "intent_card": card, "behavior": behavior}
        disclosed: set[str] = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        user_message = initial_message(
            effective,
            coarse_category(categories.get(target, [])),
            disclosed,
        )
        cumulative = {depth: set() for depth in DEPTHS}
        first_candidate_turn: int | None = None
        first_displayed_turn: int | None = None
        best_displayed_rank: int | None = None

        for turn in range(1, MAX_TURNS + 1):
            response = agent.respond(session_id, user_message, turn, TOP_K)
            ranked = normalize_recommendations(
                response.get("recommendations"), catalog_ids
            )
            trace_rows = agent.evidence_trace(session_id)["retrieval_decisions"]
            trace = trace_rows[-1] if trace_rows else {}
            candidate_ids = [
                str(value)
                for value in trace.get("candidate_ids", ranked)
                if str(value) in catalog_ids
            ][:100]

            if override_applied:
                instant: dict[str, bool] = {}
                cumulative_hit: dict[str, bool] = {}
                for depth in DEPTHS:
                    cumulative[depth].update(candidate_ids[:depth])
                    instant[str(depth)] = target in candidate_ids[:depth]
                    cumulative_hit[str(depth)] = target in cumulative[depth]
                displayed_rank = ranked.index(target) + 1 if target in ranked else None
                if target in candidate_ids and first_candidate_turn is None:
                    first_candidate_turn = turn
                if displayed_rank is not None:
                    if first_displayed_turn is None:
                        first_displayed_turn = turn
                    best_displayed_rank = (
                        displayed_rank
                        if best_displayed_rank is None
                        else min(best_displayed_rank, displayed_rank)
                    )
                turn_rows[turn].append(
                    {
                        "instant": instant,
                        "cumulative": cumulative_hit,
                        "displayed_hit": displayed_rank is not None,
                    }
                )

            if turn == MAX_TURNS:
                break
            override = effective.get("behavior", {}).get("override") or {}
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                cumulative = {depth: set() for depth in DEPTHS}
                new_value = str(override.get("new_value", ""))
                if new_value:
                    disclosed.add(new_value)
                user_message = str(
                    override.get(
                        "message",
                        "Actually, please ignore my earlier preference.",
                    )
                )
            else:
                user_message, boundary_used = customer_reply(
                    effective,
                    response.get("ask_attribute"),
                    disclosed,
                    boundary_used,
                )

        session_rows.append(
            {
                "sample_id": sample["sample_id"],
                "scenario_type": sample["scenario_type"],
                "first_candidate_turn": first_candidate_turn,
                "first_displayed_turn": first_displayed_turn,
                "best_displayed_rank": best_displayed_rank,
            }
        )

    per_turn = []
    for turn in range(1, MAX_TURNS + 1):
        rows = turn_rows.get(turn, [])
        per_turn.append(
            {
                "turn": turn,
                "eligible_session_count": len(rows),
                "instantaneous_candidate_recall": {
                    f"recall_at_{depth}": _safe_ratio(
                        sum(row["instant"][str(depth)] for row in rows), len(rows)
                    )
                    for depth in DEPTHS
                },
                "cumulative_candidate_recall": {
                    f"recall_at_{depth}": _safe_ratio(
                        sum(row["cumulative"][str(depth)] for row in rows), len(rows)
                    )
                    for depth in DEPTHS
                },
                "displayed_hit_rate": _safe_ratio(
                    sum(row["displayed_hit"] for row in rows), len(rows)
                ),
            }
        )

    candidate_hits = [row for row in session_rows if row["first_candidate_turn"] is not None]
    displayed_hits = [row for row in session_rows if row["first_displayed_turn"] is not None]
    return {
        "schema_version": 1,
        "experiment": "E4.1 strict-front candidate and oracle funnel",
        "interpretation": {
            "role": "public-label diagnostic only",
            "continues_after_official_hit": True,
            "pre_override_turns_excluded": True,
            "production_agent_receives_hidden_labels": False,
            "product_disjoint_validation": False,
        },
        "sample_count": len(samples),
        "per_turn": per_turn,
        "oracle_inclusion": {
            "candidate_pool_hit_rate": _safe_ratio(len(candidate_hits), len(samples)),
            "displayed_hit_rate": _safe_ratio(len(displayed_hits), len(samples)),
            "mean_first_candidate_turn": round(
                sum(row["first_candidate_turn"] for row in candidate_hits)
                / len(candidate_hits),
                6,
            )
            if candidate_hits
            else None,
            "oracle_rerank_mrr_if_included": _safe_ratio(
                len(candidate_hits), len(samples)
            ),
        },
        "sessions": session_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay the public set beyond hits for E4.1 funnel diagnostics."
    )
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--disable-e4-1", action="store_true")
    args = parser.parse_args()
    config = (
        e4_fallback_config()
        if args.disable_e4_1
        else e4_1_candidate_config()
    )
    report = run_report(args.catalog, args.dataset, config=config)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
