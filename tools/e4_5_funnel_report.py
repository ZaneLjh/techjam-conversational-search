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
from starter.projection import ProjectionConfig
from starter.retrieval import e4_fallback_config


DEPTHS = (10, 20, 50, 100)


def _safe_ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _safe_mean(values: list[int]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def run_report(
    catalog_path: str | Path,
    dataset_path: str | Path,
    sidecar_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    *,
    config: ProjectionConfig | None = None,
) -> dict:
    """Replay all public turns and report candidate/projection funnel diagnostics.

    This deliberately continues after an official hit. Ground-truth labels are
    used only after each response to compute diagnostics and are never supplied
    to the production Agent, its projection index, or its question policy.
    """

    if config is None:
        if sidecar_path is None or manifest_path is None:
            raise ValueError("sidecar_path and manifest_path are required")
        config = ProjectionConfig(
            enabled=True,
            sidecar_path=str(sidecar_path),
            manifest_path=str(manifest_path),
        )

    samples = load_jsonl(dataset_path)
    catalog_ids, categories, products = catalog_index(catalog_path)
    agent = Agent(
        catalog_path,
        retrieval_config=e4_fallback_config(),
        projection_config=config,
    )
    if config.enabled and not agent.projection_index.ready:
        raise ValueError(
            "projection sidecar did not validate: "
            + agent.projection_index.status_reason
        )

    turn_rows: dict[int, list[dict]] = defaultdict(list)
    session_rows: list[dict] = []

    for sample in samples:
        session_id = f"e4_5_funnel_{sample['sample_id']}"
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
        first_projection_active_turn: int | None = None
        projection_active_turn_count = 0
        projection_ranking_turn_count = 0
        projection_question_turn_count = 0
        posterior_target_turn_count = 0

        for turn in range(1, MAX_TURNS + 1):
            response = agent.respond(session_id, user_message, turn, TOP_K)
            ranked = normalize_recommendations(
                response.get("recommendations"),
                catalog_ids,
            )
            trace = agent.evidence_trace(session_id)
            retrieval_rows = trace.get("retrieval_decisions", [])
            retrieval_trace = retrieval_rows[-1] if retrieval_rows else {}
            projection_rows = trace.get("projection_decisions", [])
            projection_trace = projection_rows[-1] if projection_rows else {}
            question_rows = trace.get("projection_question_decisions", [])
            question_trace = question_rows[-1] if question_rows else {}

            raw_candidates = projection_trace.get(
                "candidate_ids",
                retrieval_trace.get("candidate_ids", ranked),
            )
            candidate_ids = list(
                dict.fromkeys(
                    str(value)
                    for value in raw_candidates
                    if str(value) in catalog_ids
                )
            )[:100]
            posterior_ids = list(
                dict.fromkeys(
                    str(value)
                    for value in projection_trace.get("posterior_ids", ())
                    if str(value) in catalog_ids
                )
            )[:100]
            projection_active = bool(projection_trace.get("active"))
            ranking_applied = bool(projection_trace.get("ranking_applied"))
            question_active = bool(question_trace.get("active"))

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
                if projection_active:
                    projection_active_turn_count += 1
                    if first_projection_active_turn is None:
                        first_projection_active_turn = turn
                if ranking_applied:
                    projection_ranking_turn_count += 1
                if question_active:
                    projection_question_turn_count += 1
                if target in posterior_ids:
                    posterior_target_turn_count += 1

                turn_rows[turn].append(
                    {
                        "instant": instant,
                        "cumulative": cumulative_hit,
                        "displayed_hit": displayed_rank is not None,
                        "projection_active": projection_active,
                        "projection_ranking_applied": ranking_applied,
                        "projection_question_active": question_active,
                        "posterior_target_hit": target in posterior_ids,
                        "posterior_size": len(posterior_ids),
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
                "first_projection_active_turn": first_projection_active_turn,
                "projection_active_turn_count": projection_active_turn_count,
                "projection_ranking_turn_count": projection_ranking_turn_count,
                "projection_question_turn_count": projection_question_turn_count,
                "posterior_target_turn_count": posterior_target_turn_count,
            }
        )

    per_turn = []
    for turn in range(1, MAX_TURNS + 1):
        rows = turn_rows.get(turn, [])
        posterior_sizes = [
            int(row["posterior_size"])
            for row in rows
            if row["projection_active"]
        ]
        per_turn.append(
            {
                "turn": turn,
                "eligible_session_count": len(rows),
                "instantaneous_candidate_recall": {
                    f"recall_at_{depth}": _safe_ratio(
                        sum(row["instant"][str(depth)] for row in rows),
                        len(rows),
                    )
                    for depth in DEPTHS
                },
                "cumulative_candidate_recall": {
                    f"recall_at_{depth}": _safe_ratio(
                        sum(row["cumulative"][str(depth)] for row in rows),
                        len(rows),
                    )
                    for depth in DEPTHS
                },
                "displayed_hit_rate": _safe_ratio(
                    sum(row["displayed_hit"] for row in rows),
                    len(rows),
                ),
                "projection_activation_rate": _safe_ratio(
                    sum(row["projection_active"] for row in rows),
                    len(rows),
                ),
                "projection_ranking_rate": _safe_ratio(
                    sum(row["projection_ranking_applied"] for row in rows),
                    len(rows),
                ),
                "projection_question_rate": _safe_ratio(
                    sum(row["projection_question_active"] for row in rows),
                    len(rows),
                ),
                "posterior_target_inclusion_rate": _safe_ratio(
                    sum(row["posterior_target_hit"] for row in rows),
                    len(rows),
                ),
                "mean_active_posterior_size": _safe_mean(posterior_sizes),
            }
        )

    candidate_hits = [
        row for row in session_rows if row["first_candidate_turn"] is not None
    ]
    displayed_hits = [
        row for row in session_rows if row["first_displayed_turn"] is not None
    ]
    activated = [
        row for row in session_rows if row["first_projection_active_turn"] is not None
    ]
    posterior_hits = [
        row for row in session_rows if row["posterior_target_turn_count"] > 0
    ]
    return {
        "schema_version": 1,
        "experiment": "E4.5 projection and exact-rollout funnel",
        "interpretation": {
            "role": "public-label diagnostic only",
            "continues_after_official_hit": True,
            "pre_override_turns_excluded": True,
            "production_agent_receives_hidden_labels": False,
            "product_disjoint_validation": False,
            "predecessor": "frozen_e4",
        },
        "projection_artifact": {
            "enabled": config.enabled,
            "ready": agent.projection_index.ready,
            "status_reason": agent.projection_index.status_reason,
        },
        "sample_count": len(samples),
        "per_turn": per_turn,
        # Keep the E4.1 funnel keys so public diagnostic consumers can compare
        # the two experiment reports without special-case parsing.
        "oracle_inclusion": {
            "candidate_pool_hit_rate": _safe_ratio(len(candidate_hits), len(samples)),
            "displayed_hit_rate": _safe_ratio(len(displayed_hits), len(samples)),
            "mean_first_candidate_turn": _safe_mean(
                [int(row["first_candidate_turn"]) for row in candidate_hits]
            ),
            "oracle_rerank_mrr_if_included": _safe_ratio(
                len(candidate_hits),
                len(samples),
            ),
        },
        "projection_funnel": {
            "session_activation_rate": _safe_ratio(len(activated), len(samples)),
            "posterior_target_hit_rate": _safe_ratio(
                len(posterior_hits),
                len(samples),
            ),
            "mean_first_active_turn": _safe_mean(
                [int(row["first_projection_active_turn"]) for row in activated]
            ),
            "total_active_turns": sum(
                int(row["projection_active_turn_count"]) for row in session_rows
            ),
            "total_reranked_turns": sum(
                int(row["projection_ranking_turn_count"]) for row in session_rows
            ),
            "total_projected_question_turns": sum(
                int(row["projection_question_turn_count"]) for row in session_rows
            ),
        },
        "sessions": session_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay the public set beyond hits for E4.5 funnel diagnostics."
    )
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument(
        "--sidecar",
        "--projection-sidecar",
        dest="sidecar",
        required=True,
    )
    parser.add_argument(
        "--manifest",
        "--projection-manifest",
        dest="manifest",
        required=True,
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--disable-e4-5", action="store_true")
    args = parser.parse_args()
    config = ProjectionConfig(
        enabled=not args.disable_e4_5,
        sidecar_path=args.sidecar,
        manifest_path=args.manifest,
    )
    report = run_report(
        args.catalog,
        args.dataset,
        args.sidecar,
        args.manifest,
        config=config,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
