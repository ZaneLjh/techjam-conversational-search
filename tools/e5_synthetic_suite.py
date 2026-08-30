from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

from evaluator.local_evaluator import (
    MAX_TURNS,
    TOP_K,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    materialize_hidden_fields,
    normalize_recommendations,
)
from starter.agent import Agent
from starter.projection import ProjectionConfig
from starter.question_policy import QuestionPolicyConfig
from starter.reranking import RerankingConfig
from starter.retrieval import e4_fallback_config


DEPTHS = (10, 20, 50, 100)
EXPECTED_SCENARIO_PERCENT = {
    "buying": 40,
    "browsing": 40,
    "intent_override": 15,
    "boundary": 5,
}
DETERMINISTIC_THRESHOLD = 0.005
LEARNED_THRESHOLD = 0.010
REQUIRED_PROMOTION_SEEDS = 3
REQUIRED_PROMOTION_FOLDS = 5
REQUIRED_PROMOTION_SESSIONS_PER_CELL = 200
REQUIRED_PROMOTION_SESSIONS = (
    REQUIRED_PROMOTION_SEEDS
    * REQUIRED_PROMOTION_FOLDS
    * REQUIRED_PROMOTION_SESSIONS_PER_CELL
)


@dataclass(frozen=True)
class VariantSpec:
    name: str
    description: str
    projection_enabled: bool
    projection_rollout: bool
    semantic_enabled: bool
    quality_enabled: bool
    learned_or_fitted: bool = False


CORE_VARIANTS = (
    VariantSpec(
        "frozen_e4",
        "Frozen E4 behavioral fallback.",
        False,
        False,
        False,
        False,
    ),
    VariantSpec(
        "projection_unique_only",
        "Exact E4.5 unique-posterior rescue; question rollout frozen off.",
        True,
        False,
        False,
        False,
    ),
    VariantSpec(
        "semantic_only_no_quality",
        "Display-membership-preserving semantic reranking on frozen E4.",
        False,
        False,
        True,
        False,
    ),
    VariantSpec(
        "guarded_hybrid",
        "Unique-posterior projection followed by semantic reranking of unlocked display positions.",
        True,
        False,
        True,
        False,
    ),
)
FULL_ONLY_VARIANTS = (
    VariantSpec(
        "guarded_hybrid_quality_on",
        "Guarded hybrid with the optional catalog-quality tie-break enabled.",
        True,
        False,
        True,
        True,
    ),
    VariantSpec(
        "guarded_hybrid_projection_rollout",
        "Guarded hybrid with E4.5 question rollout enabled.",
        True,
        True,
        True,
        False,
    ),
)


def _round(value: float) -> float:
    return round(float(value), 6)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_jsonl(path: str | Path) -> list[dict]:
    source = Path(path)
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rt", encoding="utf-8", newline="") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("synthetic dataset rows must be JSON objects")
    return rows


def _recursive_flag(value: object, key: str) -> object | None:
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for child in value.values():
            found = _recursive_flag(child, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _recursive_flag(child, key)
            if found is not None:
                return found
    return None


def _sample_partition_value(sample: Mapping[str, object], name: str) -> object | None:
    if name in sample:
        return sample[name]
    metadata = sample.get("synthetic_metadata")
    if not isinstance(metadata, dict):
        return None
    nested_name = "group_id" if name == "target_group_id" else name
    return metadata.get(nested_name)


def _sample_strata(sample: Mapping[str, object]) -> dict[str, bool]:
    direct = sample.get("synthetic_strata")
    metadata = sample.get("synthetic_metadata")
    nested = metadata.get("strata") if isinstance(metadata, dict) else None
    value = direct if isinstance(direct, dict) else nested
    if not isinstance(value, dict):
        return {}
    return {str(name): bool(enabled) for name, enabled in sorted(value.items())}


def validate_synthetic_dataset(samples: Sequence[dict], manifest: Mapping[str, object]) -> dict:
    """Reject synthetic evidence that is not truly group-disjoint and quarantined."""

    if not samples:
        raise ValueError("synthetic dataset is empty")
    if _recursive_flag(manifest, "group_disjoint_folds") is not True:
        raise ValueError("manifest must assert group_disjoint_folds=true")
    if _recursive_flag(manifest, "public_target_groups_quarantined") is not True:
        raise ValueError(
            "manifest must assert public_target_groups_quarantined=true"
        )
    if _recursive_flag(manifest, "cross_seed_disjoint") is not True:
        raise ValueError("manifest must assert cross_seed_disjoint=true")

    required = {
        "sample_id",
        "scenario_type",
        "user_profile",
        "ground_truth",
    }
    group_folds: dict[str, set[str]] = defaultdict(set)
    group_seeds: dict[str, set[str]] = defaultdict(set)
    partition_scenarios: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    partition_ids: set[tuple[str, str, str]] = set()
    target_ids: set[str] = set()
    for index, sample in enumerate(samples, start=1):
        missing = sorted(required - sample.keys())
        if missing:
            raise ValueError(f"synthetic row {index} is missing: {', '.join(missing)}")
        truth = sample.get("ground_truth")
        if not isinstance(truth, dict) or not str(truth.get("parent_asin", "")).strip():
            raise ValueError(f"synthetic row {index} has invalid ground_truth")
        partition_values = {
            name: _sample_partition_value(sample, name)
            for name in ("seed", "fold", "target_group_id")
        }
        missing_metadata = [
            name for name, value in partition_values.items() if value is None
        ]
        if missing_metadata:
            raise ValueError(
                f"synthetic row {index} is missing metadata: "
                + ", ".join(missing_metadata)
            )
        seed = str(partition_values["seed"])
        fold = str(partition_values["fold"])
        sample_id = str(sample["sample_id"])
        group_id = str(partition_values["target_group_id"])
        key = (seed, fold, sample_id)
        if key in partition_ids:
            raise ValueError(f"duplicate synthetic sample key: {key}")
        partition_ids.add(key)
        group_folds[group_id].add(fold)
        group_seeds[group_id].add(seed)
        partition_scenarios[(seed, fold)][str(sample["scenario_type"])] += 1
        target_ids.add(str(truth["parent_asin"]))

    leaking_groups = sorted(group for group, folds in group_folds.items() if len(folds) > 1)
    if leaking_groups:
        raise ValueError(
            "target groups cross folds: " + ", ".join(leaking_groups[:10])
        )
    repeated_seed_groups = sorted(
        group for group, group_seed_values in group_seeds.items()
        if len(group_seed_values) > 1
    )
    if repeated_seed_groups:
        raise ValueError(
            "target groups cross seeds: " + ", ".join(repeated_seed_groups[:10])
        )

    seeds = sorted({seed for seed, _ in partition_scenarios})
    folds = sorted({fold for _, fold in partition_scenarios})
    if len(seeds) < 3:
        raise ValueError("synthetic promotion evidence requires at least three seeds")
    if len(folds) != 5:
        raise ValueError("synthetic promotion evidence requires exactly five folds")
    claims = {
        "session_count": len(samples),
        "seed_count": len(seeds),
        "fold_count": len(folds),
    }
    for name, actual in claims.items():
        if _recursive_flag(manifest, name) != actual:
            raise ValueError(f"manifest {name} does not match the dataset")
    declared_seeds = _recursive_flag(manifest, "seeds")
    if not isinstance(declared_seeds, list) or sorted(map(str, declared_seeds)) != seeds:
        raise ValueError("manifest seeds do not match the dataset")
    scenario_percentages = _recursive_flag(manifest, "scenario_percentages")
    if scenario_percentages != EXPECTED_SCENARIO_PERCENT:
        raise ValueError("manifest scenario_percentages must be exactly 40/40/15/5")
    expected_partitions = {(seed, fold) for seed in seeds for fold in folds}
    if set(partition_scenarios) != expected_partitions:
        raise ValueError("every seed must contain every fold")
    for partition, counts in sorted(partition_scenarios.items()):
        if set(counts) != set(EXPECTED_SCENARIO_PERCENT):
            raise ValueError(f"partition {partition} has the wrong scenario strata")
        total = sum(counts.values())
        for scenario, percent in EXPECTED_SCENARIO_PERCENT.items():
            if counts[scenario] * 100 != total * percent:
                raise ValueError(
                    f"partition {partition} does not have exact 40/40/15/5 proportions"
                )

    sessions_per_cell = {
        partition: sum(counts.values())
        for partition, counts in partition_scenarios.items()
    }
    declared_sessions_per_cell = _recursive_flag(
        manifest, "sessions_per_seed_fold"
    )
    if declared_sessions_per_cell is None:
        raise ValueError("manifest sessions_per_seed_fold is required")
    if any(
        value != declared_sessions_per_cell
        for value in sessions_per_cell.values()
    ):
        raise ValueError(
            "manifest sessions_per_seed_fold does not match every seed/fold cell"
        )
    promotion_corpus_size_passed = (
        len(samples) == REQUIRED_PROMOTION_SESSIONS
        and len(seeds) == REQUIRED_PROMOTION_SEEDS
        and len(folds) == REQUIRED_PROMOTION_FOLDS
        and declared_sessions_per_cell == REQUIRED_PROMOTION_SESSIONS_PER_CELL
    )

    public_ids = _recursive_flag(manifest, "public_target_ids")
    if isinstance(public_ids, list):
        collisions = target_ids & {str(value) for value in public_ids}
        if collisions:
            raise ValueError("synthetic targets overlap quarantined public targets")
    public_groups = _recursive_flag(manifest, "public_target_group_ids")
    if isinstance(public_groups, list):
        collisions = set(group_folds) & {str(value) for value in public_groups}
        if collisions:
            raise ValueError("synthetic groups overlap quarantined public target groups")

    return {
        "validated": True,
        "sample_count": len(samples),
        "seed_count": len(seeds),
        "seeds": seeds,
        "fold_count": len(folds),
        "folds": folds,
        "target_group_count": len(group_folds),
        "scenario_proportions": EXPECTED_SCENARIO_PERCENT,
        "group_disjoint_folds": True,
        "public_target_groups_quarantined": True,
        "cross_seed_disjoint": True,
        "sessions_per_seed_fold": declared_sessions_per_cell,
        "promotion_corpus_size_passed": promotion_corpus_size_passed,
        "evidence_role": (
            "full_product_disjoint_promotion"
            if promotion_corpus_size_passed
            else "smoke_non_promotion"
        ),
    }


def _session_efficiency(row: Mapping[str, object]) -> float:
    turn = row.get("first_hit_turn")
    mttc_value = MAX_TURNS + 1 if turn is None else int(turn)
    return max(0.0, min(1.0, (11.0 - mttc_value) / 10.0))


def _technical_contribution(row: Mapping[str, object]) -> float:
    return (
        0.50 * int(bool(row.get("hit")))
        + 0.30 * float(row.get("reciprocal_rank", 0.0))
        + 0.20 * _session_efficiency(row)
    )


def _raw_technical_score(rows: Sequence[dict]) -> float:
    return (
        sum(_technical_contribution(row) for row in rows) / len(rows)
        if rows
        else 0.0
    )


def summarize_sessions(rows: Sequence[dict]) -> dict:
    if not rows:
        return {
            "sample_count": 0,
            "hit_rate_at_10": 0.0,
            "mrr": 0.0,
            "mttc": None,
            "efficiency": 0.0,
            "recommended_technical_score": 0.0,
            "retrieval": {
                "observation_window": "variant_until_first_hit_or_turn_10",
                "hit_censored": True,
                "cross_variant_comparable": False,
                "oracle_funnel": False,
                **{
                    f"observed_recall_at_{depth}_before_stop": 0.0
                    for depth in DEPTHS
                },
                "observed_conditional_mrr_at_100_before_stop": 0.0,
                "observed_retrieved_target_count_at_100_before_stop": 0,
                "observed_mean_earliest_pool_turn_before_stop": None,
                "mean_earliest_display_turn": None,
                "observed_recall_100_to_hit_gap": 0.0,
            },
        }
    count = len(rows)
    hit_rate = sum(bool(row["hit"]) for row in rows) / count
    mrr = sum(float(row["reciprocal_rank"]) for row in rows) / count
    mttc = sum(
        MAX_TURNS + 1 if row["first_hit_turn"] is None else int(row["first_hit_turn"])
        for row in rows
    ) / count
    efficiency = max(0.0, min(1.0, (11.0 - mttc) / 10.0))
    score = 0.50 * hit_rate + 0.30 * mrr + 0.20 * efficiency
    retrieved = [row for row in rows if row.get("best_pool_rank") is not None]
    pool_turns = [int(row["earliest_pool_turn"]) for row in retrieved]
    display_turns = [
        int(row["earliest_display_turn"])
        for row in rows
        if row.get("earliest_display_turn") is not None
    ]
    retrieval = {
        f"observed_recall_at_{depth}_before_stop": _round(
            sum(bool(row.get(f"retrieved_at_{depth}")) for row in rows) / count
        )
        for depth in DEPTHS
    }
    retrieval.update(
        {
            "observation_window": "variant_until_first_hit_or_turn_10",
            "hit_censored": True,
            "cross_variant_comparable": False,
            "oracle_funnel": False,
            "observed_conditional_mrr_at_100_before_stop": _round(
                sum(1.0 / int(row["best_pool_rank"]) for row in retrieved)
                / len(retrieved)
            )
            if retrieved
            else 0.0,
            "observed_retrieved_target_count_at_100_before_stop": len(retrieved),
            "observed_mean_earliest_pool_turn_before_stop": _round(
                sum(pool_turns) / len(pool_turns)
            )
            if pool_turns
            else None,
            "mean_earliest_display_turn": _round(
                sum(display_turns) / len(display_turns)
            )
            if display_turns
            else None,
            "observed_recall_100_to_hit_gap": _round(
                sum(bool(row.get("retrieved_at_100")) for row in rows) / count
                - hit_rate
            ),
        }
    )
    return {
        "sample_count": count,
        "hit_rate_at_10": _round(hit_rate),
        "mrr": _round(mrr),
        "mttc": _round(mttc),
        "efficiency": _round(efficiency),
        "recommended_technical_score": _round(score),
        "retrieval": retrieval,
    }


def _group_metrics(rows: Sequence[dict], field: str) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row[field])].append(row)
    return {key: summarize_sessions(grouped[key]) for key in sorted(grouped)}


def _stratum_metrics(rows: Sequence[dict]) -> dict[str, dict]:
    names = sorted(
        {
            str(name)
            for row in rows
            for name, active in row.get("strata", {}).items()
            if active
        }
    )
    return {
        name: summarize_sessions(
            [row for row in rows if bool(row.get("strata", {}).get(name))]
        )
        for name in names
    }


def _paired_counts(baseline: Sequence[dict], candidate: Sequence[dict]) -> dict:
    baseline_by_key = {tuple(row["pair_key"]): row for row in baseline}
    candidate_by_key = {tuple(row["pair_key"]): row for row in candidate}
    if baseline_by_key.keys() != candidate_by_key.keys():
        raise ValueError("variant session keys do not match the frozen E4 baseline")
    counts = Counter()
    strata: dict[str, Counter[str]] = defaultdict(Counter)
    synthetic_strata: dict[str, Counter[str]] = defaultdict(Counter)
    for key in sorted(baseline_by_key):
        before = baseline_by_key[key]
        after = candidate_by_key[key]
        if before["hit"] and after["hit"]:
            transition = "hit_to_hit"
        elif before["hit"]:
            transition = "hit_to_miss"
        elif after["hit"]:
            transition = "miss_to_hit"
        else:
            transition = "miss_to_miss"
        counts[transition] += 1
        delta = _technical_contribution(after) - _technical_contribution(before)
        direction = "improved" if delta > 1e-12 else "regressed" if delta < -1e-12 else "tied"
        counts[direction] += 1
        strata[str(before["scenario_type"])][transition] += 1
        strata[str(before["scenario_type"])][direction] += 1
        for name, active in before.get("strata", {}).items():
            if active:
                synthetic_strata[str(name)][transition] += 1
                synthetic_strata[str(name)][direction] += 1
    return {
        **{name: counts[name] for name in (
            "hit_to_hit", "hit_to_miss", "miss_to_hit", "miss_to_miss",
            "improved", "regressed", "tied",
        )},
        "by_scenario": {
            scenario: dict(sorted(values.items()))
            for scenario, values in sorted(strata.items())
        },
        "by_synthetic_stratum": {
            name: dict(sorted(values.items()))
            for name, values in sorted(synthetic_strata.items())
        },
    }


def compare_to_baseline(
    baseline: Sequence[dict],
    candidate: Sequence[dict],
    *,
    learned_or_fitted: bool,
    full_promotion_corpus_size: bool | None = None,
) -> dict:
    base_overall = summarize_sessions(baseline)
    candidate_overall = summarize_sessions(candidate)
    threshold = LEARNED_THRESHOLD if learned_or_fitted else DETERMINISTIC_THRESHOLD
    raw_score_delta = _raw_technical_score(candidate) - _raw_technical_score(baseline)
    raw_hr_delta = (
        sum(bool(row["hit"]) for row in candidate) / len(candidate)
        - sum(bool(row["hit"]) for row in baseline) / len(baseline)
    )
    fold_deltas: dict[str, float] = {}
    seed_deltas: dict[str, float] = {}
    raw_fold_deltas: dict[str, float] = {}
    raw_seed_deltas: dict[str, float] = {}
    for field, target, raw_target in (
        ("fold", fold_deltas, raw_fold_deltas),
        ("seed", seed_deltas, raw_seed_deltas),
    ):
        keys = sorted({str(row[field]) for row in baseline})
        for key in keys:
            base_rows = [row for row in baseline if str(row[field]) == key]
            candidate_rows = [row for row in candidate if str(row[field]) == key]
            raw_target[key] = (
                _raw_technical_score(candidate_rows) - _raw_technical_score(base_rows)
            )
            target[key] = _round(raw_target[key])
    transitions = _paired_counts(baseline, candidate)
    score_delta = _round(raw_score_delta)
    hr_delta = _round(raw_hr_delta)
    positive_folds = sum(value > 0.0 for value in raw_fold_deltas.values())
    seed_count = len(seed_deltas)
    scenario_hr_deltas: dict[str, float] = {}
    scenario_hr_checks: dict[str, bool] = {}
    for scenario in ("intent_override", "boundary"):
        base_rows = [row for row in baseline if row["scenario_type"] == scenario]
        candidate_rows = [row for row in candidate if row["scenario_type"] == scenario]
        raw_scenario_delta = (
            sum(bool(row["hit"]) for row in candidate_rows) / len(candidate_rows)
            - sum(bool(row["hit"]) for row in base_rows) / len(base_rows)
            if base_rows and candidate_rows
            else 0.0
        )
        scenario_hr_deltas[scenario] = _round(raw_scenario_delta)
        scenario_hr_checks[scenario] = (
            bool(base_rows) and bool(candidate_rows) and raw_scenario_delta >= -1e-12
        )
    error_count = sum(
        int(row.get("agent_error_count", 0))
        + int(row.get("invalid_response_count", 0))
        for row in candidate
    )
    if full_promotion_corpus_size is None:
        cell_counts = Counter(
            (str(row["seed"]), str(row["fold"])) for row in baseline
        )
        full_promotion_corpus_size = (
            len(baseline) == REQUIRED_PROMOTION_SESSIONS
            and len({seed for seed, _ in cell_counts}) == REQUIRED_PROMOTION_SEEDS
            and len({fold for _, fold in cell_counts}) == REQUIRED_PROMOTION_FOLDS
            and len(cell_counts)
            == REQUIRED_PROMOTION_SEEDS * REQUIRED_PROMOTION_FOLDS
            and all(
                count == REQUIRED_PROMOTION_SESSIONS_PER_CELL
                for count in cell_counts.values()
            )
        )
    checks = {
        "full_promotion_corpus_size": full_promotion_corpus_size,
        "technical_score_threshold": raw_score_delta >= threshold - 1e-12,
        "hit_rate_non_decreasing": raw_hr_delta >= -1e-12,
        "positive_group_disjoint_folds": positive_folds >= 4,
        "zero_hit_to_miss": transitions["hit_to_miss"] == 0,
        "three_seed_stability_reported": seed_count >= 3,
        "all_seed_deltas_nonnegative": all(
            value >= -1e-12 for value in raw_seed_deltas.values()
        ),
        "override_and_boundary_hit_rate_non_decreasing": all(
            scenario_hr_checks.values()
        ),
        "zero_runtime_or_response_errors": error_count == 0,
    }
    return {
        "technical_score_delta": score_delta,
        "hit_rate_at_10_delta": hr_delta,
        "mrr_delta": _round(candidate_overall["mrr"] - base_overall["mrr"]),
        "mttc_delta": _round(candidate_overall["mttc"] - base_overall["mttc"]),
        "candidate_inclusion_comparison": {
            "available": False,
            "reason": (
                "variant-specific first-hit censoring; run a separate fixed-dialogue "
                "all-turn replay for an unbiased comparison"
            ),
        },
        "fold_technical_score_deltas": fold_deltas,
        "positive_fold_count": positive_folds,
        "seed_technical_score_deltas": seed_deltas,
        "scenario_hit_rate_deltas": scenario_hr_deltas,
        "runtime_or_response_error_count": error_count,
        "paired_transitions": transitions,
        "promotion_gate": {
            "eligible": all(checks.values()),
            "threshold": threshold,
            "variant_class": "learned_or_fitted" if learned_or_fitted else "deterministic",
            "checks": checks,
        },
    }


def _safe_session_component(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))[:80]


def _extract_candidate_ids(trace: Mapping[str, object], catalog_ids: set[str]) -> list[str]:
    retrieval_rows = trace.get("retrieval_decisions", [])
    retrieval = retrieval_rows[-1] if isinstance(retrieval_rows, list) and retrieval_rows else {}
    raw = retrieval.get("candidate_ids", []) if isinstance(retrieval, dict) else []
    if not isinstance(raw, (list, tuple)):
        return []
    return list(
        dict.fromkeys(str(value) for value in raw if str(value) in catalog_ids)
    )[:100]


def _evaluate_variant(
    agent: Agent,
    variant: VariantSpec,
    samples: Sequence[dict],
    catalog_ids: set[str],
    categories: Mapping[str, list[str]],
    products: Mapping[str, dict],
) -> list[dict]:
    rows: list[dict] = []
    for sample in samples:
        pair_key = (
            str(_sample_partition_value(sample, "seed")),
            str(_sample_partition_value(sample, "fold")),
            str(sample["sample_id"]),
        )
        session_id = "e5_" + "_".join(
            _safe_session_component(value) for value in (variant.name, *pair_key)
        )
        agent.reset(session_id, sample["user_profile"])
        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(sample, products)  # type: ignore[arg-type]
        effective = {**sample, "intent_card": card, "behavior": behavior}
        disclosed: set[str] = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        user_message = initial_message(
            effective,
            coarse_category(categories.get(target, [])),
            disclosed,
        )
        first_hit_turn: int | None = None
        best_rank: int | None = None
        best_pool_rank: int | None = None
        earliest_pool_turn: int | None = None
        earliest_display_turn: int | None = None
        retrieved = {depth: False for depth in DEPTHS}
        agent_error_count = 0
        invalid_response_count = 0
        for turn in range(1, MAX_TURNS + 1):
            try:
                response = agent.respond(session_id, user_message, turn, TOP_K)
            except Exception:
                agent_error_count += 1
                response = {"message": "", "ask_attribute": None, "recommendations": []}
            if not isinstance(response, dict) or not isinstance(response.get("message"), str):
                invalid_response_count += 1
                response = {"message": "", "ask_attribute": None, "recommendations": []}
            ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
            try:
                trace = agent.evidence_trace(session_id)
            except Exception:
                trace = {}
            candidate_ids = _extract_candidate_ids(trace, catalog_ids)
            if override_applied:
                if target in candidate_ids:
                    pool_rank = candidate_ids.index(target) + 1
                    best_pool_rank = pool_rank if best_pool_rank is None else min(best_pool_rank, pool_rank)
                    if earliest_pool_turn is None:
                        earliest_pool_turn = turn
                for depth in DEPTHS:
                    retrieved[depth] = retrieved[depth] or target in candidate_ids[:depth]
                if target in ranked:
                    best_rank = ranked.index(target) + 1
                    first_hit_turn = turn
                    earliest_display_turn = turn
                    break
            if turn == MAX_TURNS:
                break
            override = effective.get("behavior", {}).get("override") or {}
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                best_pool_rank = None
                earliest_pool_turn = None
                retrieved = {depth: False for depth in DEPTHS}
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
        rows.append(
            {
                "pair_key": list(pair_key),
                "sample_id": str(sample["sample_id"]),
                "seed": str(_sample_partition_value(sample, "seed")),
                "fold": str(_sample_partition_value(sample, "fold")),
                "target_group_id": str(
                    _sample_partition_value(sample, "target_group_id")
                ),
                "scenario_type": str(sample["scenario_type"]),
                "strata": _sample_strata(sample),
                "hit": first_hit_turn is not None,
                "first_hit_turn": first_hit_turn,
                "best_rank": best_rank,
                "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank,
                "best_pool_rank": best_pool_rank,
                "earliest_pool_turn": earliest_pool_turn,
                "earliest_display_turn": earliest_display_turn,
                "agent_error_count": agent_error_count,
                "invalid_response_count": invalid_response_count,
                **{f"retrieved_at_{depth}": retrieved[depth] for depth in DEPTHS},
            }
        )
    return sorted(rows, key=lambda row: tuple(row["pair_key"]))


def _switch_variant(
    agent: Agent,
    projection_template: ProjectionConfig,
    reranking_template: RerankingConfig,
    variant: VariantSpec,
) -> None:
    projection = replace(
        projection_template,
        enabled=variant.projection_enabled,
        use_reranking=variant.projection_enabled,
        use_question_rollout=variant.projection_enabled and variant.projection_rollout,
        max_rerank_posterior_size=1,
    )
    reranking = replace(
        reranking_template,
        enabled=variant.semantic_enabled,
        use_quality_tiebreak=variant.semantic_enabled and variant.quality_enabled,
    )
    agent.projection_config = projection
    agent.projection_index.config = projection
    agent.reranking_config = reranking
    reranker = getattr(agent, "reranker", None) or getattr(agent, "semantic_reranker", None)
    if reranker is None:
        raise RuntimeError("Agent does not expose its E5 semantic reranker")
    reranker.config = reranking
    question_policy = replace(
        agent.question_policy_config,
        repeat_other_until_exhausted=variant.projection_rollout,
    )
    agent.question_policy_config = question_policy
    agent.question_policy.config = question_policy
    sessions = getattr(agent, "_sessions", None)
    if isinstance(sessions, dict):
        sessions.clear()


def run_suite(
    catalog_path: str | Path,
    dataset_path: str | Path,
    sidecar_path: str | Path,
    projection_manifest_path: str | Path,
    dataset_manifest_path: str | Path,
    *,
    profile: str = "core",
) -> dict:
    if profile not in {"promotion", "core", "full"}:
        raise ValueError("profile must be 'promotion', 'core', or 'full'")
    samples = _load_jsonl(dataset_path)
    with Path(dataset_manifest_path).open(encoding="utf-8") as handle:
        dataset_manifest = json.load(handle)
    if not isinstance(dataset_manifest, dict):
        raise ValueError("synthetic manifest must be a JSON object")
    validation = validate_synthetic_dataset(samples, dataset_manifest)
    dataset_sha256 = _sha256(dataset_path)
    expected_dataset_sha256 = _recursive_flag(dataset_manifest, "dataset_sha256")
    if expected_dataset_sha256 != dataset_sha256:
        raise ValueError("synthetic dataset sha256 does not match its manifest")
    catalog_sha256 = _sha256(catalog_path)
    expected_catalog_sha256 = _recursive_flag(dataset_manifest, "catalog_input_sha256")
    if expected_catalog_sha256 != catalog_sha256:
        raise ValueError("catalog sha256 does not match the synthetic manifest")
    validation["dataset_sha256_verified"] = True
    validation["catalog_sha256_verified"] = True
    catalog_ids, categories, products = catalog_index(catalog_path)

    projection_template = ProjectionConfig(
        enabled=True,
        sidecar_path=str(sidecar_path),
        manifest_path=str(projection_manifest_path),
        use_reranking=True,
        use_question_rollout=True,
        max_rerank_posterior_size=1,
    )
    reranking_template = RerankingConfig(
        enabled=True,
        enforce_projection_candidate_membership=True,
        candidate_depth=100,
        use_quality_tiebreak=True,
    )
    agent = Agent(
        catalog_path,
        retrieval_config=e4_fallback_config(),
        question_policy_config=QuestionPolicyConfig(
            repeat_other_until_exhausted=False,
        ),
        projection_config=projection_template,
        reranking_config=reranking_template,
    )
    if not agent.projection_index.ready:
        raise ValueError(
            "projection sidecar did not validate: "
            + agent.projection_index.status_reason
        )
    if profile == "promotion":
        variants = (CORE_VARIANTS[0], CORE_VARIANTS[3])
    else:
        variants = CORE_VARIANTS + (
            FULL_ONLY_VARIANTS if profile == "full" else ()
        )
    rows_by_variant: dict[str, list[dict]] = {}
    variant_reports: dict[str, dict] = {}
    for variant in variants:
        _switch_variant(agent, projection_template, reranking_template, variant)
        rows = _evaluate_variant(
            agent,
            variant,
            samples,
            catalog_ids,
            categories,
            products,
        )
        rows_by_variant[variant.name] = rows
        variant_reports[variant.name] = {
            "description": variant.description,
            "configuration": {
                "projection_enabled": variant.projection_enabled,
                "projection_rollout": variant.projection_rollout,
                "semantic_enabled": variant.semantic_enabled,
                "quality_enabled": variant.quality_enabled,
                "learned_or_fitted": variant.learned_or_fitted,
            },
            "overall": summarize_sessions(rows),
            "by_seed": _group_metrics(rows, "seed"),
            "by_fold_aggregated_across_seeds": _group_metrics(rows, "fold"),
            "by_scenario": _group_metrics(rows, "scenario_type"),
            "by_synthetic_stratum": _stratum_metrics(rows),
            "runtime_or_response_error_count": sum(
                int(row["agent_error_count"]) + int(row["invalid_response_count"])
                for row in rows
            ),
            "sessions": rows,
        }
    baseline = rows_by_variant["frozen_e4"]
    comparisons = {
        variant.name: compare_to_baseline(
            baseline,
            rows_by_variant[variant.name],
            learned_or_fitted=variant.learned_or_fitted,
            full_promotion_corpus_size=validation[
                "promotion_corpus_size_passed"
            ],
        )
        for variant in variants
        if variant.name != "frozen_e4"
    }
    return {
        "schema_version": 2,
        "experiment": "E5 group-disjoint synthetic guarded-hybrid suite",
        "profile": profile,
        "interpretation": {
            "predecessor": "frozen_e4",
            "exact_evaluator_protocol": True,
            "pre_override_turns_excluded": True,
            "production_agent_receives_hidden_labels": False,
            "public_data_used_for_tuning": False,
            "single_catalog_and_index_load": True,
            "retrieval_funnel_observation": (
                "hit-censored at each variant's first display hit; not cross-variant "
                "comparable and not used for promotion"
            ),
            "unbiased_retrieval_funnel_requirement": (
                "separate fixed-dialogue all-turn replay"
            ),
        },
        "provenance": {
            "catalog": {"path": str(catalog_path), "sha256": catalog_sha256},
            "dataset": {"path": str(dataset_path), "sha256": dataset_sha256},
            "dataset_manifest": {
                "path": str(dataset_manifest_path),
                "sha256": _sha256(dataset_manifest_path),
            },
            "projection_sidecar": {
                "path": str(sidecar_path),
                "sha256": _sha256(sidecar_path),
            },
            "projection_manifest": {
                "path": str(projection_manifest_path),
                "sha256": _sha256(projection_manifest_path),
            },
        },
        "dataset_validation": validation,
        "promotion_policy": {
            "deterministic_minimum_technical_score_delta": DETERMINISTIC_THRESHOLD,
            "learned_or_fitted_minimum_technical_score_delta": LEARNED_THRESHOLD,
            "hit_rate_at_10_must_not_decrease": True,
            "minimum_positive_group_disjoint_folds": 4,
            "maximum_hit_to_miss_transitions": 0,
            "minimum_seed_count": 3,
            "required_sessions": REQUIRED_PROMOTION_SESSIONS,
            "required_sessions_per_seed_fold": (
                REQUIRED_PROMOTION_SESSIONS_PER_CELL
            ),
        },
        "variants": variant_reports,
        "comparisons_to_frozen_e4": comparisons,
    }


def _default_dataset_manifest(dataset_path: str | Path) -> Path:
    dataset = Path(dataset_path)
    candidates = (
        dataset.parent / "e5_synthetic_manifest.json",
        Path(str(dataset) + ".manifest.json"),
        dataset.with_suffix(".manifest.json"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ValueError(
        "synthetic dataset manifest not found; pass --dataset-manifest"
    )


def _write_canonical_json(path: str | Path, value: object) -> None:
    rendered = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    Path(path).write_bytes(rendered.encode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the group-disjoint E5 synthetic promotion suite."
    )
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-manifest")
    parser.add_argument("--sidecar", required=True)
    parser.add_argument("--manifest", required=True, help="E4.5 projection manifest")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--profile",
        choices=("promotion", "core", "full"),
        default="promotion",
    )
    args = parser.parse_args()
    dataset_manifest = (
        Path(args.dataset_manifest)
        if args.dataset_manifest
        else _default_dataset_manifest(args.dataset)
    )
    report = run_suite(
        args.catalog,
        args.dataset,
        args.sidecar,
        args.manifest,
        dataset_manifest,
        profile=args.profile,
    )
    _write_canonical_json(args.output, report)
    summary = {
        "profile": args.profile,
        "sample_count": report["dataset_validation"]["sample_count"],
        "evidence_role": report["dataset_validation"]["evidence_role"],
        "promotion_corpus_size_passed": report["dataset_validation"][
            "promotion_corpus_size_passed"
        ],
        "variants": {
            name: {
                "technical_score": value["overall"]["recommended_technical_score"],
                "promotion_eligible": report["comparisons_to_frozen_e4"]
                .get(name, {})
                .get("promotion_gate", {})
                .get("eligible"),
            }
            for name, value in report["variants"].items()
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
