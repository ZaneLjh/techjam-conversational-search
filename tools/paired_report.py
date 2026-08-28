from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from tools.fold_report import METRIC_NAMES, summarize


MISS_TURN = 11
TOP_K = 10
UTILITY_EPSILON = 1e-12
UTILITY_FORMULA = (
    "0 for a miss; otherwise 0.50 + 0.30 / rank + "
    "0.20 * (11 - first_hit_turn) / 10"
)


def load_result(path: str | Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"result must be a JSON object: {path}")
    return value


def _session_map(result: dict, label: str) -> dict[str, dict]:
    sessions = result.get("sessions")
    if not isinstance(sessions, list) or not sessions:
        raise ValueError(f"{label} must contain a non-empty sessions list")

    mapped: dict[str, dict] = {}
    for session in sessions:
        if not isinstance(session, dict) or session.get("sample_id") in (None, ""):
            raise ValueError(f"{label} contains a session without a sample_id")
        sample_id = str(session["sample_id"])
        if sample_id in mapped:
            raise ValueError(f"{label} contains duplicate sample_id {sample_id}")
        mapped[sample_id] = session
    return mapped


def _outcome(session: dict) -> dict:
    hit = bool(session.get("hit"))
    reciprocal_rank = float(session.get("reciprocal_rank", 0.0))
    if hit:
        turn = session.get("first_hit_turn")
        rank = session.get("best_rank")
        if not isinstance(turn, int) or isinstance(turn, bool) or not 1 <= turn <= 10:
            raise ValueError("a hit must have first_hit_turn between 1 and 10")
        if not isinstance(rank, int) or isinstance(rank, bool) or not 1 <= rank <= TOP_K:
            raise ValueError("a hit must have best_rank between 1 and 10")
        expected_reciprocal_rank = 1.0 / rank
        if abs(reciprocal_rank - expected_reciprocal_rank) > 1e-9:
            raise ValueError("reciprocal_rank is inconsistent with best_rank")
        effective_turn = turn
    else:
        if abs(reciprocal_rank) > 1e-12:
            raise ValueError("a miss must have reciprocal_rank equal to zero")
        turn = None
        rank = None
        effective_turn = MISS_TURN

    utility = (
        0.0
        if not hit
        else 0.50
        + 0.30 * reciprocal_rank
        + 0.20 * (MISS_TURN - effective_turn) / 10.0
    )
    return {
        "hit": hit,
        "first_hit_turn": turn,
        "effective_turn": effective_turn,
        "best_rank": rank,
        "reciprocal_rank": reciprocal_rank,
        "utility": utility,
    }


def session_utility(session: dict) -> float:
    """Return this evaluator session's additive TechnicalScore contribution."""

    return float(_outcome(session)["utility"])


def _rounded(value: float) -> float:
    return round(float(value), 9)


def _row(sample_id: str, baseline: dict, candidate: dict) -> dict:
    baseline_outcome = _outcome(baseline)
    candidate_outcome = _outcome(candidate)
    return {
        "sample_id": sample_id,
        "scenario_type": str(baseline["scenario_type"]),
        "baseline": {
            "hit": baseline_outcome["hit"],
            "first_hit_turn": baseline_outcome["first_hit_turn"],
            "best_rank": baseline_outcome["best_rank"],
            "utility": _rounded(baseline_outcome["utility"]),
        },
        "candidate": {
            "hit": candidate_outcome["hit"],
            "first_hit_turn": candidate_outcome["first_hit_turn"],
            "best_rank": candidate_outcome["best_rank"],
            "utility": _rounded(candidate_outcome["utility"]),
        },
        "delta_candidate_minus_baseline": {
            "utility": _rounded(
                candidate_outcome["utility"] - baseline_outcome["utility"]
            ),
            "reciprocal_rank": _rounded(
                candidate_outcome["reciprocal_rank"]
                - baseline_outcome["reciprocal_rank"]
            ),
            "effective_first_hit_turn": (
                candidate_outcome["effective_turn"]
                - baseline_outcome["effective_turn"]
            ),
        },
    }


def compare_results(
    baseline: dict,
    candidate: dict,
    *,
    top_n: int = 10,
    include_session_deltas: bool = True,
    include_ties: bool = True,
) -> dict:
    """Build a deterministic paired comparison over matching evaluator sessions."""

    if top_n < 0:
        raise ValueError("top_n must be non-negative")
    baseline_by_id = _session_map(baseline, "baseline")
    candidate_by_id = _session_map(candidate, "candidate")
    if baseline_by_id.keys() != candidate_by_id.keys():
        raise ValueError("baseline and candidate must contain the same session IDs")

    rows: list[dict] = []
    counts = {"improved": 0, "regressed": 0, "tied": 0}
    transitions = {
        "miss_to_hit": 0,
        "hit_to_miss": 0,
        "hit_to_hit": 0,
        "miss_to_miss": 0,
    }
    scenario_rows: dict[str, list[dict]] = defaultdict(list)
    for sample_id in sorted(baseline_by_id):
        baseline_session = baseline_by_id[sample_id]
        candidate_session = candidate_by_id[sample_id]
        baseline_scenario = str(baseline_session.get("scenario_type", ""))
        candidate_scenario = str(candidate_session.get("scenario_type", ""))
        if not baseline_scenario or baseline_scenario != candidate_scenario:
            raise ValueError(f"scenario mismatch for {sample_id}")

        row = _row(sample_id, baseline_session, candidate_session)
        delta = row["delta_candidate_minus_baseline"]["utility"]
        if delta > UTILITY_EPSILON:
            classification = "improved"
        elif delta < -UTILITY_EPSILON:
            classification = "regressed"
        else:
            classification = "tied"
        row["classification"] = classification
        counts[classification] += 1

        baseline_hit = bool(row["baseline"]["hit"])
        candidate_hit = bool(row["candidate"]["hit"])
        transition = (
            "hit_to_hit"
            if baseline_hit and candidate_hit
            else "hit_to_miss"
            if baseline_hit
            else "miss_to_hit"
            if candidate_hit
            else "miss_to_miss"
        )
        transitions[transition] += 1
        rows.append(row)
        scenario_rows[baseline_scenario].append(row)

    baseline_summary = summarize(list(baseline_by_id.values()))
    candidate_summary = summarize(list(candidate_by_id.values()))
    metric_delta = {
        metric: round(candidate_summary[metric] - baseline_summary[metric], 6)
        for metric in METRIC_NAMES
    }
    utility_deltas = [
        row["delta_candidate_minus_baseline"]["utility"] for row in rows
    ]
    scenario_summary = {}
    for scenario, values in sorted(scenario_rows.items()):
        scenario_counts = {
            name: sum(row["classification"] == name for row in values)
            for name in ("improved", "regressed", "tied")
        }
        scenario_summary[scenario] = {
            "sample_count": len(values),
            "counts": scenario_counts,
            "mean_utility_delta": _rounded(
                statistics.fmean(
                    row["delta_candidate_minus_baseline"]["utility"]
                    for row in values
                )
            ),
        }

    improvements = sorted(
        (row for row in rows if row["classification"] == "improved"),
        key=lambda row: (
            -row["delta_candidate_minus_baseline"]["utility"],
            row["sample_id"],
        ),
    )[:top_n]
    regressions = sorted(
        (row for row in rows if row["classification"] == "regressed"),
        key=lambda row: (
            row["delta_candidate_minus_baseline"]["utility"],
            row["sample_id"],
        ),
    )[:top_n]
    report = {
        "sample_count": len(rows),
        "utility_formula": UTILITY_FORMULA,
        "delta_direction": (
            "candidate minus baseline; negative effective_first_hit_turn is earlier"
        ),
        "metrics": {
            "baseline": baseline_summary,
            "candidate": candidate_summary,
            "delta_candidate_minus_baseline": metric_delta,
        },
        "paired": {
            "counts": counts,
            "hit_transitions": transitions,
            "sum_utility_delta": _rounded(sum(utility_deltas)),
            "mean_utility_delta": _rounded(statistics.fmean(utility_deltas)),
            "scenario": scenario_summary,
        },
        "top_improvements": improvements,
        "top_regressions": regressions,
    }
    if include_session_deltas:
        report["session_deltas"] = [
            row for row in rows if include_ties or row["classification"] != "tied"
        ]
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two evaluator JSON files with paired session utility."
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output")
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Maximum improvements and regressions to include in each top list.",
    )
    parser.add_argument(
        "--changes-only",
        action="store_true",
        help="Omit tied rows from the per-session delta list.",
    )
    args = parser.parse_args()

    report = compare_results(
        load_result(args.baseline),
        load_result(args.candidate),
        top_n=args.top,
        include_ties=not args.changes_only,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
