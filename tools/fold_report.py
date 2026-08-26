from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


METRIC_NAMES = (
    "hit_rate_at_10",
    "mrr",
    "mttc",
    "efficiency",
    "recommended_technical_score",
)


def load_result(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def summarize(sessions: list[dict]) -> dict:
    count = len(sessions)
    hit_rate = round(sum(bool(item["hit"]) for item in sessions) / count, 6)
    mrr = round(
        statistics.fmean(float(item["reciprocal_rank"]) for item in sessions),
        6,
    )
    mttc = round(
        statistics.fmean(
            item["first_hit_turn"] if item["first_hit_turn"] is not None else 11
            for item in sessions
        ),
        6,
    )
    efficiency = max(0.0, min(1.0, (11.0 - mttc) / 10.0))
    technical_score = 0.50 * hit_rate + 0.30 * mrr + 0.20 * efficiency
    return {
        "sample_count": count,
        "hit_rate_at_10": hit_rate,
        "mrr": mrr,
        "mttc": mttc,
        "efficiency": round(efficiency, 6),
        "recommended_technical_score": round(technical_score, 6),
    }


def stratified_assignments(sessions: list[dict], folds: int) -> dict[str, int]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for session in sessions:
        grouped[str(session["scenario_type"])].append(session)

    assignments: dict[str, int] = {}
    for scenario_sessions in grouped.values():
        for index, session in enumerate(
            sorted(scenario_sessions, key=lambda item: str(item["sample_id"]))
        ):
            assignments[str(session["sample_id"])] = index % folds
    return assignments


def compare_folds(baseline: dict, candidate: dict, folds: int) -> dict:
    baseline_by_id = {
        str(item["sample_id"]): item for item in baseline.get("sessions", [])
    }
    candidate_by_id = {
        str(item["sample_id"]): item for item in candidate.get("sessions", [])
    }
    if not baseline_by_id or baseline_by_id.keys() != candidate_by_id.keys():
        raise ValueError("baseline and candidate must contain the same non-empty session IDs")

    for sample_id, baseline_session in baseline_by_id.items():
        if baseline_session["scenario_type"] != candidate_by_id[sample_id]["scenario_type"]:
            raise ValueError(f"scenario mismatch for {sample_id}")

    assignments = stratified_assignments(list(baseline_by_id.values()), folds)
    fold_rows: list[dict] = []
    for fold in range(folds):
        sample_ids = sorted(
            sample_id for sample_id, assigned in assignments.items() if assigned == fold
        )
        baseline_summary = summarize([baseline_by_id[item] for item in sample_ids])
        candidate_summary = summarize([candidate_by_id[item] for item in sample_ids])
        scenario_counts: dict[str, int] = defaultdict(int)
        for sample_id in sample_ids:
            scenario_counts[str(candidate_by_id[sample_id]["scenario_type"])] += 1
        fold_rows.append(
            {
                "fold": fold + 1,
                "scenario_counts": dict(sorted(scenario_counts.items())),
                "baseline": baseline_summary,
                "candidate": candidate_summary,
                "delta": {
                    metric: round(candidate_summary[metric] - baseline_summary[metric], 6)
                    for metric in METRIC_NAMES
                },
            }
        )

    return {
        "fold_count": folds,
        "assignment": "scenario-stratified round-robin by sample_id; evaluation only, no fitting",
        "folds": fold_rows,
        "mean_delta": {
            metric: round(statistics.fmean(row["delta"][metric] for row in fold_rows), 6)
            for metric in METRIC_NAMES
        },
        "all_public": {
            "baseline": summarize(list(baseline_by_id.values())),
            "candidate": summarize(list(candidate_by_id.values())),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two evaluator result files over deterministic scenario-stratified folds."
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")

    report = compare_folds(
        load_result(args.baseline),
        load_result(args.candidate),
        args.folds,
    )
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
