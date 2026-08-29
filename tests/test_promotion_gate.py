from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from tools.promotion_gate import evaluate_gate, render_report


def _session(
    sample_id: str,
    *,
    hit: bool = True,
    turn: int | None = 3,
    rank: int | None = 5,
) -> dict:
    return {
        "sample_id": sample_id,
        "scenario_type": "buying",
        "hit": hit,
        "first_hit_turn": turn if hit else None,
        "best_rank": rank if hit else None,
        "reciprocal_rank": 1.0 / rank if hit and rank is not None else 0.0,
    }


def _summary(sessions: list[dict]) -> dict:
    count = len(sessions)
    hit_rate = sum(bool(row["hit"]) for row in sessions) / count
    mrr = sum(float(row["reciprocal_rank"]) for row in sessions) / count
    mttc = sum(
        int(row["first_hit_turn"]) if row["first_hit_turn"] is not None else 11
        for row in sessions
    ) / count
    efficiency = max(0.0, min(1.0, (11.0 - mttc) / 10.0))
    score = 0.50 * hit_rate + 0.30 * mrr + 0.20 * efficiency
    return {
        "sample_count": count,
        "hit_rate_at_10": round(hit_rate, 6),
        "mrr": round(mrr, 6),
        "mttc": round(mttc, 6),
        "efficiency": round(efficiency, 6),
        "recommended_technical_score": round(score, 6),
    }


def _result(sessions: list[dict]) -> dict:
    return {**_summary(sessions), "sessions": sessions}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _fold_report(
    baseline: dict,
    candidate: dict,
    baseline_sha256: str,
    candidate_sha256: str,
    *,
    role: str,
) -> dict:
    if len(baseline["sessions"]) != 5 or len(candidate["sessions"]) != 5:
        raise AssertionError("test fixtures require exactly five sessions")
    held_out = role == "target_product_disjoint_held_out"
    folds = []
    for index, (baseline_row, candidate_row) in enumerate(
        zip(baseline["sessions"], candidate["sessions"]),
        start=1,
    ):
        baseline_summary = _summary([baseline_row])
        candidate_summary = _summary([candidate_row])
        folds.append(
            {
                "fold": index,
                "baseline": baseline_summary,
                "candidate": candidate_summary,
                "delta": {
                    "recommended_technical_score": round(
                        candidate_summary["recommended_technical_score"]
                        - baseline_summary["recommended_technical_score"],
                        6,
                    )
                },
            }
        )
    return {
        "fold_count": 5,
        "folds": folds,
        "provenance": {
            "evaluation_role": role,
            "split_unit": "target_product" if held_out else "public_session",
            "held_out": held_out,
            "target_product_disjoint": held_out,
            "baseline_sha256": baseline_sha256,
            "candidate_sha256": candidate_sha256,
        },
    }


class PromotionGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.baseline_sessions = [
            _session(f"S{index}", turn=3, rank=5) for index in range(1, 6)
        ]
        self.candidate_sessions = [
            _session(f"S{index}", turn=2, rank=2) for index in range(1, 6)
        ]

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _case(
        self,
        *,
        role: str = "public_consistency",
        candidate_sessions: list[dict] | None = None,
        include_optional: bool = True,
        compliance_passed: bool = True,
    ) -> tuple[dict, dict[str, Path]]:
        baseline = _result(self.baseline_sessions)
        candidate = _result(candidate_sessions or self.candidate_sessions)
        paths = {
            "baseline": self.root / "baseline.json",
            "candidate": self.root / "candidate.json",
            "repeat": self.root / "candidate_repeat.json",
            "fallback": self.root / "fallback.json",
            "compliance": self.root / "compliance.json",
            "folds": self.root / "folds.json",
        }
        _write_json(paths["baseline"], baseline)
        _write_json(paths["candidate"], candidate)
        folds = _fold_report(
            baseline,
            candidate,
            _sha256(paths["baseline"]),
            _sha256(paths["candidate"]),
            role=role,
        )
        _write_json(paths["folds"], folds)

        kwargs = {}
        if include_optional:
            paths["repeat"].write_bytes(paths["candidate"].read_bytes())
            paths["fallback"].write_bytes(paths["baseline"].read_bytes())
            _write_json(paths["compliance"], {"passed": compliance_passed})
            kwargs = {
                "candidate_repeat_path": paths["repeat"],
                "compliance_report_path": paths["compliance"],
                "fallback_result_path": paths["fallback"],
            }

        report = evaluate_gate(
            paths["baseline"],
            paths["candidate"],
            paths["folds"],
            min_delta=Decimal("0.005"),
            **kwargs,
        )
        return report, paths

    def test_public_evidence_passes_public_gate_but_cannot_promote(self) -> None:
        report, _ = self._case()

        self.assertTrue(report["public_gate_passed"])
        self.assertFalse(report["held_out_gate_passed"])
        self.assertFalse(report["deployment_promoted"])
        self.assertEqual(
            report["fold_provenance"]["classification"],
            "public_consistency",
        )
        self.assertTrue(all(row["passed"] for row in report["checks"].values()))

    def test_true_product_disjoint_held_out_evidence_can_promote(self) -> None:
        report, _ = self._case(role="target_product_disjoint_held_out")

        self.assertFalse(report["public_gate_passed"])
        self.assertTrue(report["held_out_gate_passed"])
        self.assertTrue(report["deployment_promoted"])

    def test_missing_optional_evidence_is_reported_as_gate_failure(self) -> None:
        report, _ = self._case(include_optional=False)

        self.assertFalse(report["public_gate_passed"])
        self.assertFalse(report["deployment_promoted"])
        for name in (
            "deterministic_repeat_hash_equality",
            "compliance_passed",
            "fallback_byte_equality",
        ):
            self.assertFalse(report["checks"][name]["provided"])
            self.assertFalse(report["checks"][name]["passed"])

    def test_each_optional_evidence_failure_blocks_the_gate(self) -> None:
        report, paths = self._case(compliance_passed=False)
        self.assertFalse(report["checks"]["compliance_passed"]["passed"])

        paths["repeat"].write_text("not the candidate\n", encoding="utf-8")
        paths["fallback"].write_bytes(paths["candidate"].read_bytes())
        report = evaluate_gate(
            paths["baseline"],
            paths["candidate"],
            paths["folds"],
            candidate_repeat_path=paths["repeat"],
            compliance_report_path=paths["compliance"],
            fallback_result_path=paths["fallback"],
        )

        self.assertFalse(
            report["checks"]["deterministic_repeat_hash_equality"]["passed"]
        )
        self.assertFalse(report["checks"]["fallback_byte_equality"]["passed"])
        self.assertFalse(report["public_gate_passed"])

    def test_hit_to_miss_blocks_gate_even_when_hit_rate_is_unchanged(self) -> None:
        candidate_sessions = [
            _session("S1", hit=False, turn=None, rank=None),
            _session("S2", turn=1, rank=1),
            _session("S3", turn=1, rank=1),
            _session("S4", turn=1, rank=1),
            _session("S5", turn=1, rank=1),
        ]
        baseline_sessions = list(self.baseline_sessions)
        baseline_sessions[-1] = _session("S5", hit=False, turn=None, rank=None)
        self.baseline_sessions = baseline_sessions

        report, _ = self._case(candidate_sessions=candidate_sessions)

        self.assertTrue(report["checks"]["hit_rate_nondecreasing"]["passed"])
        self.assertEqual(report["checks"]["zero_hit_to_miss"]["observed"], 1)
        self.assertFalse(report["checks"]["zero_hit_to_miss"]["passed"])
        self.assertFalse(report["public_gate_passed"])

    def test_score_threshold_and_hit_rate_regression_each_fail(self) -> None:
        unchanged_report, _ = self._case(
            candidate_sessions=list(self.baseline_sessions)
        )
        self.assertFalse(
            unchanged_report["checks"]["minimum_technical_score_delta"]["passed"]
        )
        self.assertEqual(
            unchanged_report["checks"]["minimum_technical_score_delta"][
                "required_minimum"
            ],
            0.005,
        )

        regressed_sessions = list(self.candidate_sessions)
        regressed_sessions[0] = _session("S1", hit=False, turn=None, rank=None)
        regressed_report, _ = self._case(candidate_sessions=regressed_sessions)
        self.assertFalse(
            regressed_report["checks"]["hit_rate_nondecreasing"]["passed"]
        )
        self.assertFalse(regressed_report["public_gate_passed"])

    def test_only_three_positive_folds_fails_four_of_five_requirement(self) -> None:
        candidate_sessions = [
            _session("S1", turn=1, rank=1),
            _session("S2", turn=1, rank=1),
            _session("S3", turn=1, rank=1),
            _session("S4", turn=3, rank=5),
            _session("S5", turn=3, rank=5),
        ]
        report, _ = self._case(candidate_sessions=candidate_sessions)

        check = report["checks"]["positive_folds"]
        self.assertEqual(check["observed"], 3)
        self.assertFalse(check["passed"])
        self.assertFalse(report["public_gate_passed"])

    def test_fold_hash_mismatch_and_missing_provenance_fail_closed(self) -> None:
        _, paths = self._case()
        folds = json.loads(paths["folds"].read_text(encoding="utf-8"))
        folds["provenance"]["candidate_sha256"] = "0" * 64
        _write_json(paths["folds"], folds)

        report = evaluate_gate(
            paths["baseline"],
            paths["candidate"],
            paths["folds"],
            candidate_repeat_path=paths["repeat"],
            compliance_report_path=paths["compliance"],
            fallback_result_path=paths["fallback"],
        )
        self.assertFalse(report["checks"]["fold_input_hashes_match"]["passed"])
        self.assertFalse(report["deployment_promoted"])

        folds.pop("provenance")
        _write_json(paths["folds"], folds)
        report = evaluate_gate(
            paths["baseline"],
            paths["candidate"],
            paths["folds"],
            candidate_repeat_path=paths["repeat"],
            compliance_report_path=paths["compliance"],
            fallback_result_path=paths["fallback"],
        )
        self.assertEqual(report["fold_provenance"]["classification"], "unverified")
        self.assertFalse(report["checks"]["fold_provenance_valid"]["passed"])
        self.assertFalse(report["public_gate_passed"])

    def test_rendered_output_is_sorted_and_deterministic(self) -> None:
        report, _ = self._case()

        first = render_report(report)
        second = render_report(report)

        self.assertEqual(first, second)
        self.assertTrue(first.endswith("\n"))
        self.assertLess(first.index('"checks"'), first.index('"schema_version"'))


if __name__ == "__main__":
    unittest.main()
