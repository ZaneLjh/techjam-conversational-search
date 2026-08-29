from __future__ import annotations

import argparse
import hashlib
import json
import math
from decimal import Decimal, InvalidOperation
from pathlib import Path

from tools.paired_report import compare_results


DEFAULT_MIN_DELTA = Decimal("0.005")
REQUIRED_FOLD_COUNT = 5
REQUIRED_POSITIVE_FOLDS = 4
SUMMARY_TOLERANCE = 1e-6
SHA256_HEX_LENGTH = 64


def _load_object(path: str | Path, label: str) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _files_byte_equal(first: str | Path, second: str | Path) -> bool:
    first_path = Path(first)
    second_path = Path(second)
    if first_path.stat().st_size != second_path.stat().st_size:
        return False
    with first_path.open("rb") as first_handle, second_path.open("rb") as second_handle:
        while True:
            first_chunk = first_handle.read(1024 * 1024)
            second_chunk = second_handle.read(1024 * 1024)
            if first_chunk != second_chunk:
                return False
            if not first_chunk:
                return True


def _threshold(value: Decimal | float | str) -> Decimal:
    try:
        parsed = value if isinstance(value, Decimal) else Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError("minimum TechnicalScore delta must be numeric") from exc
    if not parsed.is_finite() or parsed < 0:
        raise ValueError("minimum TechnicalScore delta must be finite and non-negative")
    return parsed


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    rendered = float(value)
    return rendered if math.isfinite(rendered) else None


def _validate_declared_summary(result: dict, computed: dict, label: str) -> None:
    declared_count = result.get("sample_count")
    if (
        isinstance(declared_count, bool)
        or not isinstance(declared_count, int)
        or declared_count != computed["sample_count"]
    ):
        raise ValueError(f"{label} sample_count disagrees with its sessions")

    for metric in (
        "hit_rate_at_10",
        "mrr",
        "mttc",
        "efficiency",
        "recommended_technical_score",
    ):
        declared = _number(result.get(metric))
        if declared is None or abs(declared - float(computed[metric])) > SUMMARY_TOLERANCE:
            raise ValueError(f"{label} {metric} disagrees with its sessions")


def _normalized_sha256(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if len(normalized) != SHA256_HEX_LENGTH:
        return None
    if any(character not in "0123456789abcdef" for character in normalized):
        return None
    return normalized


def _fold_provenance(report: dict) -> dict:
    raw = report.get("provenance")
    if not isinstance(raw, dict):
        return {
            "classification": "unverified",
            "evaluation_role": None,
            "split_unit": None,
            "held_out": None,
            "target_product_disjoint": None,
            "valid": False,
            "reason": "missing explicit fold-report provenance",
        }

    role = raw.get("evaluation_role")
    split_unit = raw.get("split_unit")
    held_out = raw.get("held_out")
    product_disjoint = raw.get("target_product_disjoint")
    public_valid = (
        role == "public_consistency"
        and split_unit == "public_session"
        and held_out is False
        and product_disjoint is False
    )
    held_out_valid = (
        role == "target_product_disjoint_held_out"
        and split_unit == "target_product"
        and held_out is True
        and product_disjoint is True
    )
    if public_valid:
        classification = "public_consistency"
        reason = "explicitly declared public consistency folds"
    elif held_out_valid:
        classification = "target_product_disjoint_held_out"
        reason = "explicitly declared target-product-disjoint held-out folds"
    else:
        classification = "unverified"
        reason = "fold provenance is incomplete, contradictory, or unsupported"
    return {
        "classification": classification,
        "evaluation_role": role if isinstance(role, str) else None,
        "split_unit": split_unit if isinstance(split_unit, str) else None,
        "held_out": held_out if isinstance(held_out, bool) else None,
        "target_product_disjoint": (
            product_disjoint if isinstance(product_disjoint, bool) else None
        ),
        "valid": public_valid or held_out_valid,
        "reason": reason,
    }


def _fold_checks(
    report: dict,
    *,
    baseline_sha256: str,
    candidate_sha256: str,
) -> tuple[dict, dict]:
    provenance = _fold_provenance(report)
    raw_provenance = report.get("provenance")
    raw_provenance = raw_provenance if isinstance(raw_provenance, dict) else {}
    linked_baseline = _normalized_sha256(raw_provenance.get("baseline_sha256"))
    linked_candidate = _normalized_sha256(raw_provenance.get("candidate_sha256"))
    hashes_match = (
        linked_baseline == baseline_sha256 and linked_candidate == candidate_sha256
    )

    folds = report.get("folds")
    declared_fold_count = report.get("fold_count")
    fold_count_valid = (
        isinstance(declared_fold_count, int)
        and not isinstance(declared_fold_count, bool)
        and declared_fold_count == REQUIRED_FOLD_COUNT
        and isinstance(folds, list)
        and len(folds) == REQUIRED_FOLD_COUNT
    )

    positive_folds = 0
    fold_schema_valid = isinstance(folds, list)
    fold_identifiers: set[int] = set()
    if isinstance(folds, list):
        for row in folds:
            if not isinstance(row, dict):
                fold_schema_valid = False
                continue
            fold_identifier = row.get("fold")
            if (
                isinstance(fold_identifier, bool)
                or not isinstance(fold_identifier, int)
                or fold_identifier in fold_identifiers
            ):
                fold_schema_valid = False
            else:
                fold_identifiers.add(fold_identifier)
            delta = row.get("delta")
            score_delta = (
                _number(delta.get("recommended_technical_score"))
                if isinstance(delta, dict)
                else None
            )
            if score_delta is None:
                fold_schema_valid = False
            elif score_delta > 0:
                positive_folds += 1
    positive_folds_passed = (
        fold_count_valid
        and fold_schema_valid
        and positive_folds >= REQUIRED_POSITIVE_FOLDS
    )

    checks = {
        "fold_provenance_valid": {
            "passed": bool(provenance["valid"]),
            "observed": provenance["classification"],
            "required": (
                "public_consistency or target_product_disjoint_held_out with "
                "internally consistent explicit provenance"
            ),
        },
        "fold_input_hashes_match": {
            "passed": hashes_match,
            "observed": {
                "baseline_sha256": linked_baseline,
                "candidate_sha256": linked_candidate,
            },
            "required": {
                "baseline_sha256": baseline_sha256,
                "candidate_sha256": candidate_sha256,
            },
        },
        "fold_count": {
            "passed": fold_count_valid and fold_schema_valid,
            "observed": {
                "declared": declared_fold_count,
                "rows": len(folds) if isinstance(folds, list) else None,
                "unique_identifiers": len(fold_identifiers),
            },
            "required": REQUIRED_FOLD_COUNT,
        },
        "positive_folds": {
            "passed": positive_folds_passed,
            "observed": positive_folds,
            "required": REQUIRED_POSITIVE_FOLDS,
            "strictly_positive_metric": "recommended_technical_score",
        },
    }
    return provenance, checks


def _repeat_check(candidate_path: Path, repeat_path: str | Path | None) -> dict:
    if repeat_path is None:
        return {
            "provided": False,
            "passed": False,
            "reason": "candidate repeat result was not supplied",
            "candidate_sha256": _sha256(candidate_path),
            "repeat_sha256": None,
        }
    repeat = Path(repeat_path)
    if not repeat.is_file():
        return {
            "provided": True,
            "passed": False,
            "reason": "candidate repeat result does not exist",
            "candidate_sha256": _sha256(candidate_path),
            "repeat_sha256": None,
        }
    candidate_sha256 = _sha256(candidate_path)
    repeat_sha256 = _sha256(repeat)
    return {
        "provided": True,
        "passed": candidate_sha256 == repeat_sha256,
        "reason": (
            "SHA-256 hashes are equal"
            if candidate_sha256 == repeat_sha256
            else "SHA-256 hashes differ"
        ),
        "candidate_sha256": candidate_sha256,
        "repeat_sha256": repeat_sha256,
    }


def _compliance_check(report_path: str | Path | None) -> dict:
    if report_path is None:
        return {
            "provided": False,
            "passed": False,
            "reason": "compliance report was not supplied",
            "report_sha256": None,
        }
    path = Path(report_path)
    if not path.is_file():
        return {
            "provided": True,
            "passed": False,
            "reason": "compliance report does not exist",
            "report_sha256": None,
        }
    report_sha256 = _sha256(path)
    try:
        report = _load_object(path, "compliance report")
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return {
            "provided": True,
            "passed": False,
            "reason": f"invalid compliance report: {type(exc).__name__}",
            "report_sha256": report_sha256,
        }
    passed = report.get("passed")
    if not isinstance(passed, bool):
        return {
            "provided": True,
            "passed": False,
            "reason": "compliance report must contain top-level boolean passed",
            "report_sha256": report_sha256,
        }
    return {
        "provided": True,
        "passed": passed,
        "reason": "compliance report passed" if passed else "compliance report failed",
        "report_sha256": report_sha256,
    }


def _fallback_check(
    baseline_path: Path,
    fallback_path: str | Path | None,
) -> dict:
    baseline_sha256 = _sha256(baseline_path)
    if fallback_path is None:
        return {
            "provided": False,
            "passed": False,
            "reason": "fallback result was not supplied",
            "baseline_sha256": baseline_sha256,
            "fallback_sha256": None,
        }
    fallback = Path(fallback_path)
    if not fallback.is_file():
        return {
            "provided": True,
            "passed": False,
            "reason": "fallback result does not exist",
            "baseline_sha256": baseline_sha256,
            "fallback_sha256": None,
        }
    fallback_sha256 = _sha256(fallback)
    equal = _files_byte_equal(baseline_path, fallback)
    return {
        "provided": True,
        "passed": equal,
        "reason": "files are byte-equal" if equal else "files differ",
        "baseline_sha256": baseline_sha256,
        "fallback_sha256": fallback_sha256,
    }


def evaluate_gate(
    baseline_path: str | Path,
    candidate_path: str | Path,
    fold_report_path: str | Path,
    *,
    min_delta: Decimal | float | str = DEFAULT_MIN_DELTA,
    candidate_repeat_path: str | Path | None = None,
    compliance_report_path: str | Path | None = None,
    fallback_result_path: str | Path | None = None,
) -> dict:
    """Evaluate deterministic evidence without treating public folds as held-out."""

    baseline_file = Path(baseline_path)
    candidate_file = Path(candidate_path)
    fold_file = Path(fold_report_path)
    threshold = _threshold(min_delta)
    baseline = _load_object(baseline_file, "baseline evaluator result")
    candidate = _load_object(candidate_file, "candidate evaluator result")
    fold_report = _load_object(fold_file, "fold report")

    paired_report = compare_results(
        baseline,
        candidate,
        top_n=0,
        include_session_deltas=False,
    )
    baseline_summary = paired_report["metrics"]["baseline"]
    candidate_summary = paired_report["metrics"]["candidate"]
    metric_delta = paired_report["metrics"]["delta_candidate_minus_baseline"]
    _validate_declared_summary(baseline, baseline_summary, "baseline")
    _validate_declared_summary(candidate, candidate_summary, "candidate")

    baseline_sha256 = _sha256(baseline_file)
    candidate_sha256 = _sha256(candidate_file)
    provenance, fold_checks = _fold_checks(
        fold_report,
        baseline_sha256=baseline_sha256,
        candidate_sha256=candidate_sha256,
    )

    score_delta = Decimal(str(metric_delta["recommended_technical_score"]))
    baseline_hit_rate = float(baseline_summary["hit_rate_at_10"])
    candidate_hit_rate = float(candidate_summary["hit_rate_at_10"])
    hit_to_miss = int(paired_report["paired"]["hit_transitions"]["hit_to_miss"])
    checks = {
        "minimum_technical_score_delta": {
            "passed": score_delta >= threshold,
            "observed": float(score_delta),
            "required_minimum": float(threshold),
        },
        "hit_rate_nondecreasing": {
            "passed": candidate_hit_rate >= baseline_hit_rate,
            "observed": {
                "baseline": baseline_hit_rate,
                "candidate": candidate_hit_rate,
                "delta": round(candidate_hit_rate - baseline_hit_rate, 6),
            },
            "required": "candidate >= baseline",
        },
        "zero_hit_to_miss": {
            "passed": hit_to_miss == 0,
            "observed": hit_to_miss,
            "required": 0,
        },
        **fold_checks,
        "deterministic_repeat_hash_equality": _repeat_check(
            candidate_file,
            candidate_repeat_path,
        ),
        "compliance_passed": _compliance_check(compliance_report_path),
        "fallback_byte_equality": _fallback_check(
            baseline_file,
            fallback_result_path,
        ),
    }
    performance_and_safety_gate_passed = all(
        bool(check["passed"]) for check in checks.values()
    )
    public_gate_passed = (
        performance_and_safety_gate_passed
        and provenance["classification"] == "public_consistency"
    )
    held_out_gate_passed = (
        performance_and_safety_gate_passed
        and provenance["classification"] == "target_product_disjoint_held_out"
    )
    deployment_promoted = held_out_gate_passed
    failed_checks = sorted(
        name for name, check in checks.items() if not bool(check["passed"])
    )
    if deployment_promoted:
        decision_reason = "true target-product-disjoint held-out gate passed"
    elif public_gate_passed:
        decision_reason = (
            "public consistency gate passed; true target-product-disjoint "
            "held-out evidence is still required for deployment promotion"
        )
    else:
        decision_reason = "one or more required promotion checks failed"

    return {
        "schema_version": 1,
        "gate": "deterministic_promotion_gate",
        "thresholds": {
            "minimum_technical_score_delta": float(threshold),
            "required_positive_folds": REQUIRED_POSITIVE_FOLDS,
            "required_fold_count": REQUIRED_FOLD_COUNT,
        },
        "input_sha256": {
            "baseline": baseline_sha256,
            "candidate": candidate_sha256,
            "fold_report": _sha256(fold_file),
        },
        "metrics": {
            "baseline": baseline_summary,
            "candidate": candidate_summary,
            "delta_candidate_minus_baseline": metric_delta,
        },
        "paired_hit_transitions": paired_report["paired"]["hit_transitions"],
        "fold_provenance": provenance,
        "checks": checks,
        "failed_checks": failed_checks,
        "performance_and_safety_gate_passed": performance_and_safety_gate_passed,
        "public_gate_passed": public_gate_passed,
        "held_out_gate_passed": held_out_gate_passed,
        "deployment_promoted": deployment_promoted,
        "decision_reason": decision_reason,
    }


def render_report(report: dict) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the deterministic promotion gate while keeping public "
            "consistency evidence separate from held-out deployment evidence."
        )
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--fold-report", required=True)
    parser.add_argument("--candidate-repeat")
    parser.add_argument("--compliance-report")
    parser.add_argument("--fallback-result")
    parser.add_argument("--min-delta", default=str(DEFAULT_MIN_DELTA))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = evaluate_gate(
        args.baseline,
        args.candidate,
        args.fold_report,
        min_delta=args.min_delta,
        candidate_repeat_path=args.candidate_repeat,
        compliance_report_path=args.compliance_report,
        fallback_result_path=args.fallback_result,
    )
    rendered = render_report(report)
    Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
