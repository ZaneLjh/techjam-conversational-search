from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.e5_synthetic_corpus import generate
from tools.e5_synthetic_suite import (
    CORE_VARIANTS,
    _write_canonical_json,
    compare_to_baseline,
    summarize_sessions,
    validate_synthetic_dataset,
)


def _synthetic_rows() -> list[dict]:
    scenarios = ["buying"] * 8 + ["browsing"] * 8 + ["intent_override"] * 3 + ["boundary"]
    rows: list[dict] = []
    for seed in (1701, 1702, 1703):
        for fold in range(5):
            for index, scenario in enumerate(scenarios):
                rows.append(
                    {
                        "sample_id": f"s{seed}_f{fold}_{index}",
                        "scenario_type": scenario,
                        "user_profile": {},
                        "ground_truth": {"parent_asin": f"P{fold}_{index}"},
                        "seed": seed,
                        "fold": fold,
                        "target_group_id": f"g{seed}_{fold}_{index}",
                    }
                )
    return rows


def _manifest() -> dict:
    return {
        "cross_seed_disjoint": True,
        "fold_count": 5,
        "group_disjoint_folds": True,
        "public_target_groups_quarantined": True,
        "scenario_percentages": {
            "buying": 40,
            "browsing": 40,
            "intent_override": 15,
            "boundary": 5,
        },
        "seed_count": 3,
        "seeds": [1701, 1702, 1703],
        "session_count": 300,
        "sessions_per_seed_fold": 20,
    }


def _metric_row(
    seed: int,
    fold: int,
    sample_id: str,
    *,
    hit: bool,
    rank: int | None,
    turn: int | None,
    scenario: str = "buying",
) -> dict:
    return {
        "pair_key": [str(seed), str(fold), sample_id],
        "sample_id": sample_id,
        "seed": str(seed),
        "fold": str(fold),
        "target_group_id": f"g{fold}_{sample_id}",
        "scenario_type": scenario,
        "hit": hit,
        "first_hit_turn": turn,
        "best_rank": rank,
        "reciprocal_rank": 0.0 if rank is None else 1.0 / rank,
        "best_pool_rank": rank,
        "earliest_pool_turn": turn,
        "earliest_display_turn": turn,
        "retrieved_at_10": rank is not None and rank <= 10,
        "retrieved_at_20": rank is not None and rank <= 20,
        "retrieved_at_50": rank is not None and rank <= 50,
        "retrieved_at_100": rank is not None and rank <= 100,
    }


class DatasetValidationTests(unittest.TestCase):
    def test_accepts_three_seed_five_fold_exact_strata(self) -> None:
        result = validate_synthetic_dataset(
            _synthetic_rows(),
            _manifest(),
        )
        self.assertTrue(result["validated"])
        self.assertEqual(result["sample_count"], 300)
        self.assertEqual(result["seed_count"], 3)
        self.assertEqual(result["fold_count"], 5)
        self.assertEqual(result["sessions_per_seed_fold"], 20)
        self.assertFalse(result["promotion_corpus_size_passed"])
        self.assertEqual(result["evidence_role"], "smoke_non_promotion")

    def test_accepts_generator_nested_metadata_schema(self) -> None:
        rows = _synthetic_rows()
        for row in rows:
            row["synthetic_metadata"] = {
                "seed": row.pop("seed"),
                "fold": row.pop("fold"),
                "group_id": row.pop("target_group_id"),
            }
        result = validate_synthetic_dataset(
            rows,
            _manifest(),
        )
        self.assertTrue(result["validated"])
        self.assertEqual(result["target_group_count"], 300)

    def test_rejects_group_crossing_folds(self) -> None:
        rows = _synthetic_rows()
        rows[20]["target_group_id"] = rows[0]["target_group_id"]
        with self.assertRaisesRegex(ValueError, "target groups cross folds"):
            validate_synthetic_dataset(
                rows,
                _manifest(),
            )

    def test_rejects_missing_quarantine_attestation(self) -> None:
        with self.assertRaisesRegex(ValueError, "public_target_groups_quarantined"):
            validate_synthetic_dataset(
                _synthetic_rows(),
                {**_manifest(), "public_target_groups_quarantined": False},
            )

    def test_rejects_inexact_scenario_mix(self) -> None:
        rows = _synthetic_rows()
        rows[0]["scenario_type"] = "boundary"
        with self.assertRaisesRegex(ValueError, "40/40/15/5"):
            validate_synthetic_dataset(
                rows,
                _manifest(),
            )

    def test_rejects_manifest_cell_size_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "sessions_per_seed_fold"):
            validate_synthetic_dataset(
                _synthetic_rows(),
                {**_manifest(), "sessions_per_seed_fold": 200},
            )

    def test_generator_output_validates_without_schema_translation(self) -> None:
        def word(index: int) -> str:
            letters = []
            value = index
            while True:
                letters.append(chr(ord("a") + value % 26))
                value //= 26
                if not value:
                    return "".join(reversed(letters))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "catalog.jsonl"
            products = []
            for index in range(301):
                marker = word(index)
                products.append(
                    {
                        "parent_asin": "PUBLIC" if index == 0 else f"P{index:04d}",
                        "title": f"Artifact {marker} jacket",
                        "features": [f"marker {marker}", "cotton shell"],
                        "details": {"Model": f"model-{marker}"},
                        "categories": ["Clothing", "Jackets"],
                        "store": f"brand-{marker}",
                    }
                )
            catalog.write_bytes(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in products).encode()
            )
            public = root / "public.jsonl"
            public.write_bytes(
                (json.dumps({"ground_truth": {"parent_asin": "PUBLIC"}}) + "\n").encode()
            )
            output = root / "generated"
            manifest = generate(
                catalog,
                public,
                output,
                folds=5,
                seeds=(1701, 1702, 1703),
                sessions_per_fold=20,
            )
            samples = [
                json.loads(line)
                for line in (output / "e5_synthetic_sessions.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line
            ]
            validation = validate_synthetic_dataset(samples, manifest)
            self.assertEqual(validation["sample_count"], 300)
            self.assertTrue(validation["cross_seed_disjoint"])


class MetricTests(unittest.TestCase):
    def test_summary_matches_official_score_and_funnel_math(self) -> None:
        rows = [
            {
                **_metric_row(1, 0, "a", hit=True, rank=2, turn=1),
                "best_pool_rank": 5,
            },
            {
                **_metric_row(1, 0, "b", hit=False, rank=None, turn=None),
                "best_pool_rank": 20,
                "earliest_pool_turn": 4,
                "retrieved_at_20": True,
                "retrieved_at_50": True,
                "retrieved_at_100": True,
            },
        ]
        summary = summarize_sessions(rows)
        self.assertEqual(summary["hit_rate_at_10"], 0.5)
        self.assertEqual(summary["mrr"], 0.25)
        self.assertEqual(summary["mttc"], 6.0)
        self.assertEqual(summary["recommended_technical_score"], 0.425)
        retrieval = summary["retrieval"]
        self.assertTrue(retrieval["hit_censored"])
        self.assertFalse(retrieval["cross_variant_comparable"])
        self.assertFalse(retrieval["oracle_funnel"])
        self.assertEqual(retrieval["observed_recall_at_10_before_stop"], 0.5)
        self.assertEqual(retrieval["observed_recall_at_100_before_stop"], 1.0)
        self.assertEqual(
            retrieval["observed_conditional_mrr_at_100_before_stop"], 0.125
        )
        self.assertEqual(
            retrieval["observed_mean_earliest_pool_turn_before_stop"], 2.5
        )
        self.assertEqual(retrieval["observed_recall_100_to_hit_gap"], 0.5)

    def test_gate_requires_threshold_folds_seeds_and_no_hit_to_miss(self) -> None:
        baseline = []
        candidate = []
        for seed in (1, 2, 3):
            for fold in range(5):
                for index in range(200):
                    sample_id = f"s{seed}f{fold}i{index}"
                    scenario = (
                        "boundary" if index < 10
                        else "intent_override" if index < 40
                        else "buying"
                    )
                    baseline.append(
                        _metric_row(
                            seed, fold, sample_id, hit=True, rank=10, turn=5,
                            scenario=scenario,
                        )
                    )
                    candidate.append(
                        _metric_row(
                            seed, fold, sample_id, hit=True, rank=1, turn=1,
                            scenario=scenario,
                        )
                    )
        comparison = compare_to_baseline(
            baseline,
            candidate,
            learned_or_fitted=False,
        )
        self.assertTrue(comparison["promotion_gate"]["eligible"])
        self.assertEqual(comparison["positive_fold_count"], 5)
        self.assertEqual(comparison["paired_transitions"]["hit_to_miss"], 0)
        self.assertEqual(len(comparison["seed_technical_score_deltas"]), 3)
        self.assertFalse(comparison["candidate_inclusion_comparison"]["available"])

    def test_gate_rejects_smoke_sized_evidence_even_when_metrics_pass(self) -> None:
        baseline = []
        candidate = []
        for seed in (1, 2, 3):
            for fold in range(5):
                for index in range(20):
                    sample_id = f"s{seed}f{fold}i{index}"
                    scenario = (
                        "boundary" if index == 0
                        else "intent_override" if index < 4
                        else "buying"
                    )
                    baseline.append(
                        _metric_row(
                            seed, fold, sample_id, hit=True, rank=10, turn=5,
                            scenario=scenario,
                        )
                    )
                    candidate.append(
                        _metric_row(
                            seed, fold, sample_id, hit=True, rank=1, turn=1,
                            scenario=scenario,
                        )
                    )
        comparison = compare_to_baseline(
            baseline,
            candidate,
            learned_or_fitted=False,
        )
        self.assertFalse(comparison["promotion_gate"]["eligible"])
        self.assertFalse(
            comparison["promotion_gate"]["checks"]["full_promotion_corpus_size"]
        )

    def test_gate_rejects_a_hit_to_miss(self) -> None:
        baseline = []
        candidate = []
        for seed in (1, 2, 3):
            for fold in range(5):
                sample_id = f"s{seed}f{fold}"
                scenario = "boundary" if fold == 0 else "intent_override" if fold == 1 else "buying"
                baseline.append(
                    _metric_row(
                        seed, fold, sample_id, hit=True, rank=10, turn=5,
                        scenario=scenario,
                    )
                )
                candidate.append(
                    _metric_row(
                        seed, fold, sample_id, hit=True, rank=1, turn=1,
                        scenario=scenario,
                    )
                )
        candidate[0] = _metric_row(
            1, 0, "s1f0", hit=False, rank=None, turn=None,
            scenario="boundary",
        )
        comparison = compare_to_baseline(
            baseline,
            candidate,
            learned_or_fitted=False,
        )
        self.assertFalse(comparison["promotion_gate"]["eligible"])
        self.assertEqual(comparison["paired_transitions"]["hit_to_miss"], 1)


class OutputContractTests(unittest.TestCase):
    def test_core_variant_order_keeps_frozen_e4_first(self) -> None:
        self.assertEqual(
            [variant.name for variant in CORE_VARIANTS],
            [
                "frozen_e4",
                "projection_unique_only",
                "semantic_only_no_quality",
                "guarded_hybrid",
            ],
        )

    def test_canonical_writer_uses_sorted_keys_and_lf(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            _write_canonical_json(output, {"z": 1, "a": 2})
            self.assertEqual(output.read_bytes(), b'{\n  "a": 2,\n  "z": 1\n}\n')
            self.assertEqual(json.loads(output.read_text()), {"a": 2, "z": 1})


if __name__ == "__main__":
    unittest.main()
