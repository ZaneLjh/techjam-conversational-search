from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.e4_ablation_suite import declared_variants, run_suite
from tools.paired_report import compare_results, session_utility


def _session(
    sample_id: str,
    scenario: str,
    turn: int | None,
    rank: int | None,
) -> dict:
    return {
        "sample_id": sample_id,
        "scenario_type": scenario,
        "hit": turn is not None,
        "first_hit_turn": turn,
        "best_rank": rank,
        "reciprocal_rank": 0.0 if rank is None else 1.0 / rank,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


class PairedReportTest(unittest.TestCase):
    def test_reports_independent_session_utility_and_top_changes(self) -> None:
        baseline = {
            "sessions": [
                _session("A", "buying", 2, 2),
                _session("B", "browsing", None, None),
                _session("C", "buying", 4, 1),
            ]
        }
        candidate = {
            "sessions": [
                _session("C", "buying", 5, 1),
                _session("B", "browsing", 10, 10),
                _session("A", "buying", 1, 1),
            ]
        }

        report = compare_results(baseline, candidate, top_n=2)

        self.assertAlmostEqual(session_utility(baseline["sessions"][0]), 0.83)
        self.assertEqual(
            report["paired"]["counts"],
            {"improved": 2, "regressed": 1, "tied": 0},
        )
        self.assertEqual(
            report["paired"]["hit_transitions"],
            {
                "miss_to_hit": 1,
                "hit_to_miss": 0,
                "hit_to_hit": 2,
                "miss_to_miss": 0,
            },
        )
        self.assertAlmostEqual(report["paired"]["sum_utility_delta"], 0.70)
        self.assertEqual(
            [row["sample_id"] for row in report["top_improvements"]],
            ["B", "A"],
        )
        self.assertEqual(report["top_regressions"][0]["sample_id"], "C")
        self.assertEqual(
            [row["sample_id"] for row in report["session_deltas"]],
            ["A", "B", "C"],
        )

    def test_rejects_duplicate_session_ids(self) -> None:
        duplicate = _session("A", "buying", 1, 1)
        with self.assertRaisesRegex(ValueError, "duplicate sample_id A"):
            compare_results(
                {"sessions": [duplicate, duplicate]},
                {"sessions": [duplicate]},
            )


class E4AblationSuiteTest(unittest.TestCase):
    def test_declares_full_compatibility_and_all_six_component_ablations(self) -> None:
        variants = {variant.name: variant.config for variant in declared_variants()}

        self.assertEqual(
            list(variants),
            [
                "full",
                "e3_compatibility",
                "no_current_turn",
                "no_ledger",
                "no_category",
                "no_facet",
                "no_constraint_reranking",
                "no_soft_relaxation",
            ],
        )
        self.assertFalse(variants["e3_compatibility"].enabled)
        self.assertFalse(variants["no_current_turn"].use_current_turn_route)
        self.assertFalse(variants["no_ledger"].use_ledger_route)
        self.assertFalse(variants["no_category"].use_category_route)
        self.assertFalse(variants["no_facet"].use_facet_route)
        self.assertFalse(
            variants["no_constraint_reranking"].use_constraint_reranking
        )
        self.assertFalse(variants["no_soft_relaxation"].use_soft_relaxation)

    def test_suite_output_is_deterministic_compact_and_paired(self) -> None:
        catalog_rows = [
            {
                "parent_asin": f"P{index:02d}",
                "title": f"Blue cotton running shoe {index}",
                "features": ["cotton", "cushioned sole"],
                "details": {"Department": "unisex"},
                "description": ["comfortable running shoe"],
                "categories": ["Clothing", "Shoes", "Running"],
                "store": "Example",
                "average_rating": 4.0,
                "rating_number": 10 + index,
                "price": 40.0 + index,
            }
            for index in range(12)
        ]
        dataset_rows = [
            {
                "sample_id": "public_test_1",
                "scenario_type": "buying",
                "user_profile": {"preference_tags": []},
                "ground_truth": {"parent_asin": "P00"},
                "intent_card": {
                    "target_category": "running shoe",
                    "hard_constraints": ["cotton"],
                    "soft_preferences": ["cushioned sole"],
                },
                "behavior": {"scenario_type": "buying"},
            },
            {
                "sample_id": "public_test_2",
                "scenario_type": "browsing",
                "user_profile": {"preference_tags": []},
                "ground_truth": {"parent_asin": "P01"},
                "intent_card": {
                    "target_category": "running shoe",
                    "hard_constraints": ["cotton"],
                    "soft_preferences": ["cushioned sole"],
                },
                "behavior": {"scenario_type": "browsing"},
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            catalog = root / "catalog.jsonl"
            dataset = root / "dataset.jsonl"
            _write_jsonl(catalog, catalog_rows)
            _write_jsonl(dataset, dataset_rows)

            first = run_suite(catalog, dataset)
            second = run_suite(catalog, dataset)

        self.assertEqual(first, second)
        self.assertEqual(first["sample_count"], 2)
        self.assertEqual(first["execution"]["agent_instances"], 1)
        self.assertEqual(first["execution"]["catalog_indexes_built"], 1)
        self.assertEqual(len(first["variants"]), 8)
        self.assertNotIn('"sessions"', json.dumps(first))
        for variant in first["variants"]:
            counts = variant["paired_full_vs_variant"]["counts"]
            self.assertEqual(sum(counts.values()), 2)


if __name__ == "__main__":
    unittest.main()
