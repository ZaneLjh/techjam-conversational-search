from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from starter.agent import Agent
from starter.retrieval import (
    RetrievalConfig,
    e4_1_strict_only_config,
    e4_fallback_config,
)
from tools.e4_1_ablation_suite import declared_variants
from tools.e4_1_compliance_suite import build_report


def _write_catalog(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _strict_front_rows() -> list[dict]:
    strict = [
        {
            "parent_asin": f"S{index:03d}",
            "title": f"Walking shoe {index}",
            "features": ["rare alpha feature"],
            "categories": ["Shoes", "Walking"],
            "rating_number": 1000 - index,
        }
        for index in range(25)
    ]
    relaxed = [
        {
            "parent_asin": f"R{index:03d}",
            "title": f"Rare alpha feature extended walking shoe {index}",
            "features": ["different beta feature"],
            "categories": ["Shoes", "Walking"],
            "rating_number": 500 - index,
        }
        for index in range(8)
    ]
    return [*strict, *relaxed]


class E41RetrievalTest(unittest.TestCase):
    def test_strict_front_uses_one_slot_then_two_after_miss(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog = Path(temp_dir) / "catalog.jsonl"
            _write_catalog(catalog, _strict_front_rows())
            agent = Agent(catalog, retrieval_config=RetrievalConfig())
            agent.reset("dynamic", {})
            first = agent.respond(
                "dynamic",
                "I'm looking for walking shoes. A key requirement is: rare alpha feature.",
                1,
                10,
            )
            first_trace = agent.evidence_trace("dynamic")["retrieval_decisions"][-1]
            second = agent.respond(
                "dynamic",
                "Those options are not quite right yet. Ask me about one specific attribute.",
                2,
                10,
            )
            second_trace = agent.evidence_trace("dynamic")["retrieval_decisions"][-1]

        self.assertEqual(first_trace["effective_relaxed_backfill_slots"], 1)
        self.assertEqual(first_trace["relaxed_candidates_used"], 1)
        self.assertTrue(first_trace["top_candidates"][0]["strict"])
        self.assertFalse(first_trace["top_candidates"][-1]["strict"])
        self.assertEqual(second_trace["effective_relaxed_backfill_slots"], 2)
        self.assertTrue(second_trace["recovery_expanded_after_miss"])
        self.assertEqual(second_trace["relaxed_candidates_used"], 2)
        self.assertEqual(len(first["recommendations"]), 10)
        self.assertEqual(len(second["recommendations"]), 10)

    def test_requested_top_one_never_replaces_strict_with_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog = Path(temp_dir) / "catalog.jsonl"
            _write_catalog(catalog, _strict_front_rows())
            agent = Agent(catalog, retrieval_config=RetrievalConfig())
            agent.reset("one", {})
            response = agent.respond(
                "one",
                "I'm looking for walking shoes. A key requirement is: rare alpha feature.",
                1,
                1,
            )
            trace = agent.evidence_trace("one")["retrieval_decisions"][-1]

        self.assertTrue(trace["top_candidates"][0]["strict"])
        self.assertEqual(response["recommendations"][0]["parent_asin"], "S000")

    def test_candidate_pool_is_bounded_unique_and_contains_display(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog = Path(temp_dir) / "catalog.jsonl"
            _write_catalog(catalog, _strict_front_rows())
            agent = Agent(
                catalog,
                retrieval_config=replace(
                    RetrievalConfig(),
                    candidate_union_depth=10,
                ),
            )
            agent.reset("pool", {})
            response = agent.respond(
                "pool",
                "I'm looking for walking shoes. A key requirement is: rare alpha feature.",
                1,
                10,
            )
            trace = agent.evidence_trace("pool")["retrieval_decisions"][-1]
            candidate_ids = trace["candidate_ids"]

        displayed = [item["parent_asin"] for item in response["recommendations"]]
        self.assertLessEqual(len(candidate_ids), 10)
        self.assertEqual(len(candidate_ids), len(set(candidate_ids)))
        self.assertTrue(set(displayed).issubset(candidate_ids))
        self.assertEqual(trace["candidate_union_count"], len(trace["candidate_ids"]))
        self.assertEqual(
            trace["eligible_candidate_count"], len(trace["eligible_candidate_ids"])
        )
        self.assertEqual(trace["recommendation_count"], len(trace["recommendation_ids"]))

    def test_master_fallback_disables_both_e4_1_components(self) -> None:
        fallback = e4_fallback_config()
        self.assertFalse(fallback.use_strict_front)
        self.assertFalse(fallback.use_auxiliary_confidence_gate)

    def test_strict_diagnostic_keeps_unknown_in_route_pool(self) -> None:
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
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog = Path(temp_dir) / "catalog.jsonl"
            _write_catalog(catalog, rows)
            agent = Agent(catalog, retrieval_config=e4_1_strict_only_config())
            agent.reset("strict", {})
            response = agent.respond(
                "strict",
                "I'm looking for walking shoes. A key requirement is: rare alpha feature.",
                1,
                10,
            )
            trace = agent.evidence_trace("strict")["retrieval_decisions"][-1]

        self.assertEqual(
            [item["parent_asin"] for item in response["recommendations"]],
            ["STRICT"],
        )
        self.assertIn("UNKNOWN", trace["candidate_ids"])
        self.assertNotIn("UNKNOWN", trace["eligible_candidate_ids"])

    def test_material_alias_is_exact_not_false_mismatch(self) -> None:
        rows = [
            {
                "parent_asin": "LEATHER",
                "title": "Walking shoe",
                "features": ["100% Leather"],
                "categories": ["Shoes", "Walking"],
            },
            {
                "parent_asin": "CANVAS",
                "title": "Walking shoe",
                "features": ["Canvas"],
                "categories": ["Shoes", "Walking"],
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog = Path(temp_dir) / "catalog.jsonl"
            _write_catalog(catalog, rows)
            agent = Agent(catalog, retrieval_config=RetrievalConfig())
            agent.reset("material", {})
            agent.respond(
                "material",
                "I'm looking for walking shoes. A key requirement is: leather.",
                1,
                2,
            )
            trace = agent.evidence_trace("material")["retrieval_decisions"][-1]
            by_id = {item["parent_asin"]: item for item in trace["top_candidates"]}

        self.assertTrue(by_id["LEATHER"]["strict"])
        self.assertEqual(by_id["LEATHER"]["mismatched_must_count"], 0)

    def test_insufficient_strict_coverage_broadens_and_override_resets_slots(self) -> None:
        rows = _strict_front_rows()
        for row in rows[2:25]:
            row["features"] = ["different beta feature"]
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog = Path(temp_dir) / "catalog.jsonl"
            _write_catalog(catalog, rows)
            agent = Agent(catalog, retrieval_config=RetrievalConfig())
            agent.reset("coverage", {})
            first = agent.respond(
                "coverage",
                "I'm looking for walking shoes. A key requirement is: rare alpha feature.",
                1,
                10,
            )
            first_trace = agent.evidence_trace("coverage")["retrieval_decisions"][-1]
            agent.respond(
                "coverage",
                "Actually, ignore my earlier preference. What I need is: different beta feature.",
                2,
                10,
            )
            override_trace = agent.evidence_trace("coverage")["retrieval_decisions"][-1]

        self.assertEqual(len(first["recommendations"]), 10)
        self.assertTrue(first_trace["broader_relaxation"])
        self.assertEqual(override_trace["effective_relaxed_backfill_slots"], 1)
        self.assertFalse(override_trace["prior_miss"])

    def test_ablation_matrix_includes_route_interactions(self) -> None:
        names = {variant.name for variant in declared_variants()}
        self.assertTrue(
            {
                "full",
                "e4_fallback",
                "strict_only_diagnostic",
                "strict_front_two_slots",
                "one_relaxed_slot",
                "zero_relaxed_slots",
                "no_strict_front",
                "no_auxiliary_gate",
                "routes_c0l0g0",
            }.issubset(names)
        )

    def test_compliance_suite_passes(self) -> None:
        report = build_report()
        self.assertTrue(report["passed"], report)


if __name__ == "__main__":
    unittest.main()
