from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path

import starter.retrieval as retrieval_module
from starter.agent import Agent
from starter.retrieval import RetrievalConfig


def _write_catalog(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


class MultiRouteRetrievalTest(unittest.TestCase):
    def test_category_only_turn_is_identical_to_e3(self) -> None:
        rows = [
            {
                "parent_asin": f"P{index:02d}",
                "title": f"Running shoe {index}",
                "categories": ["Shoes", "Running"],
            }
            for index in range(15)
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog_path = Path(temp_dir) / "catalog.jsonl"
            _write_catalog(catalog_path, rows)
            e4 = Agent(catalog_path)
            e3 = Agent(catalog_path, retrieval_config=RetrievalConfig(enabled=False))
            for agent, session in ((e4, "e4"), (e3, "e3")):
                agent.reset(session, {"preference_tags": []})

            e4_response = e4.respond(
                "e4", "I'm looking for running shoes, but I'm still exploring.", 1, 5
            )
            e3_response = e3.respond(
                "e3", "I'm looking for running shoes, but I'm still exploring.", 1, 5
            )

        self.assertEqual(e4_response, e3_response)
        trace = e4.evidence_trace("e4")["retrieval_decisions"][-1]
        self.assertEqual(trace["mode"], "legacy_category_only")

    def test_exact_constraint_phrase_breaks_lexical_coverage_tie(self) -> None:
        rows = [
            {
                "parent_asin": "DISTRACTOR",
                "title": "Rare system exact trail heel lock shoe",
                "features": ["Generic outdoor support"],
                "categories": ["Shoes", "Trail Running"],
            },
            {
                "parent_asin": "TARGET",
                "title": "Trail running shoe",
                "features": ["Rare exact heel lock system"],
                "categories": ["Shoes", "Trail Running"],
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog_path = Path(temp_dir) / "catalog.jsonl"
            _write_catalog(catalog_path, rows)
            agent = Agent(catalog_path)
            agent.reset("phrase", {"preference_tags": []})

            response = agent.respond(
                "phrase",
                "I'm looking for trail running shoes. "
                "A key requirement is: rare exact heel lock system.",
                1,
                2,
            )
            trace = agent.evidence_trace("phrase")["retrieval_decisions"][-1]

        self.assertEqual(response["recommendations"][0]["parent_asin"], "TARGET")
        self.assertGreater(trace["strict_candidate_count"], 0)
        self.assertIn("facet", trace["enabled_route_families"])
        self.assertTrue(trace["constraint_reranking"])

    def test_score_fused_relaxation_retains_partial_candidates(self) -> None:
        rows = [
            {
                "parent_asin": "STRICT",
                "title": "Blue waterproof walking shoe",
                "features": ["blue waterproof"],
                "categories": ["Shoes", "Walking"],
            },
            {
                "parent_asin": "PARTIAL_BLUE",
                "title": "Blue walking shoe",
                "categories": ["Shoes", "Walking"],
            },
            {
                "parent_asin": "PARTIAL_WATERPROOF",
                "title": "Waterproof walking shoe",
                "categories": ["Shoes", "Walking"],
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog_path = Path(temp_dir) / "catalog.jsonl"
            _write_catalog(catalog_path, rows)
            agent = Agent(catalog_path)
            agent.reset("relax", {"preference_tags": []})

            response = agent.respond(
                "relax",
                "I'm looking for walking shoes. "
                "A key requirement is: blue waterproof.",
                1,
                3,
            )
            trace = agent.evidence_trace("relax")["retrieval_decisions"][-1]

        self.assertEqual(response["recommendations"][0]["parent_asin"], "STRICT")
        self.assertEqual(len(response["recommendations"]), 3)
        self.assertEqual(trace["strict_candidate_count"], 1)
        self.assertGreaterEqual(trace["relaxed_candidates_used"], 2)

    def test_disabling_relaxation_returns_only_strict_candidates(self) -> None:
        rows = [
            {
                "parent_asin": "STRICT",
                "title": "Blue waterproof walking shoe",
                "features": ["blue waterproof"],
                "categories": ["Shoes", "Walking"],
            },
            {
                "parent_asin": "PARTIAL",
                "title": "Blue walking shoe",
                "categories": ["Shoes", "Walking"],
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog_path = Path(temp_dir) / "catalog.jsonl"
            _write_catalog(catalog_path, rows)
            config = RetrievalConfig(use_soft_relaxation=False)
            agent = Agent(catalog_path, retrieval_config=config)
            agent.reset("strict-only", {"preference_tags": []})

            response = agent.respond(
                "strict-only",
                "I'm looking for walking shoes. "
                "A key requirement is: blue waterproof.",
                1,
                2,
            )

        self.assertEqual(response["recommendations"], [{"parent_asin": "STRICT"}])

    def test_strict_filter_composes_with_legacy_union_order(self) -> None:
        rows = [
            {
                "parent_asin": "STRICT",
                "title": "Blue waterproof walking shoe",
                "features": ["blue waterproof"],
                "categories": ["Shoes", "Walking"],
            },
            {
                "parent_asin": "PARTIAL",
                "title": "Blue walking shoe",
                "categories": ["Shoes", "Walking"],
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog_path = Path(temp_dir) / "catalog.jsonl"
            _write_catalog(catalog_path, rows)
            agent = Agent(
                catalog_path,
                retrieval_config=RetrievalConfig(
                    use_constraint_reranking=False,
                    use_soft_relaxation=False,
                ),
            )
            agent.reset("strict-legacy-order", {"preference_tags": []})
            response = agent.respond(
                "strict-legacy-order",
                "I'm looking for walking shoes. "
                "A key requirement is: blue waterproof.",
                1,
                2,
            )

        self.assertEqual(response["recommendations"], [{"parent_asin": "STRICT"}])

    def test_category_exact_route_counts_as_strict_evidence(self) -> None:
        rows = [
            {
                "parent_asin": f"H{index:03d}",
                "title": "Hat with rare alpha feature",
                "features": ["rare alpha feature"],
                "categories": ["Hats"],
                "rating_number": 1000 - index,
            }
            for index in range(201)
        ]
        rows.append(
            {
                "parent_asin": "TARGET",
                "title": "Shoe with rare alpha feature",
                "features": ["rare alpha feature"],
                "categories": ["Shoes"],
                "rating_number": 1,
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog_path = Path(temp_dir) / "catalog.jsonl"
            _write_catalog(catalog_path, rows)
            agent = Agent(
                catalog_path,
                retrieval_config=RetrievalConfig(use_soft_relaxation=False),
            )
            agent.reset("category-strict", {"preference_tags": []})
            response = agent.respond(
                "category-strict",
                "I'm looking for shoes. "
                "A key requirement is: rare alpha feature.",
                1,
                10,
            )
            trace = agent.evidence_trace("category-strict")[
                "retrieval_decisions"
            ][-1]

        self.assertIn({"parent_asin": "TARGET"}, response["recommendations"])
        target_trace = next(
            item for item in trace["top_candidates"]
            if item["parent_asin"] == "TARGET"
        )
        self.assertTrue(target_trace["strict"])

    def test_candidate_union_is_bounded_and_route_trace_is_deterministic(self) -> None:
        rows = [
            {
                "parent_asin": f"P{index:03d}",
                "title": f"Blue cotton running shoe {index}",
                "features": ["Breathable mesh upper"],
                "categories": ["Shoes", "Running"],
            }
            for index in range(140)
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog_path = Path(temp_dir) / "catalog.jsonl"
            _write_catalog(catalog_path, rows)

            traces = []
            responses = []
            for session in ("one", "two"):
                agent = Agent(catalog_path)
                agent.reset(session, {"preference_tags": []})
                responses.append(
                    agent.respond(
                        session,
                        "I'm looking for running shoes. "
                        "A key requirement is: breathable mesh upper.",
                        1,
                        10,
                    )
                )
                traces.append(
                    agent.evidence_trace(session)["retrieval_decisions"][-1]
                )

        self.assertEqual(responses[0], responses[1])
        self.assertEqual(traces[0], traces[1])
        self.assertLessEqual(traces[0]["candidate_union_count"], 100)
        self.assertEqual(
            traces[0]["enabled_route_families"],
            ["current_turn", "ledger", "category", "facet"],
        )

    def test_e4_ranking_keeps_e3_question_trajectory(self) -> None:
        rows = [
            {
                "parent_asin": f"P{index:02d}",
                "title": f"{'Blue' if index % 2 == 0 else 'Black'} running shoe",
                "features": ["Breathable mesh" if index % 3 else "Cushioned sole"],
                "categories": ["Shoes", "Running"],
            }
            for index in range(24)
        ]
        messages = [
            "I'm looking for running shoes, but I'm still exploring.",
            "I don't have an additional preference for material.",
            "For that, what matters is: breathable mesh.",
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog_path = Path(temp_dir) / "catalog.jsonl"
            _write_catalog(catalog_path, rows)
            agents = [
                Agent(catalog_path),
                Agent(catalog_path, retrieval_config=RetrievalConfig(enabled=False)),
            ]
            trajectories = []
            for index, agent in enumerate(agents):
                session = f"s{index}"
                agent.reset(session, {"preference_tags": []})
                asks = [
                    agent.respond(session, message, turn, 3)["ask_attribute"]
                    for turn, message in enumerate(messages, start=1)
                ]
                trajectories.append((asks, agent.evidence_trace(session)["question_decisions"]))

        self.assertEqual(trajectories[0], trajectories[1])

    def test_question_shadow_stays_frozen_when_focus_is_removed(self) -> None:
        rows = [
            {
                "parent_asin": f"D{index:02d}",
                "title": f"Rare exact heel lock system trail shoe {index}",
                "features": ["Generic support"],
                "categories": ["Shoes", "Trail Running"],
            }
            for index in range(30)
        ]
        rows.append(
            {
                "parent_asin": "TARGET",
                "title": "Trail running shoe",
                "features": ["Rare exact heel lock system"],
                "categories": ["Shoes", "Trail Running"],
            }
        )
        messages = [
            "I'm looking for trail running shoes. "
            "A key requirement is: rare exact heel lock system.",
            "I don't have an additional preference for feature.",
            "Those options are not quite right yet.",
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog_path = Path(temp_dir) / "catalog.jsonl"
            _write_catalog(catalog_path, rows)
            agents = [
                Agent(catalog_path),
                Agent(catalog_path, retrieval_config=RetrievalConfig(enabled=False)),
            ]
            trajectories = []
            first_recommendations = []
            for index, agent in enumerate(agents):
                session = f"focus-removed-{index}"
                agent.reset(session, {"preference_tags": []})
                responses = [
                    agent.respond(session, message, turn, 1)
                    for turn, message in enumerate(messages, start=1)
                ]
                trajectories.append(
                    (
                        [response["ask_attribute"] for response in responses],
                        agent.evidence_trace(session)["question_decisions"],
                    )
                )
                first_recommendations.append(responses[0]["recommendations"])

        self.assertNotEqual(first_recommendations[0], first_recommendations[1])
        self.assertEqual(trajectories[0], trajectories[1])

    def test_question_shadow_stays_frozen_when_avoid_forces_fallback(self) -> None:
        rows = [
            {
                "parent_asin": f"D{index:02d}",
                "title": f"Rare exact heel lock system trail shoe {index}",
                "features": ["Generic support"],
                "categories": ["Shoes", "Trail Running"],
            }
            for index in range(30)
        ]
        rows.append(
            {
                "parent_asin": "TARGET",
                "title": "Trail running shoe",
                "features": ["Rare exact heel lock system"],
                "categories": ["Shoes", "Trail Running"],
            }
        )
        messages = [
            "I'm looking for trail running shoes. "
            "A key requirement is: rare exact heel lock system.",
            "I don't want leather.",
            "Those options are not quite right yet.",
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog_path = Path(temp_dir) / "catalog.jsonl"
            _write_catalog(catalog_path, rows)
            trajectories = []
            for index, config in enumerate(
                (RetrievalConfig(), RetrievalConfig(enabled=False))
            ):
                agent = Agent(catalog_path, retrieval_config=config)
                session = f"avoid-fallback-{index}"
                agent.reset(session, {"preference_tags": []})
                responses = [
                    agent.respond(session, message, turn, 1)
                    for turn, message in enumerate(messages, start=1)
                ]
                trajectories.append(
                    (
                        [response["ask_attribute"] for response in responses],
                        agent.evidence_trace(session)["question_decisions"],
                    )
                )

        self.assertEqual(trajectories[0], trajectories[1])

    def test_raw_grey_constraint_matches_catalog_exact_value(self) -> None:
        rows = [
            {
                "parent_asin": "TARGET",
                "title": "Walking shoe",
                "features": ["Slate Grey finish"],
                "categories": ["Shoes", "Walking"],
            },
            {
                "parent_asin": "DISTRACTOR",
                "title": "Gray walking shoe",
                "features": ["Generic finish"],
                "categories": ["Shoes", "Walking"],
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog_path = Path(temp_dir) / "catalog.jsonl"
            _write_catalog(catalog_path, rows)
            agent = Agent(catalog_path)
            agent.reset("grey", {"preference_tags": []})
            response = agent.respond(
                "grey",
                "I'm looking for walking shoes. "
                "A key requirement is: Slate Grey finish.",
                1,
                2,
            )

        self.assertEqual(response["recommendations"][0]["parent_asin"], "TARGET")

    def test_exclusion_fallback_refills_beyond_dominated_prefix(self) -> None:
        rows = [
            {
                "parent_asin": f"L{index:03d}",
                "title": "Leather shoe",
                "categories": ["Shoes"],
            }
            for index in range(120)
        ]
        rows.append(
            {
                "parent_asin": "ZZZ_COTTON",
                "title": "Cotton shoe " + "comfortable " * 30,
                "categories": ["Shoes"],
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog_path = Path(temp_dir) / "catalog.jsonl"
            _write_catalog(catalog_path, rows)
            agent = Agent(catalog_path)
            agent.reset("avoid", {"preference_tags": []})
            response = agent.respond(
                "avoid", "I'm looking for shoes. No leather.", 1, 1
            )
            trace = agent.evidence_trace("avoid")["retrieval_decisions"][-1]

        self.assertEqual(response["recommendations"], [{"parent_asin": "ZZZ_COTTON"}])
        self.assertEqual(trace["mode"], "legacy_exclusion_fallback")

    def test_exact_facet_routes_are_capped(self) -> None:
        features = [f"distinct feature {index}" for index in range(8)]
        rows = [
            {
                "parent_asin": "TARGET",
                "title": "Running shoe",
                "features": features,
                "categories": ["Shoes", "Running"],
            }
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog_path = Path(temp_dir) / "catalog.jsonl"
            _write_catalog(catalog_path, rows)
            agent = Agent(catalog_path)
            agent.reset("cap", {"preference_tags": []})
            agent.respond(
                "cap",
                "I'm looking for running shoes. A key requirement is: "
                + "; ".join(features)
                + ".",
                1,
                1,
            )
            routes = agent.evidence_trace("cap")["retrieval_decisions"][-1]["routes"]

        facet_routes = [route for route in routes if route["family"] == "facet"]
        self.assertLessEqual(len(facet_routes), 8)

    def test_strict_mode_uses_the_bounded_routed_must_set(self) -> None:
        features = [f"distinct feature {index}" for index in range(8)]
        rows = [
            {
                "parent_asin": "TARGET",
                "title": "Running shoe " + " ".join(features),
                "features": features,
                "categories": ["Shoes", "Running"],
            },
            {
                "parent_asin": "PARTIAL",
                "title": "Running shoe " + " ".join(features),
                "features": features[-3:],
                "categories": ["Shoes", "Running"],
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog_path = Path(temp_dir) / "catalog.jsonl"
            _write_catalog(catalog_path, rows)
            agent = Agent(
                catalog_path,
                retrieval_config=RetrievalConfig(
                    max_facet_constraints=4,
                    use_soft_relaxation=False,
                ),
            )
            agent.reset("strict-cap", {"preference_tags": []})
            response = agent.respond(
                "strict-cap",
                "I'm looking for running shoes. A key requirement is: "
                + "; ".join(features)
                + ".",
                1,
                2,
            )
            trace = agent.evidence_trace("strict-cap")["retrieval_decisions"][-1]

        self.assertEqual(response["recommendations"], [{"parent_asin": "TARGET"}])
        self.assertEqual(trace["routed_must_constraint_count"], 4)
        self.assertEqual(trace["strict_candidate_count"], 1)

    def test_full_e4_disable_keeps_legacy_e3_path(self) -> None:
        rows = [
            {
                "parent_asin": "A",
                "title": "Blue cotton running shoe",
                "categories": ["Shoes", "Running"],
            },
            {
                "parent_asin": "B",
                "title": "Black leather winter boot",
                "categories": ["Shoes", "Boots"],
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog_path = Path(temp_dir) / "catalog.jsonl"
            _write_catalog(catalog_path, rows)
            agent = Agent(
                catalog_path,
                retrieval_config=RetrievalConfig(enabled=False),
            )
            agent.reset("legacy", {"preference_tags": []})

            response = agent.respond("legacy", "I'm looking for shoes.", 1, 1)
            trace = agent.evidence_trace("legacy")

        self.assertEqual(response["recommendations"], [{"parent_asin": "A"}])
        self.assertFalse(trace["multi_route_retrieval"])
        self.assertEqual(trace["retrieval_decisions"], [])

    def test_production_retrieval_has_no_hidden_label_dependency(self) -> None:
        source = inspect.getsource(retrieval_module)
        self.assertNotIn("public_set", source)
        self.assertNotIn("ground_truth", source)
        self.assertNotIn("scenario_type", source)
        self.assertNotIn("evaluator", source)


if __name__ == "__main__":
    unittest.main()
