from __future__ import annotations

import json
import inspect
import tempfile
import unittest
from pathlib import Path

import starter.agent as agent_module
import starter.constraints as constraints_module
from starter.agent import Agent


CATALOG_ROWS = [
    {
        "parent_asin": "A",
        "title": "Blue cotton running shoe",
        "features": ["Breathable cotton upper", "Lightweight running support"],
        "description": ["Comfortable road running shoe"],
        "price": 49.0,
        "categories": ["Clothing, Shoes & Jewelry", "Shoes", "Running"],
        "details": {"Department": "Womens", "Color": "Blue"},
        "average_rating": 4.5,
        "rating_number": 100,
        "store": "Example A",
    },
    {
        "parent_asin": "B",
        "title": "Black leather winter boot",
        "features": ["Insulated leather upper", "Cold-weather traction"],
        "description": ["Warm winter boot"],
        "price": 89.0,
        "categories": ["Clothing, Shoes & Jewelry", "Shoes", "Boots"],
        "details": {"Department": "Womens", "Color": "Black"},
        "average_rating": 4.4,
        "rating_number": 80,
        "store": "Example B",
    },
]


class StatefulAgentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.catalog_path = Path(self.temp_dir.name) / "catalog.jsonl"
        self.catalog_path.write_text(
            "".join(json.dumps(row) + "\n" for row in CATALOG_ROWS),
            encoding="utf-8",
        )
        self.agent = Agent(self.catalog_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_reset_is_required(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "reset must be called"):
            self.agent.respond("missing", "I need shoes.", 1, 10)

    def test_accumulates_constraints_and_asks_structured_questions(self) -> None:
        self.agent.reset("session", {"preference_tags": []})

        first = self.agent.respond(
            "session", "I'm looking for shoes, but I'm still exploring.", 1, 10
        )
        self.assertEqual(first["ask_attribute"], "material")
        self.assertEqual(len(first["recommendations"]), 2)

        second = self.agent.respond(
            "session", "For that, what matters is: breathable cotton upper.", 2, 10
        )
        self.assertEqual(second["ask_attribute"], "feature")
        self.assertEqual(second["recommendations"][0]["parent_asin"], "A")

    def test_intent_override_discards_old_preference_but_keeps_category(self) -> None:
        self.agent.reset("override", {"preference_tags": []})
        self.agent.respond(
            "override", "I'm looking for shoes. I prefer a black leather winter boot.", 1, 10
        )
        response = self.agent.respond(
            "override",
            "Actually, ignore my earlier preference. What I need is breathable cotton.",
            2,
            10,
        )
        self.assertEqual(response["recommendations"][0]["parent_asin"], "A")

    def test_sessions_do_not_share_state(self) -> None:
        self.agent.reset("one", {"preference_tags": []})
        self.agent.reset("two", {"preference_tags": []})
        self.agent.respond("one", "I need a cotton running shoe.", 1, 10)
        response = self.agent.respond("two", "I need a leather winter boot.", 1, 10)
        self.assertEqual(response["recommendations"][0]["parent_asin"], "B")

    def test_reset_replaces_existing_session_state(self) -> None:
        self.agent.reset("reused", {"preference_tags": ["cotton"]})
        self.agent.respond("reused", "I need a cotton running shoe.", 1, 10)

        self.agent.reset("reused", {"preference_tags": ["leather"]})
        response = self.agent.respond("reused", "I need a leather winter boot.", 1, 10)

        self.assertEqual(response["recommendations"][0]["parent_asin"], "B")
        self.assertNotIn("cotton", self.agent.evidence_trace("reused")["canonical_query"])

    def test_response_contract_and_turn_ten_behavior(self) -> None:
        self.agent.reset("contract", {"preference_tags": []})
        allowed_attributes = {
            "category", "material", "color", "size", "style", "brand",
            "budget", "feature", "use_case", "other", None,
        }
        for turn in range(1, 11):
            response = self.agent.respond(
                "contract",
                "I don't have an additional preference for that attribute.",
                turn,
                10,
            )
            self.assertEqual(
                set(response),
                {"message", "ask_attribute", "recommendations", "usage"},
            )
            identifiers = [item["parent_asin"] for item in response["recommendations"]]
            self.assertLessEqual(len(identifiers), 10)
            self.assertEqual(len(identifiers), len(set(identifiers)))
            self.assertTrue(set(identifiers).issubset({"A", "B"}))
            self.assertIn(response["ask_attribute"], allowed_attributes)
            self.assertEqual(
                response["usage"], {"prompt_tokens": 0, "completion_tokens": 0}
            )
        self.assertIsNone(response["ask_attribute"])

    def test_production_agent_has_no_public_label_dependency(self) -> None:
        source = inspect.getsource(agent_module) + inspect.getsource(constraints_module)
        self.assertNotIn("public_set", source)
        self.assertNotIn("ground_truth", source)

    def test_override_preserves_clarification_and_deduplicates_new_value(self) -> None:
        self.agent.reset("trace", {"preference_tags": []})
        self.agent.respond(
            "trace", "I'm looking for shoes. I prefer a black leather winter boot.", 1, 10
        )
        self.agent.respond(
            "trace", "For that, what matters is: breathable cotton.", 2, 10
        )
        self.agent.respond(
            "trace",
            "Actually, ignore my earlier preference. What I need is: breathable cotton.",
            3,
            10,
        )

        trace = self.agent.evidence_trace("trace")
        query = trace["canonical_query"]
        self.assertIn("shoes", query)
        self.assertIn("breathable cotton", query)
        self.assertNotIn("black leather winter boot", query)
        active_cotton = [
            item
            for item in trace["constraints"]
            if item["status"] == "active"
            and item["normalized_value"] == "breathable cotton"
        ]
        self.assertEqual(len(active_cotton), 1)

    def test_continued_turn_rotates_candidates_after_a_miss(self) -> None:
        self.agent.reset("rotate", {"preference_tags": []})
        first = self.agent.respond("rotate", "I'm looking for shoes.", 1, 1)
        second = self.agent.respond(
            "rotate", "I don't have an additional preference for material.", 2, 1
        )

        first_ids = {item["parent_asin"] for item in first["recommendations"]}
        second_ids = {item["parent_asin"] for item in second["recommendations"]}
        self.assertTrue(first_ids)
        self.assertTrue(second_ids)
        self.assertTrue(first_ids.isdisjoint(second_ids))

    def test_candidate_exploration_can_be_disabled_for_ablation(self) -> None:
        agent = Agent(self.catalog_path, explore_unseen=False)
        agent.reset("stable", {"preference_tags": []})
        first = agent.respond("stable", "I'm looking for shoes.", 1, 1)
        second = agent.respond(
            "stable", "I don't have an additional preference for material.", 2, 1
        )

        self.assertEqual(first["recommendations"], second["recommendations"])
        self.assertFalse(agent.evidence_trace("stable")["candidate_exploration"])

    def test_override_reopens_candidates_from_the_previous_intent_epoch(self) -> None:
        self.agent.reset("epoch", {"preference_tags": []})
        first = self.agent.respond("epoch", "I'm looking for shoes. I prefer cotton.", 1, 2)
        self.agent.respond(
            "epoch", "I don't have an additional preference for material.", 2, 2
        )
        corrected = self.agent.respond(
            "epoch",
            "Actually, ignore my earlier preference. What I need is: cotton.",
            3,
            2,
        )

        self.assertEqual(
            [item["parent_asin"] for item in corrected["recommendations"]],
            [item["parent_asin"] for item in first["recommendations"]],
        )
        self.assertEqual(self.agent.evidence_trace("epoch")["intent_epoch"], 1)

    def test_active_avoid_constraint_filters_matching_candidates(self) -> None:
        self.agent.reset("avoid", {"preference_tags": []})
        response = self.agent.respond(
            "avoid",
            "I'm looking for shoes. No leather, please.",
            1,
            10,
        )

        self.assertEqual(
            [item["parent_asin"] for item in response["recommendations"]],
            ["A"],
        )

    def test_avoid_filter_streams_past_a_dominated_prefix(self) -> None:
        rows = []
        for index in range(120):
            rows.append(
                {
                    "parent_asin": f"L{index:03d}",
                    "title": "Leather shoe",
                    "categories": ["Shoes"],
                }
            )
        rows.append(
            {
                "parent_asin": "COTTON",
                "title": "Cotton shoe",
                "categories": ["Shoes"],
            }
        )
        catalog_path = Path(self.temp_dir.name) / "deep_catalog.jsonl"
        catalog_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        agent = Agent(catalog_path)
        agent.reset("deep-avoid", {"preference_tags": []})

        response = agent.respond(
            "deep-avoid",
            "I'm looking for shoes. No leather.",
            1,
            1,
        )

        self.assertEqual(response["recommendations"], [{"parent_asin": "COTTON"}])

    def test_avoid_filter_respects_negated_product_copy(self) -> None:
        catalog_path = Path(self.temp_dir.name) / "negated_copy.jsonl"
        catalog_path.write_text(
            json.dumps(
                {
                    "parent_asin": "RESIN",
                    "title": "Natural resin shoe",
                    "features": ["Made from natural resin, not plastic"],
                    "categories": ["Shoes"],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        agent = Agent(catalog_path)
        agent.reset("copy", {"preference_tags": []})

        response = agent.respond(
            "copy",
            "I'm looking for shoes. No plastic.",
            1,
            1,
        )

        self.assertEqual(response["recommendations"], [{"parent_asin": "RESIN"}])

    def test_product_copy_negation_does_not_bleed_into_later_material(self) -> None:
        rows = [
            {
                "parent_asin": "PERIOD",
                "title": "Shoe",
                "features": ["Not plastic. Leather upper"],
                "categories": ["Shoes"],
            },
            {
                "parent_asin": "NOT_ONLY",
                "title": "Shoe",
                "features": ["Not only comfortable but also leather"],
                "categories": ["Shoes"],
            },
            {
                "parent_asin": "NON_SLIP",
                "title": "Shoe",
                "features": ["Non-slip leather sole"],
                "categories": ["Shoes"],
            },
            {
                "parent_asin": "COTTON_ONLY",
                "title": "Cotton shoe",
                "categories": ["Shoes"],
            },
        ]
        catalog_path = Path(self.temp_dir.name) / "negation_scope.jsonl"
        catalog_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        agent = Agent(catalog_path)
        agent.reset("scope", {"preference_tags": []})

        response = agent.respond(
            "scope",
            "I'm looking for shoes. No leather.",
            1,
            10,
        )

        self.assertEqual(
            response["recommendations"],
            [{"parent_asin": "COTTON_ONLY"}],
        )


if __name__ == "__main__":
    unittest.main()
