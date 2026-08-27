from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path

import starter.question_policy as question_policy_module
from starter.agent import Agent
from starter.question_policy import (
    AdaptiveQuestionPolicy,
    QuestionCandidate,
    extract_question_facets,
)


class AdaptiveQuestionPolicyTest(unittest.TestCase):
    def test_missingness_penalizes_a_sparse_apparently_discriminative_facet(self) -> None:
        candidates = []
        for index in range(12):
            facets = {
                "material": ("cotton" if index < 6 else "leather",),
            }
            if index == 0:
                facets["size"] = ("unusually-specific-size",)
            candidates.append(QuestionCandidate(f"P{index:02d}", facets))

        decision = AdaptiveQuestionPolicy().choose(
            candidates,
            active_facets={"category"},
            asked_attributes=set(),
            turn=1,
        )

        self.assertEqual(decision.ask_attribute, "material")
        by_attribute = {item.attribute: item for item in decision.statistics}
        self.assertEqual(by_attribute["material"].missing_count, 0)
        self.assertEqual(by_attribute["size"].missing_count, 11)
        self.assertGreater(
            by_attribute["material"].selection_score,
            by_attribute["size"].selection_score,
        )

    def test_previously_asked_attribute_is_not_repeated(self) -> None:
        candidates = [
            QuestionCandidate(
                f"P{index:02d}",
                {
                    "material": ("cotton" if index < 6 else "leather",),
                    "color": ("blue" if index % 2 == 0 else "black",),
                },
            )
            for index in range(12)
        ]

        decision = AdaptiveQuestionPolicy().choose(
            candidates,
            active_facets={"category"},
            asked_attributes={"material"},
            turn=2,
        )

        self.assertNotEqual(decision.ask_attribute, "material")
        self.assertEqual(decision.ask_attribute, "color")
        material_stats = next(
            item for item in decision.statistics if item.attribute == "material"
        )
        self.assertFalse(material_stats.eligible)
        self.assertEqual(material_stats.ineligible_reason, "already_asked")

    def test_other_is_controlled_fallback_when_specific_facets_are_uninformative(self) -> None:
        candidates = [QuestionCandidate(f"P{index:02d}", {}) for index in range(12)]

        decision = AdaptiveQuestionPolicy().choose(
            candidates,
            active_facets={"category"},
            asked_attributes=set(),
            turn=1,
        )

        self.assertEqual(decision.ask_attribute, "other")
        self.assertEqual(decision.reason, "specific_facets_below_threshold")

    def test_first_turn_guardrail_overrides_the_statistical_choice(self) -> None:
        candidates = [
            QuestionCandidate(
                f"P{index:02d}",
                {
                    "feature": (f"distinct feature {index}",),
                    "material": ("cotton" if index < 10 else "leather",),
                },
            )
            for index in range(12)
        ]

        decision = AdaptiveQuestionPolicy().choose(
            candidates,
            active_facets={"category"},
            asked_attributes=set(),
            turn=1,
            guardrail_attribute="material",
        )

        self.assertEqual(decision.ask_attribute, "material")
        self.assertEqual(decision.reason, "first_turn_guardrail")

    def test_turn_ten_never_asks_even_when_guardrail_is_supplied(self) -> None:
        decision = AdaptiveQuestionPolicy().choose(
            [],
            active_facets=set(),
            asked_attributes=set(),
            turn=10,
            guardrail_attribute="material",
        )

        self.assertIsNone(decision.ask_attribute)
        self.assertEqual(decision.reason, "turn_limit")

    def test_decision_trace_is_deterministic_and_serializable(self) -> None:
        candidates = [
            QuestionCandidate(
                f"P{index:02d}",
                {"material": ("cotton" if index < 5 else "leather",)},
            )
            for index in range(10)
        ]
        policy = AdaptiveQuestionPolicy()

        first = policy.choose(
            candidates,
            active_facets={"category"},
            asked_attributes=set(),
            turn=1,
        )
        second = policy.choose(
            candidates,
            active_facets={"category"},
            asked_attributes=set(),
            turn=1,
        )

        self.assertEqual(first.as_dict(), second.as_dict())
        trace = first.as_dict()
        self.assertEqual(trace["candidate_count"], 10)
        self.assertEqual(trace["selected_attribute"], "material")
        self.assertTrue(trace["statistics"])
        material = next(
            item for item in trace["statistics"] if item["attribute"] == "material"
        )
        self.assertEqual(sum(material["answer_distribution"].values()), 10)
        self.assertIn("<missing>", material["answer_distribution"])
        self.assertIn("expected_technical_gain", material)

    def test_product_facet_extraction_uses_catalog_fields_only(self) -> None:
        facets = extract_question_facets({
            "parent_asin": "A",
            "title": "Blue cotton trail shoe",
            "features": ["Waterproof grip", "Outdoor hiking traction"],
            "details": {"Department": "Women", "Size": "8"},
            "description": ["Outdoor running trainer"],
            "price": 54.0,
            "categories": ["Shoes", "Trail Running"],
            "store": "Example Brand",
        })

        self.assertIn("cotton", facets["material"])
        self.assertIn("blue", facets["color"])
        self.assertTrue(facets["feature"])
        self.assertTrue(facets["use_case"])

    def test_policy_source_has_no_hidden_label_or_evaluator_dependency(self) -> None:
        source = inspect.getsource(question_policy_module)
        self.assertNotIn("public_set", source)
        self.assertNotIn("ground_truth", source)
        self.assertNotIn("evaluator", source)

    def test_agent_statistics_use_candidates_after_the_displayed_top_k(self) -> None:
        rows = [
            {
                "parent_asin": f"P{index:02d}",
                "title": f"Running shoe {index}",
                "features": [f"Distinct feature {index}"],
                "categories": ["Shoes", "Running"],
            }
            for index in range(15)
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog_path = Path(temp_dir) / "catalog.jsonl"
            catalog_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            agent = Agent(catalog_path)
            agent.reset("candidate-window", {"preference_tags": []})
            response = agent.respond(
                "candidate-window",
                "I'm looking for running shoes, but I'm still exploring.",
                1,
                3,
            )
            trace = agent.evidence_trace("candidate-window")

        self.assertEqual(len(response["recommendations"]), 3)
        self.assertEqual(trace["question_decisions"][0]["candidate_count"], 12)
        self.assertEqual(len(trace["shown_ids"]), 3)

    def test_agent_adapts_after_the_conservative_first_question(self) -> None:
        rows = [
            {
                "parent_asin": f"P{index:02d}",
                "title": f"{'Blue' if index % 2 == 0 else 'Black'} running shoe",
                "features": [],
                "categories": ["Shoes", "Running"],
            }
            for index in range(18)
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog_path = Path(temp_dir) / "catalog.jsonl"
            catalog_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            agent = Agent(catalog_path)
            agent.reset("adaptive", {"preference_tags": []})
            first = agent.respond(
                "adaptive",
                "I'm looking for running shoes, but I'm still exploring.",
                1,
                3,
            )
            second = agent.respond(
                "adaptive",
                "I don't have an additional preference for material.",
                2,
                3,
            )

        self.assertEqual(first["ask_attribute"], "material")
        self.assertEqual(second["ask_attribute"], "color")

    def test_reset_clears_question_history(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog_path = Path(temp_dir) / "catalog.jsonl"
            catalog_path.write_text(
                json.dumps({
                    "parent_asin": "A",
                    "title": "Cotton running shoe",
                    "features": ["Breathable"],
                    "categories": ["Shoes"],
                })
                + "\n",
                encoding="utf-8",
            )
            agent = Agent(catalog_path)
            agent.reset("reset", {"preference_tags": []})
            agent.respond("reset", "I'm looking for shoes.", 1, 10)
            agent.reset("reset", {"preference_tags": []})
            trace = agent.evidence_trace("reset")

        self.assertEqual(trace["asked_attributes"], [])
        self.assertEqual(trace["question_decisions"], [])


if __name__ == "__main__":
    unittest.main()
