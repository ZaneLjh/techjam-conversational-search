from __future__ import annotations

import unittest

from evaluator.local_evaluator import customer_reply
from starter.agent import Agent, SessionState
from starter.constraints import Facet, parse_message
from starter.question_policy import (
    AdaptiveQuestionPolicy,
    QuestionPolicyConfig,
    SPECIFIC_ATTRIBUTES,
)
from starter.projection import (
    ProjectedClue,
    ProjectedProduct,
    ProjectionConfig,
    ProjectionIndex,
    ProjectionRanking,
)


def _sample(scenario: str = "browsing") -> dict:
    return {
        "scenario_type": scenario,
        "intent_card": {
            "target_category": "Walking shoes",
            "hard_constraints": ["cotton", "color: blue"],
            "soft_preferences": ["slim fit", "outdoor use"],
        },
    }


def _clue(
    raw: str,
    facet: str,
    ordinal: int,
    *,
    role: str = "hard",
    normalized: str | None = None,
) -> ProjectedClue:
    return ProjectedClue(
        raw,
        normalized if normalized is not None else raw.casefold(),
        facet,
        role,
        ordinal,
    )


def _record(
    parent_asin: str,
    clues: tuple[ProjectedClue, ...],
) -> ProjectedProduct:
    return ProjectedProduct(
        parent_asin,
        "Women Shoes",
        "women shoes",
        clues,
    )


def _manual_index(
    records: dict[str, ProjectedProduct],
    **config_overrides: object,
) -> ProjectionIndex:
    values: dict[str, object] = {
        "enabled": True,
        "min_question_gain": 0.002,
    }
    values.update(config_overrides)
    index = object.__new__(ProjectionIndex)
    index.config = ProjectionConfig(**values)
    index.ready = True
    index.status_reason = "ready"
    index.records = records
    index._catalog_order = {
        parent_asin: order for order, parent_asin in enumerate(records)
    }
    index._category_index = {
        "women shoes": tuple(records),
    }
    index._value_index = {}
    return index


def _active_ranking(
    identifiers: tuple[str, ...],
    *,
    recommendations: tuple[str, ...] = (),
) -> ProjectionRanking:
    return ProjectionRanking(
        recommendations,
        identifiers,
        identifiers,
        True,
        {"active": True},
    )


def _tracking_agent(index: ProjectionIndex) -> Agent:
    agent = object.__new__(Agent)
    agent.projection_index = index
    return agent


class RepeatedOtherRolloutTest(unittest.TestCase):
    def test_boundary_refusal_does_not_exhaust_other(self) -> None:
        disclosed: set[str] = set()
        reply, boundary_used = customer_reply(
            _sample("boundary"),
            "other",
            disclosed,
            False,
        )
        state = SessionState(
            user_profile={},
            last_asked_attribute="other",
            infer_other_answer_facets=True,
        )
        state.add_message(reply, 2)

        self.assertTrue(boundary_used)
        self.assertEqual(
            reply,
            "I don't have a preference for other; please use your judgment.",
        )
        self.assertFalse(state.other_exhausted)

    def test_browsing_repeats_other_until_exact_exhaustion(self) -> None:
        policy = AdaptiveQuestionPolicy(
            config=QuestionPolicyConfig(repeat_other_until_exhausted=True)
        )
        asked = {*SPECIFIC_ATTRIBUTES, "other"}
        first_decision = policy.choose(
            [],
            active_facets=set(),
            asked_attributes=asked,
            turn=2,
            allow_repeated_other=True,
            other_exhausted=False,
        )
        disclosed: set[str] = set()
        first_reply, boundary = customer_reply(
            _sample(), first_decision.ask_attribute, disclosed, False
        )
        second_reply, boundary = customer_reply(
            _sample(), first_decision.ask_attribute, disclosed, boundary
        )
        exhausted_reply, _ = customer_reply(
            _sample(), first_decision.ask_attribute, disclosed, boundary
        )

        self.assertEqual(first_decision.ask_attribute, "other")
        self.assertEqual(
            first_reply,
            "For that, what matters is: cotton; color: blue.",
        )
        self.assertEqual(
            second_reply,
            "For that, what matters is: slim fit; outdoor use.",
        )
        self.assertEqual(
            exhausted_reply,
            "I don't have an additional preference for other.",
        )
        exhausted_decision = policy.choose(
            [],
            active_facets=set(),
            asked_attributes=asked,
            turn=3,
            allow_repeated_other=True,
            other_exhausted=True,
        )
        self.assertIsNone(exhausted_decision.ask_attribute)

    def test_disabled_policy_retains_legacy_one_use_other_behavior(self) -> None:
        policy = AdaptiveQuestionPolicy(
            config=QuestionPolicyConfig(repeat_other_until_exhausted=False)
        )
        decision = policy.choose(
            [],
            active_facets=set(),
            asked_attributes={*SPECIFIC_ATTRIBUTES, "other"},
            turn=2,
            allow_repeated_other=True,
            other_exhausted=False,
        )

        self.assertIsNone(decision.ask_attribute)
        self.assertEqual(decision.reason, "small_candidate_fallback")

    def test_exact_reply_partition_rollout_can_select_other(self) -> None:
        combinations = (
            ("cotton", "color: blue", "slim fit", "outdoor"),
            ("cotton", "color: red", "regular fit", "indoor"),
            ("leather", "color: blue", "slim fit", "indoor"),
            ("leather", "color: red", "regular fit", "outdoor"),
        )
        records: dict[str, ProjectedProduct] = {}
        for number, (material, color, style, use_case) in enumerate(
            combinations,
            start=1,
        ):
            parent_asin = f"P{number:09d}"
            records[parent_asin] = _record(
                parent_asin,
                (
                    _clue(material, "material", 0),
                    _clue(color, "color", 1),
                    _clue(style, "style", 0, role="soft"),
                    _clue(use_case, "use_case", 1, role="soft"),
                ),
            )
        index = _manual_index(records)
        identifiers = tuple(records)
        selected, trace = index.choose_question(
            ranking=_active_ranking(identifiers),
            constraints=(),
            disclosed_values=set(),
            asked_attributes=set(SPECIFIC_ATTRIBUTES) - {"material"},
            other_exhausted=False,
            turn=2,
            baseline_attribute="material",
        )

        self.assertEqual(selected, "other")
        self.assertEqual(trace["selected_attribute"], "other")
        self.assertEqual(trace["reason"], "gain_threshold_passed")
        self.assertGreater(trace["expected_gain"], 0)

    def test_rollout_partitions_by_rendered_reply_not_raw_tuple(self) -> None:
        records = {
            "P000000001": _record(
                "P000000001",
                (
                    _clue("a; b", "other", 0),
                    _clue("c", "other", 1),
                ),
            ),
            "P000000002": _record(
                "P000000002",
                (
                    _clue("a", "other", 0),
                    _clue("b; c", "other", 1),
                ),
            ),
        }
        index = _manual_index(records)
        selected, trace = index.choose_question(
            ranking=_active_ranking(tuple(records)),
            constraints=(),
            disclosed_values=set(),
            asked_attributes=set(SPECIFIC_ATTRIBUTES),
            other_exhausted=False,
            turn=2,
            baseline_attribute=None,
            condition_on_current_miss=False,
        )

        self.assertIsNone(selected)
        self.assertEqual(trace["expected_gain"], 0.0)

    def test_first_projected_reply_is_resolved_against_current_posterior(self) -> None:
        first = _record(
            "P000000001",
            (
                _clue("Leather", "material", 0),
                _clue("Imported", "feature", 1),
            ),
        )
        # This different raw signature renders to the same customer-visible
        # text. It must not make the active singleton posterior ambiguous.
        colliding = _record(
            "P000000002",
            (_clue("Leather; Imported", "feature", 0),),
        )
        index = _manual_index(
            {first.parent_asin: first, colliding.parent_asin: colliding}
        )
        agent = _tracking_agent(index)
        state = SessionState(
            user_profile={},
            last_asked_attribute="other",
            infer_other_answer_facets=True,
            projection_decisions=[
                {
                    "active": True,
                    "posterior_ids": [first.parent_asin],
                    "recommendation_ids": [],
                }
            ],
        )

        resolved = agent._track_projection_disclosures(
            state,
            "For that, what matters is: Leather; Imported.",
            2,
        )

        self.assertTrue(resolved)
        self.assertTrue(state.projection_template_confident)
        self.assertEqual(
            state.projection_disclosed_values,
            {"Leather", "Imported"},
        )

    def test_html_prefixed_clue_is_not_treated_as_a_rollout_sentinel(self) -> None:
        record = _record(
            "P000000001",
            (
                _clue(
                    "<p>Imported</p>",
                    "feature",
                    0,
                    normalized="<p>imported</p>",
                ),
            ),
        )
        index = _manual_index({record.parent_asin: record})
        agent = _tracking_agent(index)
        state = SessionState(
            user_profile={},
            last_asked_attribute="other",
            infer_other_answer_facets=True,
            projection_decisions=[
                {
                    "active": True,
                    "posterior_ids": [record.parent_asin],
                    "recommendation_ids": [],
                }
            ],
        )

        resolved = agent._track_projection_disclosures(
            state,
            "For that, what matters is: <p>Imported</p>.",
            2,
        )

        self.assertTrue(resolved)
        self.assertIn("<p>Imported</p>", state.projection_disclosed_values)
        self.assertEqual(
            state.projection_pending_exact_values[0].raw_value,
            "<p>Imported</p>",
        )

    def test_intent_override_initial_value_is_not_marked_disclosed(self) -> None:
        index = _manual_index({})
        agent = _tracking_agent(index)
        state = SessionState(user_profile={})

        resolved = agent._track_projection_disclosures(
            state,
            "I'm looking for Women Shoes. cotton",
            1,
        )

        self.assertTrue(resolved)
        self.assertTrue(state.projection_override_pending)
        self.assertEqual(state.projection_disclosed_values, set())
        self.assertEqual(state.projection_pending_exact_values, ())

    def test_other_payload_facet_inference_is_opt_in(self) -> None:
        message = "For that, what matters is: cotton."
        frozen = parse_message(
            message,
            2,
            "other",
            infer_other_facets=False,
        )
        projected = parse_message(
            message,
            2,
            "other",
            infer_other_facets=True,
        )

        self.assertEqual(frozen.constraints[-1].facet, Facet.OTHER)
        self.assertEqual(projected.constraints[-1].facet, Facet.MATERIAL)

    def test_override_interrupt_does_not_exhaust_pending_other(self) -> None:
        state = SessionState(
            user_profile={},
            last_asked_attribute="other",
            infer_other_answer_facets=True,
        )

        is_override = state.add_message(
            "Actually, ignore my earlier preference. What I need is: leather.",
            3,
        )

        self.assertTrue(is_override)
        self.assertFalse(state.other_exhausted)
        self.assertEqual(state.intent_epoch, 1)

    def test_reply_partition_uses_exact_raw_disclosure_membership(self) -> None:
        record = _record(
            "P000000001",
            (
                _clue("Leather", "material", 0),
                _clue("Imported", "feature", 1),
            ),
        )

        self.assertEqual(
            ProjectionIndex._reply_signature(record, "other", {"leather"}),
            ("Leather", "Imported"),
        )
        self.assertEqual(
            ProjectionIndex._reply_signature(record, None, set()),
            ("<no-question>",),
        )
        self.assertEqual(
            ProjectionIndex._reply_signature(
                record,
                "material",
                {"Leather"},
            ),
            ("<no-additional>", "material"),
        )

    def test_rollout_conditions_on_a_current_turn_miss(self) -> None:
        record = _record(
            "P000000001",
            (_clue("cotton", "material", 0),),
        )
        index = _manual_index({record.parent_asin: record})
        ranking = ProjectionRanking(
            (record.parent_asin,),
            (record.parent_asin,),
            (record.parent_asin,),
            True,
            {"active": True},
        )

        selected, conditioned = index.choose_question(
            ranking=ranking,
            constraints=(),
            asked_attributes=set(),
            other_exhausted=False,
            turn=2,
            baseline_attribute="material",
        )
        _, unconditioned = index.choose_question(
            ranking=ranking,
            constraints=(),
            asked_attributes=set(),
            other_exhausted=False,
            turn=2,
            baseline_attribute="material",
            condition_on_current_miss=False,
        )

        self.assertIsNone(selected)
        self.assertEqual(
            conditioned["reason"],
            "posterior_exhausted_by_current_display",
        )
        self.assertTrue(conditioned["conditioned_on_current_miss"])
        self.assertEqual(unconditioned["belief_count"], 1)
        self.assertFalse(unconditioned["conditioned_on_current_miss"])

    def test_rollout_preserves_first_turn_guardrail(self) -> None:
        record = _record(
            "P000000001",
            (
                _clue("cotton", "material", 0),
                _clue("color: blue", "color", 1),
            ),
        )
        index = _manual_index({record.parent_asin: record})

        selected, trace = index.choose_question(
            ranking=_active_ranking((record.parent_asin,)),
            constraints=(),
            asked_attributes=set(),
            other_exhausted=False,
            turn=1,
            baseline_attribute="material",
        )

        self.assertIsNone(selected)
        self.assertFalse(trace["active"])
        self.assertEqual(trace["reason"], "rollout_disabled_or_guardrailed")

    def test_session_tracker_preserves_duplicate_and_semicolon_raw_clues(self) -> None:
        raw = "Solids: cotton; heathers: blend"
        record = _record(
            "P000000001",
            (
                _clue(raw, "material", 0),
                _clue(raw, "material", 1),
            ),
        )
        index = _manual_index({record.parent_asin: record})
        agent = _tracking_agent(index)
        state = SessionState(
            user_profile={},
            last_asked_attribute="other",
            infer_other_answer_facets=True,
            projection_decisions=[
                {
                    "active": True,
                    "posterior_ids": [record.parent_asin],
                    "recommendation_ids": [],
                }
            ],
        )
        reply = f"For that, what matters is: {raw}; {raw}."

        self.assertTrue(agent._track_projection_disclosures(state, reply, 3))
        state.add_message(reply, 3)
        raw_values = [
            row["raw_value"] for row in state.projection_ledger.evidence_trace()
        ]

        self.assertEqual(raw_values, [raw, raw])
        self.assertNotIn("Solids: cotton", raw_values)
        self.assertNotIn("heathers: blend", raw_values)

    def test_unrecognized_reply_uses_frozen_other_parsing(self) -> None:
        state = SessionState(
            user_profile={},
            last_asked_attribute="other",
            infer_other_answer_facets=True,
        )

        state.add_message("For that, what matters is: cotton", 2)
        projected = state.projection_ledger.active()

        self.assertFalse(state.projection_template_confident)
        self.assertEqual(projected[-1].facet, Facet.OTHER)
        self.assertEqual(state.ledger.active()[-1].facet, Facet.OTHER)

    def test_whitespace_normalized_no_additional_exhausts_other(self) -> None:
        state = SessionState(
            user_profile={},
            last_asked_attribute="other",
            infer_other_answer_facets=True,
        )

        state.add_message(
            "  I don't   have an additional preference for other.   ",
            2,
        )

        self.assertTrue(state.projection_template_confident)
        self.assertTrue(state.other_exhausted)


if __name__ == "__main__":
    unittest.main()
