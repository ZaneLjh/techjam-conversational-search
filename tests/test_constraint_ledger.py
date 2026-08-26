from __future__ import annotations

import unittest

from starter.constraints import (
    ConstraintLedger,
    ConstraintStatus,
    Facet,
    Polarity,
    Strength,
    infer_facet,
    parse_message,
)


class ConstraintLedgerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ledger = ConstraintLedger()

    def apply(self, message: str, turn: int, expected: str | None = None) -> None:
        self.ledger.apply(parse_message(message, turn, expected))

    def test_accumulates_nonconflicting_constraints(self) -> None:
        self.apply("I'm looking for jackets, but I'm still exploring.", 1)
        self.apply("For that, what matters is: waterproof.", 2, "feature")
        self.apply("For that, what matters is: blue.", 3, "color")

        query = self.ledger.canonical_query()
        self.assertIn("jackets", query)
        self.assertIn("waterproof", query)
        self.assertIn("blue", query)

    def test_boundary_no_preference_is_explicit_but_not_retrieved(self) -> None:
        self.apply("I'm looking for jackets, but I'm still exploring.", 1)
        self.apply(
            "I don't have a preference for material; please use your judgment.",
            2,
            "material",
        )

        query = self.ledger.canonical_query()
        self.assertEqual(query, "jackets")
        active = self.ledger.active()
        self.assertTrue(
            any(
                item.facet is Facet.MATERIAL
                and item.strength is Strength.NO_PREFERENCE
                and item.polarity is Polarity.NEUTRAL
                for item in active
            )
        )

    def test_later_positive_reply_reactivates_declined_facet(self) -> None:
        self.apply("I'm looking for jackets, but I'm still exploring.", 1)
        self.apply("I don't have an additional preference for material.", 2, "material")
        self.apply("For that, what matters is: cotton.", 3, "material")

        self.assertIn("cotton", self.ledger.canonical_query())
        no_preference = [
            item for item in self.ledger.entries if item.strength is Strength.NO_PREFERENCE
        ]
        self.assertEqual(no_preference[-1].status, ConstraintStatus.SUPERSEDED)

    def test_no_additional_preference_preserves_known_value(self) -> None:
        self.apply("I'm looking for jackets. A key requirement is: cotton.", 1)
        self.apply("I don't have an additional preference for material.", 2, "material")

        self.assertIn("cotton", self.ledger.canonical_query())
        active_material = [
            item for item in self.ledger.active() if item.facet is Facet.MATERIAL
        ]
        self.assertTrue(
            any(item.normalized_value == "cotton" for item in active_material)
        )
        self.assertTrue(
            any(item.strength is Strength.NO_PREFERENCE for item in active_material)
        )

    def test_override_cue_inside_product_evidence_is_not_a_correction(self) -> None:
        update = parse_message(
            "For that, what matters is: a lining that actually keeps feet warm.",
            2,
            "feature",
        )

        self.assertFalse(update.is_override)
        self.assertEqual(update.constraints[-1].facet, Facet.FEATURE)

    def test_generic_rejection_does_not_pollute_query(self) -> None:
        self.apply("I'm looking for jackets, but I'm still exploring.", 1)
        before = self.ledger.canonical_query()
        self.apply(
            "Those options are not quite right yet. Ask me about one specific attribute.",
            2,
        )

        self.assertEqual(self.ledger.canonical_query(), before)

    def test_negation_words_inside_catalog_evidence_remain_positive(self) -> None:
        update = parse_message(
            "For that, what matters is: a clasp without sharp edges.",
            2,
            "feature",
        )

        self.assertEqual(len(update.constraints), 1)
        self.assertEqual(update.constraints[0].polarity, Polarity.POSITIVE)
        self.assertEqual(
            update.constraints[0].normalized_value,
            "a clasp without sharp edges",
        )

    def test_replacement_words_inside_catalog_evidence_are_not_an_override(self) -> None:
        update = parse_message(
            "For that, what matters is: stainless steel instead of alloy.",
            2,
            "feature",
        )

        self.assertFalse(update.is_override)
        self.assertEqual(len(update.constraints), 1)
        self.assertEqual(
            update.constraints[0].normalized_value,
            "stainless steel instead of alloy",
        )

    def test_override_payload_with_negation_words_remains_positive(self) -> None:
        update = parse_message(
            "Actually, ignore my earlier preference. "
            "What I need is: a coating that will not fade or tarnish.",
            3,
            "material",
        )

        self.assertTrue(update.is_override)
        self.assertTrue(update.retract_initial_preference)
        self.assertEqual(len(update.constraints), 1)
        self.assertEqual(update.constraints[0].polarity, Polarity.POSITIVE)
        self.assertEqual(
            update.constraints[0].normalized_value,
            "a coating that will not fade or tarnish",
        )

    def test_direct_no_and_dislike_phrases_are_avoid_constraints(self) -> None:
        for message in ("No leather, please.", "I don't like leather."):
            update = parse_message(message, 2)
            self.assertEqual(update.constraints[-1].strength, Strength.AVOID)
            self.assertEqual(update.constraints[-1].normalized_value, "leather")

    def test_catalog_no_closure_detail_is_not_a_customer_exclusion(self) -> None:
        update = parse_message("I'm looking for socks. No Closure closure", 1)

        self.assertEqual(len(update.constraints), 1)
        self.assertEqual(update.constraints[0].facet, Facet.CATEGORY)

    def test_curly_apostrophe_no_preference_is_recognized(self) -> None:
        update = parse_message(
            "I don’t have an additional preference for color.",
            2,
            "color",
        )

        self.assertEqual(update.constraints[-1].strength, Strength.NO_PREFERENCE)
        self.assertEqual(update.constraints[-1].facet, Facet.COLOR)

    def test_rejection_paraphrase_does_not_pollute_query(self) -> None:
        self.apply("I'm looking for jackets, but I'm still exploring.", 1)
        before = self.ledger.canonical_query()
        self.apply("The options are not quite right.", 2)

        self.assertEqual(self.ledger.canonical_query(), before)

    def test_changed_mind_retracts_initial_preference_across_facets(self) -> None:
        self.apply("I'm looking for jackets. I prefer leather.", 1)
        self.apply("I changed my mind. I want blue.", 2)

        query = self.ledger.canonical_query()
        self.assertIn("jackets", query)
        self.assertIn("blue", query)
        self.assertNotIn("leather", query)

    def test_first_turn_need_preserves_product_noun_as_category(self) -> None:
        self.apply("I need running shoes.", 1)
        self.apply("I changed my mind. I want waterproof.", 2)

        active_categories = [
            item.normalized_value
            for item in self.ledger.active()
            if item.facet is Facet.CATEGORY
        ]
        self.assertEqual(active_categories, ["shoes"])
        self.assertIn("shoes", self.ledger.canonical_query())

    def test_explicit_material_prefix_and_budget_context(self) -> None:
        self.assertEqual(infer_facet("Material: alloy"), Facet.MATERIAL)
        self.assertEqual(infer_facet("Under Armour running top"), Facet.USE_CASE)
        self.assertEqual(infer_facet("fits around the wrist"), Facet.FEATURE)
        self.assertEqual(infer_facet("under $50"), Facet.BUDGET)

    def test_override_supersedes_only_initial_preference(self) -> None:
        self.apply("I'm looking for jackets. I prefer red.", 1)
        self.apply("For that, what matters is: waterproof.", 2, "feature")
        self.apply(
            "Actually, ignore my earlier preference. What I need is: blue.",
            3,
            "material",
        )

        query = self.ledger.canonical_query()
        self.assertIn("jackets", query)
        self.assertIn("waterproof", query)
        self.assertIn("blue", query)
        self.assertNotIn("red", query)
        red = next(item for item in self.ledger.entries if item.normalized_value == "red")
        blue = next(item for item in self.ledger.entries if item.normalized_value == "blue")
        self.assertEqual(red.status, ConstraintStatus.SUPERSEDED)
        self.assertIn(red.constraint_id, blue.supersedes)

    def test_override_removes_echoed_old_value_but_keeps_other_clause(self) -> None:
        self.apply("I'm looking for jackets. Zipper closure.", 1)
        self.apply(
            "For that, what matters is: Imported; Zipper closure.",
            2,
            "feature",
        )
        self.apply(
            "Actually, ignore my earlier preference. What I need is: polyester.",
            3,
        )

        query = self.ledger.canonical_query()
        self.assertIn("jackets", query)
        self.assertIn("imported", query)
        self.assertIn("polyester", query)
        self.assertNotIn("zipper", query)

    def test_explicit_same_facet_replacement_keeps_audit_history(self) -> None:
        self.apply("I'm looking for jackets. I prefer red.", 1)
        self.apply("Actually, blue instead of red.", 2)

        query = self.ledger.canonical_query()
        self.assertIn("blue", query)
        self.assertNotIn("red", query)
        history = {item.normalized_value: item for item in self.ledger.entries}
        self.assertEqual(history["red"].status, ConstraintStatus.SUPERSEDED)
        self.assertEqual(history["blue"].status, ConstraintStatus.ACTIVE)

    def test_negation_never_becomes_a_positive_query_term(self) -> None:
        self.apply("I want leather.", 1)
        self.apply("I want cotton, not leather.", 2)

        query = self.ledger.canonical_query()
        self.assertIn("cotton", query)
        self.assertNotIn("leather", query)
        leather_entries = [
            item for item in self.ledger.entries if item.normalized_value == "leather"
        ]
        self.assertTrue(any(item.status is ConstraintStatus.NEGATED for item in leather_entries))
        self.assertTrue(
            any(
                item.status is ConstraintStatus.ACTIVE
                and item.strength is Strength.AVOID
                and item.polarity is Polarity.NEGATIVE
                for item in leather_entries
            )
        )

    def test_generic_correction_makes_latest_same_facet_value_active(self) -> None:
        self.apply("I'm looking for jackets. I prefer red.", 1)
        self.apply("Actually, I want blue.", 2)

        active_colors = [
            item.normalized_value
            for item in self.ledger.active()
            if item.facet is Facet.COLOR and item.polarity is Polarity.POSITIVE
        ]
        self.assertEqual(active_colors, ["blue"])

    def test_prefix_form_instead_of_retracts_old_value(self) -> None:
        self.apply("I'm looking for jackets. I prefer red.", 1)
        self.apply("Instead of red, I want blue.", 2)

        query = self.ledger.canonical_query()
        self.assertIn("blue", query)
        self.assertNotIn("red", query)

    def test_prefix_replacement_without_punctuation_retracts_multiple_values(self) -> None:
        self.apply("I'm looking for jackets. I prefer red.", 1)
        self.apply("For that, what matters is: green.", 2, "color")
        self.apply("Instead of red and green I want blue.", 3)

        query = self.ledger.canonical_query()
        self.assertIn("blue", query)
        self.assertNotIn("red", query)
        self.assertNotIn("green", query)

    def test_no_preference_clause_keeps_following_requirement(self) -> None:
        self.apply("I'm looking for jackets.", 1)
        self.apply("No preference for color, but it must be waterproof.", 2, "color")

        query = self.ledger.canonical_query()
        self.assertIn("waterproof", query)
        self.assertTrue(
            any(
                item.facet is Facet.COLOR
                and item.strength is Strength.NO_PREFERENCE
                for item in self.ledger.active()
            )
        )

    def test_multi_facet_boundary_keeps_budget_after_clause(self) -> None:
        self.apply("I'm looking for jackets.", 1)
        self.apply(
            "No preference for color or material, but under $50.",
            2,
        )

        active_no_preference_facets = {
            item.facet
            for item in self.ledger.active()
            if item.strength is Strength.NO_PREFERENCE
        }
        self.assertEqual(
            active_no_preference_facets,
            {Facet.COLOR, Facet.MATERIAL},
        )
        self.assertIn("under $50", self.ledger.canonical_query())

    def test_period_delimited_boundary_keeps_following_requirement(self) -> None:
        self.apply("I'm looking for jackets.", 1)
        self.apply("No preference for color. It must be waterproof.", 2, "color")

        self.assertIn("waterproof", self.ledger.canonical_query())

    def test_colonless_structured_payload_drops_scaffolding(self) -> None:
        update = parse_message("What I need is breathable cotton.", 2, "feature")

        self.assertEqual(len(update.constraints), 1)
        self.assertEqual(update.constraints[0].normalized_value, "breathable cotton")

    def test_need_form_ignores_catalog_no_closure_schema_artifact(self) -> None:
        update = parse_message("I need socks. No Closure closure", 1)

        self.assertEqual(len(update.constraints), 1)
        self.assertEqual(update.constraints[0].facet, Facet.CATEGORY)

    def test_override_supersedes_an_initial_avoid_constraint(self) -> None:
        self.apply("I'm looking for shoes. No leather or suede.", 1)
        self.apply("Actually, ignore my earlier preference. I want cotton.", 2)

        self.assertEqual(self.ledger.active_avoid_values(), [])
        self.assertIn("cotton", self.ledger.canonical_query())


if __name__ == "__main__":
    unittest.main()
