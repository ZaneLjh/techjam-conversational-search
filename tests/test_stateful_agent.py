from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
