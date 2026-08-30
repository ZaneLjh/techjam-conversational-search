from __future__ import annotations

import inspect
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import starter.agent as agent_module
import starter.reranking as reranking_module
from starter.agent import Agent
from starter.constraints import ConstraintLedger, parse_message
from starter.projection import ProjectionRanking
from starter.reranking import (
    RerankingConfig,
    RerankingResult,
    SemanticConstraintReranker,
)
from starter.retrieval import e4_fallback_config


def _constraints(message: str) -> tuple:
    ledger = ConstraintLedger()
    ledger.apply(parse_message(message, 1))
    return tuple(ledger.entries)


def _memory_index() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE products("
        "parent_asin TEXT PRIMARY KEY, title TEXT, categories TEXT, features TEXT, "
        "details TEXT, store TEXT, description TEXT)"
    )
    connection.execute(
        "CREATE TABLE retrieval_meta("
        "parent_asin TEXT PRIMARY KEY, coarse_norm TEXT, "
        "average_rating REAL, rating_number INTEGER)"
    )
    connection.execute(
        "CREATE TABLE retrieval_facet_values("
        "parent_asin TEXT, facet TEXT, lookup_norm TEXT)"
    )
    rows = [
        ("A", "plain walking shoe", "Women Shoes", "rubber sole", "", "S", ""),
        ("B", "blue cotton walking shoe", "Women Shoes", "cotton blue", "", "S", ""),
        ("C", "cotton walking shoe", "Women Shoes", "cotton", "", "S", ""),
        ("D", "candidate only cotton shoe", "Women Shoes", "cotton", "", "S", ""),
        ("E", "leather walking shoe", "Women Shoes", "leather", "", "S", ""),
    ]
    connection.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
    connection.executemany(
        "INSERT INTO retrieval_meta VALUES (?, '', ?, ?)",
        [
            ("A", 5.0, 10000),
            ("B", 1.0, 1),
            ("C", 2.0, 2),
            ("D", 5.0, 99999),
            ("E", 5.0, 999999),
        ],
    )
    connection.executemany(
        "INSERT INTO retrieval_facet_values VALUES (?, ?, ?)",
        [("B", "material", "cotton"), ("C", "material", "cotton"), ("D", "material", "cotton"), ("E", "material", "leather")],
    )
    connection.commit()
    return connection


class GuardedSemanticRerankerTest(unittest.TestCase):
    def test_candidate_only_product_cannot_enter_and_lock_stays_first(self) -> None:
        reranker = SemanticConstraintReranker(
            _memory_index(),
            RerankingConfig(enabled=True),
        )

        result = reranker.rerank(
            ("A", "B", "C"),
            ("A", "B", "C", "D"),
            _constraints(
                "I'm looking for Women Shoes. A key requirement is: cotton."
            ),
            locked_ids=("B",),
            requested_k=3,
            turn=1,
        )

        self.assertEqual(result.recommendation_ids[0], "B")
        self.assertEqual(set(result.recommendation_ids), {"A", "B", "C"})
        self.assertNotIn("D", result.recommendation_ids)
        self.assertTrue(result.trace["invariant"]["passed"])

    def test_quality_is_off_by_default_and_rating_magnitude_has_no_effect(self) -> None:
        connection = _memory_index()
        reranker = SemanticConstraintReranker(
            connection,
            RerankingConfig(
                enabled=True,
                use_exact_priority=False,
                use_fuzzy_similarity=False,
                use_candidate_idf=False,
            ),
        )
        constraints = _constraints(
            "I'm looking for Women Shoes. A key requirement is: cotton."
        )
        first = reranker.rerank(
            ("B", "C"), ("B", "C"), constraints, requested_k=2, turn=1
        )
        connection.execute(
            "UPDATE retrieval_meta SET average_rating=5.0, rating_number=999999 "
            "WHERE parent_asin='C'"
        )
        second = reranker.rerank(
            ("B", "C"), ("B", "C"), constraints, requested_k=2, turn=1
        )

        self.assertFalse(reranker.config.use_quality_tiebreak)
        self.assertEqual(first.recommendation_ids, second.recommendation_ids)

    def test_active_avoid_is_an_exact_bypass(self) -> None:
        reranker = SemanticConstraintReranker(
            _memory_index(), RerankingConfig(enabled=True)
        )
        before = ("A", "B", "C")
        result = reranker.rerank(
            before,
            (*before, "D"),
            _constraints("I'm looking for Women Shoes, but no cotton."),
            requested_k=3,
            turn=1,
            avoid_values=("cotton",),
        )

        self.assertEqual(result.recommendation_ids, before)
        self.assertEqual(result.trace["reason"], "active_avoid")

    def test_confirmed_must_mismatch_stays_below_unknown(self) -> None:
        reranker = SemanticConstraintReranker(
            _memory_index(), RerankingConfig(enabled=True)
        )
        result = reranker.rerank(
            ("E", "A", "B"),
            ("E", "A", "B"),
            _constraints(
                "I'm looking for Women Shoes. A key requirement is: cotton."
            ),
            requested_k=3,
            turn=1,
        )

        self.assertEqual(result.recommendation_ids[-1], "E")
        tiers = {
            row["parent_asin"]: row["compatibility_tier"]
            for row in result.trace["top_candidates"]
        }
        self.assertEqual(tiers["A"], "unknown")
        self.assertEqual(tiers["E"], "confirmed_mismatch")

    def test_production_modules_have_no_label_or_evaluator_dependency(self) -> None:
        source = (inspect.getsource(agent_module) + inspect.getsource(reranking_module)).lower()
        self.assertNotIn("ground_truth", source)
        self.assertNotIn("public_set", source)
        self.assertNotIn("evaluator.", source)


class AgentAdmissionGateTest(unittest.TestCase):
    @staticmethod
    def _catalog(path: Path, count: int = 14) -> None:
        rows = []
        for index in range(count):
            rows.append(
                {
                    "parent_asin": f"P{index:03d}",
                    "title": f"cotton walking shoe model {index}",
                    "features": ["cotton", f"feature {index}"],
                    "details": {"Department": "women"},
                    "categories": ["Clothing, Shoes & Jewelry", "Women", "Shoes"],
                    "average_rating": 4.0,
                    "rating_number": index + 1,
                }
            )
        path.write_bytes(
            "".join(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                for row in rows
            ).encode("utf-8")
        )

    def _agent_and_predecessor(self, catalog: Path) -> tuple[Agent, list[str], list[str]]:
        agent = Agent(
            catalog,
            retrieval_config=e4_fallback_config(),
            reranking_config=RerankingConfig(
                enabled=True,
                enforce_projection_candidate_membership=True,
            ),
        )
        agent.reset("probe", {})
        agent.respond(
            "probe",
            "I'm looking for Women Shoes. A key requirement is: cotton.",
            1,
            3,
        )
        trace = agent.evidence_trace("probe")["retrieval_decisions"][-1]
        return agent, list(trace["recommendation_ids"]), list(trace["candidate_ids"])

    def test_unique_projection_outside_predecessor_top100_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog = Path(temp_dir) / "catalog.jsonl"
            self._catalog(catalog)
            agent, _, _ = self._agent_and_predecessor(catalog)
            agent.reset("outside", {})

            def outside_projection(**kwargs: object) -> ProjectionRanking:
                recommendations = tuple(kwargs["recommendation_ids"])  # type: ignore[index]
                candidates = tuple(kwargs["candidate_ids"])  # type: ignore[index]
                return ProjectionRanking(
                    ("NOT_IN_E4_POOL", *recommendations[:-1]),
                    ("NOT_IN_E4_POOL", *candidates),
                    ("NOT_IN_E4_POOL",),
                    True,
                    {"active": True, "ranking_applied": True},
                )

            agent.projection_index.rerank = outside_projection  # type: ignore[method-assign]
            response = agent.respond(
                "outside",
                "I'm looking for Women Shoes. A key requirement is: cotton.",
                1,
                3,
            )
            trace = agent.evidence_trace("outside")

            self.assertNotIn(
                "NOT_IN_E4_POOL",
                [item["parent_asin"] for item in response["recommendations"]],
            )
            self.assertEqual(
                trace["projection_decisions"][-1]["fallback_reason"],
                "unique_outside_predecessor_pool",
            )

    def test_semantic_runtime_error_keeps_post_projection_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog = Path(temp_dir) / "catalog.jsonl"
            self._catalog(catalog)
            agent, _, _ = self._agent_and_predecessor(catalog)
            agent.reset("failure", {})

            original = agent.reranker.rerank

            def fail_semantic(*_: object, **__: object) -> RerankingResult:
                raise RuntimeError("synthetic semantic failure")

            agent.reranker.rerank = fail_semantic  # type: ignore[method-assign]
            try:
                response = agent.respond(
                    "failure",
                    "I'm looking for Women Shoes. A key requirement is: cotton.",
                    1,
                    3,
                )
            finally:
                agent.reranker.rerank = original  # type: ignore[method-assign]

            trace = agent.evidence_trace("failure")
            self.assertEqual(len(response["recommendations"]), 3)
            self.assertEqual(
                trace["reranking_decisions"][-1]["reason"],
                "runtime_error:RuntimeError",
            )


if __name__ == "__main__":
    unittest.main()
