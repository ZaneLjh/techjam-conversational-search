from __future__ import annotations

import gzip
import hashlib
import inspect
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import starter.projection as projection_module
from starter.agent import Agent
from starter.constraints import ConstraintLedger, parse_message
from starter.projection import (
    ProjectionConfig,
    ProjectionIndex,
    canonical_projection_line,
    coarse_category,
    normalize_projection_value,
    project_intent_card,
    projected_reply_values,
    projection_fingerprint,
    projection_record,
)
from starter.retrieval import e4_fallback_config
from tools.e4_5_ablation_suite import declared_variants
from tools.e4_5_projection_suite import build_projection_suite


def _write_catalog(path: Path, rows: list[dict]) -> None:
    rendered = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")
    if path.suffix == ".gz":
        with path.open("wb") as output:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=9,
                fileobj=output,
                mtime=0,
            ) as compressed:
                compressed.write(rendered)
    else:
        path.write_bytes(rendered)


def _rows() -> list[dict]:
    return [
        {
            "parent_asin": "P000000001",
            "title": "Blue cotton walking shoe",
            "features": ["100% Cotton", "color: blue", "slim fit", "outdoor use"],
            "details": {"Department": "womens"},
            "categories": ["Clothing, Shoes & Jewelry", "Women", "Shoes"],
            "rating_number": 30,
        },
        {
            "parent_asin": "P000000002",
            "title": "Black leather walking shoe",
            "features": ["100% Leather", "color: black", "regular fit", "outdoor use"],
            "details": {"Department": "womens"},
            "categories": ["Clothing, Shoes & Jewelry", "Women", "Shoes"],
            "rating_number": 20,
        },
        {
            "parent_asin": "P000000003",
            "title": "Blue cotton walking shoe alternate",
            "features": ["cotton", "color: blue", "relaxed fit", "indoor use"],
            "details": {"Department": "womens"},
            "categories": ["Clothing, Shoes & Jewelry", "Women", "Shoes"],
            "rating_number": 10,
        },
    ]


def _canonical_manifest(path: Path, manifest: dict) -> None:
    rendered = (
        json.dumps(
            manifest,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    path.write_bytes(rendered.encode("utf-8"))


def _canonical_gzip(payload: bytes) -> bytes:
    with tempfile.SpooledTemporaryFile() as output:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            compresslevel=9,
            fileobj=output,
            mtime=0,
        ) as compressed:
            compressed.write(payload)
        output.seek(0)
        return output.read()


def _replace_sidecar(
    sidecar: Path,
    manifest_path: Path,
    *,
    compressed: bytes,
    uncompressed: bytes,
) -> None:
    sidecar.write_bytes(compressed)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sidecar"]["compressed_bytes"] = len(compressed)
    manifest["sidecar"]["uncompressed_bytes"] = len(uncompressed)
    manifest["checksums"]["sidecar_gzip_sha256"] = hashlib.sha256(
        compressed
    ).hexdigest()
    manifest["checksums"]["sidecar_jsonl_sha256"] = hashlib.sha256(
        uncompressed
    ).hexdigest()
    _canonical_manifest(manifest_path, manifest)


def _build_small(root: Path) -> tuple[Path, Path, Path, dict]:
    catalog = root / "catalog.jsonl"
    sidecar = root / "projection.jsonl.gz"
    manifest = root / "projection-manifest.json"
    rows = _rows()
    _write_catalog(catalog, rows)
    report = build_projection_suite(
        catalog,
        sidecar,
        manifest,
        expected_row_count=len(rows),
        top_collision_count=3,
    )
    return catalog, sidecar, manifest, report


def _config(sidecar: Path, manifest: Path, **overrides: object) -> ProjectionConfig:
    values: dict[str, object] = {
        "enabled": True,
        "sidecar_path": str(sidecar),
        "manifest_path": str(manifest),
    }
    values.update(overrides)
    return ProjectionConfig(**values)


def _constraints(message: str, *, turn: int = 1) -> tuple:
    ledger = ConstraintLedger()
    ledger.apply(parse_message(message, turn))
    return tuple(ledger.active())


class ProjectionTransformTest(unittest.TestCase):
    def test_card_order_cleanup_and_category_projection_match_declared_rules(self) -> None:
        product = {
            "parent_asin": "P000000010",
            "title": "  Example walking shoe  ",
            "features": [
                "  Zipper   closure. ",
                " Machine Wash ",
                "slim fit",
                "outdoor use",
            ],
            "categories": [
                "Clothing, Shoes & Jewelry",
                "Women",
                "Shoes, Walking",
            ],
        }
        card = project_intent_card(product)

        self.assertEqual(
            card,
            {
                "target_category": "Example walking shoe",
                "hard_constraints": ["Zipper closure", "Machine Wash"],
                "soft_preferences": ["slim fit", "outdoor use"],
            },
        )
        self.assertEqual(
            coarse_category(product["categories"]),
            "Shoes Walking",
        )

    def test_color_stays_in_slot_one_when_no_material_is_detected(self) -> None:
        card = project_intent_card(
            {
                "parent_asin": "P000000011",
                "title": "Red walking shoe",
                "features": ["rubber sole", "machine wash"],
                "categories": ["Women", "Shoes"],
            }
        )

        self.assertEqual(card["hard_constraints"][1], "color: red")
        self.assertEqual(card["hard_constraints"][0], "rubber sole")

    def test_fingerprints_are_normalized_deterministic_and_domain_separated(self) -> None:
        composed = "Ｃotton\u00a0BLUE"
        ordinary = "cotton blue"

        self.assertEqual(
            normalize_projection_value(composed),
            normalize_projection_value(ordinary),
        )
        first = projection_fingerprint(composed, domain="constraint:material")
        self.assertEqual(
            first,
            projection_fingerprint(ordinary, domain="constraint:material"),
        )
        self.assertNotEqual(
            first,
            projection_fingerprint(ordinary, domain="category"),
        )
        self.assertEqual(len(first), 64)

    def test_material_injection_precedes_raw_leather_for_regression_product(self) -> None:
        card = project_intent_card(
            {
                "parent_asin": "P000000012",
                "title": "Leather walking shoe",
                "features": ["100% Leather", "Imported"],
                "categories": ["Women", "Shoes"],
            }
        )

        self.assertEqual(card["hard_constraints"], ["leather", "100% Leather"])
        self.assertEqual(card["soft_preferences"], ["Imported"])

    def test_production_module_has_no_restricted_data_or_harness_dependency(self) -> None:
        source = inspect.getsource(projection_module).lower()

        self.assertNotIn("public_set", source)
        self.assertNotIn("ground_truth", source)
        self.assertNotIn("intent_card(sample", source)
        self.assertNotIn("evaluator.", source)

    def test_projected_reply_values_repeat_other_until_exhaustion(self) -> None:
        card = {
            "hard_constraints": ["cotton", "color: blue"],
            "soft_preferences": ["slim fit", "outdoor use"],
        }
        disclosed: set[str] = set()

        first = projected_reply_values(card, "other", disclosed)
        disclosed.update(first)
        second = projected_reply_values(card, "other", disclosed)
        disclosed.update(second)

        self.assertEqual(first, ("cotton", "color: blue"))
        self.assertEqual(second, ("slim fit", "outdoor use"))
        self.assertEqual(projected_reply_values(card, "other", disclosed), ())

    def test_projection_record_preserves_exact_strings_with_parallel_fingerprints(self) -> None:
        product = {
            "parent_asin": "P000000013",
            "title": "Women's Café Shoe",
            "features": ["100% Cotton", "color: blue", "Machine Wash"],
            "categories": ["Women", "Shoes"],
        }
        record = projection_record(product)
        card = record["intent_card"]
        projection = record["projection"]

        self.assertEqual(record["parent_asin"], product["parent_asin"])
        self.assertEqual(card["target_category"], product["title"])
        self.assertEqual(card["hard_constraints"], ["cotton", "color: blue"])
        self.assertEqual(len(projection["hard_constraints"]), 2)
        self.assertEqual(
            canonical_projection_line(record),
            canonical_projection_line(projection_record(deepcopy(product))),
        )
        for raw, projected in zip(
            card["hard_constraints"],
            projection["hard_constraints"],
        ):
            self.assertEqual(
                projected["fingerprint"],
                projection_fingerprint(
                    raw,
                    domain=f"constraint:{projected['facet']}",
                ),
            )


class ProjectionSuiteTest(unittest.TestCase):
    def test_ablation_matrix_separates_ranking_rollout_and_fallback(self) -> None:
        variants = {
            variant.name: variant.config
            for variant in declared_variants("sidecar.gz", "manifest.json")
        }

        self.assertEqual(
            set(variants),
            {
                "full",
                "frozen_e4",
                "projection_ranking_only",
                "question_rollout_only",
                "rerank_posterior_at_most_10",
            },
        )
        self.assertTrue(variants["full"].enabled)
        self.assertFalse(variants["frozen_e4"].enabled)
        self.assertFalse(variants["projection_ranking_only"].use_question_rollout)
        self.assertFalse(variants["question_rollout_only"].use_reranking)
        self.assertEqual(
            variants["rerank_posterior_at_most_10"].max_rerank_posterior_size,
            10,
        )

    def test_default_row_count_gate_prevents_partial_catalog_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            catalog = root / "catalog.jsonl"
            sidecar = root / "projection.jsonl.gz"
            manifest = root / "projection-manifest.json"
            _write_catalog(catalog, _rows())

            with self.assertRaisesRegex(ValueError, "expected 50000 catalog rows"):
                build_projection_suite(catalog, sidecar, manifest)

            self.assertFalse(sidecar.exists())
            self.assertFalse(manifest.exists())

    def test_sidecar_builder_is_byte_deterministic_and_parity_audited(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            catalog = root / "catalog.jsonl"
            sidecar_one = root / "one.jsonl.gz"
            sidecar_two = root / "two.jsonl.gz"
            manifest_one = root / "one.json"
            manifest_two = root / "two.json"
            rows = _rows()
            _write_catalog(catalog, rows)

            first = build_projection_suite(
                catalog,
                sidecar_one,
                manifest_one,
                expected_row_count=len(rows),
            )
            second = build_projection_suite(
                catalog,
                sidecar_two,
                manifest_two,
                expected_row_count=len(rows),
            )

            self.assertEqual(sidecar_one.read_bytes(), sidecar_two.read_bytes())
            self.assertEqual(first, second)
            self.assertEqual(manifest_one.read_bytes(), manifest_two.read_bytes())
            self.assertTrue(first["passed"])
            self.assertEqual(first["parity"]["status"], "pass")
            self.assertEqual(first["parity"]["rollout_mismatches"], 0)
            self.assertEqual(first["collisions"]["true_sha256_collision_count"], 0)


class ProjectionRuntimeTest(unittest.TestCase):
    def test_duplicate_manifest_key_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog, sidecar, manifest, _ = _build_small(Path(temp_dir))
            raw = manifest.read_bytes()
            manifest.write_bytes(
                raw.replace(
                    b"{\n",
                    b'{\n  "schema_version": 1,\n',
                    1,
                )
            )

            index = ProjectionIndex(catalog, _config(sidecar, manifest))

            self.assertFalse(index.ready)
            self.assertEqual(index.records, {})
            self.assertTrue(index.status_reason.startswith("invalid_sidecar:"))

    def test_manifest_metadata_mutations_all_fail_closed(self) -> None:
        mutations = {
            "schema": lambda value: value.__setitem__("schema_version", 2),
            "passed": lambda value: value.__setitem__("passed", False),
            "algorithm": lambda value: value["projection"].__setitem__(
                "algorithm_version", "unknown"
            ),
            "parity": lambda value: value["parity"].__setitem__("status", "fail"),
            "gzip_mtime": lambda value: value["sidecar"].__setitem__("gzip_mtime", 1),
            "collision": lambda value: value["collisions"].__setitem__(
                "true_sha256_collision_count", 1
            ),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog, sidecar, manifest, report = _build_small(Path(temp_dir))
            for name, mutate in mutations.items():
                with self.subTest(name=name):
                    changed = deepcopy(report)
                    mutate(changed)
                    _canonical_manifest(manifest, changed)
                    index = ProjectionIndex(catalog, _config(sidecar, manifest))
                    self.assertFalse(index.ready)
                    self.assertEqual(index.records, {})

    def test_noncanonical_or_multimember_gzip_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            catalog, sidecar, manifest, report = _build_small(root)
            raw_jsonl = gzip.decompress(sidecar.read_bytes())

            noncanonical = root / "noncanonical.gz"
            with noncanonical.open("wb") as output:
                with gzip.GzipFile(
                    filename="",
                    mode="wb",
                    compresslevel=9,
                    fileobj=output,
                    mtime=1,
                ) as compressed:
                    compressed.write(raw_jsonl)
            changed = deepcopy(report)
            changed["sidecar"]["compressed_bytes"] = noncanonical.stat().st_size
            changed["checksums"]["sidecar_gzip_sha256"] = hashlib.sha256(
                noncanonical.read_bytes()
            ).hexdigest()
            _canonical_manifest(manifest, changed)
            self.assertFalse(
                ProjectionIndex(catalog, _config(noncanonical, manifest)).ready
            )

            _canonical_manifest(manifest, report)
            canonical_member = sidecar.read_bytes()
            multimember = canonical_member + canonical_member
            _replace_sidecar(
                sidecar,
                manifest,
                compressed=multimember,
                uncompressed=raw_jsonl + raw_jsonl,
            )
            self.assertFalse(ProjectionIndex(catalog, _config(sidecar, manifest)).ready)

    def test_sidecar_rows_must_reproduce_the_bound_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog, sidecar, manifest, _ = _build_small(Path(temp_dir))
            lines = gzip.decompress(sidecar.read_bytes()).splitlines(keepends=True)
            lines[0], lines[1] = lines[1], lines[0]
            altered = b"".join(lines)
            _replace_sidecar(
                sidecar,
                manifest,
                compressed=_canonical_gzip(altered),
                uncompressed=altered,
            )

            index = ProjectionIndex(catalog, _config(sidecar, manifest))

            self.assertFalse(index.ready)
            self.assertEqual(index.records, {})

    def test_corrupt_sidecar_and_unrecognized_template_are_exact_fallbacks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            catalog, sidecar, manifest, _ = _build_small(root)
            sidecar.write_bytes(b"not-a-gzip-stream")
            corrupt = ProjectionIndex(catalog, _config(sidecar, manifest))
            self.assertFalse(corrupt.ready)
            self.assertEqual(corrupt.records, {})

            # Rebuild and verify that arbitrary customer text cannot activate
            # the projection route even when the artifact itself is valid.
            catalog, sidecar, manifest, _ = _build_small(root)
            valid = ProjectionIndex(catalog, _config(sidecar, manifest))
            self.assertTrue(valid.ready)
            if not hasattr(valid, "rerank"):
                self.fail("ProjectionIndex.rerank is required by the E4.5 interface")
            recommendation_ids = tuple(row["parent_asin"] for row in _rows()[:2])
            candidate_ids = tuple(row["parent_asin"] for row in _rows())
            ranking = valid.rerank(
                recommendation_ids=recommendation_ids,
                candidate_ids=candidate_ids,
                constraints=_constraints(
                    "I'm looking for Women Shoes. A key requirement is: cotton."
                ),
                shown_ids=set(),
                requested_k=2,
                template_confident=False,
            )
            self.assertFalse(ranking.active)
            self.assertEqual(ranking.recommendation_ids, recommendation_ids)
            self.assertEqual(ranking.candidate_ids, candidate_ids)

    def test_valid_projection_activates_with_bounded_display_first_pool(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog, sidecar, manifest, _ = _build_small(Path(temp_dir))
            index = ProjectionIndex(catalog, _config(sidecar, manifest))
            self.assertTrue(index.ready, index.status_reason)
            recommendation_ids = ("P000000002", "P000000001")
            candidate_ids = (
                "P000000003",
                "P000000001",
                "P000000002",
            )
            ranking = index.rerank(
                recommendation_ids=recommendation_ids,
                candidate_ids=candidate_ids,
                constraints=_constraints(
                    "I'm looking for Women Shoes. A key requirement is: cotton."
                ),
                shown_ids=set(),
                requested_k=2,
                template_confident=True,
            )

            self.assertLessEqual(len(ranking.candidate_ids), 100)
            self.assertEqual(len(ranking.candidate_ids), len(set(ranking.candidate_ids)))
            self.assertEqual(ranking.candidate_ids[:2], recommendation_ids)
            self.assertTrue(
                set(ranking.recommendation_ids).issubset(ranking.candidate_ids)
            )
            self.assertIsNone(ranking.trace["fallback_reason"])

    def test_large_category_clue_collision_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog, sidecar, manifest, _ = _build_small(Path(temp_dir))
            index = ProjectionIndex(
                catalog,
                _config(sidecar, manifest, max_posterior_size=1),
            )
            ranking = index.rerank(
                recommendation_ids=("P000000001",),
                candidate_ids=("P000000001", "P000000002", "P000000003"),
                constraints=_constraints(
                    "I'm looking for Women Shoes. A key requirement is: cotton."
                ),
                shown_ids=set(),
                requested_k=1,
                template_confident=True,
            )

            self.assertFalse(ranking.active)
            self.assertEqual(ranking.recommendation_ids, ("P000000001",))
            self.assertEqual(
                ranking.trace["fallback_reason"],
                "oversized_category_clue_intersection",
            )

    def test_bounded_union_scores_before_cap_and_keeps_predecessor_ties(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog, sidecar, manifest, _ = _build_small(Path(temp_dir))
            index = ProjectionIndex(
                catalog,
                _config(
                    sidecar,
                    manifest,
                    candidate_depth=10,
                    max_rerank_posterior_size=3,
                ),
            )
            predecessor = ("P000000002", "P000000003", "P000000001")
            ranking = index.rerank(
                recommendation_ids=predecessor[:2],
                candidate_ids=predecessor,
                constraints=_constraints(
                    "I'm looking for Women Shoes. A key requirement is: cotton."
                ),
                shown_ids=set(),
                requested_k=2,
                template_confident=True,
            )

            self.assertEqual(len(ranking.candidate_ids), len(set(ranking.candidate_ids)))
            self.assertLessEqual(len(ranking.candidate_ids), 10)
            self.assertTrue(ranking.trace["ranking_applied"])
            self.assertEqual(
                ranking.recommendation_ids,
                ("P000000003", "P000000001"),
            )

    def test_question_runtime_error_rolls_back_the_complete_turn(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog, sidecar, manifest, _ = _build_small(Path(temp_dir))
            agent = Agent(
                catalog,
                retrieval_config=e4_fallback_config(),
                projection_config=_config(sidecar, manifest),
            )
            agent.reset("rollback", {})
            state = agent._sessions["rollback"]
            before = deepcopy(state)

            original = agent.projection_index.choose_question

            def fail_question(**_: object) -> tuple[str | None, dict]:
                raise RuntimeError("synthetic rollout failure")

            agent.projection_index.choose_question = fail_question  # type: ignore[method-assign]
            try:
                response = agent.respond(
                    "rollback",
                    "I'm looking for Women Shoes. A key requirement is: cotton.",
                    1,
                    2,
                )
            finally:
                agent.projection_index.choose_question = original  # type: ignore[method-assign]

            after = agent._sessions["rollback"]
            self.assertEqual(response["ask_attribute"], "material")
            self.assertEqual(after.intent_epoch, before.intent_epoch)
            self.assertFalse(after.projection_question_decisions[-1]["active"])
            self.assertEqual(
                after.projection_question_decisions[-1]["reason"],
                "runtime_error:RuntimeError",
            )
            self.assertEqual(
                after.projection_decisions[-1]["fallback_reason"],
                "question_runtime_error:RuntimeError",
            )


if __name__ == "__main__":
    unittest.main()
