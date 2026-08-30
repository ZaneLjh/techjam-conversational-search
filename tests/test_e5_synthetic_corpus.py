from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path

from evaluator.local_evaluator import behavior_for, intent_card, materialize_hidden_fields
from tools.e5_synthetic_corpus import build_groups, generate


def _product(index: int, *, title: str | None = None, store: str | None = None) -> dict:
    return {
        "average_rating": 4.0 + (index % 9) / 10,
        "categories": ["Clothing, Shoes & Jewelry", "Women", "Shirts"],
        "description": [f"A deliberately detailed description for synthetic item {index} with comfortable fabric and durable construction."],
        "details": {"Item model number": f"MODEL-{index}", "Manufacturer": store or f"Maker {index}"},
        "features": ["Cotton", f"feature number {index}", "Machine washable"],
        "parent_asin": f"ASIN{index:04d}",
        "price": 20.0 + index,
        "rating_number": 10 + index,
        "store": store or f"Brand {index}",
        "title": title or f"Distinctive Synthetic Shirt Alpha Item {index}",
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


class E5SyntheticCorpusTests(unittest.TestCase):
    def _inputs(self, root: Path, *, shuffled: bool = False) -> tuple[Path, Path, list[dict]]:
        products = [_product(index) for index in range(100)]
        products[1] = _product(1, title="Public Variant Shirt Black XL", store="Variant Brand")
        products[2] = _product(2, title="Public Variant Shirt Blue Small", store="Variant Brand")
        # Keep the same leaf/store and a Jaccard-1.0 title after conservative variant stripping.
        products[1]["details"]["Item model number"] = "PUBLIC-BLACK"
        products[2]["details"]["Item model number"] = "PUBLIC-BLUE"
        if shuffled:
            products = list(reversed(products))
        catalog = root / "catalog.jsonl"
        public = root / "public.jsonl"
        _write_jsonl(catalog, products)
        _write_jsonl(public, [{
            "ground_truth": {"parent_asin": "ASIN0001"},
            "sample_id": "public_1",
            "scenario_type": "buying",
            "user_profile": {},
        }])
        return catalog, public, products

    def test_exact_ratios_determinism_quarantine_and_disjoint_folds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog, public, _ = self._inputs(root)
            first = root / "first"
            second = root / "second"
            generate(catalog, public, first, folds=2, seeds=(1701, 1702), sessions_per_fold=20)
            generate(catalog, public, second, folds=2, seeds=(1701, 1702), sessions_per_fold=20)

            for name in (
                "e5_synthetic_sessions.jsonl",
                "e5_synthetic_group_map.jsonl",
                "e5_synthetic_leakage_audit.json",
                "e5_synthetic_manifest.json",
                "SHA256SUMS",
            ):
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())

            samples = [json.loads(line) for line in (first / "e5_synthetic_sessions.jsonl").read_text().splitlines()]
            by_cell: dict[tuple[int, int], Counter] = defaultdict(Counter)
            groups_by_fold: dict[int, set[str]] = defaultdict(set)
            targets = set()
            for sample in samples:
                metadata = sample["synthetic_metadata"]
                by_cell[(metadata["seed"], metadata["fold"])][sample["scenario_type"]] += 1
                groups_by_fold[metadata["fold"]].add(metadata["group_id"])
                targets.add(sample["ground_truth"]["parent_asin"])
            self.assertEqual(len(samples), 80)
            for counts in by_cell.values():
                self.assertEqual(counts, {"buying": 8, "browsing": 8, "intent_override": 3, "boundary": 1})
            self.assertFalse(groups_by_fold[0] & groups_by_fold[1])
            self.assertNotIn("ASIN0001", targets)
            self.assertNotIn("ASIN0002", targets)

            manifest = json.loads((first / "e5_synthetic_manifest.json").read_text())
            audit = json.loads((first / "e5_synthetic_leakage_audit.json").read_text())
            self.assertTrue(manifest["group_disjoint_folds"])
            self.assertTrue(manifest["public_target_groups_quarantined"])
            self.assertTrue(audit["group_disjoint_folds"])
            self.assertTrue(audit["public_target_groups_quarantined"])

    def test_grouping_and_output_are_invariant_to_catalog_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ordered = root / "ordered"
            reversed_root = root / "reversed"
            ordered.mkdir()
            reversed_root.mkdir()
            catalog_a, public_a, _ = self._inputs(ordered)
            catalog_b, public_b, _ = self._inputs(reversed_root, shuffled=True)
            output_a = root / "output_a"
            output_b = root / "output_b"
            generate(catalog_a, public_a, output_a, folds=2, seeds=(1701,), sessions_per_fold=20)
            generate(catalog_b, public_b, output_b, folds=2, seeds=(1701,), sessions_per_fold=20)
            for name in (
                "e5_synthetic_sessions.jsonl",
                "e5_synthetic_group_map.jsonl",
                "e5_synthetic_leakage_audit.json",
            ):
                self.assertEqual((output_a / name).read_bytes(), (output_b / name).read_bytes())
            manifest_a = json.loads((output_a / "e5_synthetic_manifest.json").read_text())
            manifest_b = json.loads((output_b / "e5_synthetic_manifest.json").read_text())
            self.assertEqual(
                manifest_a["catalog_canonical_sorted_records_sha256"],
                manifest_b["catalog_canonical_sorted_records_sha256"],
            )
            self.assertNotEqual(manifest_a["catalog_input_sha256"], manifest_b["catalog_input_sha256"])

    def test_public_target_quarantines_whole_near_duplicate_group(self) -> None:
        products = {
            product["parent_asin"]: product
            for product in (
                _product(1, title="Trail Running Shirt Black XL", store="Trail Co"),
                _product(2, title="Trail Running Shirt Blue Small", store="Trail Co"),
                _product(3, title="Formal Silk Blouse", store="Formal Co"),
            )
        }
        products["ASIN0001"]["details"]["Item model number"] = "A"
        products["ASIN0002"]["details"]["Item model number"] = "B"
        groups, asin_to_group = build_groups(products)
        self.assertEqual(asin_to_group["ASIN0001"], asin_to_group["ASIN0002"])
        self.assertEqual(len(groups[asin_to_group["ASIN0001"]]), 2)

    def test_explicit_hidden_fields_match_evaluator_protocol(self) -> None:
        product = _product(7)
        card = intent_card(product)
        for scenario in ("buying", "browsing", "intent_override", "boundary"):
            import random

            behavior = behavior_for(scenario, card, random.Random("parity"))
            sample = {
                "behavior": behavior,
                "ground_truth": {"parent_asin": product["parent_asin"]},
                "intent_card": card,
                "sample_id": "parity",
                "scenario_type": scenario,
            }
            effective_card, effective_behavior = materialize_hidden_fields(sample, {product["parent_asin"]: product})
            self.assertEqual(effective_card, card)
            self.assertEqual(effective_behavior, behavior)

    def test_invalid_cell_size_and_insufficient_groups_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog, public, _ = self._inputs(root)
            with self.assertRaisesRegex(ValueError, "multiple of 20"):
                generate(catalog, public, root / "bad_ratio", folds=2, seeds=(1,), sessions_per_fold=19)
            with self.assertRaisesRegex(ValueError, "eligible groups"):
                generate(catalog, public, root / "too_many", folds=5, seeds=(1,), sessions_per_fold=20)


if __name__ == "__main__":
    unittest.main()
