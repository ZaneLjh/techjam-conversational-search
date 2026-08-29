from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import statistics
import tempfile
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Iterator

import starter.projection as projection_module
from evaluator.local_evaluator import (
    classify_constraint as reference_classify_constraint,
    coarse_category as reference_coarse_category,
    customer_reply as reference_customer_reply,
    intent_card as reference_intent_card,
)
from starter.projection import (
    PROJECTION_CANONICAL_JSON,
    PROJECTION_MANIFEST_SCHEMA_VERSION,
    PROJECTION_SCHEMA_VERSION,
    PROJECTION_VERSION,
    canonical_projection_line,
    classify_constraint,
    coarse_category,
    normalize_projection_value,
    project_intent_card,
    projected_reply_values,
    projection_record,
    strict_json_loads,
)


EXPECTED_CATALOG_ROWS = 50_000
TOP_COLLISION_COUNT = 20
ROLLOUT_ATTRIBUTES = (
    "material",
    "feature",
    "color",
    "style",
    "size",
    "use_case",
    "brand",
    "budget",
    "other",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def _catalog_lines(path: Path) -> Iterator[BinaryIO]:
    if path.suffix.lower() == ".gz":
        with gzip.open(path, "rb") as handle:
            yield handle
    else:
        with path.open("rb") as handle:
            yield handle


@contextmanager
def _temporary_output(final_path: Path) -> Iterator[Path]:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{final_path.name}.",
        suffix=".tmp",
        dir=final_path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        yield temporary_path
    finally:
        temporary_path.unlink(missing_ok=True)


def _nearest_rank(values: list[int], quantile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, math.ceil(quantile * len(ordered)) - 1),
    )
    return ordered[index]


def _collision_summary(
    products_by_fingerprint: dict[str, set[str]],
    representative_by_fingerprint: dict[str, tuple[str, str]],
    *,
    top_count: int,
) -> dict:
    frequencies = [len(products) for products in products_by_fingerprint.values()]
    largest = sorted(
        (
            (
                len(products),
                representative_by_fingerprint[fingerprint][0],
                representative_by_fingerprint[fingerprint][1],
                fingerprint,
            )
            for fingerprint, products in products_by_fingerprint.items()
        ),
        key=lambda item: (-item[0], item[1], item[2], item[3]),
    )[:top_count]
    return {
        "distinct_fingerprint_count": len(frequencies),
        "singleton_fingerprint_count": sum(value == 1 for value in frequencies),
        "median_document_frequency": statistics.median(frequencies) if frequencies else 0,
        "p90_document_frequency": _nearest_rank(frequencies, 0.90),
        "p95_document_frequency": _nearest_rank(frequencies, 0.95),
        "p99_document_frequency": _nearest_rank(frequencies, 0.99),
        "maximum_document_frequency": max(frequencies, default=0),
        "largest_collision_sets": [
            {
                "document_frequency": frequency,
                "domain": domain,
                "normalized_value": normalized_value,
                "fingerprint": fingerprint,
            }
            for frequency, domain, normalized_value, fingerprint in largest
        ],
    }


def _register_collision(
    *,
    parent_asin: str,
    fingerprint: str,
    domain: str,
    value: str,
    products_by_fingerprint: dict[str, set[str]],
    representative_by_fingerprint: dict[str, tuple[str, str]],
    conflicting_fingerprints: set[str],
) -> None:
    canonical = (domain, normalize_projection_value(value))
    existing = representative_by_fingerprint.setdefault(fingerprint, canonical)
    if existing != canonical:
        conflicting_fingerprints.add(fingerprint)
    products_by_fingerprint[fingerprint].add(parent_asin)


def _projected_reply(
    card: dict[str, object],
    attribute: str,
    disclosed: set[str],
) -> str:
    values = projected_reply_values(card, attribute, disclosed)
    if not values:
        return f"I don't have an additional preference for {attribute}."
    disclosed.update(values)
    return "For that, what matters is: " + "; ".join(values) + "."


def build_projection_suite(
    catalog_path: str | Path,
    sidecar_path: str | Path,
    manifest_path: str | Path,
    *,
    expected_row_count: int = EXPECTED_CATALOG_ROWS,
    top_collision_count: int = TOP_COLLISION_COUNT,
) -> dict:
    catalog = Path(catalog_path)
    sidecar = Path(sidecar_path)
    manifest = Path(manifest_path)

    if not catalog.is_file():
        raise FileNotFoundError(f"catalog not found: {catalog}")
    if expected_row_count < 1:
        raise ValueError("expected_row_count must be at least one")
    if top_collision_count < 0:
        raise ValueError("top_collision_count must be non-negative")
    resolved_paths = {catalog.resolve(), sidecar.resolve(), manifest.resolve()}
    if len(resolved_paths) != 3:
        raise ValueError("catalog, sidecar, and manifest paths must be distinct")

    catalog_input_sha256 = _sha256_file(catalog)
    catalog_input_bytes = catalog.stat().st_size
    catalog_content_digest = hashlib.sha256()
    catalog_content_bytes = 0
    sidecar_jsonl_digest = hashlib.sha256()
    sidecar_jsonl_bytes = 0
    row_count = 0
    constraint_occurrences = 0
    classifier_checks = 0
    rollout_checks = 0
    parent_asins: set[str] = set()

    mismatch_examples: dict[str, list[str]] = {
        "intent_card": [],
        "coarse_category": [],
        "classifier": [],
        "rollout": [],
    }
    mismatch_counts = {
        "intent_card": 0,
        "coarse_category": 0,
        "classifier": 0,
        "rollout": 0,
    }

    constraint_products: dict[str, set[str]] = defaultdict(set)
    category_products: dict[str, set[str]] = defaultdict(set)
    constraint_representatives: dict[str, tuple[str, str]] = {}
    category_representatives: dict[str, tuple[str, str]] = {}
    conflicting_fingerprints: set[str] = set()

    with _temporary_output(sidecar) as temporary_sidecar:
        with temporary_sidecar.open("wb") as compressed_handle:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=9,
                fileobj=compressed_handle,
                mtime=0,
            ) as sidecar_handle:
                with _catalog_lines(catalog) as catalog_handle:
                    for raw_line in catalog_handle:
                        catalog_content_digest.update(raw_line)
                        catalog_content_bytes += len(raw_line)
                        if not raw_line.strip():
                            continue

                        product = strict_json_loads(raw_line)
                        if not isinstance(product, dict):
                            raise ValueError("catalog rows must be JSON objects")
                        record = projection_record(product)
                        parent_asin_value = record["parent_asin"]
                        if not isinstance(parent_asin_value, str):
                            raise ValueError("projection parent_asin must be a string")
                        parent_asin = parent_asin_value
                        if parent_asin in parent_asins:
                            raise ValueError(f"duplicate parent_asin: {parent_asin}")
                        parent_asins.add(parent_asin)
                        row_count += 1

                        card = project_intent_card(product)
                        reference_card = reference_intent_card(product)
                        if card != reference_card:
                            mismatch_counts["intent_card"] += 1
                            if len(mismatch_examples["intent_card"]) < 10:
                                mismatch_examples["intent_card"].append(parent_asin)

                        categories = [
                            str(value) for value in product.get("categories") or []
                        ]
                        category = coarse_category(categories)
                        reference_category = reference_coarse_category(categories)
                        if category != reference_category:
                            mismatch_counts["coarse_category"] += 1
                            if len(mismatch_examples["coarse_category"]) < 10:
                                mismatch_examples["coarse_category"].append(parent_asin)

                        projection = record["projection"]
                        _register_collision(
                            parent_asin=parent_asin,
                            fingerprint=str(projection["category_fingerprint"]),
                            domain="category",
                            value=category,
                            products_by_fingerprint=category_products,
                            representative_by_fingerprint=category_representatives,
                            conflicting_fingerprints=conflicting_fingerprints,
                        )

                        for card_key, projection_key in (
                            ("hard_constraints", "hard_constraints"),
                            ("soft_preferences", "soft_preferences"),
                        ):
                            values = [str(value) for value in card.get(card_key, [])]
                            projected_values = projection[projection_key]
                            for value, projected_value in zip(values, projected_values):
                                classifier_checks += 1
                                constraint_occurrences += 1
                                facet = classify_constraint(value)
                                reference_facet = reference_classify_constraint(value)
                                if facet != reference_facet:
                                    mismatch_counts["classifier"] += 1
                                    if len(mismatch_examples["classifier"]) < 10:
                                        mismatch_examples["classifier"].append(parent_asin)
                                _register_collision(
                                    parent_asin=parent_asin,
                                    fingerprint=str(projected_value["fingerprint"]),
                                    domain=f"constraint:{facet}",
                                    value=value,
                                    products_by_fingerprint=constraint_products,
                                    representative_by_fingerprint=constraint_representatives,
                                    conflicting_fingerprints=conflicting_fingerprints,
                                )

                        rollout_sample = {
                            "scenario_type": "browsing",
                            "intent_card": card,
                        }
                        for attribute in ROLLOUT_ATTRIBUTES:
                            rollout_checks += 1
                            reference_reply = reference_customer_reply(
                                rollout_sample,
                                attribute,
                                set(),
                                False,
                            )[0]
                            projected_reply = _projected_reply(card, attribute, set())
                            if projected_reply != reference_reply:
                                mismatch_counts["rollout"] += 1
                                if len(mismatch_examples["rollout"]) < 10:
                                    mismatch_examples["rollout"].append(parent_asin)

                        reference_disclosed: set[str] = set()
                        projected_disclosed: set[str] = set()
                        for _ in range(3):
                            rollout_checks += 1
                            reference_reply = reference_customer_reply(
                                rollout_sample,
                                "other",
                                reference_disclosed,
                                False,
                            )[0]
                            projected_reply = _projected_reply(
                                card,
                                "other",
                                projected_disclosed,
                            )
                            if projected_reply != reference_reply:
                                mismatch_counts["rollout"] += 1
                                if len(mismatch_examples["rollout"]) < 10:
                                    mismatch_examples["rollout"].append(parent_asin)

                        rendered = canonical_projection_line(record)
                        sidecar_handle.write(rendered)
                        sidecar_jsonl_digest.update(rendered)
                        sidecar_jsonl_bytes += len(rendered)

        if row_count != expected_row_count:
            raise ValueError(
                f"expected {expected_row_count} catalog rows, found {row_count}"
            )
        if any(mismatch_counts.values()):
            raise ValueError(
                "projection parity failed: "
                + ", ".join(
                    f"{name}={count}"
                    for name, count in sorted(mismatch_counts.items())
                )
            )
        if conflicting_fingerprints:
            raise ValueError(
                f"detected {len(conflicting_fingerprints)} SHA-256 fingerprint collisions"
            )

        sidecar_gzip_sha256 = _sha256_file(temporary_sidecar)
        sidecar_gzip_bytes = temporary_sidecar.stat().st_size
        report = {
            "schema_version": PROJECTION_MANIFEST_SCHEMA_VERSION,
            "passed": True,
            "experiment": "E4.5 catalog-derived intent-card projection",
            "projection": {
                "record_schema_version": PROJECTION_SCHEMA_VERSION,
                "algorithm_version": PROJECTION_VERSION,
                "canonical_json": PROJECTION_CANONICAL_JSON,
            },
            "catalog": {
                "input_format": (
                    "gzip-jsonl" if catalog.suffix.lower() == ".gz" else "jsonl"
                ),
                "input_bytes": catalog_input_bytes,
                "content_bytes": catalog_content_bytes,
                "row_count": row_count,
                "unique_parent_asin_count": len(parent_asins),
            },
            "sidecar": {
                "format": "gzip-jsonl",
                "gzip_mtime": 0,
                "compression_level": 9,
                "row_count": row_count,
                "uncompressed_bytes": sidecar_jsonl_bytes,
                "compressed_bytes": sidecar_gzip_bytes,
            },
            "checksums": {
                "catalog_input_sha256": catalog_input_sha256,
                "catalog_content_sha256": catalog_content_digest.hexdigest(),
                "sidecar_jsonl_sha256": sidecar_jsonl_digest.hexdigest(),
                "sidecar_gzip_sha256": sidecar_gzip_sha256,
                "transform_source_sha256": _sha256_file(
                    Path(projection_module.__file__)
                ),
            },
            "parity": {
                "status": "pass",
                "intent_card_mismatches": mismatch_counts["intent_card"],
                "coarse_category_mismatches": mismatch_counts["coarse_category"],
                "classifier_mismatches": mismatch_counts["classifier"],
                "rollout_mismatches": mismatch_counts["rollout"],
                "classifier_checks": classifier_checks,
                "rollout_checks": rollout_checks,
                "mismatch_examples": mismatch_examples,
            },
            "collisions": {
                "constraint_occurrences": constraint_occurrences,
                "true_sha256_collision_count": len(conflicting_fingerprints),
                "constraints": _collision_summary(
                    constraint_products,
                    constraint_representatives,
                    top_count=top_collision_count,
                ),
                "categories": _collision_summary(
                    category_products,
                    category_representatives,
                    top_count=top_collision_count,
                ),
            },
        }
        rendered_manifest = (
            json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
        )
        with _temporary_output(manifest) as temporary_manifest:
            temporary_manifest.write_bytes(rendered_manifest.encode("utf-8"))
            os.replace(temporary_sidecar, sidecar)
            os.replace(temporary_manifest, manifest)
        return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and audit the deterministic E4.5 catalog intent-card sidecar."
    )
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument(
        "--sidecar",
        default="results/e4_5_intent_projection.jsonl.gz",
    )
    parser.add_argument(
        "--manifest",
        default="results/e4_5_projection_manifest.json",
    )
    parser.add_argument(
        "--expected-row-count",
        type=int,
        default=EXPECTED_CATALOG_ROWS,
    )
    parser.add_argument(
        "--top-collisions",
        type=int,
        default=TOP_COLLISION_COUNT,
    )
    args = parser.parse_args()
    report = build_projection_suite(
        args.catalog,
        args.sidecar,
        args.manifest,
        expected_row_count=args.expected_row_count,
        top_collision_count=args.top_collisions,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
