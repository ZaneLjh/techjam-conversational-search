from __future__ import annotations

import gzip
import hashlib
import itertools
import json
import math
import re
import statistics
import unicodedata
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Mapping, Sequence

from starter.constraints import Constraint, ConstraintStatus, Facet, Polarity, Strength


PROJECTION_SCHEMA_VERSION = 1
PROJECTION_MANIFEST_SCHEMA_VERSION = 1
PROJECTION_VERSION = "e4.5-intent-card-v1"
PROJECTION_CANONICAL_JSON = (
    "UTF-8 JSON Lines; keys sorted; compact separators; LF endings; "
    "catalog row order"
)
SEARCH_FIELDS = (
    "title",
    "features",
    "details",
    "description",
    "categories",
    "store",
)
MATERIALS = (
    "cotton",
    "polyester",
    "nylon",
    "leather",
    "wool",
    "spandex",
    "silk",
    "rayon",
    "fabric",
)
MATERIAL_RE = re.compile(
    r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric)\b",
    re.IGNORECASE,
)
COLOR_RE = re.compile(
    r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\b",
    re.IGNORECASE,
)
# Unicode letters and digits, excluding underscore. NFKC turns mathematical
# presentation forms into their ordinary equivalents without erasing CJK text.
TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
QUESTION_ATTRIBUTES = (
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

_MANIFEST_TOP_LEVEL_KEYS = {
    "catalog",
    "checksums",
    "collisions",
    "experiment",
    "parity",
    "passed",
    "projection",
    "schema_version",
    "sidecar",
}
_CATALOG_MANIFEST_KEYS = {
    "content_bytes",
    "input_bytes",
    "input_format",
    "row_count",
    "unique_parent_asin_count",
}
_CHECKSUM_KEYS = {
    "catalog_content_sha256",
    "catalog_input_sha256",
    "sidecar_gzip_sha256",
    "sidecar_jsonl_sha256",
    "transform_source_sha256",
}
_PARITY_KEYS = {
    "classifier_checks",
    "classifier_mismatches",
    "coarse_category_mismatches",
    "intent_card_mismatches",
    "mismatch_examples",
    "rollout_checks",
    "rollout_mismatches",
    "status",
}
_PROJECTION_MANIFEST_KEYS = {
    "algorithm_version",
    "canonical_json",
    "record_schema_version",
}
_SIDECAR_MANIFEST_KEYS = {
    "compressed_bytes",
    "compression_level",
    "format",
    "gzip_mtime",
    "row_count",
    "uncompressed_bytes",
}
_COLLISION_KEYS = {
    "categories",
    "constraint_occurrences",
    "constraints",
    "true_sha256_collision_count",
}
_MISMATCH_EXAMPLE_KEYS = {
    "classifier",
    "coarse_category",
    "intent_card",
    "rollout",
}
_COLLISION_SUMMARY_KEYS = {
    "distinct_fingerprint_count",
    "singleton_fingerprint_count",
    "median_document_frequency",
    "p90_document_frequency",
    "p95_document_frequency",
    "p99_document_frequency",
    "maximum_document_frequency",
    "largest_collision_sets",
}
_COLLISION_ROW_KEYS = {
    "document_frequency",
    "domain",
    "normalized_value",
    "fingerprint",
}


def searchable_text(product: Mapping[str, object]) -> str:
    """Join the catalog fields used to detect material and color clues."""

    parts: list[str] = []
    for field in SEARCH_FIELDS:
        value = product.get(field)
        if isinstance(value, dict):
            parts.extend(f"{key} {item}" for key, item in value.items())
        elif isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value is not None:
            parts.append(str(value))
    return " ".join(parts).strip()


def _flatten_values(value: object) -> list[str]:
    if isinstance(value, dict):
        return [
            f"{key}: {item}"
            for key, item in value.items()
            if item not in (None, "", [])
        ]
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)] if value not in (None, "") else []


def _clean_constraint(value: str, limit: int) -> str:
    return re.sub(r"\s+", " ", value).strip(" -;,.\t\n")[:limit].rstrip()


def project_intent_card(
    product: Mapping[str, object],
    limit: int = 180,
) -> dict[str, object]:
    """Project the deterministic title, hard clues, and soft clues for a product."""

    title = _clean_constraint(str(product.get("title") or "product"), limit)
    candidates = [
        *_flatten_values(product.get("features")),
        *_flatten_values(product.get("details")),
    ]
    corpus = searchable_text(product)
    material = MATERIAL_RE.search(corpus)
    color = COLOR_RE.search(corpus)
    if material:
        candidates.insert(0, material.group(1).lower())
    if color:
        # Match the released projection exactly: list.insert(1, ...) appends
        # when no material clue occupies slot zero.
        candidates.insert(1, f"color: {color.group(1).lower()}")
    if product.get("price") not in (None, ""):
        candidates.append(f"budget around ${product['price']}")

    cleaned = list(
        dict.fromkeys(
            _clean_constraint(item, limit)
            for item in candidates
            if _clean_constraint(item, limit)
        )
    )
    if not cleaned:
        cleaned = [title]
    return {
        "target_category": title,
        "hard_constraints": cleaned[:2],
        "soft_preferences": cleaned[2:4] or cleaned[:1],
    }


def coarse_category(values: Sequence[str]) -> str:
    """Return the last two non-generic category fragments."""

    excluded = {
        "clothing",
        "clothing shoes & jewelry",
        "clothing, shoes & jewelry",
    }
    cleaned: list[str] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part and part.lower() not in excluded:
                cleaned.append(part)
    return " ".join(cleaned[-2:]) if cleaned else "clothing item"


def classify_constraint(value: str) -> str:
    """Classify one projected clue into the structured question vocabulary."""

    lowered = value.lower()
    if "budget" in lowered or re.search(r"(?:\$|<=|under)\s*\d", lowered):
        return "budget"
    if any(material in lowered for material in MATERIALS):
        return "material"
    if any(
        word in lowered
        for word in ("color", "black", "white", "blue", "red", "pink", "green")
    ):
        return "color"
    if any(word in lowered for word in ("size", "sizing", "width", "wide", "narrow")):
        return "size"
    if any(word in lowered for word in ("department", "style", "fit", "sleeve", "neck")):
        return "style"
    if any(word in lowered for word in ("hiking", "running", "gym", "winter", "outdoor", "work")):
        return "use_case"
    return "feature"


def projected_reply_values(
    card: Mapping[str, object],
    attribute: str,
    disclosed: set[str],
) -> tuple[str, ...]:
    """Return the next two exact undisclosed values for one question action."""

    values = [
        *[str(value) for value in card.get("hard_constraints", [])],
        *[str(value) for value in card.get("soft_preferences", [])],
    ]
    return tuple(
        value
        for value in values
        if value not in disclosed
        and (attribute == "other" or classify_constraint(value) == attribute)
    )[:2]


def normalize_projection_value(value: str) -> str:
    """Normalize Unicode text before stable hashing and exact inversion."""

    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    tokens = TOKEN_RE.findall(normalized)
    if tokens:
        return " ".join(tokens)
    # Do not collapse distinct non-empty punctuation-only catalog clues onto
    # the empty key. Whitespace itself remains empty and cannot activate.
    return re.sub(r"\s+", " ", normalized).strip()


def projection_fingerprint(value: str, *, domain: str) -> str:
    """Return a domain-separated SHA-256 fingerprint for one projected value."""

    normalized_domain = str(domain).strip().casefold()
    if not normalized_domain:
        raise ValueError("fingerprint domain must be non-empty")
    normalized_value = normalize_projection_value(value)
    payload = (
        f"{PROJECTION_VERSION}\0{normalized_domain}\0{normalized_value}"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _constraint_projection(value: str) -> dict[str, str]:
    facet = classify_constraint(value)
    return {
        "facet": facet,
        "fingerprint": projection_fingerprint(
            value,
            domain=f"constraint:{facet}",
        ),
    }


def _require_parent_asin(product: Mapping[str, object]) -> str:
    if "parent_asin" not in product:
        raise ValueError("catalog product is missing parent_asin")
    value = product["parent_asin"]
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(
            "catalog parent_asin must be a non-empty, unpadded string"
        )
    return value


def projection_record(product: Mapping[str, object]) -> dict[str, object]:
    """Return one canonical sidecar record from catalog-visible fields."""

    parent_asin = _require_parent_asin(product)
    card = project_intent_card(product)
    raw_categories = product.get("categories")
    if raw_categories is None:
        raw_categories = []
    if not isinstance(raw_categories, list):
        raise ValueError("catalog categories must be an array")
    categories = [str(value) for value in raw_categories]
    category = coarse_category(categories)
    hard = [str(value) for value in card["hard_constraints"]]
    soft = [str(value) for value in card["soft_preferences"]]
    canonical_card = json.dumps(
        card,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "schema_version": PROJECTION_SCHEMA_VERSION,
        "parent_asin": parent_asin,
        "coarse_category": category,
        "intent_card": card,
        "projection": {
            "algorithm_version": PROJECTION_VERSION,
            "card_fingerprint": projection_fingerprint(
                canonical_card,
                domain="intent-card",
            ),
            "category_fingerprint": projection_fingerprint(
                category,
                domain="category",
            ),
            "hard_constraints": [
                _constraint_projection(value)
                for value in hard
            ],
            "soft_preferences": [
                _constraint_projection(value)
                for value in soft
            ],
        },
    }


def canonical_projection_line(record: Mapping[str, object]) -> bytes:
    """Serialize one projection row in the only accepted sidecar encoding."""

    return (
        json.dumps(
            record,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


@dataclass(frozen=True)
class ProjectionConfig:
    """Fail-closed E4.5 runtime configuration."""

    enabled: bool = False
    sidecar_path: str | None = None
    manifest_path: str | None = None
    use_reranking: bool = True
    use_question_rollout: bool = True
    candidate_depth: int = 100
    max_rerank_posterior_size: int = 1
    max_posterior_size: int = 100
    max_sidecar_bytes: int = 32 * 1024 * 1024
    max_uncompressed_sidecar_bytes: int = 64 * 1024 * 1024
    max_manifest_bytes: int = 1024 * 1024
    max_catalog_content_bytes: int = 64 * 1024 * 1024
    max_catalog_rows: int = 50_000
    max_catalog_row_bytes: int = 16 * 1024
    max_sidecar_row_bytes: int = 16 * 1024
    min_exact_clues: int = 1
    min_question_gain: float = 0.002

    def __post_init__(self) -> None:
        for name, value in (
            ("candidate_depth", self.candidate_depth),
            ("max_rerank_posterior_size", self.max_rerank_posterior_size),
            ("max_posterior_size", self.max_posterior_size),
            ("max_sidecar_bytes", self.max_sidecar_bytes),
            (
                "max_uncompressed_sidecar_bytes",
                self.max_uncompressed_sidecar_bytes,
            ),
            ("max_manifest_bytes", self.max_manifest_bytes),
            ("max_catalog_content_bytes", self.max_catalog_content_bytes),
            ("max_catalog_rows", self.max_catalog_rows),
            ("max_catalog_row_bytes", self.max_catalog_row_bytes),
            ("max_sidecar_row_bytes", self.max_sidecar_row_bytes),
            ("min_exact_clues", self.min_exact_clues),
        ):
            if type(value) is not int:
                raise ValueError(f"{name} must be an integer")
        if not 10 <= self.candidate_depth <= 100:
            raise ValueError("candidate_depth must be between 10 and 100")
        if not 1 <= self.max_posterior_size <= 100:
            raise ValueError("max_posterior_size must be between 1 and 100")
        if not 1 <= self.max_rerank_posterior_size <= self.max_posterior_size:
            raise ValueError(
                "max_rerank_posterior_size must be between 1 and max_posterior_size"
            )
        for name, value in (
            ("max_sidecar_bytes", self.max_sidecar_bytes),
            (
                "max_uncompressed_sidecar_bytes",
                self.max_uncompressed_sidecar_bytes,
            ),
            ("max_manifest_bytes", self.max_manifest_bytes),
            ("max_catalog_content_bytes", self.max_catalog_content_bytes),
            ("max_catalog_rows", self.max_catalog_rows),
            ("max_catalog_row_bytes", self.max_catalog_row_bytes),
            ("max_sidecar_row_bytes", self.max_sidecar_row_bytes),
            ("min_exact_clues", self.min_exact_clues),
        ):
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if not isinstance(self.min_question_gain, (int, float)) or isinstance(
            self.min_question_gain, bool
        ):
            raise ValueError("min_question_gain must be numeric")
        if not math.isfinite(self.min_question_gain) or self.min_question_gain < 0:
            raise ValueError("min_question_gain must be finite and non-negative")


@dataclass(frozen=True)
class ProjectedClue:
    raw_value: str
    normalized_value: str
    facet: str
    role: str
    ordinal: int


@dataclass(frozen=True)
class ProjectedProduct:
    parent_asin: str
    category: str
    category_norm: str
    clues: tuple[ProjectedClue, ...]


@dataclass(frozen=True)
class ProjectionRanking:
    recommendation_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    posterior_ids: tuple[str, ...]
    active: bool
    trace: dict


def _sha256_file(path: Path, *, max_bytes: int | None = None) -> str:
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            byte_count += len(chunk)
            if max_bytes is not None and byte_count > max_bytes:
                raise ValueError("file exceeds the configured hash limit")
            digest.update(chunk)
    return digest.hexdigest()


def canonical_source_sha256(path: str | Path) -> str:
    """Hash UTF-8 source after normalizing platform line endings to LF."""

    text = Path(path).read_bytes().decode("utf-8")
    canonical = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _read_bounded_file(path: Path, *, max_bytes: int) -> bytes:
    with path.open("rb") as handle:
        value = handle.read(max_bytes + 1)
    if len(value) > max_bytes:
        raise ValueError("file exceeds the configured read limit")
    return value


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def _finite_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite JSON number is not allowed: {value}")
    return result


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json_loads(raw: bytes | str) -> object:
    if isinstance(raw, bytes):
        text = raw.decode("utf-8")
    else:
        text = raw
    return json.loads(
        text,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_constant,
        parse_float=_finite_float,
    )


def _require_exact_keys(value: Mapping[str, object], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} keys are not canonical")


def _manifest_int(section: Mapping[str, object], key: str) -> int:
    value = section.get(key)
    if type(value) is not int:
        raise ValueError(f"manifest {key} must be an integer")
    return value


def _nearest_rank(values: Sequence[int], quantile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, math.ceil(quantile * len(ordered)) - 1),
    )
    return ordered[index]


def _validate_collision_summary(value: object, name: str) -> int:
    if not isinstance(value, dict):
        raise ValueError(f"manifest {name} collision summary must be an object")
    _require_exact_keys(value, _COLLISION_SUMMARY_KEYS, name)
    for key in (
        "distinct_fingerprint_count",
        "singleton_fingerprint_count",
        "p90_document_frequency",
        "p95_document_frequency",
        "p99_document_frequency",
        "maximum_document_frequency",
    ):
        if type(value.get(key)) is not int or int(value[key]) < 0:
            raise ValueError(f"manifest {name}.{key} must be a non-negative integer")
    median = value.get("median_document_frequency")
    if (
        not isinstance(median, (int, float))
        or isinstance(median, bool)
        or not math.isfinite(median)
        or median < 0
    ):
        raise ValueError(
            f"manifest {name}.median_document_frequency must be non-negative"
        )
    largest = value.get("largest_collision_sets")
    if not isinstance(largest, list):
        raise ValueError(f"manifest {name}.largest_collision_sets must be an array")
    if len(largest) > int(value["distinct_fingerprint_count"]):
        raise ValueError(f"manifest {name} collision list is too long")
    for row in largest:
        if not isinstance(row, dict):
            raise ValueError(f"manifest {name} collision row must be an object")
        _require_exact_keys(row, _COLLISION_ROW_KEYS, f"{name} collision row")
        if type(row.get("document_frequency")) is not int or int(
            row["document_frequency"]
        ) < 1:
            raise ValueError(
                f"manifest {name} collision frequency must be a positive integer"
            )
        for key in ("domain", "normalized_value", "fingerprint"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(f"manifest {name} collision {key} is invalid")
        fingerprint = str(row["fingerprint"])
        if not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
            raise ValueError(f"manifest {name} collision fingerprint is invalid")
    return len(largest)


def _collision_summary(
    rows: Mapping[tuple[str, str], Sequence[str]],
    *,
    top_count: int,
) -> dict[str, object]:
    products_by_fingerprint: dict[str, set[str]] = {}
    representatives: dict[str, tuple[str, str]] = {}
    for (domain, normalized_value), parent_asins in rows.items():
        fingerprint = projection_fingerprint(normalized_value, domain=domain)
        representative = (domain, normalized_value)
        existing = representatives.setdefault(fingerprint, representative)
        if existing != representative:
            raise ValueError("projection fingerprint collision detected at runtime")
        products_by_fingerprint.setdefault(fingerprint, set()).update(parent_asins)

    frequencies = [len(products) for products in products_by_fingerprint.values()]
    largest = sorted(
        (
            (
                len(products),
                representatives[fingerprint][0],
                representatives[fingerprint][1],
                fingerprint,
            )
            for fingerprint, products in products_by_fingerprint.items()
        ),
        key=lambda item: (-item[0], item[1], item[2], item[3]),
    )[:top_count]
    return {
        "distinct_fingerprint_count": len(frequencies),
        "singleton_fingerprint_count": sum(value == 1 for value in frequencies),
        "median_document_frequency": (
            statistics.median(frequencies) if frequencies else 0
        ),
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


def _canonical_json_value(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _content_digest(
    path: Path,
    *,
    compressed: bool,
    max_content_bytes: int,
) -> tuple[str, int]:
    """Hash an exact JSONL byte stream without unbounded decompression."""

    digest = hashlib.sha256()
    byte_count = 0
    opener = gzip.open if compressed else Path.open
    with opener(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            byte_count += len(chunk)
            if byte_count > max_content_bytes:
                raise ValueError("catalog content exceeds the configured limit")
            digest.update(chunk)
    return digest.hexdigest(), byte_count


def _single_gzip_content_size(
    path: Path,
    *,
    max_compressed_bytes: int,
    max_uncompressed_sidecar_bytes: int,
) -> int:
    """Bound inflation and reject trailing data or concatenated gzip members."""

    decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
    input_bytes = 0
    output_bytes = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            input_bytes += len(chunk)
            if input_bytes > max_compressed_bytes:
                raise ValueError("projection sidecar exceeds the configured limit")
            remaining = chunk
            while remaining:
                output = decompressor.decompress(
                    remaining,
                    max_uncompressed_sidecar_bytes - output_bytes + 1,
                )
                output_bytes += len(output)
                if output_bytes > max_uncompressed_sidecar_bytes:
                    raise ValueError("projection sidecar expands beyond the configured limit")
                remaining = decompressor.unconsumed_tail
                if decompressor.eof:
                    if decompressor.unused_data or remaining or handle.read(1):
                        raise ValueError("projection sidecar must contain one gzip member")
                    break
            if decompressor.eof:
                break
    if not decompressor.eof:
        raise ValueError("projection sidecar gzip stream is truncated")
    return output_bytes


def bounded_sidecar_lines(
    handle: BinaryIO,
    *,
    max_sidecar_row_bytes: int,
):
    """Yield newline-terminated records without materializing oversized lines."""

    while True:
        raw_line = handle.readline(max_sidecar_row_bytes + 1)
        if not raw_line:
            return
        if len(raw_line) > max_sidecar_row_bytes:
            raise ValueError("projection row exceeds the configured limit")
        if not raw_line.endswith(b"\n"):
            raise ValueError("projection row is not LF terminated")
        yield raw_line


def bounded_catalog_lines(
    handle: BinaryIO,
    *,
    max_catalog_row_bytes: int,
    max_catalog_rows: int,
):
    """Yield nonblank catalog rows under explicit row and count bounds."""

    row_count = 0
    while True:
        raw_line = handle.readline(max_catalog_row_bytes + 1)
        if not raw_line:
            return
        if len(raw_line) > max_catalog_row_bytes:
            raise ValueError("catalog row exceeds the configured limit")
        if not raw_line.strip():
            continue
        row_count += 1
        if row_count > max_catalog_rows:
            raise ValueError("catalog row count exceeds the configured limit")
        yield raw_line


def is_projection_template_message(message: str, turn: int) -> bool:
    """Recognize only the released structured wrappers, failing closed."""

    value = re.sub(r"\s+", " ", str(message)).strip()
    if not value:
        return False
    if turn == 1:
        return bool(
            re.fullmatch(
                r"I'm looking for .+(?:\. A key requirement is: .+\.|"
                r", but I'm still exploring\.|\. .+)",
                value,
            )
        )
    return bool(
        re.fullmatch(
            r"(?:For that, what matters is: .+\.|"
            r"I don't have an additional preference for [a-z_]+\.|"
            r"I don't have a preference for [a-z_]+; please use your judgment\.|"
            r"Those options are not quite right yet\. Ask me about one specific attribute\.|"
            r"Actually, ignore my earlier preference\. What I need is: .+\.)",
            value,
            flags=re.IGNORECASE,
        )
    )


class ProjectionIndex:
    """Validated exact projection index with deterministic E4 fallback."""

    def __init__(self, catalog_path: str | Path, config: ProjectionConfig) -> None:
        self.config = config
        self.ready = False
        self.status_reason = "disabled"
        self.records: dict[str, ProjectedProduct] = {}
        self._catalog_order: dict[str, int] = {}
        self._category_index: dict[str, tuple[str, ...]] = {}
        self._value_index: dict[tuple[str, str], tuple[str, ...]] = {}
        if not config.enabled:
            return
        try:
            self._load(Path(catalog_path))
        except Exception as exc:  # fail closed; no partial route survives
            self.records.clear()
            self._catalog_order.clear()
            self._category_index.clear()
            self._value_index.clear()
            self.ready = False
            self.status_reason = f"invalid_sidecar:{type(exc).__name__}"

    def _load(self, catalog_path: Path) -> None:
        if not self.config.sidecar_path or not self.config.manifest_path:
            raise ValueError("sidecar and manifest are required")
        sidecar = Path(self.config.sidecar_path)
        manifest_path = Path(self.config.manifest_path)
        if not catalog_path.is_file() or not sidecar.is_file() or not manifest_path.is_file():
            raise FileNotFoundError("projection input is missing")
        if catalog_path.stat().st_size > self.config.max_catalog_content_bytes:
            raise ValueError("catalog input exceeds the configured limit")
        if sidecar.stat().st_size > self.config.max_sidecar_bytes:
            raise ValueError("projection sidecar exceeds the configured limit")
        if manifest_path.stat().st_size > self.config.max_manifest_bytes:
            raise ValueError("projection manifest exceeds the configured limit")

        manifest_bytes = _read_bounded_file(
            manifest_path,
            max_bytes=self.config.max_manifest_bytes,
        )
        manifest = strict_json_loads(manifest_bytes)
        if not isinstance(manifest, dict):
            raise ValueError("projection manifest must be an object")
        canonical_manifest = (
            json.dumps(
                manifest,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        if manifest_bytes != canonical_manifest:
            raise ValueError("projection manifest is not canonically encoded")
        _require_exact_keys(manifest, _MANIFEST_TOP_LEVEL_KEYS, "manifest")
        if type(manifest.get("schema_version")) is not int or manifest.get(
            "schema_version"
        ) != PROJECTION_MANIFEST_SCHEMA_VERSION:
            raise ValueError("projection manifest schema mismatch")
        if manifest.get("passed") is not True:
            raise ValueError("projection manifest did not pass")
        if manifest.get("experiment") != "E4.5 catalog-derived intent-card projection":
            raise ValueError("projection experiment mismatch")

        projection = manifest.get("projection")
        checksums = manifest.get("checksums")
        parity = manifest.get("parity")
        catalog = manifest.get("catalog")
        sidecar_meta = manifest.get("sidecar")
        collisions = manifest.get("collisions")
        if not all(
            isinstance(value, dict)
            for value in (
                projection,
                checksums,
                parity,
                catalog,
                sidecar_meta,
                collisions,
            )
        ):
            raise ValueError("projection manifest sections are missing")
        _require_exact_keys(projection, _PROJECTION_MANIFEST_KEYS, "projection")
        _require_exact_keys(checksums, _CHECKSUM_KEYS, "checksums")
        _require_exact_keys(parity, _PARITY_KEYS, "parity")
        _require_exact_keys(catalog, _CATALOG_MANIFEST_KEYS, "catalog")
        _require_exact_keys(sidecar_meta, _SIDECAR_MANIFEST_KEYS, "sidecar")
        _require_exact_keys(collisions, _COLLISION_KEYS, "collisions")
        if parity.get("mismatch_examples") != {
            key: [] for key in _MISMATCH_EXAMPLE_KEYS
        }:
            raise ValueError("projection parity mismatch examples are not canonical")
        category_collision_count = _validate_collision_summary(
            collisions.get("categories"),
            "categories",
        )
        constraint_collision_count = _validate_collision_summary(
            collisions.get("constraints"),
            "constraints",
        )
        collision_top_count = max(
            category_collision_count,
            constraint_collision_count,
        )
        if type(projection.get("record_schema_version")) is not int or projection.get(
            "record_schema_version"
        ) != PROJECTION_SCHEMA_VERSION:
            raise ValueError("projection schema mismatch")
        if projection.get("algorithm_version") != PROJECTION_VERSION:
            raise ValueError("projection algorithm mismatch")
        if projection.get("canonical_json") != PROJECTION_CANONICAL_JSON:
            raise ValueError("projection canonical encoding mismatch")

        expected_catalog_format = (
            "gzip-jsonl" if catalog_path.suffix.lower() == ".gz" else "jsonl"
        )
        if catalog.get("input_format") != expected_catalog_format:
            raise ValueError("catalog format mismatch")
        if _manifest_int(catalog, "input_bytes") != catalog_path.stat().st_size:
            raise ValueError("catalog input size mismatch")
        declared_rows = _manifest_int(catalog, "row_count")
        declared_unique = _manifest_int(catalog, "unique_parent_asin_count")
        declared_catalog_content_bytes = _manifest_int(catalog, "content_bytes")
        if not 1 <= declared_rows <= self.config.max_catalog_rows:
            raise ValueError("projection catalog row count is outside the configured bound")
        if not (
            1
            <= declared_catalog_content_bytes
            <= self.config.max_catalog_content_bytes
        ):
            raise ValueError("catalog content size is outside the configured bound")
        if declared_unique != declared_rows:
            raise ValueError("projection catalog IDs are not unique")

        if sidecar_meta.get("format") != "gzip-jsonl":
            raise ValueError("sidecar format mismatch")
        if _manifest_int(sidecar_meta, "gzip_mtime") != 0:
            raise ValueError("sidecar gzip timestamp mismatch")
        if _manifest_int(sidecar_meta, "compression_level") != 9:
            raise ValueError("sidecar compression setting mismatch")
        if _manifest_int(sidecar_meta, "compressed_bytes") != sidecar.stat().st_size:
            raise ValueError("sidecar compressed size mismatch")
        declared_uncompressed = _manifest_int(sidecar_meta, "uncompressed_bytes")
        if not (
            1
            <= declared_uncompressed
            <= self.config.max_uncompressed_sidecar_bytes
        ):
            raise ValueError("sidecar content size is outside the configured bound")
        if _manifest_int(sidecar_meta, "row_count") != declared_rows:
            raise ValueError("sidecar row count mismatch")

        with sidecar.open("rb") as handle:
            header = handle.read(10)
        if header != b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x02\xff":
            raise ValueError("sidecar gzip header is not canonical")
        expanded_bytes = _single_gzip_content_size(
            sidecar,
            max_compressed_bytes=self.config.max_sidecar_bytes,
            max_uncompressed_sidecar_bytes=(
                self.config.max_uncompressed_sidecar_bytes
            ),
        )
        if expanded_bytes != declared_uncompressed:
            raise ValueError("sidecar expanded size mismatch")

        if checksums.get("catalog_input_sha256") != _sha256_file(
            catalog_path,
            max_bytes=self.config.max_catalog_content_bytes,
        ):
            raise ValueError("catalog checksum mismatch")
        actual_sidecar_sha256 = _sha256_file(
            sidecar,
            max_bytes=self.config.max_sidecar_bytes,
        )
        if checksums.get("sidecar_gzip_sha256") != actual_sidecar_sha256:
            raise ValueError("sidecar checksum mismatch")
        if checksums.get("transform_source_sha256") != canonical_source_sha256(
            Path(__file__)
        ):
            raise ValueError("projection transform checksum mismatch")
        catalog_content_sha256, catalog_content_bytes = _content_digest(
            catalog_path,
            compressed=expected_catalog_format == "gzip-jsonl",
            max_content_bytes=self.config.max_catalog_content_bytes,
        )
        if declared_catalog_content_bytes != catalog_content_bytes:
            raise ValueError("catalog content size mismatch")
        if checksums.get("catalog_content_sha256") != catalog_content_sha256:
            raise ValueError("catalog content checksum mismatch")
        if parity.get("status") != "pass" or any(
            _manifest_int(parity, name) != 0
            for name in (
                "intent_card_mismatches",
                "coarse_category_mismatches",
                "classifier_mismatches",
                "rollout_mismatches",
            )
        ):
            raise ValueError("projection parity did not pass")
        if _manifest_int(collisions, "true_sha256_collision_count") != 0:
            raise ValueError("projection fingerprint collision audit did not pass")

        category_rows: dict[str, list[str]] = {}
        value_rows: dict[tuple[str, str], list[str]] = {}
        sidecar_content_digest = hashlib.sha256()
        sidecar_content_bytes = 0
        clue_count = 0
        catalog_opener = gzip.open if expected_catalog_format == "gzip-jsonl" else Path.open
        missing = object()
        with catalog_opener(catalog_path, "rb") as catalog_handle, gzip.open(
            sidecar, "rb"
        ) as sidecar_handle:
            catalog_lines = bounded_catalog_lines(
                catalog_handle,
                max_catalog_row_bytes=self.config.max_catalog_row_bytes,
                max_catalog_rows=self.config.max_catalog_rows,
            )
            sidecar_lines = bounded_sidecar_lines(
                sidecar_handle,
                max_sidecar_row_bytes=self.config.max_sidecar_row_bytes,
            )
            paired_lines = itertools.zip_longest(
                catalog_lines,
                sidecar_lines,
                fillvalue=missing,
            )
            for order, (catalog_line, raw_line) in enumerate(paired_lines):
                    if catalog_line is missing or raw_line is missing:
                        raise ValueError("catalog and sidecar row counts disagree")
                    if not isinstance(catalog_line, bytes) or not isinstance(raw_line, bytes):
                        raise ValueError("catalog or sidecar row is not bytes")
                    sidecar_content_digest.update(raw_line)
                    sidecar_content_bytes += len(raw_line)
                    product = strict_json_loads(catalog_line)
                    if not isinstance(product, dict):
                        raise ValueError("catalog row must be an object")
                    expected_record = projection_record(product)
                    expected_line = canonical_projection_line(expected_record)
                    if raw_line != expected_line:
                        raise ValueError("projection row does not match catalog transform")
                    raw = expected_record
                    parent_asin = str(raw.get("parent_asin", ""))
                    if not parent_asin or parent_asin in self.records:
                        raise ValueError("projection parent_asin is empty or duplicated")
                    if raw.get("schema_version") != PROJECTION_SCHEMA_VERSION:
                        raise ValueError("projection row schema mismatch")
                    raw_projection = raw.get("projection")
                    card = raw.get("intent_card")
                    if not isinstance(raw_projection, dict) or not isinstance(card, dict):
                        raise ValueError("projection row sections are missing")
                    if raw_projection.get("algorithm_version") != PROJECTION_VERSION:
                        raise ValueError("projection row algorithm mismatch")
                    if not isinstance(card.get("target_category"), str):
                        raise ValueError("projection target category is invalid")
                    canonical_card = json.dumps(
                        card,
                        allow_nan=False,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    if raw_projection.get("card_fingerprint") != projection_fingerprint(
                        canonical_card,
                        domain="intent-card",
                    ):
                        raise ValueError("projection card fingerprint mismatch")
                    category_value = str(raw.get("coarse_category", ""))
                    if not category_value:
                        raise ValueError("projection category is empty")
                    if raw_projection.get("category_fingerprint") != projection_fingerprint(
                        category_value,
                        domain="category",
                    ):
                        raise ValueError("projection category fingerprint mismatch")
                    category_norm = normalize_projection_value(category_value)
                    clues: list[ProjectedClue] = []
                    for role, card_key, projection_key in (
                        ("hard", "hard_constraints", "hard_constraints"),
                        ("soft", "soft_preferences", "soft_preferences"),
                    ):
                        raw_values = card.get(card_key)
                        projected_values = raw_projection.get(projection_key)
                        if not isinstance(raw_values, list) or not isinstance(projected_values, list):
                            raise ValueError("projection clue arrays are invalid")
                        if len(raw_values) != len(projected_values):
                            raise ValueError("projection clue arrays disagree")
                        for ordinal, (raw_value, projected) in enumerate(
                            zip(raw_values, projected_values)
                        ):
                            if not isinstance(projected, dict):
                                raise ValueError("projection clue is invalid")
                            if not isinstance(raw_value, str) or not raw_value:
                                raise ValueError("projection clue value is invalid")
                            value = str(raw_value)
                            facet = str(projected.get("facet", ""))
                            if facet != classify_constraint(value):
                                raise ValueError("projection clue classifier mismatch")
                            expected_fingerprint = projection_fingerprint(
                                value,
                                domain=f"constraint:{facet}",
                            )
                            if projected.get("fingerprint") != expected_fingerprint:
                                raise ValueError("projection clue fingerprint mismatch")
                            normalized_value = normalize_projection_value(value)
                            if not normalized_value:
                                raise ValueError("projection clue normalization is empty")
                            clue = ProjectedClue(
                                raw_value=value,
                                normalized_value=normalized_value,
                                facet=facet,
                                role=role,
                                ordinal=ordinal,
                            )
                            clues.append(clue)
                            clue_count += 1
                            value_rows.setdefault(
                                (clue.facet, clue.normalized_value), []
                            ).append(parent_asin)
                    record = ProjectedProduct(
                        parent_asin=parent_asin,
                        category=category_value,
                        category_norm=category_norm,
                        clues=tuple(clues),
                    )
                    self.records[parent_asin] = record
                    self._catalog_order[parent_asin] = order
                    category_rows.setdefault(category_norm, []).append(parent_asin)

        if len(self.records) != declared_rows or len(self.records) != declared_unique:
            raise ValueError("projection catalog row count mismatch")
        if sidecar_content_bytes != declared_uncompressed:
            raise ValueError("sidecar content size mismatch")
        if checksums.get("sidecar_jsonl_sha256") != sidecar_content_digest.hexdigest():
            raise ValueError("sidecar content checksum mismatch")
        if _manifest_int(parity, "classifier_checks") != clue_count:
            raise ValueError("projection classifier audit count mismatch")
        expected_rollout_checks = len(self.records) * (len(QUESTION_ATTRIBUTES) + 3)
        if _manifest_int(parity, "rollout_checks") != expected_rollout_checks:
            raise ValueError("projection rollout audit count mismatch")
        if _manifest_int(collisions, "constraint_occurrences") != clue_count:
            raise ValueError("projection collision occurrence count mismatch")
        self._category_index = {
            key: tuple(dict.fromkeys(values)) for key, values in category_rows.items()
        }
        self._value_index = {
            key: tuple(dict.fromkeys(values)) for key, values in value_rows.items()
        }
        expected_category_collisions = _collision_summary(
            {
                ("category", normalized): parent_asins
                for normalized, parent_asins in self._category_index.items()
            },
            top_count=collision_top_count,
        )
        expected_constraint_collisions = _collision_summary(
            {
                (f"constraint:{facet}", normalized): parent_asins
                for (facet, normalized), parent_asins in self._value_index.items()
            },
            top_count=collision_top_count,
        )
        if _canonical_json_value(
            collisions.get("categories")
        ) != _canonical_json_value(expected_category_collisions):
            raise ValueError("projection category collision audit mismatch")
        if _canonical_json_value(
            collisions.get("constraints")
        ) != _canonical_json_value(expected_constraint_collisions):
            raise ValueError("projection constraint collision audit mismatch")
        self.ready = True
        self.status_reason = "ready"

    @staticmethod
    def _positive_constraints(
        constraints: Sequence[Constraint],
    ) -> list[Constraint]:
        return [
            item
            for item in constraints
            if item.status is ConstraintStatus.ACTIVE
            and item.polarity is Polarity.POSITIVE
            and item.strength in {Strength.MUST, Strength.SHOULD}
            and item.normalized_value
        ]

    def _fallback(
        self,
        recommendation_ids: Sequence[str],
        candidate_ids: Sequence[str],
        reason: str,
    ) -> ProjectionRanking:
        return ProjectionRanking(
            tuple(recommendation_ids),
            tuple(candidate_ids),
            (),
            False,
            {
                "enabled": self.config.enabled,
                "ready": self.ready,
                "active": False,
                "fallback_reason": reason,
                "predecessor_recommendation_count": len(recommendation_ids),
                "predecessor_candidate_count": len(candidate_ids),
                "recommendation_ids": list(recommendation_ids),
                "candidate_ids": list(candidate_ids),
            },
        )

    def rerank(
        self,
        *,
        recommendation_ids: Sequence[str],
        candidate_ids: Sequence[str],
        constraints: Sequence[Constraint],
        shown_ids: set[str],
        requested_k: int,
        template_confident: bool,
    ) -> ProjectionRanking:
        """Apply exact category-plus-clue projection or return the predecessor."""

        if not self.config.enabled:
            return self._fallback(recommendation_ids, candidate_ids, "disabled")
        if not self.ready:
            return self._fallback(
                recommendation_ids,
                candidate_ids,
                self.status_reason,
            )
        if not template_confident:
            return self._fallback(
                recommendation_ids,
                candidate_ids,
                "template_not_recognized",
            )
        if len(recommendation_ids) > 10 or len(candidate_ids) > 100:
            return self._fallback(
                recommendation_ids,
                candidate_ids,
                "predecessor_pool_outside_bound",
            )
        if any(
            item.status is ConstraintStatus.ACTIVE
            and item.polarity is Polarity.NEGATIVE
            for item in constraints
        ):
            return self._fallback(
                recommendation_ids,
                candidate_ids,
                "active_exclusion",
            )

        positive = self._positive_constraints(constraints)
        category = next(
            (item for item in reversed(positive) if item.facet is Facet.CATEGORY),
            None,
        )
        if category is None:
            return self._fallback(
                recommendation_ids,
                candidate_ids,
                "missing_exact_category",
            )
        category_norm = normalize_projection_value(category.raw_value)
        if not category_norm:
            return self._fallback(
                recommendation_ids,
                candidate_ids,
                "empty_category_normalization",
            )
        category_matches = set(self._category_index.get(category_norm, ()))
        if not category_matches:
            return self._fallback(
                recommendation_ids,
                candidate_ids,
                "category_not_projected",
            )

        active_clues: list[tuple[Constraint, str, str, set[str]]] = []
        for item in positive:
            if item.facet is Facet.CATEGORY:
                continue
            normalized = normalize_projection_value(item.raw_value)
            if not normalized:
                return self._fallback(
                    recommendation_ids,
                    candidate_ids,
                    "active_clue_normalization_empty",
                )
            facet = classify_constraint(item.raw_value)
            matches = set(self._value_index.get((facet, normalized), ()))
            if not matches:
                return self._fallback(
                    recommendation_ids,
                    candidate_ids,
                    "active_clue_not_projected",
                )
            active_clues.append((item, facet, normalized, matches))
        if len(active_clues) < self.config.min_exact_clues:
            return self._fallback(
                recommendation_ids,
                candidate_ids,
                "insufficient_exact_clues",
            )

        posterior = set(category_matches)
        for _, _, _, matches in active_clues:
            posterior.intersection_update(matches)
        posterior.difference_update(shown_ids)
        if not posterior:
            return self._fallback(
                recommendation_ids,
                candidate_ids,
                "empty_category_clue_intersection",
            )
        if len(posterior) > self.config.max_posterior_size:
            return self._fallback(
                recommendation_ids,
                candidate_ids,
                "oversized_category_clue_intersection",
            )

        projection_matches = sorted(
            posterior,
            key=lambda value: (self._catalog_order.get(value, 10**9), value),
        )
        if (
            not self.config.use_reranking
            or len(projection_matches) > self.config.max_rerank_posterior_size
        ):
            stable_candidates = list(
                dict.fromkeys(
                    [
                        *recommendation_ids,
                        *projection_matches,
                        *(value for value in candidate_ids if value not in shown_ids),
                    ]
                )
            )[: self.config.candidate_depth]
            return ProjectionRanking(
                tuple(recommendation_ids),
                tuple(stable_candidates),
                tuple(projection_matches),
                True,
                {
                    "enabled": True,
                    "ready": True,
                    "active": True,
                    "ranking_applied": False,
                    "ranking_skip_reason": (
                        "reranking_disabled"
                        if not self.config.use_reranking
                        else "posterior_above_rerank_limit"
                    ),
                    "fallback_reason": None,
                    "exact_category": category.raw_value,
                    "exact_clue_count": len(active_clues),
                    "posterior_count": len(projection_matches),
                    "posterior_ids": projection_matches,
                    "predecessor_recommendation_count": len(recommendation_ids),
                    "predecessor_candidate_count": len(candidate_ids),
                    "recommendation_ids": list(recommendation_ids),
                    "candidate_count": len(stable_candidates),
                    "candidate_ids": stable_candidates,
                },
            )

        predecessor_order = {
            value: rank for rank, value in enumerate(candidate_ids, start=1)
        }
        pool = list(
            dict.fromkeys(
                [
                    *(value for value in candidate_ids if value not in shown_ids),
                    *projection_matches,
                ]
            )
        )
        active_replacements = {facet for _, facet, _, _ in active_clues}
        superseded = [
            item
            for item in constraints
            if item.status is ConstraintStatus.SUPERSEDED
            and item.polarity is Polarity.POSITIVE
            and classify_constraint(item.raw_value) in active_replacements
        ]

        def score(parent_asin: str) -> tuple[float, int, int, str]:
            record = self.records.get(parent_asin)
            if record is None:
                return (
                    0.0,
                    predecessor_order.get(parent_asin, 10**9),
                    10**9,
                    parent_asin,
                )
            value = 1.0 if record.category_norm == category_norm else 0.0
            clue_keys = {
                (clue.facet, clue.normalized_value): clue for clue in record.clues
            }
            clue_facets = {clue.facet for clue in record.clues}
            for item, facet, normalized, _ in active_clues:
                clue = clue_keys.get((facet, normalized))
                if clue is not None:
                    # Weight visible evidence, not hidden card position, so
                    # observational ties retain frozen predecessor order.
                    value += 2.0 if item.strength is Strength.MUST else 1.0
                elif facet in clue_facets:
                    value -= 0.5
            for item in superseded:
                key = (
                    classify_constraint(item.raw_value),
                    normalize_projection_value(item.raw_value),
                )
                if key in clue_keys:
                    value -= 0.25
            return (
                -value,
                predecessor_order.get(parent_asin, 10**9),
                self._catalog_order.get(parent_asin, 10**9),
                parent_asin,
            )

        # The two bounded sources contain at most 100 rows apiece. Score their
        # complete union before applying the configured output cap.
        ranked = sorted(pool, key=score)[: self.config.candidate_depth]
        final_recommendations = ranked[:requested_k]
        used = set(final_recommendations)
        final_candidates = [
            *final_recommendations,
            *(value for value in ranked if value not in used),
        ][: self.config.candidate_depth]
        return ProjectionRanking(
            tuple(final_recommendations),
            tuple(final_candidates),
            tuple(projection_matches),
            True,
            {
                "enabled": True,
                "ready": True,
                "active": True,
                "ranking_applied": True,
                "ranking_skip_reason": None,
                "fallback_reason": None,
                "exact_category": category.raw_value,
                "exact_clue_count": len(active_clues),
                "posterior_count": len(projection_matches),
                "posterior_ids": projection_matches,
                "predecessor_recommendation_count": len(recommendation_ids),
                "predecessor_candidate_count": len(candidate_ids),
                "candidate_count": len(final_candidates),
                "recommendation_count": len(final_recommendations),
                "recommendation_ids": final_recommendations,
                "candidate_ids": final_candidates,
            },
        )

    @staticmethod
    def _utility(rank: int, reply_turn: int) -> float:
        if rank > 10:
            return 0.0
        return 0.50 + 0.30 / rank + 0.20 * (11 - min(reply_turn, 10)) / 10

    @staticmethod
    def _reply_values(
        record: ProjectedProduct,
        attribute: str | None,
        disclosed: set[str],
    ) -> tuple[str, ...]:
        if attribute is None:
            return ()
        return tuple(
            clue.raw_value
            for clue in record.clues
            if clue.raw_value not in disclosed
            and (attribute == "other" or clue.facet == attribute)
        )[:2]

    @classmethod
    def _reply_signature(
        cls,
        record: ProjectedProduct,
        attribute: str | None,
        disclosed: set[str],
    ) -> tuple[str, ...]:
        """Compatibility tuple used by disclosure tracking."""

        if attribute is None:
            return ("<no-question>",)
        matches = cls._reply_values(record, attribute, disclosed)
        return matches if matches else ("<no-additional>", attribute)

    @classmethod
    def _render_reply_signature(
        cls,
        record: ProjectedProduct,
        attribute: str | None,
        disclosed: set[str],
    ) -> str:
        """Render exactly what the simulator would emit for one action."""

        if attribute is None:
            return (
                "Those options are not quite right yet. "
                "Ask me about one specific attribute."
            )
        matches = cls._reply_values(record, attribute, disclosed)
        if not matches:
            return f"I don't have an additional preference for {attribute}."
        return "For that, what matters is: " + "; ".join(matches) + "."

    def choose_question(
        self,
        *,
        ranking: ProjectionRanking,
        constraints: Sequence[Constraint],
        disclosed_values: set[str] | None = None,
        asked_attributes: set[str],
        other_exhausted: bool,
        turn: int,
        baseline_attribute: str | None,
        condition_on_current_miss: bool = True,
    ) -> tuple[str | None, dict]:
        """Choose a question by exact projected rendered-reply partitions."""

        if (
            not ranking.active
            or not self.config.use_question_rollout
            or turn <= 1
            or turn >= 10
        ):
            return None, {
                "active": False,
                "reason": (
                    "projection_inactive"
                    if not ranking.active
                    else "rollout_disabled_or_guardrailed"
                ),
            }
        posterior = set(ranking.posterior_ids)
        displayed = (
            set(ranking.recommendation_ids)
            if condition_on_current_miss
            else set()
        )
        belief = [
            parent_asin
            for parent_asin in ranking.posterior_ids
            if parent_asin not in displayed and parent_asin in self.records
        ]
        if not belief:
            return None, {
                "active": False,
                "reason": "posterior_exhausted_by_current_display",
                "displayed_posterior_count": len(displayed & posterior),
                "conditioned_on_current_miss": condition_on_current_miss,
            }
        disclosed = (
            set(disclosed_values)
            if disclosed_values is not None
            else {
                item.raw_value
                for item in self._positive_constraints(constraints)
                if item.facet is not Facet.CATEGORY
            }
        )
        actions = [
            attribute
            for attribute in QUESTION_ATTRIBUTES
            if (
                (attribute == "other" and not other_exhausted)
                or (attribute != "other" and attribute not in asked_attributes)
            )
        ]
        if not actions:
            return None, {"active": False, "reason": "no_eligible_action"}

        prior_weights = [1.0 / math.sqrt(rank) for rank in range(1, len(belief) + 1)]
        prior_total = sum(prior_weights)

        def expected_utility(attribute: str | None) -> float:
            signatures = {
                parent_asin: self._render_reply_signature(
                    self.records[parent_asin],
                    attribute,
                    disclosed,
                )
                for parent_asin in belief
            }
            partition_rank: dict[str, int] = {}
            seen: dict[str, int] = {}
            for parent_asin in belief:
                signature = signatures[parent_asin]
                seen[signature] = seen.get(signature, 0) + 1
                partition_rank[parent_asin] = seen[signature]
            return sum(
                (weight / prior_total)
                * self._utility(partition_rank[parent_asin], turn + 1)
                for parent_asin, weight in zip(belief, prior_weights)
            )

        baseline_utility = expected_utility(baseline_attribute)
        scored = [(attribute, expected_utility(attribute)) for attribute in actions]
        priority = {
            attribute: index for index, attribute in enumerate(QUESTION_ATTRIBUTES)
        }
        best_attribute, best_utility = min(
            scored,
            key=lambda item: (-item[1], priority[item[0]]),
        )
        gain = best_utility - baseline_utility
        selected = (
            best_attribute
            if best_attribute != baseline_attribute
            and gain >= self.config.min_question_gain
            else None
        )
        return selected, {
            "active": True,
            "reason": "gain_threshold_passed" if selected else "retain_predecessor",
            "belief_count": len(belief),
            "displayed_posterior_count": len(displayed & posterior),
            "conditioned_on_current_miss": condition_on_current_miss,
            "baseline_attribute": baseline_attribute,
            "selected_attribute": selected,
            "baseline_expected_utility": round(baseline_utility, 9),
            "best_expected_utility": round(best_utility, 9),
            "expected_gain": round(gain, 9),
            "minimum_gain": self.config.min_question_gain,
            "action_expected_utility": {
                attribute: round(value, 9) for attribute, value in scored
            },
        }
