from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, TextIO

from evaluator.local_evaluator import (
    behavior_for,
    coarse_category,
    customer_reply,
    initial_message,
    intent_card,
)


DEFAULT_SEEDS = (1701, 1702, 1703)
SCENARIO_SHARES = (
    ("buying", 40),
    ("browsing", 40),
    ("intent_override", 15),
    ("boundary", 5),
)
JACCARD_THRESHOLD = 0.80
LENGTH_RATIO_THRESHOLD = 0.75
MIN_BLOCKED_TOKENS = 3

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_VARIANT_TOKENS = {
    "black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey",
    "purple", "yellow", "orange", "beige", "navy", "small", "medium", "large", "xl",
    "xxl", "xs", "xxs", "xxx", "inch", "inches", "cm", "mm", "oz", "ounce", "ounces",
    "pack", "packs", "count", "piece", "pieces", "pair", "pairs", "set", "sets",
}
_TITLE_STOPWORDS = {"a", "an", "and", "for", "in", "of", "the", "to", "with"}
_GENERIC_BRANDS = {"", "amazon", "generic", "none", "unknown", "unbranded"}
_SPARSE_FIELDS = ("features", "details", "description", "price", "store")


def _open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_bytes((_canonical_json(value) + "\n").encode("utf-8"))


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("wb") as handle:
        for row in rows:
            handle.write((_canonical_json(row) + "\n").encode("utf-8"))


def _normalize(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(_TOKEN_RE.findall(text))


def _tokens(value: object) -> tuple[str, ...]:
    return tuple(_normalize(value).split())


def _variant_stripped_title(value: object) -> tuple[str, ...]:
    result: list[str] = []
    for token in _tokens(value):
        if token in _TITLE_STOPWORDS or token in _VARIANT_TOKENS:
            continue
        if any(character.isdigit() for character in token):
            continue
        result.append(token)
    return tuple(dict.fromkeys(result))


def _flatten(value: object) -> Iterator[str]:
    if isinstance(value, dict):
        for key in sorted(value):
            item = value[key]
            if item not in (None, "", []):
                yield f"{key}: {item}"
    elif isinstance(value, list):
        for item in value:
            if item not in (None, ""):
                yield str(item)
    elif value not in (None, ""):
        yield str(value)


def _leaf_category(product: dict) -> str:
    categories = product.get("categories") or []
    if not isinstance(categories, list) or not categories:
        return ""
    parts = [part.strip() for part in str(categories[-1]).split(",") if part.strip()]
    return _normalize(parts[-1] if parts else categories[-1])


def _manufacturer(product: dict) -> str:
    details = product.get("details")
    if not isinstance(details, dict):
        return ""
    for key, value in details.items():
        if _normalize(key) == "manufacturer":
            return _normalize(value)
    return ""


def _brand(product: dict) -> str:
    brand = _normalize(product.get("store")) or _manufacturer(product)
    return "" if brand in _GENERIC_BRANDS else brand


def _model(product: dict) -> str:
    details = product.get("details")
    if not isinstance(details, dict):
        return ""
    for key, value in details.items():
        if _normalize(key) in {"item model number", "model number", "model"}:
            return _normalize(value)
    return ""


def _long_content(product: dict) -> str:
    content = " ".join([*_flatten(product.get("features")), *_flatten(product.get("description"))])
    normalized = _normalize(content)
    return normalized if len(normalized) >= 80 else ""


def _card_signature(product: dict) -> str:
    card = intent_card(product)
    normalized = {
        # The released initial_message() ignores intent_card.target_category
        # and renders the catalog-derived coarse category instead. Excluding
        # the title here groups products that are observationally equivalent
        # under the exact evaluator transcript even when their titles differ.
        "coarse_category": _normalize(
            coarse_category([str(value) for value in product.get("categories") or []])
        ),
        "hard_constraints": [_normalize(value) for value in card.get("hard_constraints", [])],
        "soft_preferences": [_normalize(value) for value in card.get("soft_preferences", [])],
    }
    return _canonical_json(normalized)


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        materialized = tuple(values)
        self.parent = {value: value for value in materialized}
        self.rank = {value: 0 for value in materialized}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


@dataclass(frozen=True)
class ProductFeatures:
    asin: str
    title: str
    card_signature: str
    brand: str
    model: str
    leaf: str
    long_content: str
    blocked_tokens: tuple[str, ...]


def _features(product: dict) -> ProductFeatures:
    return ProductFeatures(
        asin=str(product["parent_asin"]),
        title=_normalize(product.get("title")),
        card_signature=_card_signature(product),
        brand=_brand(product),
        model=_model(product),
        leaf=_leaf_category(product),
        long_content=_long_content(product),
        blocked_tokens=_variant_stripped_title(product.get("title")),
    )


def _union_equal_keys(uf: _UnionFind, rows: list[ProductFeatures], key_name: str) -> None:
    first: dict[object, str] = {}
    for row in rows:
        key = getattr(row, key_name)
        if not key:
            continue
        previous = first.setdefault(key, row.asin)
        uf.union(previous, row.asin)


def _union_compound_keys(
    uf: _UnionFind,
    rows: list[ProductFeatures],
    key_fn,
) -> None:
    first: dict[tuple[str, ...], str] = {}
    for row in rows:
        key = key_fn(row)
        if not key or any(not item for item in key):
            continue
        previous = first.setdefault(key, row.asin)
        uf.union(previous, row.asin)


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    return len(left & right) / len(left | right)


def _union_near_duplicate_blocks(uf: _UnionFind, rows: list[ProductFeatures]) -> None:
    blocks: dict[tuple[str, str], list[ProductFeatures]] = defaultdict(list)
    for row in rows:
        if row.brand and row.leaf and len(row.blocked_tokens) >= MIN_BLOCKED_TOKENS:
            blocks[(row.brand, row.leaf)].append(row)

    for block_rows in blocks.values():
        token_frequency = Counter(token for row in block_rows for token in set(row.blocked_tokens))
        inverted: dict[str, list[tuple[str, frozenset[str]]]] = defaultdict(list)
        ordered = sorted(block_rows, key=lambda row: (len(set(row.blocked_tokens)), row.asin))
        for row in ordered:
            token_set = frozenset(row.blocked_tokens)
            ordered_tokens = sorted(token_set, key=lambda token: (token_frequency[token], token))
            prefix_size = len(token_set) - math.ceil(JACCARD_THRESHOLD * len(token_set)) + 1
            candidates: dict[str, frozenset[str]] = {}
            for token in ordered_tokens[:prefix_size]:
                for candidate_asin, candidate_tokens in inverted[token]:
                    candidates[candidate_asin] = candidate_tokens
            for candidate_asin, candidate_tokens in candidates.items():
                length_ratio = min(len(token_set), len(candidate_tokens)) / max(len(token_set), len(candidate_tokens))
                if length_ratio >= LENGTH_RATIO_THRESHOLD and _jaccard(token_set, candidate_tokens) >= JACCARD_THRESHOLD:
                    uf.union(row.asin, candidate_asin)
            for token in ordered_tokens[:prefix_size]:
                inverted[token].append((row.asin, token_set))


def build_groups(products: dict[str, dict]) -> tuple[dict[str, tuple[str, ...]], dict[str, str]]:
    rows = [_features(products[asin]) for asin in sorted(products)]
    uf = _UnionFind(row.asin for row in rows)
    _union_equal_keys(uf, rows, "title")
    _union_equal_keys(uf, rows, "card_signature")
    _union_compound_keys(uf, rows, lambda row: (row.brand, row.model))
    _union_compound_keys(uf, rows, lambda row: (row.brand, row.leaf, row.long_content))
    _union_near_duplicate_blocks(uf, rows)

    components: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        components[uf.find(row.asin)].append(row.asin)
    groups: dict[str, tuple[str, ...]] = {}
    asin_to_group: dict[str, str] = {}
    for members_list in components.values():
        members = tuple(sorted(members_list))
        group_id = hashlib.sha256("\0".join(members).encode("utf-8")).hexdigest()
        groups[group_id] = members
        for asin in members:
            asin_to_group[asin] = group_id
    return dict(sorted(groups.items())), asin_to_group


def _load_catalog(path: Path) -> dict[str, dict]:
    products: dict[str, dict] = {}
    with _open_text(path) as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            product = json.loads(line)
            asin = str(product.get("parent_asin") or "").strip()
            if not asin:
                raise ValueError(f"catalog line {line_number} has no parent_asin")
            if asin in products:
                raise ValueError(f"duplicate catalog parent_asin: {asin}")
            products[asin] = product
    if not products:
        raise ValueError("catalog is empty")
    return products


def _load_public_targets(path: Path) -> set[str]:
    targets: set[str] = set()
    with _open_text(path) as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            sample = json.loads(line)
            ground_truth = sample.get("ground_truth")
            target = str(ground_truth.get("parent_asin") if isinstance(ground_truth, dict) else "").strip()
            if not target:
                raise ValueError(f"public dataset line {line_number} has no ground-truth parent_asin")
            targets.add(target)
    if not targets:
        raise ValueError("public dataset is empty")
    return targets


def _canonical_content_hash(products: dict[str, dict]) -> str:
    payload = "".join(_canonical_json(products[asin]) + "\n" for asin in sorted(products)).encode("utf-8")
    return _sha256_bytes(payload)


def _scenario_schedule(sessions_per_fold: int, seed: int, fold: int) -> list[str]:
    if sessions_per_fold <= 0 or sessions_per_fold % 20:
        raise ValueError("sessions-per-fold must be a positive multiple of 20")
    schedule: list[str] = []
    for scenario, percent in SCENARIO_SHARES:
        schedule.extend([scenario] * (sessions_per_fold * percent // 100))
    random.Random(f"e5-scenarios\0{seed}\0{fold}").shuffle(schedule)
    return schedule


def _user_profile(seed: int, fold: int, ordinal: int) -> dict:
    # Profiles are sampled independently of the target product. This prevents
    # a later personalization experiment from treating target-derived tags as
    # an accidental label channel even though the current E5 ignores profiles.
    tag_profiles = (
        ("fit", "comfort", "durability"),
        ("style", "comfort"),
        ("material", "fit"),
        ("value", "durability"),
    )
    tags = list(tag_profiles[(seed + fold + ordinal) % len(tag_profiles)])
    styles = ((5.0, "usually positive"), (3.0, "mixed"), (1.0, "critical"))
    average, rating_style = styles[(seed + fold + ordinal) % len(styles)]
    summary = f"Prior purchases emphasize {', '.join(tags)}; ratings are {rating_style}."
    return {
        "average_prior_rating": average,
        "preference_tags": tags,
        "purchase_frequency": "3-4 prior purchases",
        "rating_style": rating_style,
        "summary": summary,
    }


def _missing_fields(product: dict) -> list[str]:
    return [field for field in _SPARSE_FIELDS if product.get(field) in (None, "", [], {})]


def _sample(
    product: dict,
    scenario: str,
    seed: int,
    fold: int,
    ordinal: int,
    group_id: str,
    group_size: int,
) -> dict:
    card = intent_card(product)
    behavior = behavior_for(scenario, card, random.Random(f"e5-behavior\0{seed}\0{fold}\0{ordinal}"))
    missing = _missing_fields(product)
    probe_sample = {"behavior": behavior, "intent_card": card, "scenario_type": scenario}
    disclosed: set[str] = set()
    boundary_used = False
    initial_message(probe_sample, _leaf_category(product) or "clothing", disclosed)
    positive_other_replies = 0
    for _ in range(3):
        reply, boundary_used = customer_reply(probe_sample, "other", disclosed, boundary_used)
        positive_other_replies += int(reply.startswith("For that, what matters is:"))
    return {
        "behavior": behavior,
        "category_bucket": _leaf_category(product) or "clothing",
        "difficulty_bucket": "hard" if group_size > 1 or missing else "easy",
        "ground_truth": {"parent_asin": str(product["parent_asin"])},
        "intent_card": card,
        "sample_id": f"synthetic_s{seed}_f{fold}_{ordinal + 1:04d}",
        "scenario_type": scenario,
        "synthetic_metadata": {
            "fold": fold,
            "group_id": group_id,
            "seed": seed,
            "strata": {
                "collision_group": group_size > 1,
                "missing_fields": missing,
                "repeated_other_compatible": positive_other_replies >= 2,
                "repeated_other_positive_replies": positive_other_replies,
                "sparse_metadata": bool(missing),
            },
        },
        "user_profile": _user_profile(seed, fold, ordinal),
    }


def generate(
    catalog_path: str | Path,
    public_dataset_path: str | Path,
    output_dir: str | Path,
    folds: int = 5,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    sessions_per_fold: int = 200,
) -> dict:
    if folds < 2:
        raise ValueError("folds must be at least 2")
    seed_values = tuple(sorted(int(seed) for seed in seeds))
    if not seed_values or len(seed_values) != len(set(seed_values)):
        raise ValueError("seeds must be a non-empty list of unique integers")
    _scenario_schedule(sessions_per_fold, seed_values[0], 0)

    catalog_path = Path(catalog_path)
    public_dataset_path = Path(public_dataset_path)
    output_dir = Path(output_dir)
    products = _load_catalog(catalog_path)
    public_targets = _load_public_targets(public_dataset_path)
    missing_public_targets = sorted(public_targets - products.keys())
    if missing_public_targets:
        raise ValueError(f"public targets absent from catalog: {missing_public_targets[:5]}")

    groups, asin_to_group = build_groups(products)
    quarantined_groups = {asin_to_group[asin] for asin in public_targets}
    eligible_groups = [group_id for group_id in groups if group_id not in quarantined_groups]
    fold_order = sorted(
        eligible_groups,
        key=lambda group_id: (hashlib.sha256(f"e5-fold\0{group_id}".encode()).hexdigest(), group_id),
    )
    group_fold = {group_id: index % folds for index, group_id in enumerate(fold_order)}
    groups_by_fold = {
        fold: sorted(group_id for group_id, owner in group_fold.items() if owner == fold)
        for fold in range(folds)
    }
    for fold, fold_groups in groups_by_fold.items():
        required_groups = sessions_per_fold * len(seed_values)
        if len(fold_groups) < required_groups:
            raise ValueError(
                f"fold {fold} has {len(fold_groups)} eligible groups; {required_groups} required "
                "for cross-seed-disjoint selection"
            )

    samples: list[dict] = []
    selected_by_group: dict[str, list[dict[str, int]]] = defaultdict(list)
    cell_counts: list[dict] = []
    selected_groups_by_seed_fold: dict[tuple[int, int], list[str]] = {}
    for fold in range(folds):
        ranked_groups = sorted(
            groups_by_fold[fold],
            key=lambda group_id: (
                hashlib.sha256(f"e5-select\0{fold}\0{group_id}".encode()).hexdigest(),
                group_id,
            ),
        )
        for seed_index, seed in enumerate(seed_values):
            start = seed_index * sessions_per_fold
            selected_groups_by_seed_fold[(seed, fold)] = ranked_groups[start:start + sessions_per_fold]

    for seed in seed_values:
        for fold in range(folds):
            selected_groups = selected_groups_by_seed_fold[(seed, fold)]
            scenarios = _scenario_schedule(sessions_per_fold, seed, fold)
            scenario_counts = Counter(scenarios)
            for ordinal, (group_id, scenario) in enumerate(zip(selected_groups, scenarios)):
                members = groups[group_id]
                target_index = int(
                    hashlib.sha256(f"e5-target\0{seed}\0{fold}\0{group_id}".encode()).hexdigest(), 16
                ) % len(members)
                target = members[target_index]
                samples.append(_sample(products[target], scenario, seed, fold, ordinal, group_id, len(members)))
                selected_by_group[group_id].append({"fold": fold, "seed": seed})
            cell_counts.append({
                "fold": fold,
                "scenario_counts": dict(sorted(scenario_counts.items())),
                "seed": seed,
                "session_count": len(selected_groups),
            })

    samples.sort(key=lambda sample: sample["sample_id"])
    selected_target_ids = {sample["ground_truth"]["parent_asin"] for sample in samples}
    selected_groups_by_fold = {
        fold: {
            sample["synthetic_metadata"]["group_id"]
            for sample in samples
            if sample["synthetic_metadata"]["fold"] == fold
        }
        for fold in range(folds)
    }
    fold_pair_overlaps = []
    for left in range(folds):
        for right in range(left + 1, folds):
            fold_pair_overlaps.append({
                "folds": [left, right],
                "overlap_count": len(selected_groups_by_fold[left] & selected_groups_by_fold[right]),
            })
    public_quarantine_ok = not (selected_target_ids & public_targets) and not (
        set().union(*selected_groups_by_fold.values()) & quarantined_groups
    )
    group_disjoint = all(item["overlap_count"] == 0 for item in fold_pair_overlaps)
    cross_seed_overlaps = []
    for fold in range(folds):
        for left_index, left_seed in enumerate(seed_values):
            for right_seed in seed_values[left_index + 1:]:
                left_groups = set(selected_groups_by_seed_fold[(left_seed, fold)])
                right_groups = set(selected_groups_by_seed_fold[(right_seed, fold)])
                left_targets = {
                    sample["ground_truth"]["parent_asin"]
                    for sample in samples
                    if sample["synthetic_metadata"]["fold"] == fold
                    and sample["synthetic_metadata"]["seed"] == left_seed
                }
                right_targets = {
                    sample["ground_truth"]["parent_asin"]
                    for sample in samples
                    if sample["synthetic_metadata"]["fold"] == fold
                    and sample["synthetic_metadata"]["seed"] == right_seed
                }
                cross_seed_overlaps.append({
                    "fold": fold,
                    "group_overlap_count": len(left_groups & right_groups),
                    "seeds": [left_seed, right_seed],
                    "target_overlap_count": len(left_targets & right_targets),
                })
    cross_seed_disjoint = all(
        item["group_overlap_count"] == 0 and item["target_overlap_count"] == 0
        for item in cross_seed_overlaps
    )
    if not public_quarantine_ok or not group_disjoint or not cross_seed_disjoint:
        raise RuntimeError("synthetic split leakage detected")

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "e5_synthetic_sessions.jsonl"
    group_map_path = output_dir / "e5_synthetic_group_map.jsonl"
    audit_path = output_dir / "e5_synthetic_leakage_audit.json"
    manifest_path = output_dir / "e5_synthetic_manifest.json"
    checksums_path = output_dir / "SHA256SUMS"
    _write_jsonl(dataset_path, samples)

    group_rows = []
    for group_id, members in groups.items():
        group_rows.append({
            "fold": group_fold.get(group_id),
            "group_id": group_id,
            "member_asins": list(members),
            "quarantined": group_id in quarantined_groups,
            "selected_by": sorted(selected_by_group.get(group_id, []), key=lambda item: (item["seed"], item["fold"])),
        })
    _write_jsonl(group_map_path, group_rows)

    strata_counts = Counter()
    for sample in samples:
        strata = sample["synthetic_metadata"]["strata"]
        for name in ("collision_group", "repeated_other_compatible", "sparse_metadata"):
            strata_counts[name] += int(strata[name])
        strata_counts["correction_intent_override"] += int(sample["scenario_type"] == "intent_override")
    audit = {
        "cross_seed_disjoint": cross_seed_disjoint,
        "cross_seed_overlaps_by_fold": cross_seed_overlaps,
        "fold_pair_overlaps": fold_pair_overlaps,
        "group_disjoint_folds": group_disjoint,
        "public_target_count": len(public_targets),
        "public_target_groups_quarantined": public_quarantine_ok,
        "quarantined_group_count": len(quarantined_groups),
        "selected_public_target_count": len(selected_target_ids & public_targets),
    }
    _write_json(audit_path, audit)

    manifest = {
        "catalog_canonical_sorted_records_sha256": _canonical_content_hash(products),
        "catalog_input_sha256": _sha256_file(catalog_path),
        "cell_counts": cell_counts,
        "dataset": dataset_path.name,
        "dataset_sha256": _sha256_file(dataset_path),
        "dialogue_probe_protocol": {
            "repeated_other_compatible": (
                "evaluator initial_message followed by three customer_reply calls with ask_attribute=other; "
                "at least two replies disclose constraints"
            ),
        },
        "fold_count": folds,
        "fold_group_counts": {str(fold): len(groups_by_fold[fold]) for fold in range(folds)},
        "group_count": len(groups),
        "group_disjoint_folds": group_disjoint,
        "cross_seed_disjoint": cross_seed_disjoint,
        "grouping_protocol": {
            "blocked_jaccard_threshold": JACCARD_THRESHOLD,
            "blocked_length_ratio_threshold": LENGTH_RATIO_THRESHOLD,
            "blocked_minimum_tokens": MIN_BLOCKED_TOKENS,
            "equivalence_rules": [
                "exact_normalized_title",
                "exact_evaluator_intent_card",
                "same_non_generic_brand_or_manufacturer_and_model",
                "same_brand_leaf_and_exact_long_feature_description",
                "same_brand_leaf_and_blocked_variant_stripped_title_similarity",
            ],
            "group_id": "sha256(sorted member parent_asin values joined by NUL)",
        },
        "public_dataset_input_sha256": _sha256_file(public_dataset_path),
        "public_target_set_sha256": _sha256_bytes(
            "".join(sorted(public_targets)).encode("utf-8")
        ),
        "public_target_groups_quarantined": public_quarantine_ok,
        "scenario_percentages": {scenario: percent for scenario, percent in SCENARIO_SHARES},
        "seed_count": len(seed_values),
        "seeds": list(seed_values),
        "selection_protocol": {
            "collision_group": "evaluation-equivalence group has more than one catalog product",
            "cross_seed_reuse": "forbidden within every fold",
            "sparse_metadata": f"one or more fields absent: {', '.join(_SPARSE_FIELDS)}",
        },
        "session_count": len(samples),
        "sessions_per_seed_fold": sessions_per_fold,
        "strata_counts": dict(sorted(strata_counts.items())),
    }
    _write_json(manifest_path, manifest)

    checksum_files = (dataset_path, group_map_path, audit_path, manifest_path)
    checksums = "".join(f"{_sha256_file(path)}  {path.name}\n" for path in checksum_files)
    checksums_path.write_bytes(checksums.encode("ascii"))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic, leakage-audited E5 synthetic sessions")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--public-dataset", default="data/public_set.jsonl")
    parser.add_argument("--output-dir", default="results/e5_synthetic")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--sessions-per-fold", type=int, default=200)
    args = parser.parse_args()
    manifest = generate(
        args.catalog,
        args.public_dataset,
        args.output_dir,
        folds=args.folds,
        seeds=args.seeds,
        sessions_per_fold=args.sessions_per_fold,
    )
    print(_canonical_json(manifest))


if __name__ == "__main__":
    main()
