from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from typing import Mapping, Sequence

from starter.constraints import (
    COLOR_RE,
    MATERIAL_RE,
    Constraint,
    Facet,
    Polarity,
    Strength,
    infer_facet,
)


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for",
    "from", "i", "in", "is", "it", "me", "my", "of", "on", "or",
    "please", "some", "that", "the", "this", "to", "want", "with",
    "would", "you", "looking", "about", "additional", "ask", "attribute",
    "don", "earlier", "have", "ignore", "judgment", "matters", "need",
    "not", "one", "options", "preference", "quite", "right", "specific",
    "those", "yet",
}
ROUTE_FAMILY_ORDER = ("current_turn", "ledger", "category", "facet")
BM25_ORDER = "bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0)"
GENERIC_CATEGORIES = {
    "clothing", "clothing shoes jewelry", "clothing shoes and jewelry",
}


@dataclass(frozen=True)
class RetrievalConfig:
    """Declared E4/E4.1 switches shared by production and ablations."""

    enabled: bool = True
    use_current_turn_route: bool = True
    use_ledger_route: bool = True
    use_category_route: bool = True
    use_facet_route: bool = True
    use_constraint_reranking: bool = True
    use_soft_relaxation: bool = True
    use_strict_front: bool = True
    use_auxiliary_confidence_gate: bool = True
    relaxed_backfill_slots: int = 2
    candidate_union_depth: int = 100
    route_depth: int = 200
    max_facet_constraints: int = 4
    rrf_k: float = 60.0

    def __post_init__(self) -> None:
        if not 1 <= self.candidate_union_depth <= 100:
            raise ValueError("candidate_union_depth must be between 1 and 100")
        if self.route_depth < self.candidate_union_depth:
            raise ValueError("route_depth must cover candidate_union_depth")
        if self.max_facet_constraints < 1:
            raise ValueError("max_facet_constraints must be positive")
        if self.rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        if not 0 <= self.relaxed_backfill_slots <= 2:
            raise ValueError("relaxed_backfill_slots must be between 0 and 2")
        if self.enabled and not any(
            (
                self.use_current_turn_route,
                self.use_ledger_route,
                self.use_category_route,
                self.use_facet_route,
            )
        ):
            raise ValueError("at least one retrieval route must be enabled")


def e4_fallback_config() -> RetrievalConfig:
    """Return the frozen full-E4 ranking configuration."""

    return RetrievalConfig(
        use_strict_front=False,
        use_auxiliary_confidence_gate=False,
    )


def e4_1_candidate_config() -> RetrievalConfig:
    """Return the complete E4.1 strict-front/recall-backfill experiment."""

    return RetrievalConfig()


def e4_1_strict_only_config() -> RetrievalConfig:
    """Return the public-only strict diagnostic; not a promotable E4.1 policy."""

    return RetrievalConfig(use_soft_relaxation=False)


@dataclass(frozen=True)
class RouteSpec:
    name: str
    family: str
    expression: str
    fallback_expression: str | None
    weight: float


@dataclass(frozen=True)
class RetrievalResult:
    recommendation_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    trace: dict


@dataclass
class _Candidate:
    parent_asin: str
    route_ranks: dict[str, int]
    fusion_score: float = 0.0
    facet_fusion_score: float = 0.0
    auxiliary_fusion_score: float = 0.0
    gated_fusion_score: float = 0.0
    exact_coverage: float = 0.0
    current_turn_exact_coverage: float = 0.0
    normalized_fusion_score: float = 0.0
    evidence_confidence: float = 0.0
    strict: bool = False
    unknown_must_count: int = 0
    mismatched_must_count: int = 0


def _raw_terms(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def _search_terms(text: str) -> list[str]:
    return [
        token
        for token in _raw_terms(text)
        if len(token) > 1 and token not in STOPWORDS
    ]


def _unique(values: Sequence[str], limit: int | None = None) -> list[str]:
    result = list(dict.fromkeys(value for value in values if value))
    return result if limit is None else result[:limit]


def _catalog_norm(value: object) -> str:
    return " ".join(_raw_terms(str(value)))


def _normalization_aliases(value: str) -> tuple[str, ...]:
    normalized = _catalog_norm(value)
    return tuple(
        _unique(
            [
                normalized,
                re.sub(r"\bgrey\b", "gray", normalized),
                re.sub(r"\bgray\b", "grey", normalized),
            ]
        )
    )


def _coarse_category(values: object) -> str:
    if not isinstance(values, list):
        return ""
    cleaned: list[str] = []
    for value in values:
        for part in str(value).split(","):
            normalized = _catalog_norm(part)
            if normalized and normalized not in GENERIC_CATEGORIES:
                cleaned.append(part.strip())
    return " ".join(cleaned[-2:]) if cleaned else ""


def _scalar_values(product: Mapping[str, object]) -> list[str]:
    values: list[str] = []
    features = product.get("features")
    if isinstance(features, list):
        values.extend(str(value) for value in features if value not in (None, ""))
    details = product.get("details")
    if isinstance(details, dict):
        for key, value in details.items():
            if value not in (None, "", []):
                # Preserve the catalog scalar exactly as customer constraints
                # expose it (for example, "Department: Women"). A second bare
                # value row duplicates the same evidence at substantial index
                # cost and is intentionally omitted.
                values.append(f"{key}: {value}")
    title = product.get("title")
    if title not in (None, ""):
        values.append(str(title))
    price = product.get("price")
    if price not in (None, ""):
        values.append(f"budget around ${price}")
    return values


def retrieval_index_rows(
    product: Mapping[str, object],
) -> tuple[tuple[str, str, float, int], list[tuple[str, str]]]:
    """Return catalog-only side-table rows for exact/facet retrieval."""

    parent_asin = str(product["parent_asin"])
    try:
        average_rating = float(product.get("average_rating") or 0.0)
    except (TypeError, ValueError):
        average_rating = 0.0
    try:
        rating_number = max(0, int(product.get("rating_number") or 0))
    except (TypeError, ValueError):
        rating_number = 0
    meta = (
        parent_asin,
        _catalog_norm(_coarse_category(product.get("categories"))),
        average_rating,
        rating_number,
    )
    lookups: list[tuple[str, str]] = []
    seen: set[str] = set()
    for scalar in _scalar_values(product):
        compact = re.sub(r"\s+", " ", scalar).strip(" -;,.\t\n")
        for value in (compact, compact[:180].rstrip()):
            for alias in _normalization_aliases(value):
                if alias and alias not in seen:
                    lookups.append((alias, parent_asin))
                    seen.add(alias)
    return meta, lookups


def retrieval_presence_rows(product: Mapping[str, object]) -> list[tuple[str, str]]:
    """Return catalog-observable facet-presence rows for UNKNOWN handling.

    Presence is deliberately weaker than a match.  A candidate lacking any
    observable value for a routed facet is UNKNOWN and remains neutral; a
    candidate with observable evidence that does not match is a mismatch.
    Price is represented as budget evidence only when a usable value exists.
    """

    parent_asin = str(product["parent_asin"])
    facets = {
        Facet(facet)
        for _, facet, _ in retrieval_facet_value_rows(product)
    }
    if product.get("categories"):
        facets.add(Facet.CATEGORY)
    return [(parent_asin, facet.value) for facet in sorted(facets, key=lambda item: item.value)]


def retrieval_facet_value_rows(
    product: Mapping[str, object],
) -> list[tuple[str, str, str]]:
    """Return auditable facet/value evidence used only for compatibility.

    Whole catalog scalars retain E4's equality semantics. Material and color
    tokens also receive canonical rows so a visible structured clue such as
    ``leather`` is compatible with a catalog scalar such as ``100% Leather``.
    These rows do not add retrieval routes, preserving frozen-E4 ranking.
    """

    parent_asin = str(product["parent_asin"])
    rows: set[tuple[str, str, str]] = set()
    for scalar in _scalar_values(product):
        compact = re.sub(r"\s+", " ", scalar).strip(" -;,.\t\n")
        facet = infer_facet(compact)
        for alias in _normalization_aliases(compact):
            if alias:
                rows.add((parent_asin, facet.value, alias))
        for match in MATERIAL_RE.finditer(compact):
            rows.add((parent_asin, Facet.MATERIAL.value, _catalog_norm(match.group(0))))
        for match in COLOR_RE.finditer(compact):
            normalized = _catalog_norm(match.group(0))
            for alias in _normalization_aliases(normalized):
                rows.add((parent_asin, Facet.COLOR.value, alias))
    return sorted(rows, key=lambda row: (row[1], row[2]))


def has_focused_evidence(constraints: Sequence[Constraint]) -> bool:
    return any(
        item.facet is not Facet.CATEGORY
        and item.polarity is Polarity.POSITIVE
        and item.strength in {Strength.MUST, Strength.SHOULD}
        and item.normalized_value
        for item in constraints
    )


def _positive_constraints(constraints: Sequence[Constraint]) -> list[Constraint]:
    return [
        item
        for item in constraints
        if item.polarity is Polarity.POSITIVE
        and item.strength in {Strength.MUST, Strength.SHOULD}
        and item.normalized_value
    ]


def _or_expression(terms: Sequence[str], column: str | None = None) -> str:
    body = " OR ".join(f'"{term}"' for term in _unique(terms, 80))
    return f"{column} : ({body})" if body and column else body


def _and_expression(terms: Sequence[str], column: str | None = None) -> str:
    body = " AND ".join(f'"{term}"' for term in _unique(terms, 40))
    return f"{column} : ({body})" if body and column else body


def _entry_weight(item: Constraint, turn: int) -> float:
    strength = 1.35 if item.strength is Strength.MUST else 0.85
    recency = 1.25 if item.source_turn == turn else 1.0
    return strength * recency * max(0.0, min(1.0, float(item.confidence)))


def _route_specs(
    constraints: Sequence[Constraint],
    turn: int,
    config: RetrievalConfig,
) -> tuple[RouteSpec, ...]:
    positive = _positive_constraints(constraints)
    noncategory = [item for item in positive if item.facet is not Facet.CATEGORY]
    specs: list[RouteSpec] = []
    if config.use_ledger_route:
        terms = _search_terms(" ".join(item.normalized_value for item in positive))
        if expression := _or_expression(terms):
            specs.append(RouteSpec("ledger", "ledger", expression, None, 1.00))
    if config.use_category_route:
        terms = _search_terms(
            " ".join(
                item.raw_value
                for item in positive
                if item.facet is Facet.CATEGORY
            )
        )
        if expression := _and_expression(terms, "categories"):
            specs.append(
                RouteSpec(
                    "category", "category", expression,
                    _or_expression(terms, "categories"), 1.10,
                )
            )
    if config.use_current_turn_route:
        current = [item for item in noncategory if item.source_turn == turn]
        terms = _search_terms(" ".join(item.raw_value for item in current))
        if expression := _or_expression(terms):
            specs.append(RouteSpec("current_turn", "current_turn", expression, None, 1.25))
    deduplicated: list[RouteSpec] = []
    seen: set[str] = set()
    for spec in specs:
        if spec.expression in seen:
            continue
        deduplicated.append(spec)
        seen.add(spec.expression)
    return tuple(deduplicated)


class MultiRouteRetriever:
    """Fuse bounded catalog-only routes and deterministically rerank the union."""

    def __init__(self, connection: sqlite3.Connection, config: RetrievalConfig) -> None:
        self.connection = connection
        self.config = config

    def _fts_rank(self, spec: RouteSpec, raw_limit: int) -> tuple[list[str], bool]:
        sql = (
            "SELECT parent_asin FROM products WHERE products MATCH ? "
            f"ORDER BY {BM25_ORDER}, parent_asin LIMIT ?"
        )
        rows = self.connection.execute(sql, (spec.expression, raw_limit)).fetchall()
        used_fallback = False
        if not rows and spec.fallback_expression and spec.fallback_expression != spec.expression:
            rows = self.connection.execute(
                sql, (spec.fallback_expression, raw_limit)
            ).fetchall()
            used_fallback = True
        return [str(row[0]) for row in rows], used_fallback

    def _exact_rank(
        self,
        item: Constraint,
        category: Constraint | None,
        raw_limit: int,
    ) -> tuple[list[str], list[str]]:
        aliases = _unique(
            [
                *_normalization_aliases(item.raw_value),
                *_normalization_aliases(item.normalized_value),
            ]
        )
        if not aliases:
            return [], []
        placeholders = ",".join("?" for _ in aliases)
        exact = [
            str(row[0])
            for row in self.connection.execute(
                "SELECT DISTINCT rv.parent_asin FROM retrieval_values rv "
                "JOIN retrieval_meta rm ON rm.parent_asin=rv.parent_asin "
                f"WHERE rv.lookup_norm IN ({placeholders}) "
                "ORDER BY rm.rating_number DESC, rm.average_rating DESC, "
                "rv.parent_asin LIMIT ?",
                (*aliases, raw_limit),
            )
        ]
        category_exact: list[str] = []
        if category is not None:
            category_aliases = _unique(
                [
                    *_normalization_aliases(category.raw_value),
                    *_normalization_aliases(category.normalized_value),
                ]
            )
            category_placeholders = ",".join("?" for _ in category_aliases)
            category_exact = [
                str(row[0])
                for row in self.connection.execute(
                    "SELECT DISTINCT rv.parent_asin FROM retrieval_values rv "
                    "JOIN retrieval_meta rm ON rm.parent_asin=rv.parent_asin "
                    f"WHERE rv.lookup_norm IN ({placeholders}) "
                    f"AND rm.coarse_norm IN ({category_placeholders}) "
                    "ORDER BY rm.rating_number DESC, rm.average_rating DESC, "
                    "rv.parent_asin LIMIT ?",
                    (*aliases, *category_aliases, raw_limit),
                )
            ]
        return exact, category_exact

    @staticmethod
    def _eligible_ranked(ranked: Sequence[str], shown_ids: set[str]) -> list[str]:
        return [parent_asin for parent_asin in ranked if parent_asin not in shown_ids]

    def _facet_values(
        self,
        identifiers: Sequence[str],
    ) -> dict[str, dict[str, set[str]]]:
        """Load bounded compatibility evidence without SQLite bind overflow."""

        values: dict[str, dict[str, set[str]]] = {
            str(identifier): {} for identifier in identifiers
        }
        for start in range(0, len(identifiers), 500):
            chunk = [str(identifier) for identifier in identifiers[start : start + 500]]
            if not chunk:
                continue
            placeholders = ",".join("?" for _ in chunk)
            rows = self.connection.execute(
                "SELECT parent_asin, facet, lookup_norm FROM retrieval_facet_values "
                f"WHERE parent_asin IN ({placeholders})",
                chunk,
            ).fetchall()
            for parent_asin, facet, lookup_norm in rows:
                values.setdefault(str(parent_asin), {}).setdefault(
                    str(facet), set()
                ).add(str(lookup_norm))
        return values

    def search(
        self,
        *,
        constraints: Sequence[Constraint],
        avoid_values: Sequence[str],
        shown_ids: set[str],
        explore_unseen: bool,
        turn: int,
        requested_k: int,
        ledger_ranking: Sequence[str] | None = None,
    ) -> RetrievalResult:
        if avoid_values:
            raise ValueError("E4 retrieval expects the exclusion-safe E3 fallback")
        requested_k = min(requested_k, self.config.candidate_union_depth)
        prior_miss = bool(shown_ids)
        effective_relaxed_slots = min(
            self.config.relaxed_backfill_slots,
            self.config.relaxed_backfill_slots if prior_miss else 1,
        )
        specs = _route_specs(constraints, turn, self.config)
        positive = _positive_constraints(constraints)
        noncategory = [item for item in positive if item.facet is not Facet.CATEGORY]
        category = next(
            (item for item in positive if item.facet is Facet.CATEGORY), None
        )
        raw_limit = min(
            self.config.route_depth
            + (len(shown_ids) if explore_unseen else 0),
            1000,
        )
        route_rankings: list[tuple[str, str, float, list[str], bool]] = []
        for spec in specs:
            if spec.name == "ledger" and ledger_ranking is not None:
                ranked = list(ledger_ranking[:raw_limit])
                used_fallback = False
            else:
                ranked, used_fallback = self._fts_rank(spec, raw_limit)
            eligible = self._eligible_ranked(ranked, shown_ids) if explore_unseen else ranked
            route_rankings.append(
                (spec.name, spec.family, spec.weight, eligible, used_fallback)
            )
        facet_items = sorted(
            noncategory,
            key=lambda item: (item.source_turn, item.constraint_id),
            reverse=True,
        )[: self.config.max_facet_constraints]
        if self.config.use_facet_route:
            for item in reversed(facet_items):
                exact, category_exact = self._exact_rank(item, category, raw_limit)
                weight = _entry_weight(item, turn)
                if exact:
                    eligible = self._eligible_ranked(exact, shown_ids) if explore_unseen else exact
                    route_rankings.append(
                        (f"facet_exact:{item.constraint_id}", "facet", 1.35 * weight, eligible, False)
                    )
                if category_exact:
                    eligible = self._eligible_ranked(category_exact, shown_ids) if explore_unseen else category_exact
                    route_rankings.append(
                        (f"facet_category_exact:{item.constraint_id}", "facet", 2.00 * weight, eligible, False)
                    )

        candidates: dict[str, _Candidate] = {}
        route_traces: list[dict] = []
        for name, family, weight, ranked, used_fallback in route_rankings:
            for rank, parent_asin in enumerate(ranked, start=1):
                candidate = candidates.setdefault(parent_asin, _Candidate(parent_asin, {}))
                candidate.route_ranks[name] = rank
                contribution = weight / (self.config.rrf_k + rank)
                candidate.fusion_score += contribution
                if family == "facet":
                    candidate.facet_fusion_score += contribution
                else:
                    candidate.auxiliary_fusion_score += contribution
            route_traces.append(
                {
                    "name": name,
                    "family": family,
                    "weight": round(weight, 9),
                    "eligible_count": len(ranked),
                    "used_fallback": used_fallback,
                }
            )

        max_fusion = max((item.fusion_score for item in candidates.values()), default=1.0)
        # Price is always soft in E4.1. Missing or inconsistent price metadata
        # must never turn an otherwise compatible parent product into a hard
        # violation.
        routed_must = [
            item
            for item in facet_items
            if (
                item.strength is Strength.MUST
                and item.facet is not Facet.BUDGET
                and self.config.use_facet_route
            )
        ]
        facet_values = self._facet_values(tuple(candidates))
        for candidate in candidates.values():
            exact_matches = 0
            current_matches = 0
            strict_matches: list[bool] = []
            unknown_must_count = 0
            mismatched_must_count = 0
            # Confidence only covers constraints that actually received exact
            # routes. Older ledger evidence is still useful for recall, but it
            # cannot dilute this bounded exact-evidence denominator.
            confidence_items = [
                item for item in facet_items if item.facet is not Facet.BUDGET
            ]
            for item in noncategory:
                observed_values = facet_values.get(candidate.parent_asin, {}).get(
                    item.facet.value, set()
                )
                aliases = {
                    *_normalization_aliases(item.raw_value),
                    *_normalization_aliases(item.normalized_value),
                }
                exact = bool(observed_values.intersection(aliases))
                if item in confidence_items:
                    exact_matches += int(exact)
                    if exact and item.source_turn == turn:
                        current_matches += 1
                if item in routed_must:
                    known = bool(observed_values)
                    if not exact and not known:
                        unknown_must_count += 1
                    elif not exact:
                        mismatched_must_count += 1
                    # UNKNOWN remains eligible and unpenalized, but only a
                    # confirmed exact match belongs in the strict front.
                    strict_matches.append(exact)
            candidate.exact_coverage = (
                exact_matches / len(confidence_items) if confidence_items else 0.0
            )
            current_count = sum(item.source_turn == turn for item in confidence_items)
            candidate.current_turn_exact_coverage = (
                current_matches / current_count if current_count else 0.0
            )
            candidate.normalized_fusion_score = (
                candidate.fusion_score / max_fusion if max_fusion else 0.0
            )
            candidate.evidence_confidence = candidate.exact_coverage
            gate = 0.15 + 0.85 * candidate.evidence_confidence
            mismatch_gate = 0.50 ** mismatched_must_count
            candidate.gated_fusion_score = (
                candidate.facet_fusion_score
                + candidate.auxiliary_fusion_score * gate * mismatch_gate
            )
            candidate.strict = bool(strict_matches) and all(strict_matches)
            candidate.unknown_must_count = unknown_must_count
            candidate.mismatched_must_count = mismatched_must_count

        strict_count = sum(item.strict for item in candidates.values())
        if self.config.use_constraint_reranking:
            if self.config.use_auxiliary_confidence_gate:
                ranked_candidates = sorted(
                    candidates.values(),
                    key=lambda item: (
                        -item.gated_fusion_score,
                        -item.current_turn_exact_coverage,
                        -item.exact_coverage,
                        item.route_ranks.get("ledger", 10**9),
                        min(item.route_ranks.values()),
                        item.parent_asin,
                    ),
                )
            else:
                ranked_candidates = sorted(
                    candidates.values(),
                    key=lambda item: (
                        -item.fusion_score,
                        item.route_ranks.get("ledger", 10**9),
                        min(item.route_ranks.values()),
                        item.parent_asin,
                    ),
                )
        else:
            ranked_candidates = sorted(
                candidates.values(),
                key=lambda item: (
                    item.route_ranks.get("ledger", 10**9),
                    min(item.route_ranks.values()),
                    item.parent_asin,
                ),
            )
        route_ranked_candidates = list(ranked_candidates)
        if routed_must and strict_count and not self.config.use_soft_relaxation:
            ranked_candidates = [item for item in ranked_candidates if item.strict]
        strict_front_applied = bool(
            self.config.use_strict_front
            and routed_must
            and strict_count
            and self.config.use_soft_relaxation
        )
        if strict_front_applied:
            strict_candidates = [item for item in ranked_candidates if item.strict]
            relaxed_candidates = [item for item in ranked_candidates if not item.strict]
            pool_relaxed = min(
                effective_relaxed_slots,
                len(relaxed_candidates),
                self.config.candidate_union_depth,
            )
            bounded = [
                *strict_candidates[: self.config.candidate_union_depth - pool_relaxed],
                *relaxed_candidates[:pool_relaxed],
            ]
            if len(bounded) < self.config.candidate_union_depth:
                used = {item.parent_asin for item in bounded}
                bounded.extend(
                    item
                    for item in [*strict_candidates, *relaxed_candidates]
                    if item.parent_asin not in used
                )
                bounded = bounded[: self.config.candidate_union_depth]
        else:
            strict_candidates = []
            relaxed_candidates = []
            bounded = ranked_candidates[: self.config.candidate_union_depth]

        broader_relaxation = False
        if strict_front_applied:
            # A strict candidate always owns the leading slot. Relaxed safety
            # slots are lower-ranked recovery positions, never the whole page.
            reserved = min(effective_relaxed_slots, max(0, requested_k - 1))
            strict_target = requested_k - reserved
            if len(strict_candidates) >= strict_target:
                recommendation_candidates = [
                    *strict_candidates[:strict_target],
                    *relaxed_candidates[:reserved],
                ]
                if len(recommendation_candidates) < requested_k:
                    used = {item.parent_asin for item in recommendation_candidates}
                    recommendation_candidates.extend(
                        item
                        for item in strict_candidates
                        if item.parent_asin not in used
                    )
            else:
                broader_relaxation = True
                recommendation_candidates = [
                    *strict_candidates,
                    *relaxed_candidates[: requested_k - len(strict_candidates)],
                ]
            recommendation_candidates = recommendation_candidates[:requested_k]
        else:
            broader_relaxation = bool(routed_must and not strict_count)
            recommendation_candidates = bounded[:requested_k]
        if strict_front_applied:
            used = {item.parent_asin for item in recommendation_candidates}
            bounded = [
                *recommendation_candidates,
                *(
                    item
                    for item in [*strict_candidates, *relaxed_candidates]
                    if item.parent_asin not in used
                ),
            ][: self.config.candidate_union_depth]
        eligible_candidate_ids = [item.parent_asin for item in bounded]
        recommendation_ids = [
            item.parent_asin for item in recommendation_candidates
        ]
        used_candidate_ids = set(recommendation_ids)
        candidate_ids = [
            *recommendation_ids,
            *(
                item.parent_asin
                for item in route_ranked_candidates
                if item.parent_asin not in used_candidate_ids
            ),
        ][: self.config.candidate_union_depth]
        route_candidate_ids = [
            item.parent_asin
            for item in route_ranked_candidates[: self.config.candidate_union_depth]
        ]
        enabled_families = [
            family
            for family in ROUTE_FAMILY_ORDER
            if any(route[1] == family for route in route_rankings)
        ]
        trace = {
            "turn": turn,
            "enabled_route_families": enabled_families,
            "routes": route_traces,
            "raw_union_count": len(candidates),
            "candidate_union_count": len(candidate_ids),
            "candidate_ids": candidate_ids,
            "route_candidate_count": len(route_candidate_ids),
            "route_candidate_ids": route_candidate_ids,
            "eligible_candidate_count": len(eligible_candidate_ids),
            "eligible_candidate_ids": eligible_candidate_ids,
            "recommendation_count": len(recommendation_ids),
            "recommendation_ids": recommendation_ids,
            "ranked_candidate_count": len(ranked_candidates),
            "strict_candidate_count": strict_count,
            "relaxed_candidates_used": (
                sum(not item.strict for item in recommendation_candidates)
                if routed_must
                else 0
            ),
            "routed_must_constraint_count": len(routed_must),
            "constraint_reranking": self.config.use_constraint_reranking,
            "soft_relaxation": self.config.use_soft_relaxation,
            "strict_front": self.config.use_strict_front,
            "strict_front_applied": strict_front_applied,
            "configured_relaxed_backfill_slots": self.config.relaxed_backfill_slots,
            "effective_relaxed_backfill_slots": effective_relaxed_slots,
            "prior_miss": prior_miss,
            "recovery_expanded_after_miss": bool(
                prior_miss and effective_relaxed_slots > 1
            ),
            "broader_relaxation": broader_relaxation,
            "auxiliary_confidence_gate": self.config.use_auxiliary_confidence_gate,
            "top_candidates": [
                {
                    "parent_asin": item.parent_asin,
                    "fusion_score": round(item.fusion_score, 12),
                    "exact_coverage": round(item.exact_coverage, 9),
                    "current_turn_exact_coverage": round(
                        item.current_turn_exact_coverage, 9
                    ),
                    "normalized_fusion_score": round(
                        item.normalized_fusion_score, 9
                    ),
                    "gated_fusion_score": round(item.gated_fusion_score, 12),
                    "evidence_confidence": round(item.evidence_confidence, 9),
                    "strict": item.strict,
                    "unknown_must_count": item.unknown_must_count,
                    "mismatched_must_count": item.mismatched_must_count,
                    "route_ranks": dict(sorted(item.route_ranks.items())),
                }
                for item in recommendation_candidates
            ],
        }
        return RetrievalResult(
            tuple(recommendation_ids),
            tuple(candidate_ids),
            trace,
        )
