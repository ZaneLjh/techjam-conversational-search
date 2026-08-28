from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from typing import Mapping, Sequence

from starter.constraints import Constraint, Facet, Polarity, Strength


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
    """Declared E4 switches shared by production and controlled ablations."""

    enabled: bool = True
    use_current_turn_route: bool = True
    use_ledger_route: bool = True
    use_category_route: bool = True
    use_facet_route: bool = True
    use_constraint_reranking: bool = True
    use_soft_relaxation: bool = True
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
        if self.enabled and not any(
            (
                self.use_current_turn_route,
                self.use_ledger_route,
                self.use_category_route,
                self.use_facet_route,
            )
        ):
            raise ValueError("at least one retrieval route must be enabled")


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
    trace: dict


@dataclass
class _Candidate:
    parent_asin: str
    route_ranks: dict[str, int]
    fusion_score: float = 0.0
    exact_coverage: float = 0.0
    current_turn_exact_coverage: float = 0.0
    normalized_fusion_score: float = 0.0
    strict: bool = False


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
                candidate.fusion_score += weight / (self.config.rrf_k + rank)
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
        routed_must = [
            item
            for item in facet_items
            if item.strength is Strength.MUST and self.config.use_facet_route
        ]
        for candidate in candidates.values():
            exact_matches = 0
            current_matches = 0
            strict_matches: list[bool] = []
            for item in noncategory:
                exact = any(
                    route_name in candidate.route_ranks
                    for route_name in (
                        f"facet_exact:{item.constraint_id}",
                        f"facet_category_exact:{item.constraint_id}",
                    )
                )
                exact_matches += int(exact)
                if exact and item.source_turn == turn:
                    current_matches += 1
                if item in routed_must:
                    strict_matches.append(exact)
            candidate.exact_coverage = (
                exact_matches / len(noncategory) if noncategory else 0.0
            )
            current_count = sum(item.source_turn == turn for item in noncategory)
            candidate.current_turn_exact_coverage = (
                current_matches / current_count if current_count else 0.0
            )
            candidate.normalized_fusion_score = (
                candidate.fusion_score / max_fusion if max_fusion else 0.0
            )
            candidate.strict = bool(strict_matches) and all(strict_matches)

        strict_count = sum(item.strict for item in candidates.values())
        if self.config.use_constraint_reranking:
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
        if routed_must and strict_count and not self.config.use_soft_relaxation:
            ranked_candidates = [item for item in ranked_candidates if item.strict]
        bounded = ranked_candidates[: self.config.candidate_union_depth]
        recommendation_candidates = bounded[:requested_k]
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
            "candidate_union_count": len(bounded),
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
                    "strict": item.strict,
                    "route_ranks": dict(sorted(item.route_ranks.items())),
                }
                for item in bounded[:10]
            ],
        }
        return RetrievalResult(
            tuple(item.parent_asin for item in recommendation_candidates),
            trace,
        )
