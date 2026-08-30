from __future__ import annotations

import math
import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from typing import Sequence

from starter.constraints import (
    Constraint,
    ConstraintStatus,
    Facet,
    Polarity,
    Strength,
)
from starter.retrieval import _normalization_aliases


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
TOKEN_ALIASES = {
    "grey": "gray",
    "colour": "color",
    "tee": "shirt",
    "tshirt": "shirt",
    "trainer": "sneaker",
    "trainers": "sneaker",
    "sneakers": "sneaker",
    "rucksack": "backpack",
    "handbag": "purse",
    "trousers": "pant",
    "pants": "pant",
}


@dataclass(frozen=True)
class RerankingConfig:
    """Bounded semantic switches for the guarded E5 experiment."""

    enabled: bool = False
    enforce_projection_candidate_membership: bool = False
    candidate_depth: int = 100
    max_constraints: int = 6
    use_exact_priority: bool = True
    use_fuzzy_similarity: bool = True
    use_candidate_idf: bool = True
    use_quality_tiebreak: bool = False
    fuzzy_threshold: float = 0.72

    def __post_init__(self) -> None:
        if not 1 <= self.candidate_depth <= 100:
            raise ValueError("candidate_depth must be between 1 and 100")
        if self.max_constraints < 1:
            raise ValueError("max_constraints must be positive")
        if not 0.5 <= self.fuzzy_threshold <= 1.0:
            raise ValueError("fuzzy_threshold must be between 0.5 and 1.0")


@dataclass(frozen=True)
class RerankingResult:
    recommendation_ids: tuple[str, ...]
    trace: dict


@dataclass(frozen=True)
class _QueryConstraint:
    constraint_id: int
    facet: Facet
    strength: Strength
    phrase: str
    lookup_values: tuple[str, ...]
    tokens: tuple[str, ...]
    weight: float


@dataclass(frozen=True)
class _Document:
    parent_asin: str
    full_text: str
    full_tokens: frozenset[str]
    category_text: str
    category_tokens: frozenset[str]
    average_rating: float
    rating_number: int


@dataclass
class _ScoredCandidate:
    parent_asin: str
    original_rank: int
    exact_priority: float = 0.0
    semantic_score: float = 0.0
    quality_score: float = 0.0
    matched_must_count: int = 0
    unknown_must_count: int = 0
    mismatched_must_count: int = 0
    exact_constraint_ids: tuple[int, ...] = ()
    constraint_scores: tuple[float, ...] = ()


def _normalize(value: object) -> str:
    text = (
        unicodedata.normalize("NFKD", str(value))
        .encode("ascii", "ignore")
        .decode()
    )
    tokens = [
        TOKEN_ALIASES.get(token.lower(), token.lower())
        for token in TOKEN_RE.findall(text)
    ]
    return " ".join(tokens)


def _stem(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _tokens(value: object) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            _stem(token)
            for token in _normalize(value).split()
            if len(token) > 1 and token not in STOPWORDS
        )
    )


def _trigrams(token: str) -> frozenset[str]:
    padded = f"  {token}  "
    return frozenset(padded[index : index + 3] for index in range(len(padded) - 2))


def _trigram_similarity(left: str, right: str) -> float:
    left_grams = _trigrams(left)
    right_grams = _trigrams(right)
    if not left_grams or not right_grams:
        return 0.0
    return 2.0 * len(left_grams & right_grams) / (len(left_grams) + len(right_grams))


def _contains_phrase(text: str, phrase: str) -> bool:
    """Return contiguous normalized-token containment, never substrings."""

    return bool(phrase) and f" {phrase} " in f" {text} "


def _positive_constraints(
    constraints: Sequence[Constraint],
    turn: int,
    limit: int,
) -> tuple[_QueryConstraint, ...]:
    active = [
        item
        for item in constraints
        if item.status is ConstraintStatus.ACTIVE
        and item.polarity is Polarity.POSITIVE
        and item.strength in {Strength.MUST, Strength.SHOULD}
        and item.normalized_value
    ]
    categories = [item for item in active if item.facet is Facet.CATEGORY]
    focused = sorted(
        (item for item in active if item.facet is not Facet.CATEGORY),
        key=lambda item: (item.source_turn, item.constraint_id),
        reverse=True,
    )[: max(0, limit - bool(categories))]
    selected = [*categories[-1:], *reversed(focused)]
    result: list[_QueryConstraint] = []
    for item in selected:
        query_tokens = _tokens(item.raw_value or item.normalized_value)
        if not query_tokens:
            continue
        strength = 1.35 if item.strength is Strength.MUST else 0.85
        recency = 1.15 if item.source_turn == turn else 1.0
        facet = 0.55 if item.facet is Facet.CATEGORY else 1.0
        result.append(
            _QueryConstraint(
                constraint_id=item.constraint_id,
                facet=item.facet,
                strength=item.strength,
                phrase=_normalize(item.raw_value or item.normalized_value),
                lookup_values=tuple(
                    dict.fromkeys(
                        (
                            *_normalization_aliases(item.raw_value),
                            *_normalization_aliases(item.normalized_value),
                        )
                    )
                ),
                tokens=query_tokens,
                weight=(
                    strength
                    * recency
                    * facet
                    * max(0.0, min(1.0, float(item.confidence)))
                ),
            )
        )
    return tuple(result)


class SemanticConstraintReranker:
    """Rerank only the post-projection display set using catalog evidence.

    ``candidate_ids`` supplies bounded local IDF evidence, but it is never an
    admission pool. Products absent from ``display_ids`` cannot enter the
    output. A projection-confirmed product may be locked ahead of the semantic
    ordering without letting semantic or quality signals demote it.
    """

    def __init__(
        self,
        connection: sqlite3.Connection,
        config: RerankingConfig = RerankingConfig(),
    ) -> None:
        self.connection = connection
        self.config = config

    def _documents(self, identifiers: Sequence[str]) -> dict[str, _Document]:
        if not identifiers:
            return {}
        placeholders = ",".join("?" for _ in identifiers)
        rows = self.connection.execute(
            "SELECT p.parent_asin, p.title, p.categories, p.features, p.details, "
            "p.store, p.description, COALESCE(rm.average_rating, 0.0), "
            "COALESCE(rm.rating_number, 0) "
            "FROM products p LEFT JOIN retrieval_meta rm "
            "ON rm.parent_asin=p.parent_asin "
            f"WHERE p.parent_asin IN ({placeholders})",
            tuple(identifiers),
        ).fetchall()
        documents: dict[str, _Document] = {}
        for row in rows:
            parent_asin = str(row[0])
            category_text = _normalize(" ".join(str(value or "") for value in row[1:3]))
            full_text = _normalize(" ".join(str(value or "") for value in row[1:7]))
            documents[parent_asin] = _Document(
                parent_asin=parent_asin,
                full_text=full_text,
                full_tokens=frozenset(_tokens(full_text)),
                category_text=category_text,
                category_tokens=frozenset(_tokens(category_text)),
                average_rating=float(row[7] or 0.0),
                rating_number=max(0, int(row[8] or 0)),
            )
        return documents

    def _idf(
        self,
        query: Sequence[_QueryConstraint],
        documents: Sequence[_Document],
    ) -> dict[str, float]:
        query_tokens = set(token for item in query for token in item.tokens)
        count = max(1, len(documents))
        result: dict[str, float] = {}
        for token in query_tokens:
            frequency = sum(token in document.full_tokens for document in documents)
            result[token] = (
                math.log((count + 1.0) / (frequency + 1.0)) + 1.0
                if self.config.use_candidate_idf
                else 1.0
            )
        return result

    def _must_compatibility(
        self,
        identifiers: Sequence[str],
        query: Sequence[_QueryConstraint],
    ) -> dict[str, tuple[int, int, int]]:
        """Return (matched, unknown, mismatched) bounded MUST evidence counts.

        UNKNOWN is deliberately neutral. A candidate is a confirmed mismatch
        only when the catalog exposes at least one value for the same facet and
        none equals the requested visible value. Budget remains soft because
        catalog price metadata is not a reliable hard-compatibility signal.
        """

        must_items = tuple(
            item
            for item in query
            if item.strength is Strength.MUST
            and item.facet not in {Facet.CATEGORY, Facet.BUDGET}
        )
        result = {str(parent_asin): (0, 0, 0) for parent_asin in identifiers}
        if not identifiers or not must_items:
            return result
        placeholders = ",".join("?" for _ in identifiers)
        try:
            rows = self.connection.execute(
                "SELECT parent_asin, facet, lookup_norm "
                "FROM retrieval_facet_values "
                f"WHERE parent_asin IN ({placeholders})",
                tuple(identifiers),
            ).fetchall()
        except sqlite3.OperationalError:
            # A compatibility side table is unavailable only on legacy or
            # deliberately minimal test indexes. Treat the evidence as UNKNOWN
            # instead of manufacturing mismatches from absent metadata.
            return {
                str(parent_asin): (0, len(must_items), 0)
                for parent_asin in identifiers
            }
        observed: dict[str, dict[str, set[str]]] = {
            str(parent_asin): {} for parent_asin in identifiers
        }
        for parent_asin, facet, lookup_norm in rows:
            observed.setdefault(str(parent_asin), {}).setdefault(
                str(facet), set()
            ).add(str(lookup_norm))
        for parent_asin in identifiers:
            matched = 0
            unknown = 0
            mismatched = 0
            by_facet = observed.get(str(parent_asin), {})
            for item in must_items:
                values = by_facet.get(item.facet.value, set())
                if not values:
                    unknown += 1
                elif values.intersection(item.lookup_values):
                    matched += 1
                else:
                    mismatched += 1
            result[str(parent_asin)] = (matched, unknown, mismatched)
        return result

    def _token_score(
        self,
        query_tokens: Sequence[str],
        document_tokens: frozenset[str],
        idf: dict[str, float],
    ) -> float:
        denominator = sum(idf[token] for token in query_tokens)
        if denominator <= 0.0:
            return 0.0
        matched = 0.0
        candidate_tokens = tuple(document_tokens)
        for token in query_tokens:
            if token in document_tokens:
                matched += idf[token]
                continue
            if not self.config.use_fuzzy_similarity or len(token) < 4:
                continue
            best = 0.0
            for candidate in candidate_tokens:
                if abs(len(candidate) - len(token)) > 2:
                    continue
                if candidate[:1] != token[:1]:
                    continue
                best = max(best, _trigram_similarity(token, candidate))
            if best >= self.config.fuzzy_threshold:
                matched += idf[token] * (0.72 * best)
        return matched / denominator

    @staticmethod
    def _invariants(
        before_ids: Sequence[str],
        after_ids: Sequence[str],
        locked_ids: Sequence[str],
    ) -> dict[str, object]:
        membership_preserved = (
            len(after_ids) == len(before_ids) and set(after_ids) == set(before_ids)
        )
        locked_prefix_preserved = tuple(after_ids[: len(locked_ids)]) == tuple(locked_ids)
        return {
            "display_membership_preserved": membership_preserved,
            "candidate_only_ids_introduced": sorted(set(after_ids) - set(before_ids)),
            "locked_prefix_preserved": locked_prefix_preserved,
            "passed": membership_preserved and locked_prefix_preserved,
        }

    def rerank(
        self,
        display_ids: Sequence[str],
        candidate_ids: Sequence[str],
        constraints: Sequence[Constraint],
        *,
        locked_ids: Sequence[str] = (),
        requested_k: int,
        turn: int,
        avoid_values: Sequence[str] = (),
    ) -> RerankingResult:
        if requested_k < 1:
            raise ValueError("requested_k must be positive")

        before_ids = tuple(dict.fromkeys(display_ids))[:requested_k]
        display_set = set(before_ids)
        locked_set = set(locked_ids)
        display_locked_ids = tuple(
            parent_asin for parent_asin in before_ids if parent_asin in locked_set
        )
        unlocked_ids = tuple(
            parent_asin for parent_asin in before_ids if parent_asin not in locked_set
        )
        bounded_candidate_ids = tuple(dict.fromkeys(candidate_ids))[
            : self.config.candidate_depth
        ]
        query = _positive_constraints(constraints, turn, self.config.max_constraints)
        focused = tuple(item for item in query if item.facet is not Facet.CATEGORY)
        active_avoid_constraints = tuple(
            item
            for item in constraints
            if item.status is ConstraintStatus.ACTIVE
            and (item.strength is Strength.AVOID or item.polarity is Polarity.NEGATIVE)
        )
        has_active_avoid = bool(avoid_values or active_avoid_constraints)

        def trace_for(
            after_ids: Sequence[str],
            *,
            applied: bool,
            reason: str,
            top_candidates: Sequence[dict] = (),
        ) -> dict:
            after = tuple(after_ids)
            before_ranks = {value: rank for rank, value in enumerate(before_ids, start=1)}
            after_ranks = {value: rank for rank, value in enumerate(after, start=1)}
            promoted = [
                {
                    "parent_asin": value,
                    "before_rank": before_ranks[value],
                    "after_rank": after_ranks[value],
                }
                for value in after
                if after_ranks[value] < before_ranks[value]
            ]
            invariant = self._invariants(before_ids, after, display_locked_ids)
            return {
                "turn": turn,
                "enabled": self.config.enabled,
                "projection_candidate_membership_enforced": (
                    self.config.enforce_projection_candidate_membership
                ),
                "applied": applied,
                "reason": reason,
                "candidate_depth": self.config.candidate_depth,
                "candidate_window_count": len(bounded_candidate_ids),
                "display_count": len(before_ids),
                "constraint_count": len(query),
                "focused_constraint_count": len(focused),
                "active_avoid_count": len(tuple(avoid_values))
                + len(active_avoid_constraints),
                "before_ids": list(before_ids),
                "after_ids": list(after),
                "locked_ids": list(display_locked_ids),
                "ignored_locked_ids": sorted(locked_set - display_set),
                "promoted_ids": [item["parent_asin"] for item in promoted],
                "promotions": promoted,
                "membership_invariant": invariant["display_membership_preserved"],
                "invariant": invariant,
                "use_exact_priority": self.config.use_exact_priority,
                "use_fuzzy_similarity": self.config.use_fuzzy_similarity,
                "use_candidate_idf": self.config.use_candidate_idf,
                "use_quality_tiebreak": self.config.use_quality_tiebreak,
                "top_candidates": list(top_candidates),
            }

        bypass_reason = None
        if not self.config.enabled:
            bypass_reason = "disabled"
        elif has_active_avoid:
            bypass_reason = "active_avoid"
        elif len(unlocked_ids) < 2:
            bypass_reason = "insufficient_unlocked_candidates"
        elif not focused:
            bypass_reason = "no_focused_constraint"
        if bypass_reason is not None:
            trace = trace_for(before_ids, applied=False, reason=bypass_reason)
            return RerankingResult(before_ids, trace)

        document_ids = tuple(
            dict.fromkeys((*bounded_candidate_ids, *before_ids))
        )
        documents_by_id = self._documents(document_ids)
        evidence_documents = [
            documents_by_id[parent_asin]
            for parent_asin in bounded_candidate_ids
            if parent_asin in documents_by_id
        ]
        if not evidence_documents:
            trace = trace_for(
                before_ids,
                applied=False,
                reason="catalog_metadata_unavailable",
            )
            return RerankingResult(before_ids, trace)

        idf = self._idf(query, evidence_documents)
        compatibility = self._must_compatibility(unlocked_ids, query)
        exact_frequency: dict[int, int] = {}
        exact_flags: dict[tuple[str, int], bool] = {}
        for item in query:
            count = 0
            for document in evidence_documents:
                text = (
                    document.category_text
                    if item.facet is Facet.CATEGORY
                    else document.full_text
                )
                tokens = (
                    document.category_tokens
                    if item.facet is Facet.CATEGORY
                    else document.full_tokens
                )
                exact = _contains_phrase(text, item.phrase) or set(
                    item.tokens
                ).issubset(tokens)
                exact_flags[(document.parent_asin, item.constraint_id)] = exact
                count += int(exact)
            exact_frequency[item.constraint_id] = count

        window_count = max(1, len(evidence_documents))
        scored: list[_ScoredCandidate] = []
        for rank, parent_asin in enumerate(unlocked_ids, start=1):
            document = documents_by_id.get(parent_asin)
            candidate = _ScoredCandidate(parent_asin, rank)
            (
                candidate.matched_must_count,
                candidate.unknown_must_count,
                candidate.mismatched_must_count,
            ) = compatibility.get(parent_asin, (0, 0, 0))
            if document is None:
                scored.append(candidate)
                continue
            exact_ids: list[int] = []
            scores: list[float] = []
            weighted_score = 0.0
            total_weight = 0.0
            for item in query:
                tokens = (
                    document.category_tokens
                    if item.facet is Facet.CATEGORY
                    else document.full_tokens
                )
                exact = exact_flags.get(
                    (parent_asin, item.constraint_id),
                    _contains_phrase(
                        document.category_text
                        if item.facet is Facet.CATEGORY
                        else document.full_text,
                        item.phrase,
                    )
                    or set(item.tokens).issubset(tokens),
                )
                token_score = self._token_score(item.tokens, tokens, idf)
                match_score = 1.0 if exact else token_score
                scores.append(match_score)
                weighted_score += item.weight * match_score
                total_weight += item.weight
                if exact:
                    exact_ids.append(item.constraint_id)
                    rarity = math.log(
                        (window_count + 1.0)
                        / (exact_frequency[item.constraint_id] + 1.0)
                    ) + 1.0
                    candidate.exact_priority += item.weight * rarity
            candidate.semantic_score = (
                weighted_score / total_weight if total_weight else 0.0
            )
            candidate.exact_constraint_ids = tuple(exact_ids)
            candidate.constraint_scores = tuple(scores)
            if self.config.use_quality_tiebreak:
                candidate.quality_score = (
                    math.log1p(document.rating_number)
                    + 0.25 * document.average_rating
                )
            scored.append(candidate)

        scored.sort(
            key=lambda item: (
                item.mismatched_must_count,
                -item.matched_must_count,
                -item.exact_priority if self.config.use_exact_priority else 0.0,
                -item.semantic_score,
                -item.quality_score if self.config.use_quality_tiebreak else 0.0,
                item.original_rank,
                item.parent_asin,
            )
        )
        after_ids = (*display_locked_ids, *(item.parent_asin for item in scored))
        invariant = self._invariants(before_ids, after_ids, display_locked_ids)
        if not invariant["passed"]:
            raise AssertionError("E5 violated the display or projection-lock invariant")

        top_candidates = [
            {
                "parent_asin": item.parent_asin,
                "original_unlocked_rank": item.original_rank,
                "reranked_unlocked_rank": rank,
                "exact_priority": round(item.exact_priority, 9),
                "semantic_score": round(item.semantic_score, 9),
                "quality_score": round(item.quality_score, 9),
                "compatibility_tier": (
                    "confirmed_mismatch"
                    if item.mismatched_must_count
                    else "confirmed_match"
                    if item.matched_must_count
                    else "unknown"
                ),
                "matched_must_count": item.matched_must_count,
                "unknown_must_count": item.unknown_must_count,
                "mismatched_must_count": item.mismatched_must_count,
                "exact_constraint_ids": list(item.exact_constraint_ids),
                "constraint_scores": [
                    round(value, 9) for value in item.constraint_scores
                ],
            }
            for rank, item in enumerate(scored, start=1)
        ]
        trace = trace_for(
            after_ids,
            applied=True,
            reason="bounded_catalog_alignment",
            top_candidates=top_candidates,
        )
        return RerankingResult(tuple(after_ids), trace)
