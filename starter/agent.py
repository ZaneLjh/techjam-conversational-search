from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from starter.constraints import ConstraintLedger, parse_message
from starter.question_policy import (
    AdaptiveQuestionPolicy,
    QuestionCandidate,
    extract_question_facets,
)
from starter.retrieval import (
    MultiRouteRetriever,
    RetrievalConfig,
    e4_fallback_config,
    has_focused_evidence,
    retrieval_index_rows,
    retrieval_facet_value_rows,
    retrieval_presence_rows,
)


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
    # Conversation scaffolding emitted by the public simulator. These words do
    # not describe products and otherwise dilute later constraint messages.
    "about", "additional", "ask", "attribute", "don", "earlier", "have",
    "ignore", "judgment", "matters", "need", "not", "one", "options",
    "preference", "quite", "right", "specific", "those", "yet",
}

QUESTION_CANDIDATE_DEPTH = 80


@dataclass
class SessionState:
    """Typed per-session memory; no labels or evaluator internals are stored."""

    user_profile: dict
    ledger: ConstraintLedger = field(default_factory=ConstraintLedger)
    last_asked_attribute: str | None = None
    shown_ids: set[str] = field(default_factory=set)
    asked_attributes: set[str] = field(default_factory=set)
    question_decisions: list[dict] = field(default_factory=list)
    retrieval_decisions: list[dict] = field(default_factory=list)
    question_shown_ids: set[str] = field(default_factory=set)
    intent_epoch: int = 0

    def add_message(self, message: str, turn: int) -> bool:
        update = parse_message(message, turn, self.last_asked_attribute)
        self.ledger.apply(update)
        if update.is_override:
            # A pre-override target is deliberately unscoreable. Re-open the
            # candidate pool when the corrected intent becomes active.
            self.shown_ids.clear()
            self.question_shown_ids.clear()
            self.intent_epoch += 1
        return update.is_override

    def query_text(self) -> str:
        return self.ledger.canonical_query()

    def avoid_values(self) -> list[str]:
        return self.ledger.active_avoid_values()

    def active_facets(self) -> set[str]:
        return {entry.facet.value for entry in self.ledger.active()}


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


def _initial_question_guardrail(turn: int) -> str | None:
    """Preserve E2's low-risk first ask before adapting to the reply."""

    if turn != 1:
        return None
    return "material"


def _contains_avoided_terms(
    product_text: str,
    avoid_term_sets: list[set[str]],
) -> bool:
    """Match exclusions while ignoring locally negated product-copy terms."""

    lowered = product_text.lower()
    positive_terms: set[str] = set()
    delimiters = iter(re.finditer(r"[.!?;,]|\bbut\b", lowered))
    next_delimiter = next(delimiters, None)
    recent_clause_tokens: list[re.Match[str]] = []
    persistent_negation = False
    for match in TOKEN_RE.finditer(lowered):
        token = match.group(0)
        while next_delimiter is not None and next_delimiter.end() <= match.start():
            recent_clause_tokens.clear()
            persistent_negation = False
            next_delimiter = next(delimiters, None)

        # The bounded suffix is sufficient for the "no/not + up to three
        # words" rules. Persistent clause negators are tracked separately, so
        # the scan remains linear even for unusually long product copy.
        local_start = (
            recent_clause_tokens[-4].start()
            if len(recent_clause_tokens) >= 4
            else (
                recent_clause_tokens[0].start()
                if recent_clause_tokens
                else match.start()
            )
        )
        clause_suffix = lowered[local_start : match.start()]
        clause_suffix = re.sub(r"\bnot\s+only\b", "", clause_suffix)
        negated = bool(
            persistent_negation
            or re.search(r"\b(?:no|never)\s+(?:\w+\s+){0,3}$", clause_suffix)
            or re.search(r"\bnot\s+(?:\w+\s+){0,3}$", clause_suffix)
            or re.search(r"\bnon[-\s]*$", clause_suffix)
            or re.match(r"\s*-?\s*free\b", lowered[match.end() :])
        )
        if negated:
            pass
        elif len(token) > 1 and token not in STOPWORDS:
            positive_terms.add(token)
        if token in {"without", "excluding", "except"}:
            persistent_negation = True
        recent_clause_tokens.append(match)
    return any(avoided <= positive_terms for avoided in avoid_term_sets)


class Agent:
    """E4 multi-route retrieval with E3's adaptive-question control."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        *,
        explore_unseen: bool = True,
        retrieval_config: RetrievalConfig | None = None,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.explore_unseen = explore_unseen
        # E4.1 remains a provisional experiment until true product-disjoint
        # validation. Applying the patch therefore preserves full E4 by
        # default; experiment tools pass the E4.1 candidate explicitly.
        self.retrieval_config = retrieval_config or e4_fallback_config()
        self.connection = sqlite3.connect(":memory:")
        self._sessions: dict[str, SessionState] = {}
        self.question_policy = AdaptiveQuestionPolicy()
        self._build_index()
        self.retriever = MultiRouteRetriever(self.connection, self.retrieval_config)

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        cursor.execute(
            "CREATE TABLE question_facets("
            "parent_asin TEXT PRIMARY KEY, facets_json TEXT NOT NULL)"
        )
        if self.retrieval_config.enabled:
            cursor.execute(
                "CREATE TABLE retrieval_meta("
                "parent_asin TEXT PRIMARY KEY, coarse_norm TEXT NOT NULL, "
                "average_rating REAL NOT NULL, rating_number INTEGER NOT NULL)"
            )
            cursor.execute(
                "CREATE TABLE retrieval_values("
                "lookup_norm TEXT NOT NULL, parent_asin TEXT NOT NULL, "
                "PRIMARY KEY(lookup_norm, parent_asin)) WITHOUT ROWID"
            )
            cursor.execute(
                "CREATE TABLE retrieval_facet_presence("
                "parent_asin TEXT NOT NULL, facet TEXT NOT NULL, "
                "PRIMARY KEY(parent_asin, facet)) WITHOUT ROWID"
            )
            cursor.execute(
                "CREATE TABLE retrieval_facet_values("
                "parent_asin TEXT NOT NULL, facet TEXT NOT NULL, "
                "lookup_norm TEXT NOT NULL, "
                "PRIMARY KEY(parent_asin, facet, lookup_norm)) WITHOUT ROWID"
            )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        question_batch: list[tuple[str, str]] = []
        retrieval_meta_batch: list[tuple[str, str, float, int]] = []
        retrieval_value_batch: list[tuple[str, str]] = []
        retrieval_presence_batch: list[tuple[str, str]] = []
        retrieval_facet_value_batch: list[tuple[str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                parent_asin = str(product["parent_asin"])
                batch.append(
                    (
                        parent_asin,
                        _text(product.get("title")),
                        _text(product.get("categories")),
                        _text(product.get("features")),
                        _text(product.get("details")),
                        _text(product.get("store")),
                        _text(product.get("description")),
                    )
                )
                question_batch.append(
                    (
                        parent_asin,
                        json.dumps(
                            extract_question_facets(product),
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    )
                )
                if self.retrieval_config.enabled:
                    meta_row, value_rows = retrieval_index_rows(product)
                    retrieval_meta_batch.append(meta_row)
                    retrieval_value_batch.extend(value_rows)
                    retrieval_presence_batch.extend(retrieval_presence_rows(product))
                    retrieval_facet_value_batch.extend(
                        retrieval_facet_value_rows(product)
                    )
                if len(batch) >= 1000:
                    cursor.executemany(
                        "INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)",
                        batch,
                    )
                    cursor.executemany(
                        "INSERT INTO question_facets VALUES (?, ?)",
                        question_batch,
                    )
                    if self.retrieval_config.enabled:
                        cursor.executemany(
                            "INSERT INTO retrieval_meta VALUES (?, ?, ?, ?)",
                            retrieval_meta_batch,
                        )
                        cursor.executemany(
                            "INSERT OR IGNORE INTO retrieval_values VALUES (?, ?)",
                            retrieval_value_batch,
                        )
                        cursor.executemany(
                            "INSERT OR IGNORE INTO retrieval_facet_presence VALUES (?, ?)",
                            retrieval_presence_batch,
                        )
                        cursor.executemany(
                            "INSERT OR IGNORE INTO retrieval_facet_values VALUES (?, ?, ?)",
                            retrieval_facet_value_batch,
                        )
                    batch.clear()
                    question_batch.clear()
                    retrieval_meta_batch.clear()
                    retrieval_value_batch.clear()
                    retrieval_presence_batch.clear()
                    retrieval_facet_value_batch.clear()
        if batch:
            cursor.executemany(
                "INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)",
                batch,
            )
            cursor.executemany(
                "INSERT INTO question_facets VALUES (?, ?)",
                question_batch,
            )
            if self.retrieval_config.enabled:
                cursor.executemany(
                    "INSERT INTO retrieval_meta VALUES (?, ?, ?, ?)",
                    retrieval_meta_batch,
                )
                cursor.executemany(
                    "INSERT OR IGNORE INTO retrieval_values VALUES (?, ?)",
                    retrieval_value_batch,
                )
                cursor.executemany(
                    "INSERT OR IGNORE INTO retrieval_facet_presence VALUES (?, ?)",
                    retrieval_presence_batch,
                )
                cursor.executemany(
                    "INSERT OR IGNORE INTO retrieval_facet_values VALUES (?, ?, ?)",
                    retrieval_facet_value_batch,
                )
        self.connection.commit()

    def _question_candidates(self, identifiers: list[str]) -> list[QuestionCandidate]:
        if not identifiers:
            return []
        placeholders = ",".join("?" for _ in identifiers)
        rows = self.connection.execute(
            f"SELECT parent_asin, facets_json FROM question_facets "
            f"WHERE parent_asin IN ({placeholders})",
            identifiers,
        ).fetchall()
        by_identifier = {str(row[0]): str(row[1]) for row in rows}
        candidates: list[QuestionCandidate] = []
        for identifier in identifiers:
            try:
                facets = json.loads(by_identifier.get(identifier, "{}"))
            except (TypeError, ValueError, json.JSONDecodeError):
                facets = {}
            if not isinstance(facets, dict):
                facets = {}
            candidates.append(
                QuestionCandidate(
                    parent_asin=identifier,
                    facets={
                        str(attribute): tuple(str(value) for value in values)
                        for attribute, values in facets.items()
                        if isinstance(values, list)
                    },
                )
            )
        return candidates

    def reset(self, session_id: str, user_profile: dict) -> None:
        # The profile is anonymized. E4 retains it for the later personalization
        # experiment but does not yet use it for ranking.
        self._sessions[session_id] = SessionState(user_profile=dict(user_profile))

    def _legacy_candidates(
        self,
        state: SessionState,
        requested_k: int,
        shown_ids: set[str],
        *,
        ranked_ids: list[str] | None = None,
    ) -> tuple[list[str], list[str]]:
        """Run the byte-compatible E3 ranking used by the declared ablation.

        E4 also uses this independent candidate window for question selection,
        keeping the adaptive-question mechanism fixed while ranking changes.
        """

        avoid_term_sets = [set(_terms(value)) for value in state.avoid_values()]
        avoid_term_sets = [terms for terms in avoid_term_sets if terms]
        if ranked_ids is not None:
            if avoid_term_sets:
                raise ValueError("precomputed legacy rankings cannot apply exclusions")
            rows = ((parent_asin,) for parent_asin in ranked_ids)
        else:
            unique_terms = list(dict.fromkeys(_terms(state.query_text())))[:80]
            expression = " OR ".join(f'"{term}"' for term in unique_terms)
            if not expression:
                return [], []

            selected_columns = (
                "parent_asin, title, categories, features, details, store, description"
                if avoid_term_sets
                else "parent_asin"
            )
            ranked_sql = (
                f"SELECT {selected_columns} FROM products WHERE products MATCH ? "
                "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0)"
            )
            if avoid_term_sets:
                rows = self.connection.execute(ranked_sql, (expression,))
            else:
                candidate_limit = (
                    min(
                        requested_k
                        + QUESTION_CANDIDATE_DEPTH
                        + (len(shown_ids) if self.explore_unseen else 0),
                        1000,
                    )
                    if self.explore_unseen
                    else requested_k + QUESTION_CANDIDATE_DEPTH
                )
                rows = self.connection.execute(
                    ranked_sql + " LIMIT ?",
                    (expression, candidate_limit),
                )

        identifiers: list[str] = []
        question_identifiers: list[str] = []
        for row in rows:
            identifier = str(row[0])
            if self.explore_unseen and identifier in shown_ids:
                continue
            if avoid_term_sets:
                product_text = ". ".join(str(value) for value in row[1:7])
                if _contains_avoided_terms(product_text, avoid_term_sets):
                    continue
            if len(identifiers) < requested_k:
                identifiers.append(identifier)
                continue
            question_identifiers.append(identifier)
            if len(question_identifiers) >= QUESTION_CANDIDATE_DEPTH:
                break
        if self.explore_unseen and len(identifiers) == requested_k:
            shown_ids.update(identifiers)
        return identifiers, question_identifiers

    def _shared_legacy_ranking(
        self,
        state: SessionState,
        requested_k: int,
    ) -> list[str]:
        """Fetch one E3-ordered ledger stream for E3 questions and E4 fusion."""

        unique_terms = list(dict.fromkeys(_terms(state.query_text())))[:80]
        expression = " OR ".join(f'"{term}"' for term in unique_terms)
        if not expression:
            return []
        shadow_depth = requested_k + QUESTION_CANDIDATE_DEPTH
        route_depth = (
            self.retrieval_config.route_depth
            if self.retrieval_config.use_ledger_route
            else 0
        )
        if self.explore_unseen:
            shadow_depth += len(state.question_shown_ids)
            route_depth += len(state.shown_ids)
        candidate_limit = min(max(shadow_depth, route_depth), 1000)
        rows = self.connection.execute(
            "SELECT parent_asin FROM products WHERE products MATCH ? "
            "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) "
            "LIMIT ?",
            (expression, candidate_limit),
        )
        return [str(row[0]) for row in rows]

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        state = self._sessions.get(session_id)
        if state is None:
            raise RuntimeError("reset must be called before respond")

        state.add_message(user_message, turn)
        requested_k = min(max(int(top_k), 1), 10)
        if self.retrieval_config.enabled:
            active_constraints = state.ledger.active()
            avoid_values = state.avoid_values()
            if not has_focused_evidence(active_constraints) or avoid_values:
                # Category-only turns and explicit exclusions retain the exact
                # E3 ordering. This is both a recall guardrail and the safe
                # deep-streaming path for dominated exclusions.
                if state.shown_ids == state.question_shown_ids:
                    identifiers, question_identifiers = self._legacy_candidates(
                        state,
                        requested_k,
                        state.shown_ids,
                    )
                    state.question_shown_ids = set(state.shown_ids)
                else:
                    # Once E4 and E3 have displayed different pages, advance
                    # the frozen E3 shadow independently. This matters when a
                    # later no-preference or AVOID update returns display
                    # ranking to the legacy-safe path.
                    _, question_identifiers = self._legacy_candidates(
                        state,
                        requested_k,
                        state.question_shown_ids,
                    )
                    identifiers, _ = self._legacy_candidates(
                        state,
                        requested_k,
                        state.shown_ids,
                    )
                legacy_candidate_ids = list(
                    dict.fromkeys([*identifiers, *question_identifiers])
                )[:100]
                state.retrieval_decisions.append(
                    {
                        "turn": turn,
                        "mode": (
                            "legacy_exclusion_fallback"
                            if avoid_values
                            else "legacy_category_only"
                        ),
                        "enabled_route_families": ["ledger"],
                        "candidate_union_count": len(legacy_candidate_ids),
                        "candidate_ids": legacy_candidate_ids,
                        "route_candidate_count": len(legacy_candidate_ids),
                        "route_candidate_ids": legacy_candidate_ids,
                        "eligible_candidate_count": len(legacy_candidate_ids),
                        "eligible_candidate_ids": legacy_candidate_ids,
                        "recommendation_count": len(identifiers),
                        "recommendation_ids": list(identifiers),
                        "constraint_reranking": False,
                        "soft_relaxation": True,
                    }
                )
            else:
                # The E3 candidate window remains the sole input to E3's
                # question policy, isolating E4's displayed ranking.
                shared_legacy_ranking = self._shared_legacy_ranking(
                    state,
                    requested_k,
                )
                _, question_identifiers = self._legacy_candidates(
                    state,
                    requested_k,
                    state.question_shown_ids,
                    ranked_ids=shared_legacy_ranking,
                )
                retrieval = self.retriever.search(
                    constraints=active_constraints,
                    avoid_values=(),
                    shown_ids=state.shown_ids,
                    explore_unseen=self.explore_unseen,
                    turn=turn,
                    requested_k=requested_k,
                    ledger_ranking=shared_legacy_ranking,
                )
                identifiers = list(retrieval.recommendation_ids)
                if self.explore_unseen and identifiers and (
                    len(identifiers) == requested_k
                    or self.retrieval_config.use_strict_front
                    or not self.retrieval_config.use_soft_relaxation
                ):
                    state.shown_ids.update(identifiers)
                state.retrieval_decisions.append(retrieval.trace)
            recommendations = [
                {"parent_asin": parent_asin}
                for parent_asin in identifiers
            ]
        else:
            identifiers, question_identifiers = self._legacy_candidates(
                state,
                requested_k,
                state.shown_ids,
            )
            recommendations = [
                {"parent_asin": parent_asin}
                for parent_asin in identifiers
            ]

        question_candidates = self._question_candidates(question_identifiers)
        decision = self.question_policy.choose(
            question_candidates,
            active_facets=state.active_facets(),
            asked_attributes=state.asked_attributes,
            turn=turn,
            guardrail_attribute=_initial_question_guardrail(turn),
        )
        ask_attribute = decision.ask_attribute
        message = decision.message
        if ask_attribute is not None:
            state.asked_attributes.add(ask_attribute)
        state.question_decisions.append(decision.as_dict())
        state.last_asked_attribute = ask_attribute

        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }

    def evidence_trace(self, session_id: str) -> dict:
        """Return a JSON-serializable state trace for tests and demonstrations."""

        state = self._sessions.get(session_id)
        if state is None:
            raise RuntimeError("reset must be called before evidence_trace")
        return {
            "canonical_query": state.query_text(),
            "candidate_exploration": self.explore_unseen,
            "multi_route_retrieval": self.retrieval_config.enabled,
            "question_candidate_ranking": "e3_frozen",
            "last_asked_attribute": state.last_asked_attribute,
            "intent_epoch": state.intent_epoch,
            "shown_ids": sorted(state.shown_ids),
            "asked_attributes": sorted(state.asked_attributes),
            "question_decisions": list(state.question_decisions),
            "retrieval_decisions": list(state.retrieval_decisions),
            "constraints": state.ledger.evidence_trace(),
        }
