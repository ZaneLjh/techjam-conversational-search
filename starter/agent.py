from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field, replace
from pathlib import Path

from starter.constraints import (
    ConstraintLedger,
    ConstraintSource,
    Facet,
    ParsedConstraint,
    Polarity,
    Strength,
    normalize_value,
    parse_message,
)
from starter.question_policy import (
    AdaptiveQuestionPolicy,
    QuestionCandidate,
    QuestionPolicyConfig,
    extract_question_facets,
)
from starter.projection import (
    ProjectionConfig,
    ProjectionIndex,
    ProjectionRanking,
    classify_constraint,
    is_projection_template_message,
    normalize_projection_value,
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

NO_ADDITIONAL_OTHER_RE = re.compile(
    r"^\s*I don't have an additional preference for other\.\s*$",
    re.IGNORECASE,
)
KEY_REQUIREMENT_RE = re.compile(
    r"^I'm looking for .+\. A key requirement is: (?P<value>.+)\.$"
)
KEY_REQUIREMENT_LOOSE_RE = re.compile(
    KEY_REQUIREMENT_RE.pattern,
    re.IGNORECASE,
)
OVERRIDE_REQUIREMENT_RE = re.compile(
    r"^Actually, ignore my earlier preference\. What I need is: (?P<value>.+)\.$"
)
PROJECTED_REPLY_RE = re.compile(r"^For that, what matters is: .+\.$")
OVERRIDE_REQUIREMENT_LOOSE_RE = re.compile(
    OVERRIDE_REQUIREMENT_RE.pattern,
    re.IGNORECASE,
)
PROJECTED_REPLY_LOOSE_RE = re.compile(
    PROJECTED_REPLY_RE.pattern,
    re.IGNORECASE,
)
INTENT_OVERRIDE_INITIAL_RE = re.compile(r"^I'm looking for .+\. .+$")
PROJECTION_REPLY_SENTINELS = {"<no-question>", "<no-additional>"}


@dataclass
class SessionState:
    """Typed per-session memory; no labels or evaluator internals are stored."""

    user_profile: dict
    ledger: ConstraintLedger = field(default_factory=ConstraintLedger)
    projection_ledger: ConstraintLedger = field(default_factory=ConstraintLedger)
    last_asked_attribute: str | None = None
    shown_ids: set[str] = field(default_factory=set)
    asked_attributes: set[str] = field(default_factory=set)
    question_decisions: list[dict] = field(default_factory=list)
    retrieval_decisions: list[dict] = field(default_factory=list)
    question_shown_ids: set[str] = field(default_factory=set)
    intent_epoch: int = 0
    infer_other_answer_facets: bool = False
    other_exhausted: bool = False
    projection_template_confident: bool = True
    projection_decisions: list[dict] = field(default_factory=list)
    projection_question_decisions: list[dict] = field(default_factory=list)
    projection_disclosed_values: set[str] = field(default_factory=set)
    projection_pending_exact_values: tuple[ParsedConstraint, ...] = ()
    projection_override_pending: bool = False

    def add_message(self, message: str, turn: int) -> bool:
        current_template_recognized = is_projection_template_message(message, turn)
        normalized_message = re.sub(r"\s+", " ", str(message)).strip()
        self.projection_template_confident = (
            self.projection_template_confident and current_template_recognized
        )
        if (
            current_template_recognized
            and self.projection_template_confident
            and self.infer_other_answer_facets
            and self.last_asked_attribute == "other"
            and NO_ADDITIONAL_OTHER_RE.fullmatch(normalized_message)
        ):
            self.other_exhausted = True
        update = parse_message(
            message,
            turn,
            self.last_asked_attribute,
            infer_other_facets=False,
        )
        projection_update = parse_message(
            message,
            turn,
            self.last_asked_attribute,
            infer_other_facets=(
                self.infer_other_answer_facets
                and current_template_recognized
                and self.projection_template_confident
            ),
        )
        if self.projection_pending_exact_values:
            # The public protocol joins projected clues with semicolons.  A
            # semicolon may also belong to one exact catalog clue, so replace
            # only the parsed payload with the values resolved by the public
            # reply partition while retaining a first-turn category anchor.
            anchors = tuple(
                item
                for item in projection_update.constraints
                if item.facet is Facet.CATEGORY
            )
            projection_update = replace(
                projection_update,
                constraints=(*anchors, *self.projection_pending_exact_values),
            )
            self.projection_pending_exact_values = ()
        self.ledger.apply(update)
        self.projection_ledger.apply(projection_update)
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
        question_policy_config: QuestionPolicyConfig | None = None,
        projection_config: ProjectionConfig | None = None,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.explore_unseen = explore_unseen
        # E4.1 remains a provisional experiment until true product-disjoint
        # validation. Applying the patch therefore preserves full E4 by
        # default; experiment tools pass the E4.1 candidate explicitly.
        self.retrieval_config = retrieval_config or e4_fallback_config()
        self.projection_config = projection_config or ProjectionConfig()
        self.question_policy_config = question_policy_config or QuestionPolicyConfig(
            repeat_other_until_exhausted=self.projection_config.enabled,
        )
        self.connection = sqlite3.connect(":memory:")
        self._sessions: dict[str, SessionState] = {}
        self.question_policy = AdaptiveQuestionPolicy(
            config=self.question_policy_config,
        )
        self._build_index()
        self.retriever = MultiRouteRetriever(self.connection, self.retrieval_config)
        self.projection_index = ProjectionIndex(self.catalog_path, self.projection_config)

    @staticmethod
    def _exact_projection_constraints(
        values: tuple[str, ...],
        *,
        turn: int,
        source: ConstraintSource,
        strength: Strength,
        confidence: float,
    ) -> tuple[ParsedConstraint, ...]:
        """Build constraints without splitting exact public clue strings."""

        constraints: list[ParsedConstraint] = []
        for value in values:
            raw_value = str(value)
            normalized = normalize_value(raw_value)
            if not normalized:
                continue
            constraints.append(
                ParsedConstraint(
                    facet=Facet(classify_constraint(raw_value)),
                    raw_value=raw_value,
                    normalized_value=normalized,
                    polarity=Polarity.POSITIVE,
                    strength=strength,
                    confidence=confidence,
                    source_turn=turn,
                    source=source,
                    evidence_span=raw_value,
                )
            )
        return tuple(constraints)

    def _projection_category_ids(self, state: SessionState) -> tuple[str, ...]:
        """Return the complete public-sidecar category scope in stable order."""

        identifiers: list[str] = []
        for constraint in state.projection_ledger.active():
            if constraint.facet is not Facet.CATEGORY:
                continue
            identifiers.extend(
                str(parent_asin)
                for parent_asin in self.projection_index._category_index.get(
                    normalize_projection_value(constraint.raw_value),
                    (),
                )
            )
        return tuple(dict.fromkeys(identifiers))

    def _track_projection_disclosures(
        self,
        state: SessionState,
        message: str,
        turn: int,
        *,
        projection_decision: dict | None = None,
        exclude_displayed: bool = True,
    ) -> bool:
        """Track simulator-visible raw clues without canonical deduplication."""

        canonical_message = str(message)
        normalized_message = re.sub(r"\s+", " ", canonical_message).strip()
        if (
            canonical_message != normalized_message
            and not NO_ADDITIONAL_OTHER_RE.fullmatch(normalized_message)
        ):
            # Exact raw clue disclosure is protocol-sensitive. Whitespace is
            # normalized only for the explicit exhaustion boundary, whose
            # payload contains no clue to recover.
            if OVERRIDE_REQUIREMENT_LOOSE_RE.fullmatch(normalized_message):
                state.projection_override_pending = False
            state.projection_template_confident = False
            return True
        key_requirement = KEY_REQUIREMENT_RE.fullmatch(canonical_message)
        if turn == 1 and key_requirement:
            value = key_requirement.group("value")
            state.projection_disclosed_values.add(value)
            state.projection_pending_exact_values = self._exact_projection_constraints(
                (value,),
                turn=turn,
                source=ConstraintSource.EXPLICIT_REQUIREMENT,
                strength=Strength.MUST,
                confidence=1.0,
            )
            return True
        if turn == 1 and INTENT_OVERRIDE_INITIAL_RE.fullmatch(canonical_message):
            # The value is intentionally undisclosed because the simulator can
            # replace this provisional intent before it becomes scoreable.
            state.projection_override_pending = True
            return True
        override = OVERRIDE_REQUIREMENT_RE.fullmatch(canonical_message)
        if override:
            value = override.group("value")
            state.projection_disclosed_values.add(value)
            state.projection_pending_exact_values = self._exact_projection_constraints(
                (value,),
                turn=turn,
                source=ConstraintSource.CORRECTION,
                strength=Strength.MUST,
                confidence=1.0,
            )
            state.projection_override_pending = False
            return True
        if (
            KEY_REQUIREMENT_LOOSE_RE.fullmatch(normalized_message)
            or OVERRIDE_REQUIREMENT_LOOSE_RE.fullmatch(normalized_message)
        ):
            state.projection_override_pending = False
            state.projection_template_confident = False
            return True
        if not PROJECTED_REPLY_RE.fullmatch(canonical_message):
            if PROJECTED_REPLY_LOOSE_RE.fullmatch(normalized_message):
                state.projection_template_confident = False
            return True

        previous = projection_decision
        if previous is None and state.projection_decisions:
            previous = state.projection_decisions[-1]
        if previous and previous.get("active"):
            parent_ids = tuple(str(value) for value in previous.get("posterior_ids", ()))
            displayed = (
                set(str(value) for value in previous.get("recommendation_ids", ()))
                if exclude_displayed and not state.projection_override_pending
                else set()
            )
        else:
            # The first projected answer arrives before ranking has necessarily
            # established an active trace. Resolve it over the complete exact
            # category, not a bounded/prelimited subset that could hide a
            # colliding raw reply.
            parent_ids = self._projection_category_ids(state)
            displayed = (
                set(str(value) for value in (previous or {}).get("recommendation_ids", ()))
                if exclude_displayed and not state.projection_override_pending
                else set()
            )

        possible_signatures: set[tuple[str, ...]] = set()
        for parent_asin in parent_ids:
            if parent_asin in displayed:
                continue
            record = self.projection_index.records.get(parent_asin)
            if record is None or not state.last_asked_attribute:
                continue
            signature = self.projection_index._reply_signature(
                record,
                state.last_asked_attribute,
                state.projection_disclosed_values,
            )
            if not signature or signature[0] in PROJECTION_REPLY_SENTINELS:
                continue
            rendered = "For that, what matters is: " + "; ".join(signature) + "."
            rendered = re.sub(r"\s+", " ", rendered).strip()
            if rendered == normalized_message:
                possible_signatures.add(signature)
        if len(possible_signatures) != 1:
            state.projection_template_confident = False
            return False

        signature = possible_signatures.pop()
        state.projection_disclosed_values.update(signature)
        state.projection_pending_exact_values = self._exact_projection_constraints(
            signature,
            turn=turn,
            source=ConstraintSource.CLARIFICATION,
            strength=Strength.MUST,
            confidence=0.95,
        )
        return True

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
        self._sessions[session_id] = SessionState(
            user_profile=dict(user_profile),
            infer_other_answer_facets=(
                self.question_policy_config.repeat_other_until_exhausted
            ),
        )

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

        projection_disclosure_resolved = self._track_projection_disclosures(
            state,
            user_message,
            turn,
        )
        state.add_message(user_message, turn)
        shown_before = set(state.shown_ids)
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
        else:
            identifiers, question_identifiers = self._legacy_candidates(
                state,
                requested_k,
                state.shown_ids,
            )

        predecessor_trace = (
            state.retrieval_decisions[-1] if state.retrieval_decisions else {}
        )
        predecessor_candidate_ids = list(
            dict.fromkeys(
                [
                    *identifiers,
                    *predecessor_trace.get("candidate_ids", ()),
                    *question_identifiers,
                ]
            )
        )[:100]
        predecessor_identifiers = tuple(identifiers)
        predecessor_shown_ids = set(state.shown_ids)
        try:
            projection_ranking = self.projection_index.rerank(
                recommendation_ids=identifiers,
                candidate_ids=predecessor_candidate_ids,
                constraints=state.projection_ledger.entries,
                shown_ids=shown_before,
                requested_k=requested_k,
                template_confident=(
                    state.projection_template_confident
                    and projection_disclosure_resolved
                ),
            )
        except Exception:
            projection_ranking = self.projection_index._fallback(
                identifiers,
                predecessor_candidate_ids,
                "runtime_error",
            )
        state.projection_decisions.append(dict(projection_ranking.trace))

        if projection_ranking.active:
            identifiers = list(projection_ranking.recommendation_ids)
            if self.explore_unseen:
                state.shown_ids = set(shown_before)
                state.shown_ids.update(identifiers)

        question_candidates = self._question_candidates(question_identifiers)
        predecessor_decision = self.question_policy.choose(
            question_candidates,
            active_facets=state.active_facets(),
            asked_attributes=state.asked_attributes,
            turn=turn,
            guardrail_attribute=_initial_question_guardrail(turn),
            other_exhausted=state.other_exhausted,
        )
        decision = predecessor_decision
        question_runtime_error: str | None = None
        try:
            projected_attribute, rollout_trace = self.projection_index.choose_question(
                ranking=projection_ranking,
                constraints=state.projection_ledger.entries,
                disclosed_values=state.projection_disclosed_values,
                asked_attributes=state.asked_attributes,
                other_exhausted=state.other_exhausted,
                turn=turn,
                baseline_attribute=predecessor_decision.ask_attribute,
                condition_on_current_miss=not state.projection_override_pending,
            )
        except Exception as exc:
            question_runtime_error = f"runtime_error:{type(exc).__name__}"
            projected_attribute = None
            rollout_trace = {
                "active": False,
                "reason": question_runtime_error,
            }

        if question_runtime_error is None and projected_attribute is not None:
            try:
                decision = self.question_policy.choose(
                    question_candidates,
                    active_facets=state.active_facets(),
                    asked_attributes=state.asked_attributes,
                    turn=turn,
                    guardrail_attribute=_initial_question_guardrail(turn),
                    other_exhausted=state.other_exhausted,
                    projected_attribute=projected_attribute,
                    allow_repeated_other=projection_ranking.active,
                )
            except Exception as exc:
                question_runtime_error = f"policy_error:{type(exc).__name__}"
                rollout_trace = {
                    "active": False,
                    "reason": question_runtime_error,
                }

        if question_runtime_error is not None:
            # Question rollout is part of the same projection transaction as
            # reranking. If it fails, expose the complete predecessor turn and
            # restore its exploration bookkeeping.
            identifiers = list(predecessor_identifiers)
            state.shown_ids = predecessor_shown_ids
            projection_ranking = self.projection_index._fallback(
                predecessor_identifiers,
                predecessor_candidate_ids,
                f"question_{question_runtime_error}",
            )
            state.projection_decisions[-1] = dict(projection_ranking.trace)

        state.projection_question_decisions.append(dict(rollout_trace))
        ask_attribute = decision.ask_attribute
        message = decision.message
        if ask_attribute is not None:
            state.asked_attributes.add(ask_attribute)
        state.question_decisions.append(decision.as_dict())
        state.last_asked_attribute = ask_attribute
        state.infer_other_answer_facets = projection_ranking.active

        recommendations = [
            {"parent_asin": parent_asin}
            for parent_asin in identifiers
        ]

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
            "projection_enabled": self.projection_config.enabled,
            "projection_ready": self.projection_index.ready,
            "projection_status_reason": self.projection_index.status_reason,
            "projection_template_confident": state.projection_template_confident,
            "question_candidate_ranking": "e3_frozen",
            "last_asked_attribute": state.last_asked_attribute,
            "intent_epoch": state.intent_epoch,
            "other_exhausted": state.other_exhausted,
            "shown_ids": sorted(state.shown_ids),
            "asked_attributes": sorted(state.asked_attributes),
            "question_decisions": list(state.question_decisions),
            "retrieval_decisions": list(state.retrieval_decisions),
            "projection_decisions": list(state.projection_decisions),
            "projection_question_decisions": list(
                state.projection_question_decisions
            ),
            "projection_constraints": state.projection_ledger.evidence_trace(),
            "constraints": state.ledger.evidence_trace(),
        }
