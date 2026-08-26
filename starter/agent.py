from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from starter.constraints import ConstraintLedger, parse_message


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

QUESTION_SCHEDULE: tuple[tuple[str, str], ...] = (
    ("material", "Do you have a material preference?"),
    ("feature", "Which product feature matters most to you?"),
    ("color", "Do you have a color preference?"),
    ("style", "Which style or fit do you prefer?"),
    ("size", "Do you have a size or sizing preference?"),
    ("use_case", "What will you mainly use the product for?"),
    ("brand", "Do you prefer a particular brand?"),
    ("budget", "What budget should I stay within?"),
    ("other", "What other requirement matters most?"),
)


@dataclass
class SessionState:
    """Typed per-session memory; no labels or evaluator internals are stored."""

    user_profile: dict
    ledger: ConstraintLedger = field(default_factory=ConstraintLedger)
    last_asked_attribute: str | None = None
    shown_ids: set[str] = field(default_factory=set)
    intent_epoch: int = 0

    def add_message(self, message: str, turn: int) -> bool:
        update = parse_message(message, turn, self.last_asked_attribute)
        self.ledger.apply(update)
        if update.is_override:
            # A pre-override target is deliberately unscoreable. Re-open the
            # candidate pool when the corrected intent becomes active.
            self.shown_ids.clear()
            self.intent_epoch += 1
        return update.is_override

    def query_text(self) -> str:
        return self.ledger.canonical_query()

    def avoid_values(self) -> list[str]:
        return self.ledger.active_avoid_values()


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


def _contains_avoided_terms(
    product_text: str,
    avoid_term_sets: list[set[str]],
) -> bool:
    """Match exclusions while ignoring locally negated product-copy terms."""

    lowered = product_text.lower()
    positive_terms: set[str] = set()
    for match in TOKEN_RE.finditer(lowered):
        token = match.group(0)
        prefix = lowered[: match.start()]
        delimiters = list(re.finditer(r"[.!?;,]|\bbut\b", prefix))
        clause_prefix = prefix[delimiters[-1].end() :] if delimiters else prefix
        clause_prefix = re.sub(r"\bnot\s+only\b", "", clause_prefix)
        negated = bool(
            re.search(r"\b(?:no|never)\s+(?:\w+\s+){0,3}$", clause_prefix)
            or re.search(r"\bnot\s+(?:\w+\s+){0,3}$", clause_prefix)
            or re.search(
                r"\b(?:without|excluding|except)\b[^.!?;,]*$",
                clause_prefix,
            )
            or re.search(r"\bnon[-\s]*$", clause_prefix)
            or re.match(r"\s*-?\s*free\b", lowered[match.end() :])
        )
        if negated:
            continue
        if len(token) > 1 and token not in STOPWORDS:
            positive_terms.add(token)
    return any(avoided <= positive_terms for avoided in avoid_term_sets)


class Agent:
    """E2: constraint-ledger BM25 with deterministic clarification questions."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        *,
        explore_unseen: bool = True,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.explore_unseen = explore_unseen
        self.connection = sqlite3.connect(":memory:")
        self._sessions: dict[str, SessionState] = {}
        self._build_index()

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                batch.append(
                    (
                        str(product["parent_asin"]),
                        _text(product.get("title")),
                        _text(product.get("categories")),
                        _text(product.get("features")),
                        _text(product.get("details")),
                        _text(product.get("store")),
                        _text(product.get("description")),
                    )
                )
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def reset(self, session_id: str, user_profile: dict) -> None:
        # The profile is anonymized. E1 retains it for the later personalization
        # experiment but does not yet use it for ranking.
        self._sessions[session_id] = SessionState(user_profile=dict(user_profile))

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
        unique_terms = list(dict.fromkeys(_terms(state.query_text())))[:80]
        avoid_term_sets = [set(_terms(value)) for value in state.avoid_values()]
        avoid_term_sets = [terms for terms in avoid_term_sets if terms]
        expression = " OR ".join(f'"{term}"' for term in unique_terms)
        if not expression:
            recommendations: list[dict] = []
        else:
            requested_k = min(max(int(top_k), 1), 10)
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
                # Stream until enough compliant rows are found. A fixed buffer
                # can fail when a common excluded value dominates early ranks.
                rows = self.connection.execute(ranked_sql, (expression,))
            else:
                candidate_limit = (
                    min(requested_k + len(state.shown_ids), 1000)
                    if self.explore_unseen
                    else requested_k
                )
                rows = self.connection.execute(
                    ranked_sql + " LIMIT ?",
                    (expression, candidate_limit),
                )
            identifiers = []
            for row in rows:
                identifier = str(row[0])
                if self.explore_unseen and identifier in state.shown_ids:
                    continue
                if avoid_term_sets:
                    product_text = ". ".join(str(value) for value in row[1:])
                    if _contains_avoided_terms(product_text, avoid_term_sets):
                        continue
                identifiers.append(identifier)
                if len(identifiers) == requested_k:
                    break
            recommendations = [{"parent_asin": parent_asin} for parent_asin in identifiers]
            # If the query cannot fill the requested list there is no useful
            # deeper page to expose, so retain the stable fallback ordering.
            if self.explore_unseen and len(identifiers) == requested_k:
                state.shown_ids.update(identifiers)

        if turn < 10:
            schedule_index = min(max(turn - 1, 0), len(QUESTION_SCHEDULE) - 1)
            ask_attribute, message = QUESTION_SCHEDULE[schedule_index]
        else:
            ask_attribute = None
            message = "Here are the closest matches I found."

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
            "last_asked_attribute": state.last_asked_attribute,
            "intent_epoch": state.intent_epoch,
            "shown_ids": sorted(state.shown_ids),
            "constraints": state.ledger.evidence_trace(),
        }
