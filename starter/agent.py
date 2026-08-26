from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path


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

OVERRIDE_RE = re.compile(
    r"\b(?:actually|changed\s+my\s+mind|ignore|instead|rather)\b",
    re.IGNORECASE,
)

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
    """Small per-session memory; no labels or evaluator internals are stored."""

    user_profile: dict
    messages: list[str] = field(default_factory=list)
    category_anchor: str = ""

    def add_message(self, message: str) -> None:
        normalized = re.sub(r"\s+", " ", message).strip()
        if not normalized:
            return

        if not self.messages:
            # The simulator places the coarse category in the first sentence.
            # Preserve that sentence so an intent override can discard the old
            # preference without losing the product category.
            first_sentence = re.match(r"^.*?[.!?](?:\s|$)", normalized)
            self.category_anchor = (
                first_sentence.group(0).strip() if first_sentence else normalized
            )
        elif OVERRIDE_RE.search(normalized):
            self.messages = [self.category_anchor] if self.category_anchor else []

        self.messages.append(normalized)

    def query_text(self) -> str:
        return " ".join(self.messages)


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


class Agent:
    """E1: stateful BM25 retrieval with deterministic clarification questions."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
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

        state.add_message(user_message)
        unique_terms = list(dict.fromkeys(_terms(state.query_text())))[:80]
        expression = " OR ".join(f'"{term}"' for term in unique_terms)
        if not expression:
            recommendations: list[dict] = []
        else:
            rows = self.connection.execute(
                "SELECT parent_asin FROM products WHERE products MATCH ? "
                "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT ?",
                (expression, min(max(int(top_k), 1), 10)),
            ).fetchall()
            recommendations = [{"parent_asin": str(row[0])} for row in rows]

        if turn < 10:
            schedule_index = min(max(turn - 1, 0), len(QUESTION_SCHEDULE) - 1)
            ask_attribute, message = QUESTION_SCHEDULE[schedule_index]
        else:
            ask_attribute = None
            message = "Here are the closest matches I found."

        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
