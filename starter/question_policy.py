from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
PRICE_RE = re.compile(r"(?:\$|£|€)?\s*(\d+(?:\.\d+)?)")
MATERIAL_RE = re.compile(
    r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric)\b",
    re.IGNORECASE,
)
COLOR_RE = re.compile(
    r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|"
    r"orange)\b",
    re.IGNORECASE,
)
SIZE_RE = re.compile(
    r"\b(?:size|sizing|width|wide|narrow|petite|plus size|one size|"
    r"xxs|xs|small|medium|large|xl|xxl|xxxl)\b",
    re.IGNORECASE,
)
STYLE_RE = re.compile(
    r"\b(?:department|style|fit|fitted|relaxed|slim|regular|sleeve|neck|"
    r"crewneck|v-neck|pattern|formal|casual|vintage|classic|modern)\b",
    re.IGNORECASE,
)
USE_CASE_RE = re.compile(
    r"\b(?:hiking|running|walking|gym|fitness|training|winter|summer|"
    r"outdoor|work|office|travel|cycling|dance|yoga|sports?|trail|road|"
    r"wedding|party|school)\b",
    re.IGNORECASE,
)

SPECIFIC_ATTRIBUTES: tuple[str, ...] = (
    "material",
    "feature",
    "color",
    "style",
    "size",
    "use_case",
    "brand",
    "budget",
)
FALLBACK_ORDER: tuple[str, ...] = (*SPECIFIC_ATTRIBUTES, "other")
QUESTION_MESSAGES: Mapping[str | None, str] = {
    "material": "Do you have a material preference?",
    "feature": "Which product feature matters most to you?",
    "color": "Do you have a color preference?",
    "style": "Which style or fit do you prefer?",
    "size": "Do you have a size or sizing preference?",
    "use_case": "What will you mainly use the product for?",
    "brand": "Do you prefer a particular brand?",
    "budget": "What budget should I stay within?",
    "other": "What other requirement matters most?",
    None: "Here are the closest matches I found.",
}
MISSING_BUCKET = "<missing>"

# These weights describe evidence quality, not answer popularity. Exact lexical
# facets receive more trust than broad or sparse metadata fields.
ATTRIBUTE_RELIABILITY: Mapping[str, float] = {
    "material": 1.00,
    "color": 0.95,
    "feature": 0.80,
    "use_case": 0.65,
    "size": 0.55,
    "style": 0.50,
    "budget": 0.05,
    "brand": 0.00,
}
MIN_ADAPTIVE_CANDIDATES = 8
MIN_SELECTION_SCORE = 0.006


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key}: {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _values(value: object) -> list[str]:
    if isinstance(value, dict):
        return [
            f"{key}: {item}"
            for key, item in value.items()
            if item not in (None, "", [])
        ]
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)] if value not in (None, "") else []


def _normalized_phrase(value: str, limit: int = 12) -> str:
    tokens = [token.lower() for token in TOKEN_RE.findall(value)][:limit]
    return " ".join(tokens)


def _deduplicated(values: Sequence[str], limit: int = 4) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = _normalized_phrase(value)
        if normalized and normalized not in seen:
            result.append(normalized)
            seen.add(normalized)
        if len(result) >= limit:
            break
    return tuple(result)


def _classify_catalog_value(value: str) -> str:
    lowered = value.lower()
    if "budget" in lowered or re.search(r"(?:\$|<=|under)\s*\d", lowered):
        return "budget"
    if MATERIAL_RE.search(value):
        return "material"
    if COLOR_RE.search(value) or "color" in lowered or "colour" in lowered:
        return "color"
    if SIZE_RE.search(value):
        return "size"
    if STYLE_RE.search(value):
        return "style"
    if USE_CASE_RE.search(value):
        return "use_case"
    return "feature"


def _price_bucket(value: object) -> str | None:
    try:
        price = float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        match = PRICE_RE.search(str(value))
        price = float(match.group(1)) if match else None
    if price is None:
        return None
    if price < 25:
        return "under 25"
    if price < 50:
        return "25 to 49"
    if price < 100:
        return "50 to 99"
    return "100 or more"


def extract_question_facets(product: Mapping[str, object]) -> dict[str, tuple[str, ...]]:
    """Extract deterministic candidate evidence from public catalog fields only."""

    search_text = " ".join(
        _text(product.get(field))
        for field in (
            "title",
            "features",
            "details",
            "description",
            "categories",
            "store",
        )
    )
    salient_values = [
        *_values(product.get("features")),
        *_values(product.get("details")),
    ]
    material = MATERIAL_RE.search(search_text)
    color = COLOR_RE.search(search_text)
    if material:
        salient_values.insert(0, material.group(1).lower())
    if color:
        salient_values.insert(1 if material else 0, f"color: {color.group(1).lower()}")
    if product.get("price") not in (None, ""):
        salient_values.append(f"budget around ${product['price']}")

    # Question answerability should follow the product's most salient catalog
    # evidence. Considering every metadata value makes ubiquitous fields such
    # as Department look far more useful than they are to a customer.
    salient_values = list(dict.fromkeys(salient_values))[:4]
    collected: dict[str, list[str]] = defaultdict(list)
    for value in salient_values:
        attribute = _classify_catalog_value(value)
        if attribute == "material":
            match = MATERIAL_RE.search(value)
            collected[attribute].append(match.group(1) if match else value)
        elif attribute == "color":
            match = COLOR_RE.search(value)
            collected[attribute].append(match.group(1) if match else value)
        elif attribute == "budget":
            collected[attribute].append(_price_bucket(product.get("price")) or value)
        else:
            collected[attribute].append(value)

    return {
        attribute: values
        for attribute, raw_values in collected.items()
        if (values := _deduplicated(raw_values))
    }


@dataclass(frozen=True)
class QuestionCandidate:
    parent_asin: str
    facets: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True)
class FacetStatistics:
    attribute: str
    candidate_count: int
    observed_count: int
    missing_count: int
    answer_distribution: Mapping[str, int]
    entropy: float
    expected_technical_gain: float
    selection_score: float
    eligible: bool
    ineligible_reason: str | None = None
    already_constrained: bool = False

    def as_dict(self) -> dict:
        return {
            "attribute": self.attribute,
            "candidate_count": self.candidate_count,
            "observed_count": self.observed_count,
            "missing_count": self.missing_count,
            "missing_rate": round(
                self.missing_count / self.candidate_count
                if self.candidate_count
                else 1.0,
                6,
            ),
            "answer_distribution": dict(sorted(self.answer_distribution.items())),
            "entropy": round(self.entropy, 6),
            "expected_technical_gain": round(self.expected_technical_gain, 6),
            "selection_score": round(self.selection_score, 6),
            "eligible": self.eligible,
            "ineligible_reason": self.ineligible_reason,
            "already_constrained": self.already_constrained,
        }


@dataclass(frozen=True)
class QuestionDecision:
    ask_attribute: str | None
    message: str
    turn: int
    candidate_count: int
    reason: str
    statistics: tuple[FacetStatistics, ...]

    def as_dict(self) -> dict:
        return {
            "turn": self.turn,
            "candidate_count": self.candidate_count,
            "selected_attribute": self.ask_attribute,
            "message": self.message,
            "reason": self.reason,
            "statistics": [item.as_dict() for item in self.statistics],
        }


def _retrieval_utility(rank: int, hit_turn: int) -> float:
    if rank > 10:
        return 0.0
    efficiency = 0.20 * max(0.0, min(1.0, (11.0 - hit_turn) / 10.0))
    return 0.5 + 0.3 / rank + efficiency


def _normalized_entropy(counts: Mapping[str, int], candidate_count: int) -> float:
    if candidate_count <= 1:
        return 0.0
    nonzero = [count for count in counts.values() if count]
    if len(nonzero) <= 1:
        return 0.0
    entropy = -sum(
        (count / candidate_count) * math.log(count / candidate_count)
        for count in nonzero
    )
    return entropy / math.log(min(candidate_count, len(nonzero)))


class AdaptiveQuestionPolicy:
    """Choose the next clarification from current candidate uncertainty."""

    def __init__(
        self,
        *,
        min_adaptive_candidates: int = MIN_ADAPTIVE_CANDIDATES,
        min_selection_score: float = MIN_SELECTION_SCORE,
    ) -> None:
        self.min_adaptive_candidates = min_adaptive_candidates
        self.min_selection_score = min_selection_score

    @staticmethod
    def _representative(candidate: QuestionCandidate, attribute: str) -> str:
        values = candidate.facets.get(attribute, ())
        return values[0] if values else MISSING_BUCKET

    def _statistics(
        self,
        candidates: Sequence[QuestionCandidate],
        attribute: str,
        active_facets: set[str],
        asked_attributes: set[str],
        turn: int,
    ) -> FacetStatistics:
        representatives = [self._representative(item, attribute) for item in candidates]
        distribution = Counter(representatives)
        distribution.setdefault(MISSING_BUCKET, 0)
        missing = distribution[MISSING_BUCKET]
        observed = len(candidates) - missing
        entropy = _normalized_entropy(distribution, len(candidates))

        groups: dict[str, list[int]] = defaultdict(list)
        for rank, value in enumerate(representatives, start=1):
            groups[value].append(rank)
        within_group_rank: dict[int, int] = {}
        for value, ranks in groups.items():
            if value == MISSING_BUCKET:
                continue
            for new_rank, old_rank in enumerate(ranks, start=1):
                within_group_rank[old_rank] = new_rank

        if candidates:
            prior_weights = [1.0 / math.sqrt(rank) for rank in range(1, len(candidates) + 1)]
            prior_total = sum(prior_weights)
            expected_gain = 0.0
            for old_rank, (value, prior) in enumerate(
                zip(representatives, prior_weights),
                start=1,
            ):
                new_rank = old_rank if value == MISSING_BUCKET else within_group_rank[old_rank]
                reply_turn = min(turn + 1, 10)
                gain = max(
                    0.0,
                    _retrieval_utility(new_rank, reply_turn)
                    - _retrieval_utility(old_rank, reply_turn),
                )
                expected_gain += (prior / prior_total) * gain
        else:
            expected_gain = 0.0

        observed_rate = observed / len(candidates) if candidates else 0.0
        constrained_factor = 0.60 if attribute in active_facets else 1.0
        reliability = ATTRIBUTE_RELIABILITY[attribute]
        score = (
            reliability
            * constrained_factor
            * observed_rate**1.5
            * (expected_gain + 0.025 * entropy)
        )
        eligible = attribute not in asked_attributes
        return FacetStatistics(
            attribute=attribute,
            candidate_count=len(candidates),
            observed_count=observed,
            missing_count=missing,
            answer_distribution=dict(distribution),
            entropy=entropy,
            expected_technical_gain=expected_gain,
            selection_score=score if eligible else 0.0,
            eligible=eligible,
            ineligible_reason=None if eligible else "already_asked",
            already_constrained=attribute in active_facets,
        )

    def choose(
        self,
        candidates: Sequence[QuestionCandidate],
        *,
        active_facets: set[str],
        asked_attributes: set[str],
        turn: int,
        guardrail_attribute: str | None = None,
    ) -> QuestionDecision:
        statistics = tuple(
            self._statistics(
                candidates,
                attribute,
                active_facets,
                asked_attributes,
                turn,
            )
            for attribute in SPECIFIC_ATTRIBUTES
        )
        eligible = [item for item in statistics if item.eligible]

        if turn >= 10:
            selected: str | None = None
            reason = "turn_limit"
        elif (
            guardrail_attribute in SPECIFIC_ATTRIBUTES
            and guardrail_attribute not in asked_attributes
        ):
            selected = guardrail_attribute
            reason = "first_turn_guardrail"
        elif len(candidates) < self.min_adaptive_candidates:
            selected = next(
                (attribute for attribute in FALLBACK_ORDER if attribute not in asked_attributes),
                None,
            )
            reason = "small_candidate_fallback"
        else:
            priority = {attribute: index for index, attribute in enumerate(SPECIFIC_ATTRIBUTES)}
            ranked = sorted(
                eligible,
                key=lambda item: (-item.selection_score, priority[item.attribute]),
            )
            best = ranked[0] if ranked else None
            if best is not None and best.selection_score >= self.min_selection_score:
                selected = best.attribute
                reason = "highest_expected_technical_gain"
            elif "other" not in asked_attributes:
                selected = "other"
                reason = "specific_facets_below_threshold"
            else:
                selected = best.attribute if best is not None else None
                reason = "fallback_exhausted"

        return QuestionDecision(
            ask_attribute=selected,
            message=QUESTION_MESSAGES[selected],
            turn=turn,
            candidate_count=len(candidates),
            reason=reason,
            statistics=statistics,
        )
