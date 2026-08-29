from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field, replace
from enum import Enum


class Facet(str, Enum):
    CATEGORY = "category"
    MATERIAL = "material"
    COLOR = "color"
    SIZE = "size"
    STYLE = "style"
    BRAND = "brand"
    BUDGET = "budget"
    FEATURE = "feature"
    USE_CASE = "use_case"
    OTHER = "other"


class Strength(str, Enum):
    MUST = "MUST"
    SHOULD = "SHOULD"
    AVOID = "AVOID"
    NO_PREFERENCE = "NO_PREFERENCE"


class Polarity(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class ConstraintStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    NEGATED = "negated"


class ConstraintSource(str, Enum):
    CATEGORY_ANCHOR = "category_anchor"
    INITIAL_PREFERENCE = "initial_preference"
    EXPLICIT_REQUIREMENT = "explicit_requirement"
    CLARIFICATION = "clarification"
    CORRECTION = "correction"
    NEGATION = "negation"
    BOUNDARY = "boundary"


SPACE_RE = re.compile(r"\s+")
LOOKING_FOR_RE = re.compile(
    r"\b(?:i(?:'m| am)\s+)?(?:looking|shopping|searching)\s+for\s+"
    r"(?:an?\s+|some\s+)?(?P<value>[^,.!?;]+)",
    re.IGNORECASE,
)
NEED_RE = re.compile(
    r"^\s*(?:i\s+)?(?:need|want|would\s+like)\s+"
    r"(?:an?\s+|some\s+)?(?P<value>.+?)[.!]?\s*$",
    re.IGNORECASE,
)
CATEGORY_NOUN_RE = re.compile(
    r"\b(?:shoes?|boots?|shirts?|t-?shirts?|tees?|jackets?|coats?|pants?|"
    r"dresses?|wallets?|watches?|socks?|hats?|caps?|bags?|bras?|underwear|"
    r"sandals?|slippers?|sneakers?|hoodies?|sweaters?|skirts?|shorts?|belts?|"
    r"gloves?|scarves?|jewelry|bracelets?|necklaces?|earrings?|rings?)\b",
    re.IGNORECASE,
)
PAYLOAD_RE = re.compile(
    r"\b(?:a\s+key\s+requirement\s+is|what\s+matters\s+is|"
    r"what\s+i\s+need\s+is|my\s+priority\s+is)\s*:?\s*(?P<value>.+)$",
    re.IGNORECASE,
)
OVERRIDE_RE = re.compile(
    r"^\s*(?:(?:well\s*,?\s*)?actually\b|(?:i\s+)?changed\s+my\s+mind\b|"
    r"(?:please\s+)?(?:ignore|forget|disregard)\s+(?:my\s+)?"
    r"(?:earlier|previous|old)\s+preference\b)",
    re.IGNORECASE,
)
IGNORE_EARLIER_RE = re.compile(
    r"\b(?:ignore|forget|disregard)\s+(?:my\s+)?(?:earlier|previous|old)\s+preference\b",
    re.IGNORECASE,
)
CHANGED_MIND_RE = re.compile(
    r"^\s*(?:i\s+)?changed\s+my\s+mind\b",
    re.IGNORECASE,
)
NO_PREFERENCE_RE = re.compile(
    r"\b(?:no\s+preference|"
    r"(?:i\s+)?(?:do\s+not|don't)\s+have\s+(?:an?\s+|any\s+)?(?:additional\s+)?preference|"
    r"(?:i\s+)?(?:do\s+not|don't)\s+mind|"
    r"(?:any|either)\s+\w+\s+(?:is|are)\s+fine|"
    r"use\s+your\s+judg(?:e)?ment)\b",
    re.IGNORECASE,
)
GENERIC_REJECTION_RE = re.compile(
    r"^(?:(?:those|these|the)\s+(?:options|results|recommendations)\s+"
    r"(?:are\s+)?not\s+(?:quite\s+)?right(?:\s+yet)?|"
    r"none\s+of\s+(?:these|those)\s+(?:options|results|recommendations)\s+work)"
    r"(?:[.!]\s*(?:ask\s+me\s+about\s+one\s+specific\s+attribute\.?)?)?$",
    re.IGNORECASE,
)
NEGATION_RE = re.compile(
    r"\b(?:do\s+not\s+(?:want|like)|don't\s+(?:want|like)|"
    r"would\s+not\s+like|wouldn't\s+like|dislike|anything\s+but|without|"
    r"avoid|except|not|no)\s+(?P<value>[^,.;]+)",
    re.IGNORECASE,
)
INSTEAD_RE = re.compile(
    r"(?P<new>.+?)\s+(?:instead\s+of|rather\s+than)\s+(?P<old>[^,.;]+)",
    re.IGNORECASE,
)
PREFIX_INSTEAD_RE = re.compile(
    r"^\s*(?:instead\s+of|rather\s+than)\s+(?P<old>.+?)\s*(?:[,;]\s*)?"
    r"(?P<new>(?:i\s+)?(?:want|prefer|need|require|would\s+like)\s+.+?)"
    r"[.!]?\s*$",
    re.IGNORECASE,
)
DISCOURSE_PREFIX_RE = re.compile(
    r"^(?:actually|instead|please|for\s+that|now|well|"
    r"(?:i\s+)?changed\s+my\s+mind)\b[\s,.:;-]*",
    re.IGNORECASE,
)
PREFERENCE_PREFIX_RE = re.compile(
    r"^(?:i\s+)?(?:prefer|want|would\s+like|need|require)\s+"
    r"(?:an?\s+|some\s+|it\s+to\s+be\s+)?",
    re.IGNORECASE,
)

MATERIAL_RE = re.compile(
    r"\b(?:cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric|"
    r"acrylic|rubber|synthetic|textile|canvas|suede|fur|steel|alloy|metal|"
    r"modal|viscose|eva|pvc|pu|polyurethane|plastic|resin|fleece|denim|linen|"
    r"latex)\b",
    re.IGNORECASE,
)
COLOR_RE = re.compile(
    r"\b(?:colou?r|black|white|blue|red|pink|green|brown|gr[ae]y|purple|"
    r"yellow|orange|beige|navy|gold|silver)\b",
    re.IGNORECASE,
)
SIZE_RE = re.compile(
    r"\b(?:size|sizing|width|wide|narrow|petite|plus\s+size|small|medium|large|"
    r"\d+(?:\.\d+)?\s*(?:inch|inches|cm|mm))\b",
    re.IGNORECASE,
)
STYLE_RE = re.compile(
    r"\b(?:style|fit|fitted|loose|slim|department|sleeve|neck|closure|rise)\b",
    re.IGNORECASE,
)
USE_CASE_RE = re.compile(
    r"\b(?:hiking|running|walking|gym|winter|outdoor|work|office|travel|sport|"
    r"waterproof|rain|wedding|party|casual)\b",
    re.IGNORECASE,
)
BUDGET_RE = re.compile(
    r"(?:\b(?:budget|price|cost)\b|[$£€]\s*\d|"
    r"\b(?:under|below|less\s+than|around)\s*[$£€]?\s*\d)",
    re.IGNORECASE,
)
BRAND_RE = re.compile(r"\b(?:brand|store|maker|manufacturer)\b", re.IGNORECASE)


def normalize_value(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value))
    normalized = SPACE_RE.sub(" ", normalized).strip(" \t\r\n.,;:!?-\"")
    normalized = re.sub(r"\bgrey\b", "gray", normalized, flags=re.IGNORECASE)
    return normalized.casefold()


def _facet(value: str | Facet | None) -> Facet | None:
    if isinstance(value, Facet):
        return value
    if value is None:
        return None
    try:
        return Facet(value)
    except ValueError:
        return None


def infer_facet(value: str, expected: Facet | None = None) -> Facet:
    if expected is not None:
        return expected
    if re.match(r"^\s*(?:material|fabric)\s*:", value, re.IGNORECASE):
        return Facet.MATERIAL
    if BUDGET_RE.search(value):
        return Facet.BUDGET
    if MATERIAL_RE.search(value):
        return Facet.MATERIAL
    if COLOR_RE.search(value):
        return Facet.COLOR
    if SIZE_RE.search(value):
        return Facet.SIZE
    if BRAND_RE.search(value):
        return Facet.BRAND
    if STYLE_RE.search(value):
        return Facet.STYLE
    if USE_CASE_RE.search(value):
        return Facet.USE_CASE
    return Facet.FEATURE


def _mentioned_facets(message: str, fallback: Facet | None) -> list[Facet]:
    lowered = message.lower()
    mentions: list[tuple[int, Facet]] = []
    for facet in Facet:
        aliases = {facet.value, facet.value.replace("_", " ")}
        positions = [
            match.start()
            for alias in aliases
            if (match := re.search(rf"\b{re.escape(alias)}\b", lowered))
        ]
        if positions:
            mentions.append((min(positions), facet))
    if not mentions:
        return [fallback or Facet.OTHER]
    return [facet for _, facet in sorted(mentions)]


def _clean_preference_text(value: str) -> str:
    cleaned = SPACE_RE.sub(" ", value).strip(" \t\r\n.,;:!?-")
    previous = None
    while cleaned and cleaned != previous:
        previous = cleaned
        cleaned = DISCOURSE_PREFIX_RE.sub("", cleaned).strip(" \t\r\n.,;:!?-")
        cleaned = PREFERENCE_PREFIX_RE.sub("", cleaned).strip(" \t\r\n.,;:!?-")
        cleaned = re.sub(r"^(?:but|and)\b[\s,:-]*", "", cleaned, flags=re.IGNORECASE)
    return cleaned


def _no_preference_remainder(message: str, match: re.Match[str]) -> str:
    """Return a real requirement that follows a no-preference clause, if any."""

    tail = message[match.end() :]
    delimiter = re.search(r"[,;.]|\bbut\b", tail, flags=re.IGNORECASE)
    if delimiter is None:
        return ""
    remainder = re.sub(
        r"^\s*(?:[,;.]\s*)?(?:but\s+)?",
        "",
        tail[delimiter.start() :],
        flags=re.IGNORECASE,
    )
    if not remainder or NO_PREFERENCE_RE.search(remainder):
        return ""
    return remainder


def _no_preference_clause(message: str, match: re.Match[str]) -> str:
    tail = message[match.end() :]
    delimiter = re.search(r"[,;.]|\bbut\b", tail, flags=re.IGNORECASE)
    end = len(message) if delimiter is None else match.end() + delimiter.start()
    return message[match.start() : end]


@dataclass(frozen=True)
class ParsedConstraint:
    facet: Facet
    raw_value: str
    normalized_value: str
    polarity: Polarity
    strength: Strength
    confidence: float
    source_turn: int
    source: ConstraintSource
    evidence_span: str


@dataclass(frozen=True)
class Retraction:
    facet: Facet
    normalized_value: str


@dataclass(frozen=True)
class ParseResult:
    constraints: tuple[ParsedConstraint, ...]
    is_override: bool = False
    retract_initial_preference: bool = False
    retractions: tuple[Retraction, ...] = ()


@dataclass(frozen=True)
class Constraint:
    constraint_id: int
    facet: Facet
    normalized_value: str
    raw_value: str
    polarity: Polarity
    strength: Strength
    confidence: float
    source_turn: int
    status: ConstraintStatus
    supersedes: tuple[int, ...]
    evidence_span: str
    source: ConstraintSource

    def as_dict(self) -> dict:
        return {
            "constraint_id": self.constraint_id,
            "facet": self.facet.value,
            "normalized_value": self.normalized_value,
            "raw_value": self.raw_value,
            "polarity": self.polarity.value,
            "strength": self.strength.value,
            "confidence": self.confidence,
            "source_turn": self.source_turn,
            "status": self.status.value,
            "supersedes": list(self.supersedes),
            "evidence_span": self.evidence_span,
            "source": self.source.value,
        }


def _parsed(
    *,
    facet: Facet,
    value: str,
    turn: int,
    source: ConstraintSource,
    evidence: str,
    strength: Strength,
    polarity: Polarity = Polarity.POSITIVE,
    confidence: float = 0.9,
) -> ParsedConstraint | None:
    raw_value = _clean_preference_text(value)
    normalized = normalize_value(raw_value)
    if not normalized and strength is not Strength.NO_PREFERENCE:
        return None
    return ParsedConstraint(
        facet=facet,
        raw_value=raw_value,
        normalized_value=normalized,
        polarity=polarity,
        strength=strength,
        confidence=confidence,
        source_turn=turn,
        source=source,
        evidence_span=evidence,
    )


def parse_message(
    message: str,
    turn: int,
    expected_facet: str | Facet | None = None,
    *,
    infer_other_facets: bool = False,
) -> ParseResult:
    """Parse only customer-visible text; no labels or evaluator state are used."""

    normalized_message = SPACE_RE.sub(
        " ", message.translate(str.maketrans({"’": "'", "‘": "'"}))
    ).strip()
    expected = _facet(expected_facet)
    if not normalized_message:
        return ParseResult(())
    if GENERIC_REJECTION_RE.fullmatch(normalized_message):
        return ParseResult(())

    parsed: list[ParsedConstraint] = []
    is_override = bool(OVERRIDE_RE.search(normalized_message))
    category_match = LOOKING_FOR_RE.search(normalized_message)
    need_match = NEED_RE.match(normalized_message) if turn == 1 and not category_match else None
    need_category_match = None
    if need_match:
        noun_matches = list(CATEGORY_NOUN_RE.finditer(need_match.group("value")))
        need_category_match = noun_matches[-1] if noun_matches else None

    if category_match or need_category_match:
        category_value = (
            category_match.group("value")
            if category_match
            else need_category_match.group(0)
        )
        category_evidence = (
            category_match.group(0) if category_match else need_match.group(0)
        )
        category = _parsed(
            facet=Facet.CATEGORY,
            value=category_value,
            turn=turn,
            source=ConstraintSource.CATEGORY_ANCHOR,
            evidence=category_evidence,
            strength=Strength.MUST,
            confidence=1.0,
        )
        if category:
            parsed.append(category)

    no_preference_match = NO_PREFERENCE_RE.search(normalized_message)
    boundary_remainder = ""
    if no_preference_match:
        boundary_clause = _no_preference_clause(
            normalized_message,
            no_preference_match,
        )
        for facet in _mentioned_facets(boundary_clause, expected):
            no_preference = _parsed(
                facet=facet,
                value="",
                turn=turn,
                source=ConstraintSource.BOUNDARY,
                evidence=boundary_clause,
                strength=Strength.NO_PREFERENCE,
                polarity=Polarity.NEUTRAL,
                confidence=1.0,
            )
            if no_preference:
                parsed.append(no_preference)
        boundary_remainder = _no_preference_remainder(
            normalized_message,
            no_preference_match,
        )
        if not boundary_remainder:
            return ParseResult(tuple(parsed), is_override=is_override)

    payload_match = PAYLOAD_RE.search(normalized_message)
    payload = payload_match.group("value") if payload_match else ""

    residual = boundary_remainder or normalized_message
    if category_match:
        residual = residual[: category_match.start()] + " " + residual[category_match.end() :]
    elif need_match and need_category_match:
        need_value = need_match.group("value")
        residual = (
            need_value[: need_category_match.start()]
            + " "
            + need_value[need_category_match.end() :]
        )
    if payload_match:
        residual = payload
    residual = re.sub(
        r"\b(?:a\s+key\s+requirement\s+is|what\s+matters\s+is|"
        r"what\s+i\s+need\s+is|my\s+priority\s+is)\s*:?\s*",
        "",
        residual,
        flags=re.IGNORECASE,
    )
    residual = re.sub(
        r"\b(?:i(?:'m| am)\s+still\s+exploring|ignore\s+my\s+earlier\s+preference)\b",
        "",
        residual,
        flags=re.IGNORECASE,
    )
    prefix_instead_match = (
        None if payload_match else PREFIX_INSTEAD_RE.search(residual)
    )
    residual = _clean_preference_text(residual)

    retractions: list[Retraction] = []
    instead_match = (
        None
        if payload_match
        else prefix_instead_match or INSTEAD_RE.search(residual)
    )
    if instead_match:
        new_value = _clean_preference_text(instead_match.group("new"))
        old_value = _clean_preference_text(instead_match.group("old"))
        new_facet = infer_facet(new_value)
        old_facet = infer_facet(old_value)
        correction = _parsed(
            facet=new_facet,
            value=new_value,
            turn=turn,
            source=ConstraintSource.CORRECTION,
            evidence=instead_match.group(0),
            strength=Strength.MUST,
            confidence=1.0,
        )
        if correction:
            parsed.append(correction)
        if old_value:
            old_values = [old_value]
            old_values.extend(
                item
                for item in re.split(r"\s+(?:and|or)\s+", old_value)
                if item and item != old_value
            )
            for retracted_value in dict.fromkeys(old_values):
                retractions.append(
                    Retraction(
                        infer_facet(retracted_value, old_facet),
                        normalize_value(retracted_value),
                    )
                )
        return ParseResult(
            tuple(parsed),
            is_override=True,
            retractions=tuple(retractions),
        )

    # Protocol-scaffolded payloads are catalog evidence. Product copy such as
    # "will not fade" or "without sharp edges" is a positive target clue, not
    # a customer-level exclusion. Parse negation only in direct utterances.
    negations = [] if payload_match else list(NEGATION_RE.finditer(residual))
    positive_residual = residual
    for match in negations:
        # The simulator may expose a catalog detail rendered as
        # "No Closure closure" in the initial intent card. It is product
        # evidence, not a customer exclusion, so discard that schema artifact.
        if (
            turn == 1
            and (category_match is not None or need_category_match is not None)
            and re.fullmatch(
                r"no\s+closure\s+closure",
                match.group(0).strip(),
                flags=re.IGNORECASE,
            )
        ):
            positive_residual = positive_residual.replace(match.group(0), " ")
            continue
        value = _clean_preference_text(match.group("value"))
        negative_values = [
            item
            for item in re.split(r"\s+(?:and|or)\s+", value)
            if item.strip()
        ]
        for negative_value in negative_values:
            negative = _parsed(
                facet=infer_facet(negative_value),
                value=negative_value,
                turn=turn,
                source=ConstraintSource.NEGATION,
                evidence=match.group(0),
                strength=Strength.AVOID,
                polarity=Polarity.NEGATIVE,
                confidence=1.0,
            )
            if negative:
                parsed.append(negative)
        positive_residual = positive_residual.replace(match.group(0), " ")

    positive_residual = _clean_preference_text(positive_residual)
    if positive_residual.lower() in {"i", "me", "please"}:
        positive_residual = ""
    if positive_residual and not re.fullmatch(
        r"(?:those\s+options\s+are\s+)?(?:quite\s+)?right\s+yet",
        positive_residual,
        flags=re.IGNORECASE,
    ):
        if is_override:
            source = ConstraintSource.CORRECTION
            strength = Strength.MUST
            confidence = 1.0
        elif payload_match and "key requirement" in payload_match.group(0).lower():
            source = ConstraintSource.EXPLICIT_REQUIREMENT
            strength = Strength.MUST
            confidence = 1.0
        elif payload_match:
            source = ConstraintSource.CLARIFICATION
            strength = Strength.MUST
            confidence = 0.95
        elif turn == 1:
            source = ConstraintSource.INITIAL_PREFERENCE
            strength = Strength.SHOULD
            confidence = 0.85
        else:
            source = ConstraintSource.CLARIFICATION
            strength = Strength.SHOULD
            confidence = 0.8
        positive_values = (
            [item for item in re.split(r";\s+", positive_residual) if item.strip()]
            if payload_match
            else [positive_residual]
        )
        for positive_value in positive_values:
            direct_answer = payload_match is not None and not is_override
            expected_answer = expected if direct_answer else None
            if infer_other_facets and expected_answer is Facet.OTHER:
                expected_answer = None
            facet = infer_facet(positive_value, expected_answer)
            positive = _parsed(
                facet=facet,
                value=positive_value,
                turn=turn,
                source=source,
                evidence=positive_value,
                strength=strength,
                confidence=confidence,
            )
            if positive:
                parsed.append(positive)

    return ParseResult(
        tuple(parsed),
        is_override=is_override,
        retract_initial_preference=(
            is_override
            and bool(
                IGNORE_EARLIER_RE.search(normalized_message)
                or CHANGED_MIND_RE.search(normalized_message)
            )
        ),
        retractions=tuple(retractions),
    )


@dataclass
class ConstraintLedger:
    entries: list[Constraint] = field(default_factory=list)
    _next_id: int = 1

    def _change_status(
        self,
        predicate,
        status: ConstraintStatus,
    ) -> list[int]:
        changed: list[int] = []
        for index, entry in enumerate(self.entries):
            if entry.status is ConstraintStatus.ACTIVE and predicate(entry):
                self.entries[index] = replace(entry, status=status)
                changed.append(entry.constraint_id)
        return changed

    def apply(self, update: ParseResult) -> None:
        superseded: list[int] = []
        if update.retract_initial_preference:
            initial_values = {
                entry.normalized_value
                for entry in self.entries
                if entry.source_turn == 1
                and entry.facet is not Facet.CATEGORY
                and entry.strength is not Strength.NO_PREFERENCE
                and entry.normalized_value
            }
            if initial_values:
                superseded.extend(
                    self._change_status(
                        lambda item: item.normalized_value in initial_values,
                        ConstraintStatus.SUPERSEDED,
                    )
                )

        for retraction in update.retractions:
            superseded.extend(
                self._change_status(
                    lambda item, old=retraction: (
                        item.facet is old.facet
                        and item.polarity is Polarity.POSITIVE
                        and item.normalized_value == old.normalized_value
                    ),
                    ConstraintStatus.SUPERSEDED,
                )
            )

        has_targeted_retraction = bool(update.retract_initial_preference or update.retractions)
        for parsed in update.constraints:
            entry_supersedes = list(superseded)
            superseded.clear()

            if parsed.strength is Strength.NO_PREFERENCE:
                no_additional_value = bool(
                    re.search(r"\badditional\b", parsed.evidence_span, re.IGNORECASE)
                )
                entry_supersedes.extend(
                    self._change_status(
                        lambda item, facet=parsed.facet, preserve=no_additional_value: (
                            item.facet is facet
                            and (not preserve or item.polarity is not Polarity.POSITIVE)
                        ),
                        ConstraintStatus.SUPERSEDED,
                    )
                )
            elif parsed.polarity is Polarity.NEGATIVE:
                entry_supersedes.extend(
                    self._change_status(
                        lambda item, current=parsed: (
                            item.facet is current.facet
                            and item.polarity is Polarity.POSITIVE
                            and item.normalized_value == current.normalized_value
                        ),
                        ConstraintStatus.NEGATED,
                    )
                )
            else:
                entry_supersedes.extend(
                    self._change_status(
                        lambda item, current=parsed: (
                            item.facet is current.facet
                            and item.polarity is not Polarity.POSITIVE
                            and (
                                not item.normalized_value
                                or item.normalized_value == current.normalized_value
                            )
                        ),
                        ConstraintStatus.SUPERSEDED,
                    )
                )
                duplicate_ids = self._change_status(
                    lambda item, current=parsed: (
                        item.facet is current.facet
                        and item.polarity is Polarity.POSITIVE
                        and item.normalized_value == current.normalized_value
                    ),
                    ConstraintStatus.SUPERSEDED,
                )
                entry_supersedes.extend(duplicate_ids)
                if parsed.source is ConstraintSource.CORRECTION and not has_targeted_retraction:
                    entry_supersedes.extend(
                        self._change_status(
                            lambda item, facet=parsed.facet: (
                                item.facet is facet and item.polarity is Polarity.POSITIVE
                            ),
                            ConstraintStatus.SUPERSEDED,
                        )
                    )

            self.entries.append(
                Constraint(
                    constraint_id=self._next_id,
                    facet=parsed.facet,
                    normalized_value=parsed.normalized_value,
                    raw_value=parsed.raw_value,
                    polarity=parsed.polarity,
                    strength=parsed.strength,
                    confidence=parsed.confidence,
                    source_turn=parsed.source_turn,
                    status=ConstraintStatus.ACTIVE,
                    supersedes=tuple(dict.fromkeys(entry_supersedes)),
                    evidence_span=parsed.evidence_span,
                    source=parsed.source,
                )
            )
            self._next_id += 1

    def active(self) -> list[Constraint]:
        return [entry for entry in self.entries if entry.status is ConstraintStatus.ACTIVE]

    def canonical_query(self) -> str:
        positive = [
            entry
            for entry in self.active()
            if entry.polarity is Polarity.POSITIVE
            and entry.strength in {Strength.MUST, Strength.SHOULD}
            and entry.normalized_value
        ]
        strength_order = {Strength.MUST: 0, Strength.SHOULD: 1}
        positive.sort(
            key=lambda item: (
                0 if item.facet is Facet.CATEGORY else 1,
                strength_order[item.strength],
                item.source_turn,
                item.constraint_id,
            )
        )
        values = list(dict.fromkeys(entry.normalized_value for entry in positive))
        return " ".join(values)

    def active_avoid_values(self) -> list[str]:
        """Return active customer exclusions for candidate-level filtering."""

        return list(
            dict.fromkeys(
                entry.normalized_value
                for entry in self.active()
                if entry.polarity is Polarity.NEGATIVE
                and entry.strength is Strength.AVOID
                and entry.normalized_value
            )
        )

    def evidence_trace(self) -> list[dict]:
        return [entry.as_dict() for entry in self.entries]
