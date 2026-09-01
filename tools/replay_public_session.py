from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any

from evaluator.local_evaluator import (
    MAX_TURNS,
    TOP_K,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
    normalize_recommendations,
)
from starter.agent import Agent


def _recorded_session(results_path: Path, sample_id: str) -> dict[str, Any]:
    with results_path.open(encoding="utf-8") as handle:
        result = json.load(handle)
    for session in result.get("sessions", []):
        if session.get("sample_id") == sample_id:
            return session
    raise SystemExit(f"sample {sample_id!r} is absent from {results_path}")


def _sample(dataset_path: Path, sample_id: str) -> dict[str, Any]:
    for sample in load_jsonl(dataset_path):
        if sample.get("sample_id") == sample_id:
            return sample
    raise SystemExit(f"sample {sample_id!r} is absent from {dataset_path}")


def _append_unique(values: list[str], value: object) -> None:
    text = str(value).strip()
    if text and text not in values:
        values.append(text)


def replay(
    catalog_path: Path,
    dataset_path: Path,
    results_path: Path,
    sample_id: str,
) -> dict[str, Any]:
    sample = _sample(dataset_path, sample_id)
    recorded = _recorded_session(results_path, sample_id)
    if sample.get("scenario_type") != "intent_override":
        raise SystemExit(
            "the screenshot workflow expects an Intent Override sample; "
            f"{sample_id!r} is {sample.get('scenario_type')!r}"
        )

    catalog_ids, categories, products = catalog_index(catalog_path)
    agent = Agent(str(catalog_path))

    # Ground truth and the derived simulator card remain outside the Agent API.
    target = str(sample["ground_truth"]["parent_asin"])
    intent_card, behavior = materialize_hidden_fields(sample, products)
    effective_sample = {**sample, "intent_card": intent_card, "behavior": behavior}

    display_session_id = f"demo_{sample_id}"
    agent.reset(display_session_id, sample["user_profile"])

    disclosed: set[str] = set()
    boundary_used = False
    override_applied = False
    category = coarse_category(categories.get(target, []))
    user_message = initial_message(effective_sample, category, disclosed)

    override = behavior.get("override") or {}
    old_value = str(override.get("old_value", "")).strip()
    new_value = str(override.get("new_value", "")).strip()
    active: list[str] = [f"category: {category}"]
    superseded: list[str] = []
    no_preference: list[str] = []
    _append_unique(active, old_value)
    for value in sorted(disclosed):
        _append_unique(active, value)

    turns: list[dict[str, Any]] = []
    hit_turn: int | None = None
    best_rank: int | None = None

    for turn in range(1, MAX_TURNS + 1):
        try:
            response = agent.respond(display_session_id, user_message, turn, TOP_K)
        except Exception as exc:  # mirror official invalid-response handling, but record the cause
            response = {"message": "", "ask_attribute": None, "recommendations": []}
            response_error = f"{type(exc).__name__}: {exc}"
        else:
            response_error = None

        if not isinstance(response, dict) or not isinstance(response.get("message"), str):
            response = {"message": "", "ask_attribute": None, "recommendations": []}
            response_error = response_error or "invalid response payload"

        ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
        current_rank = ranked.index(target) + 1 if override_applied and target in ranked else None
        if current_rank is not None:
            hit_turn = turn
            best_rank = current_rank

        turns.append(
            {
                "turn": turn,
                "user_message": user_message,
                "agent_message": response.get("message", ""),
                "ask_attribute": response.get("ask_attribute"),
                "ranked_parent_asins": ranked,
                "top_three_parent_asins": ranked[:3],
                "active_visible_constraints": list(active),
                "superseded_visible_preferences": list(superseded),
                "no_preference_attributes": list(no_preference),
                "public_target_hit": current_rank is not None,
                "public_target_rank": current_rank,
                "response_error": response_error,
            }
        )

        if current_rank is not None or turn == MAX_TURNS:
            break

        if turn + 1 == int(override.get("turn", 3)):
            override_applied = True
            user_message = str(
                override.get(
                    "message",
                    "Actually, please ignore my earlier preference.",
                )
            )
            if old_value in active:
                active.remove(old_value)
            _append_unique(superseded, old_value)
            _append_unique(active, new_value)
            if new_value:
                disclosed.add(new_value)
        else:
            before = set(disclosed)
            user_message, boundary_used = customer_reply(
                effective_sample,
                response.get("ask_attribute"),
                disclosed,
                boundary_used,
            )
            for value in sorted(disclosed - before):
                _append_unique(active, value)
            if user_message.startswith("I don't have a preference for "):
                _append_unique(no_preference, response.get("ask_attribute"))

    expected_turn = recorded.get("first_hit_turn")
    expected_rank = recorded.get("best_rank")
    if any(item["response_error"] for item in turns):
        raise SystemExit("demo replay encountered an Agent response error")
    if (hit_turn, best_rank) != (expected_turn, expected_rank):
        raise SystemExit(
            "replay disagrees with the frozen public result: "
            f"replay={(hit_turn, best_rank)}, recorded={(expected_turn, expected_rank)}"
        )

    return {
        "title": "ShopSIFT demonstrated public session",
        "source": "released 200-session public development set",
        "sample_id": sample_id,
        "scenario_type": sample["scenario_type"],
        "agent_configuration": "frozen E5 guarded deterministic hybrid",
        "final_or_private_data_used": False,
        "annotation_notice": (
            "Conversation-state annotation reconstructed from visible user messages; "
            "not an additional Agent input or internal debug export."
        ),
        "label_notice": (
            "The released public target was checked only after each Agent response "
            "and was never passed to reset() or respond()."
        ),
        "turns": turns,
        "outcome": {
            "hit": hit_turn is not None,
            "first_hit_turn": hit_turn,
            "best_rank": best_rank,
            "matched_recorded_public_result": True,
        },
    }


def _md(text: object) -> str:
    return str(text).replace("|", "\\|").replace("\n", "<br>")


def write_markdown(trace: dict[str, Any], path: Path) -> None:
    lines = [
        "# ShopSIFT demonstrated public session",
        "",
        f"- Source: {trace['source']}",
        f"- Public sample: `{trace['sample_id']}`",
        f"- Scenario: {trace['scenario_type'].replace('_', ' ').title()}",
        f"- Agent configuration: {trace['agent_configuration']}",
        "- Final/private evaluation data used: No",
        f"- Label handling: {trace['label_notice']}",
        f"- State annotation: {trace['annotation_notice']}",
        "",
        "| Turn | Visible user message | Agent message | `ask_attribute` | Top three ranked ASINs | Visible state transition |",
        "|---:|---|---|---|---|---|",
    ]
    for item in trace["turns"]:
        active = ", ".join(item["active_visible_constraints"]) or "—"
        superseded = ", ".join(item["superseded_visible_preferences"]) or "—"
        state = f"Active: {active}; superseded: {superseded}"
        top_three = ", ".join(item["top_three_parent_asins"]) or "—"
        ask = item["ask_attribute"] if item["ask_attribute"] is not None else "null"
        lines.append(
            "| {turn} | {user} | {agent} | `{ask}` | {top} | {state} |".format(
                turn=item["turn"],
                user=_md(item["user_message"]),
                agent=_md(item["agent_message"]),
                ask=_md(ask),
                top=_md(top_three),
                state=_md(state),
            )
        )
    outcome = trace["outcome"]
    lines.extend(
        [
            "",
            f"- Outcome: public target first appeared on turn `{outcome['first_hit_turn']}` at rank `{outcome['best_rank']}`.",
            "- Validation: replayed `first_hit_turn` and `best_rank` matched the recorded public-session summary.",
            "- Recommendation validation: ranking was normalized to catalog-valid, unique ASINs in scoring order.",
            "",
            "> This demonstration is presentation evidence only. Official metrics come from the unmodified full public evaluator.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _wrapped(label: str, value: str, width: int = 72) -> list[str]:
    wrapped = textwrap.wrap(value, width=width) or ["—"]
    return [label, *wrapped]


def print_screen(trace: dict[str, Any], screen: int) -> None:
    turns = trace["turns"]
    if screen < 1 or screen > len(turns):
        raise SystemExit(f"screen must be between 1 and {len(turns)}")
    item = turns[screen - 1]
    total = len(turns)
    ask = item["ask_attribute"] if item["ask_attribute"] is not None else "null"

    print("=" * 78)
    print(f"SHOPSIFT | RELEASED PUBLIC DEMO | TURN {item['turn']}/{total}")
    print("=" * 78)
    print(*_wrapped("USER", item["user_message"]), sep="\n")
    print()
    print(*_wrapped("AGENT", item["agent_message"]), sep="\n")
    print()
    print("STRUCTURED OUTPUT")
    print(f"ask_attribute: {ask}")
    for rank, asin in enumerate(item["top_three_parent_asins"], start=1):
        print(f"{rank}. {asin}")
    if not item["top_three_parent_asins"]:
        print("—")
    print()
    print("VISIBLE STATE ANNOTATION")
    print("active: " + (", ".join(item["active_visible_constraints"]) or "—"))
    print("superseded: " + (", ".join(item["superseded_visible_preferences"]) or "—"))
    print()
    if item["public_target_hit"]:
        print(
            "PUBLIC TARGET HIT | "
            f"TURN {item['turn']} | RANK {item['public_target_rank']}"
        )
    else:
        print("OUTCOME | no valid public-target hit on this turn")
    print("=" * 78)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay one released public session for presentation evidence"
    )
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument(
        "--results",
        default=str(
            Path.home()
            / "shopsift_submission_evidence"
            / "e5_final_public_run1.json"
        ),
        help="full public-evaluator result used to verify turn and rank",
    )
    parser.add_argument("--sample-id", default="public_0002")
    parser.add_argument(
        "--output-json",
        default=str(
            Path.home()
            / "shopsift_submission_evidence"
            / "e5_demo_public_session.json"
        ),
    )
    parser.add_argument("--output-markdown", default="docs/demo_session.md")
    parser.add_argument(
        "--input-json",
        help="render a previously saved trace without constructing the Agent",
    )
    parser.add_argument("--show-turn", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input_json:
        with Path(args.input_json).open(encoding="utf-8") as handle:
            trace = json.load(handle)
    else:
        trace = replay(
            Path(args.catalog),
            Path(args.dataset),
            Path(args.results),
            args.sample_id,
        )
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
        write_markdown(trace, Path(args.output_markdown))

    if args.show_turn is not None:
        print_screen(trace, args.show_turn)
    else:
        for index in range(1, len(trace["turns"]) + 1):
            print_screen(trace, index)


if __name__ == "__main__":
    main()
