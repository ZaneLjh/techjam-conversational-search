# Capture one complete public multi-turn session

This procedure creates presentation evidence from one **released public** session without modifying the official evaluator or fabricating terminal output.

The recommended example is `public_0002`, an Intent Override session. Its deterministic visible sequence begins with a belt request and replaces the earlier “Buckle closure” preference with “leather” on turn 3. Use it only if the frozen E5 replayed hit turn and rank match the recorded public-session summary.

## 1. Produce the frozen public result

From the clean E5 submission branch:

```bash
EVIDENCE_DIR="$HOME/shopsift_submission_evidence"
mkdir -p "$EVIDENCE_DIR"

python -m evaluator.local_evaluator \
  --catalog data/catalog.jsonl \
  --dataset data/public_set.jsonl \
  --output "$EVIDENCE_DIR/e5_final_public_run1.json"

python -m evaluator.local_evaluator \
  --catalog data/catalog.jsonl \
  --dataset data/public_set.jsonl \
  --output "$EVIDENCE_DIR/e5_final_public_run2.json"

if ! cmp -s \
  "$EVIDENCE_DIR/e5_final_public_run1.json" \
  "$EVIDENCE_DIR/e5_final_public_run2.json"; then
  echo "STOP: public runs differ"
  exit 1
fi

echo "PASS: public runs are byte-identical"
```

Stop if the two results differ.

## 2. Confirm `public_0002` is suitable

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path.home() / "shopsift_submission_evidence" / "e5_final_public_run1.json"
with path.open(encoding="utf-8") as handle:
    result = json.load(handle)

for session in result["sessions"]:
    if session["sample_id"] == "public_0002":
        print(session)
        break
else:
    raise SystemExit("public_0002 not found")
PY
```

Prefer it when `hit` is `true`, `first_hit_turn` is 3 or 4, and `best_rank` is small. If it is unsuitable, list alternatives:

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path.home() / "shopsift_submission_evidence" / "e5_final_public_run1.json"
with path.open(encoding="utf-8") as handle:
    result = json.load(handle)

for session in result["sessions"]:
    if (
        session["scenario_type"] == "intent_override"
        and session["hit"]
        and session["first_hit_turn"] in (3, 4)
        and session["best_rank"] <= 3
    ):
        print(
            session["sample_id"],
            "turn=", session["first_hit_turn"],
            "rank=", session["best_rank"],
        )
PY
```

Choose one listed public sample. Do not use any final/private session.

## 3. Generate the real transcript

The supplied replay utility imports the official simulator helpers but does not edit the evaluator. Public ground truth is checked only after each Agent response and is never passed to `reset()` or `respond()`.

```bash
python -m tools.replay_public_session \
  --catalog data/catalog.jsonl \
  --dataset data/public_set.jsonl \
  --results "$HOME/shopsift_submission_evidence/e5_final_public_run1.json" \
  --sample-id public_0002 \
  --output-json "$HOME/shopsift_submission_evidence/e5_demo_public_session.json" \
  --output-markdown docs/demo_session.md
```

If you selected another sample, change only `--sample-id`.

The command hard-fails if replayed `first_hit_turn` or `best_rank` differs from the recorded public-session summary. The official result does not contain a full message or ranking trace. When the check succeeds, review:

```bash
sed -n '1,240p' docs/demo_session.md
```

The JSON trace is retained as local evidence. `docs/demo_session.md` is the public transcript to commit.

## 4. Render one screenshot-ready turn at a time

This mode reads the saved trace, so it does not rebuild the E5 index:

```bash
clear
python -m tools.replay_public_session \
  --input-json "$HOME/shopsift_submission_evidence/e5_demo_public_session.json" \
  --show-turn 1
```

Capture the terminal, then repeat with `--show-turn 2`, `3`, and `4` only if the trace contains that turn.

Each frame contains:

1. the visible user message;
2. the Agent's actual customer-facing message;
3. structured `ask_attribute`;
4. the actual first three normalized recommendations;
5. a clearly labelled visible-state annotation; and
6. the public hit turn/rank only after the Agent response.

The visible-state annotation is reconstructed from visible user messages for presentation. It is not an additional Agent input or a claim that the Agent exposes an internal debug field.

## 5. Take the screenshots on Windows

1. Open Windows Terminal or Git Bash.
2. Use a dark theme and increase the font until the content is legible at thumbnail size.
3. Resize the terminal to roughly 78–84 columns and 24–30 visible lines.
4. Run `clear`, then the single-turn render command.
5. Press **Win+Shift+S** and capture only the output card.
6. Save files as `demo_turn_1.png`, `demo_turn_2.png`, and so on.
7. Crop out the shell prompt, username, absolute path, unrelated logs, environment variables, and tokens.
8. Do not edit or retype the messages, ASINs, `ask_attribute`, turn, or rank.

Use three screenshots when the hit occurs on turn 3; use four when it occurs on turn 4.

## 6. Build the horizontal sequence

Create a 16:9 canvas in PowerPoint, Canva, Figma, or another layout tool:

- three screenshots: use `1920 × 1080`;
- four screenshots: use `2560 × 1440` for better readability;
- place screenshots left to right in turn order with equal width and consistent margins;
- for a turn-3 hit, remove the fourth panel and enlarge the three remaining panels to fill the canvas;
- title: **One public session. Intent changes. Ranking adapts.**
- footer: **Released public development session · target used only for post-response scoring · no final/private data**.

Recommended sequence:

1. broad initial request and first recommendations;
2. structured clarification and its `ask_attribute`;
3. user answer or explicit intent override with active/superseded annotation; and
4. final ranked hit with turn and rank, if it occurs on a separate turn.

Export the real composite as `02_public_demo.png`. Use `docs/project_media/02_demo_storyboard_GUIDE.png` only as the layout reference; never upload the guide in place of the real terminal composite.

## 7. Privacy and integrity check

Do not show:

- ground truth before the Agent responds;
- a complete public dataset dump;
- any final/private dataset, session, or label;
- raw usernames or absolute local paths;
- credentials, tokens, or environment-variable values;
- invented terminal output; or
- an internal state claim that the Agent does not actually expose.

Before committing:

```bash
if rg -q 'DEMO_NOT_CAPTURED' docs/demo_session.md; then
  echo "STOP: the real public demonstration has not replaced the template"
else
  echo "PASS: real demo transcript present"
fi
```
