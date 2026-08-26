# TechJam Conversational E-Commerce Search Challenge

Build an AI shopping agent that asks useful follow-up questions and recommends the customer's hidden target product within at most 10 turns.

## What You Receive

- A frozen catalog of 50,000 products from the `Clothing_Shoes_and_Jewelry` category of Amazon Reviews 2023.
- 200 labeled public sessions for local development.
- A weak BM25 starter agent and deterministic local evaluator.
- The Agent API contract and scoring rules.

The organizer keeps 800 additional sessions private for final evaluation.

## Task

For each session, your agent receives an anonymized preference profile and a short customer message. Raw user IDs, review text, timestamps, and purchase history are never disclosed. On every turn the agent may:

- ask a natural clarification question in `message` and identify one requested field in `ask_attribute`;
- return a ranked list of up to 10 catalog `parent_asin` values;
- do both in the same response.

The session ends when the target product appears in the scored Top 10 or after turn 10. Sessions cover Buying, Browsing, Intent Override, and Boundary behavior.

## Download the Catalog

Download `catalog.jsonl.gz` from the GitHub Release attached to this repository, then run:

```bash
gzip -dk catalog.jsonl.gz
mv catalog.jsonl data/catalog.jsonl
```

Verify the downloaded file using the published `SHA256SUMS` file.

## Run the Starter

Python 3.10 or later is required for E2. The agent uses only the Python standard
library, so there is no dependency-install step.

```bash
python3 -m evaluator.local_evaluator
```

Edit `starter/agent.py` to implement your system. Do not edit the evaluator or public labels when reporting your local score.
The command writes per-session results and aggregate metrics to `results.json`.

The included weak BM25 starter scores Hit Rate@10 `0.125`, MRR `0.068034`, and
MTTC `9.81` on the released public set. See `docs/baseline_results.json`.

## Current solution: E2 constraint ledger

This branch implements E2 as an offline, standard-library-only retrieval agent.
It replaces append-only conversation text with a typed constraint ledger that
tracks facet, strength (`MUST`, `SHOULD`, `AVOID`, `NO_PREFERENCE`), polarity,
source turn, status, and supersession history. Deterministic parsing handles
clarifications, explicit boundaries, negations, and intent corrections. Only
active positive constraints enter the BM25 query; active exclusions filter
matching candidates. Continued turns explore deeper unseen results, and an
intent override opens a new candidate epoch.

The catalog, evaluator, FTS5 fields, BM25 weights, and question schedule remain
unchanged from E1. E2 makes no network calls and reports zero model tokens.
It uses no model or API, so estimated model/API cost is `$0`.

| Metric | E1 | E2 |
| --- | ---: | ---: |
| Hit Rate@10 | 0.865000 | 0.985000 |
| MRR | 0.522867 | 0.594141 |
| MTTC | 4.320000 | 3.170000 |
| Efficiency | 0.668000 | 0.783000 |
| TechnicalScore | 0.722960 | 0.827342 |

Reproduce the implementation evidence from the repository root:

```bash
python -m unittest discover -v
python -m evaluator.local_evaluator --output results/e2_structured_constraint_ledger.json
python -m tools.evaluate_variant --disable-candidate-exploration \
  --output results/e2a_ledger_only.json
python -m tools.fold_report \
  --baseline results/e1_stateful_lexical.json \
  --candidate results/e2_structured_constraint_ledger.json \
  --folds 5 --output results/e2_five_fold_report.json
python -m tools.benchmark_agent --output results/e2_benchmark.json
```

See `docs/experiments.md` for the controlled ablation and limitations, and
`docs/e2_review_guide.md` for the review upload manifest.

## Agent Interface

```python
class Agent:
    def reset(self, session_id: str, user_profile: dict) -> None:
        ...

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return {
            "message": "Do you have a material preference?",
            "ask_attribute": "material",
            "recommendations": [
                {"parent_asin": "B000..."},
                {"parent_asin": "B001..."}
            ],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30}
        }
```

`ask_attribute` is one of `category`, `material`, `color`, `size`, `style`, `brand`, `budget`, `feature`, `use_case`, `other`, or `null`. See `docs/agent_api_contract.json`.

## Technical Metrics

- **Hit Rate@10:** fraction of sessions that find the target within 10 turns.
- **MRR:** mean reciprocal rank of the target; a miss contributes zero.
- **MTTC:** mean first-hit turn; a miss is assigned turn 11.
- **Reported token usage:** prompt and completion tokens returned by the team's model client.

```text
TechnicalScore = 0.50 × HitRate@10 + 0.30 × MRR + 0.20 × Efficiency
Efficiency = clip((11 - MTTC) / 10, 0, 1)
```

Only exact `parent_asin` equality produces a hit. Core metrics are also reported by scenario.

## Model Choice and Cost

Teams may use any legally accessible LLM API or local model. Teams manage their own credentials and must never commit API keys. Model choice, estimated cost, token usage, and latency must be disclosed. Token usage is a feasibility metric, not part of the core technical score. The organizer may reimburse model costs through prizes instead of issuing API keys.

## Files

```text
data/public_set.jsonl             200 labeled development sessions
docs/competition_specification.md participant rules and evaluation protocol
docs/agent_api_contract.json      machine-readable Agent contract
docs/evaluation_config.json       scoring configuration
docs/baseline_results.json        reproducible weak-starter reference score
starter/agent.py                  E2 retrieval agent and session state
starter/constraints.py            typed constraint parser and ledger
evaluator/local_evaluator.py      public-set simulator and scorer
tools/evaluate_variant.py         declared E2 ablation runner
tools/fold_report.py              scenario-stratified stability report
tools/benchmark_agent.py          latency and resource benchmark
```

## Judging and Submission Policy

- Participant submission requirements: `docs/submission_rules.md`
- Participant release checklist: `docs/participant_release_checklist.md`
- Organizer-only final judging controls: `organizer/JUDGING_RUNBOOK.md`
- Organizer private release checklist: `organizer/private_release_checklist.md`
- Judging day operations SOP: `organizer/JUDGING_DAY_SOP.md`

## Data Source

The catalog and sessions are derived from Amazon Reviews 2023 by McAuley Lab, UCSD. See `DATA_ATTRIBUTION.md` before using or redistributing the data.
Sessions are sampled deterministically from the official Clothing 5-core leave-last-out split and joined to the frozen catalog.
