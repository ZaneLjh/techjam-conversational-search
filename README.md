# TechJam Conversational E-Commerce Search Challenge

Build an AI shopping agent that asks useful follow-up questions and recommends the customer's hidden target product within at most 10 turns.

## What You Receive

- A frozen catalog of 50,000 products from the `Clothing_Shoes_and_Jewelry` category of Amazon Reviews 2023.
- 200 labeled public sessions for local development.
- A weak BM25 starter agent and deterministic local evaluator.
- The Agent API contract and scoring rules.

The organizer keeps 800 additional sessions unreleased until the Devpost submission deadline. After the deadline, the final evaluation package will be released and teams will run the unmodified official evaluator in their own environments using their frozen submitted commit.

See [`docs/final_evaluation_faq.md`](docs/final_evaluation_faq.md) for the final evaluation, network, credentials, hardware, data, and scoring policy.

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

Python 3.10 or later is required for E4. The agent uses only the Python standard
library, so there is no dependency-install step.

```bash
python3 -m evaluator.local_evaluator
```

Edit `starter/agent.py` to implement your system. Do not edit the evaluator or public labels when reporting your local score.
The command writes per-session results and aggregate metrics to `results.json`.

The included weak BM25 starter scores Hit Rate@10 `0.125`, MRR `0.068034`, and
MTTC `9.81` on the released public set. See `docs/baseline_results.json`.

## Current solution: E4 multi-route retrieval and deterministic reranking

E4 keeps E3's typed constraint ledger and adaptive-question policy fixed while
changing only the displayed recommendation ranking. An independent E3 shadow
candidate stream remains the sole input to the question policy, so an E4 rank
change cannot silently change the next simulated customer reply.

For turns with non-category positive evidence, E4 combines four catalog-only
route families: the accumulated ledger query, current-turn evidence, a
category-column FTS query, and exact normalized feature/detail scalar lookups.
Up to four recent facet constraints receive exact and category-plus-exact
routes.
Weighted reciprocal-rank fusion scores the complete bounded route union, then a
stable tie-break uses legacy rank, best route rank, and `parent_asin`. The final
union is capped at 100 before returning at most 10 recommendations.

Category-only turns retain the exact E3 order. Any active `AVOID` constraint
uses E3's exclusion-safe streaming fallback and is never relaxed. The default
fusion can retain partial positive matches after stronger exact evidence; the
declared `no_soft_relaxation` ablation keeps only candidates matching every
bounded routed `MUST` constraint when such candidates exist. Route decisions,
weights, ranks, exact coverage, and relaxation counts are available through
`Agent.evidence_trace()`.

The catalog, evaluator, public labels, scoring configuration, Agent contract,
constraint parser, and adaptive-question implementation remain unchanged. E4
uses only the Python standard library, makes no network or model calls, reports
zero model tokens, and has estimated model/API cost of `$0`.

| Metric | E3 | E4 | Delta |
| --- | ---: | ---: | ---: |
| Hit Rate@10 | 0.985000 | 1.000000 | +0.015000 |
| MRR | 0.596099 | 0.798198 | +0.202099 |
| MTTC | 3.065000 | 2.410000 | -0.655000 |
| Efficiency | 0.793500 | 0.859000 | +0.065500 |
| TechnicalScore | 0.830030 | 0.911259 | +0.081229 |

Reproduce the implementation evidence from the repository root:

```bash
mkdir -p results
python -m compileall -q starter tests tools
python -m unittest discover -v
python -m evaluator.local_evaluator \
  --output results/e4_multi_route_reranking.json
python -m evaluator.local_evaluator \
  --output results/e4_multi_route_reranking_repeat.json
cmp results/e4_multi_route_reranking.json \
  results/e4_multi_route_reranking_repeat.json
python -m tools.evaluate_variant --disable-multi-route-ranking \
  --output results/e4a_e3_ranking_ablation.json
python -m tools.e4_ablation_suite \
  --output results/e4_ablation_suite.json
python -m tools.paired_report \
  --baseline results/e4a_e3_ranking_ablation.json \
  --candidate results/e4_multi_route_reranking.json \
  --changes-only --output results/e4_vs_e3_paired_report.json
python -m tools.fold_report \
  --baseline results/e4a_e3_ranking_ablation.json \
  --candidate results/e4_multi_route_reranking.json \
  --folds 5 --output results/e4_vs_e3_five_fold_report.json
python -m tools.benchmark_agent --output results/e4_benchmark.json
```

The 200 released sessions are development evidence, not an estimate of the 800
hidden sessions. The five-fold report is only a deterministic stability slice
of that same development set; it is not cross-validation or held-out
validation. See `docs/experiments.md` for the fusion formula, controlled
ablations, scenario results, performance, and limitations.

## E4.1 provisional compliance repair

E4.1 adds a three-state compatibility check (exact, unknown, mismatch), keeps
price evidence soft, advances non-empty partial strict pages, and exposes a
bounded candidate pool for funnel diagnostics. It also implements the proposed
strict-front/recall-backfill cascade and confidence-gated auxiliary fusion.
The frozen E3 ledger and question-policy path remains unchanged.

On the released 200-session development set, the complete E4.1 configuration
preserves Hit Rate and improves MRR slightly, but does not clear its promotion
threshold:

| Metric | Frozen E4 | E4.1 candidate | Delta |
| --- | ---: | ---: | ---: |
| Hit Rate@10 | 1.000000 | 1.000000 | 0.000000 |
| MRR | 0.798198 | 0.807163 | +0.008965 |
| MTTC | 2.410000 | 2.420000 | +0.010000 |
| TechnicalScore | 0.911259 | 0.913749 | +0.002490 |

The required score delta is `+0.005`; only three of five public consistency
slices are positive, and those slices are not product-held-out validation.
Therefore the E4.1 promotion decision is **reject**. `Agent()` still defaults to
the byte-compatible frozen E4 configuration. E4.1 is enabled only by an
explicit experiment configuration. The strict-only configuration is retained
as a diagnostic, not mislabeled as a promotable UNKNOWN-neutral policy.

```bash
python -m compileall -q starter tests tools
python -m unittest discover -v
python -m tools.evaluate_variant --e4-1-candidate \
  --output results/e4_1_strict_front.json
python -m tools.evaluate_variant --e4-1-candidate \
  --output results/e4_1_strict_front_repeat.json
cmp results/e4_1_strict_front.json \
  results/e4_1_strict_front_repeat.json
python -m tools.evaluate_variant --e4-1-strict-only-diagnostic \
  --output results/e4_1_strict_only_diagnostic.json
python -m tools.evaluate_variant --disable-e4-1 \
  --output results/e4_1a_e4_fallback.json
cmp results/e4_multi_route_reranking.json \
  results/e4_1a_e4_fallback.json
python -m tools.e4_1_compliance_suite \
  --output results/e4_1_compliance_suite.json
python -m tools.e4_1_ablation_suite \
  --output results/e4_1_ablation_suite.json
python -m tools.e4_1_funnel_report \
  --output results/e4_1_candidate_funnel.json
python -m tools.fold_report \
  --baseline results/e4_multi_route_reranking.json \
  --candidate results/e4_1_strict_front.json --folds 5 \
  --output results/e4_1_vs_e4_public_folds.json
python -m tools.promotion_gate \
  --baseline results/e4_multi_route_reranking.json \
  --candidate results/e4_1_strict_front.json \
  --candidate-repeat results/e4_1_strict_front_repeat.json \
  --fold-report results/e4_1_vs_e4_public_folds.json \
  --compliance-report results/e4_1_compliance_suite.json \
  --fallback-result results/e4_1a_e4_fallback.json \
  --output results/e4_1_promotion_gate.json
```

Do not call E4.1 deployed or promoted. A future revision must first clear the
same public gate and then be positive on at least four of five true
target-product-disjoint folds with no Hit Rate loss. Full E4 remains the
kill-switch fallback.

## E4.5 provisional intent-card projection

E4.5 is a deterministic, catalog-derived experiment layered on the E4.1
source tree. Its behavioral predecessor is still frozen full E4: the
no-argument `Agent()` and `--disable-e4-5` paths do not enable E4.1 or E4.5.
The E4.5 candidate activates only when explicitly configured with a sidecar
and manifest built from the exact catalog path and bytes used at evaluation.
An absent, corrupt, mismatched, oversized, or noncanonical artifact fails
closed to the complete frozen-E4 turn.

The sidecar contains exact intent-card projections from public catalog fields.
It supports a bounded category-plus-clue posterior, uniqueness-gated display
reranking, repeated `other` questions until the exact no-additional reply, and
question selection over exact rendered-reply partitions. The rollout score is
a TechnicalScore-shaped expected-utility proxy over deterministic posterior
order; it is not learned and is not an exact next-turn target-utility estimate.
Learned rankers, LambdaMART, or Ettin model work remain outside E4.5.

On the released 200-session development set, the fresh frozen-source result is:

| Metric | Frozen E4 | E4.5 candidate | Delta |
| --- | ---: | ---: | ---: |
| Hit Rate@10 | 1.000000 | 1.000000 | 0.000000 |
| MRR | 0.798198 | 0.813317 | +0.015119 |
| MTTC | 2.410000 | 2.390000 | -0.020000 |
| TechnicalScore | 0.911259 | 0.916195 | +0.004936 |

The required score delta is `+0.005`. E4.5 falls short by `0.000064`, so its
promotion decision is **reject** before held-out validation. Public five-slice
TechnicalScore deltas are `0.000000`, `+0.007929`, `0.000000`, `+0.003750`,
and `+0.013000`, so only 3/5 are strictly positive. These are consistency
diagnostics, never cross-validation or deployment evidence. Promotion would
additionally require non-decreasing Hit Rate, no hit-to-miss transition,
deterministic/fallback/compliance checks, and positive delta on at least four of
five true target-product-disjoint held-out folds.

Build the generated artifact from the same plain catalog used by the evaluator,
then run the candidate explicitly:

```bash
mkdir -p results
python -m tools.e4_5_projection_suite \
  --catalog data/catalog.jsonl \
  --sidecar results/e4_5_intent_projection.jsonl.gz \
  --manifest results/e4_5_projection_manifest.json
python -m tools.evaluate_variant --e4-5-candidate \
  --catalog data/catalog.jsonl \
  --projection-sidecar results/e4_5_intent_projection.jsonl.gz \
  --projection-manifest results/e4_5_projection_manifest.json \
  --output results/e4_5_candidate.json
python -m tools.evaluate_variant --e4-5-candidate \
  --catalog data/catalog.jsonl \
  --projection-sidecar results/e4_5_intent_projection.jsonl.gz \
  --projection-manifest results/e4_5_projection_manifest.json \
  --output results/e4_5_candidate_repeat.json
cmp results/e4_5_candidate.json results/e4_5_candidate_repeat.json
python -m tools.evaluate_variant --disable-e4-5 \
  --output results/e4_5a_e4_fallback.json
python -m tools.e4_5_ablation_suite \
  --catalog data/catalog.jsonl \
  --sidecar results/e4_5_intent_projection.jsonl.gz \
  --manifest results/e4_5_projection_manifest.json \
  --output results/e4_5_ablation_suite.json
python -m tools.e4_5_funnel_report \
  --catalog data/catalog.jsonl \
  --sidecar results/e4_5_intent_projection.jsonl.gz \
  --manifest results/e4_5_projection_manifest.json \
  --output results/e4_5_candidate_funnel.json
```

The sidecar itself is a reproducible build product and is not intended for the
source patch. Runtime validation binds its exact bytes and canonical JSONL
content to the catalog and manifest, enforces explicit size/row limits, and
rejects extra gzip members. Manifest `compression_level: 9` records build
provenance; it is not a cryptographic proof of the DEFLATE encoder setting.

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
docs/final_evaluation_faq.md      final evaluation and judging clarifications
docs/agent_api_contract.json      machine-readable Agent contract
docs/evaluation_config.json       scoring configuration
docs/baseline_results.json        reproducible weak-starter reference score
starter/agent.py                  E4 retrieval agent and session state
starter/constraints.py            typed constraint parser and ledger
starter/question_policy.py        candidate-aware question selection and traces
starter/retrieval.py              bounded routes, exact index, fusion, and reranking
evaluator/local_evaluator.py      public-set simulator and scorer
tools/evaluate_variant.py         single-switch E2/E4 variant runner
tools/e4_ablation_suite.py        fixed E4 component-ablation matrix
tools/e4_1_ablation_suite.py      E4.1 strict/gate/route interaction matrix
tools/e4_1_compliance_suite.py    deterministic compatibility probes
tools/e4_1_funnel_report.py       public-only candidate/oracle diagnostics
starter/projection.py             E4.5 exact projection and bounded rollout
tools/e4_5_projection_suite.py    deterministic sidecar build/parity audit
tools/e4_5_ablation_suite.py      fixed E4.5 projection/rollout matrix
tools/e4_5_funnel_report.py       E4.5 activation and posterior diagnostics
tools/promotion_gate.py           fail-closed public/held-out promotion gate
tools/paired_report.py             paired per-session utility comparison
tools/fold_report.py              scenario-stratified stability report
tools/benchmark_agent.py          latency and resource benchmark
```

## Judging and Submission Policy

- Participant submission requirements: `docs/submission_rules.md`
- Participant release checklist: `docs/participant_release_checklist.md`
- Organizer-only final judging controls: `organizer/JUDGING_RUNBOOK.md`
- Organizer private release checklist: `organizer/private_release_checklist.md`
- Judging day operations SOP: `organizer/JUDGING_DAY_SOP.md`
- Final evaluation FAQ: `docs/final_evaluation_faq.md`

## Data Source

The catalog and sessions are derived from Amazon Reviews 2023 by McAuley Lab, UCSD. See `DATA_ATTRIBUTION.md` before using or redistributing the data.
Sessions are sampled deterministically from the official Clothing 5-core leave-last-out split and joined to the frozen catalog.
