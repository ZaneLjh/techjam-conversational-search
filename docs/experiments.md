# Experiment Log

## Rules for every reported run

- Use the frozen organizer catalog and public set without modification.
- Run the unmodified `evaluator/local_evaluator.py`.
- Keep the production Agent independent of `data/public_set.jsonl` and its labels.
- Record the Git commit, configuration, overall metrics, scenario metrics, latency,
  dependencies, and network/model usage.
- Report synthetic stress-set results separately from official public-set results.

## E0 - Organizer weak BM25 baseline

Source tag: `participant-kit` (`2a6cc8e776da66ce69b1cbd237838fbc43f32587`)

| Metric | Result |
| --- | ---: |
| Hit Rate@10 | 0.125000 |
| MRR | 0.068034 |
| MTTC | 9.810000 |
| Efficiency | 0.119000 |
| TechnicalScore | 0.106710 |

## E1 - Stateful lexical retrieval and structured clarification

Hypothesis: accumulating customer constraints, asking one simulator-recognized
attribute per turn, and continuing to recommend on every turn will improve target
coverage and conversion speed without introducing model cost or network risk.

Controlled changes from E0:

1. Retain customer messages within each session.
2. Clear obsolete preference text when an intent-override message arrives while
   preserving the initial category sentence.
3. Query the unchanged SQLite FTS5/BM25 index with accumulated terms.
4. Ask a deterministic attribute sequence: material, feature, color, style, size,
   use case, brand, budget, then other.
5. Return up to 10 recommendations on every turn.

Unchanged:

- catalog and public labels;
- evaluator and scoring formula;
- FTS5 index fields and BM25 field weights;
- standard-library-only, offline execution;
- zero reported model tokens.

Run:

```bash
mkdir -p results
python -m unittest discover -v
python -m evaluator.local_evaluator --output results/e1_stateful_lexical.json
```

Acceptance gates:

- all tests pass;
- TechnicalScore exceeds 0.65 on the released public set;
- no scenario crashes or invalid responses;
- `git diff participant-kit -- evaluator data/public_set.jsonl` is empty;
- a second run reproduces identical aggregate and per-scenario metrics.

Observed public-set result:

| Metric | E0 baseline | E1 |
| --- | ---: | ---: |
| Hit Rate@10 | 0.125000 | 0.865000 |
| MRR | 0.068034 | 0.522867 |
| MTTC | 9.810000 | 4.320000 |
| Efficiency | 0.119000 | 0.668000 |
| TechnicalScore | 0.106710 | 0.722960 |

| Scenario | Hit Rate@10 | MRR | MTTC |
| --- | ---: | ---: | ---: |
| Buying | 0.900000 | 0.535481 | 3.350000 |
| Browsing | 0.912500 | 0.526429 | 3.850000 |
| Intent Override | 0.666667 | 0.429021 | 7.500000 |
| Boundary | 0.800000 | 0.675000 | 6.300000 |

The repeated run was byte-identical. E1 passes its acceptance gates. Intent
Override is the weakest scenario and is the primary target for E2.

## E2 - Typed constraint ledger and candidate exploration

Hypothesis: replacing append-only query text with auditable, facet-local state
will prevent stale or declined preferences from polluting retrieval. After a
miss, exposing deeper unseen candidates should improve coverage without changing
the underlying ranker or using private labels.

### E2a - Ledger-only ablation

Controlled changes from E1:

1. Parse customer-visible text into typed constraints with facet, polarity,
   strength, confidence, source turn, evidence span, and status.
2. Preserve history while applying exact retractions, same-facet supersession,
   explicit negation, and no-preference boundaries.
3. Build the canonical BM25 query from active positive `MUST` and `SHOULD`
   constraints only.
4. Filter candidates that contain every normalized token of an active `AVOID`
   value.

Candidate exploration is disabled in this ablation, so the same ranked page is
returned while the active query is unchanged.

```bash
python -m tools.evaluate_variant --disable-candidate-exploration \
  --output results/e2a_ledger_only.json
```

| Metric | E1 | E2a ledger only | Delta |
| --- | ---: | ---: | ---: |
| Hit Rate@10 | 0.865000 | 0.860000 | -0.005000 |
| MRR | 0.522867 | 0.572677 | +0.049810 |
| MTTC | 4.320000 | 4.190000 | -0.130000 |
| Efficiency | 0.668000 | 0.681000 | +0.013000 |
| TechnicalScore | 0.722960 | 0.738003 | +0.015043 |

The small Hit Rate change reflects removal of no-preference/scaffolding tokens
that sometimes produced an accidental match in E1. The higher MRR and score are
attributable to cleaner state rather than a changed index.

### E2b - Epoch-aware unseen-candidate exploration

Full E2 additionally pages past identifiers already shown in the current intent
epoch. An explicit override clears the shown set because pre-override results are
not scoreable. This adds `+0.089339` TechnicalScore over E2a and accounts for most
of the total E2 gain.

Unchanged from E1:

- frozen catalog, public labels, evaluator, and scoring formula;
- SQLite FTS5 fields and BM25 field weights;
- deterministic question schedule;
- offline, standard-library-only execution;
- no neural model, LLM, reinforcement learning, fitting, or training;
- zero network calls and zero reported model tokens.

Run and reproduce:

```bash
python -m unittest discover -v
python -m evaluator.local_evaluator \
  --output results/e2_structured_constraint_ledger.json
python -m evaluator.local_evaluator \
  --output results/e2_structured_constraint_ledger_repeat.json
sha256sum results/e2_structured_constraint_ledger*.json
python -m tools.fold_report \
  --baseline results/e1_stateful_lexical.json \
  --candidate results/e2_structured_constraint_ledger.json \
  --folds 5 --output results/e2_five_fold_report.json
python -m tools.benchmark_agent --output results/e2_benchmark.json
```

Verification:

- `47/47` unit tests pass.
- Two evaluator runs are byte-identical with SHA-256
  `af20738d40241258f0ae84a4ccc022d8c100ec52963004354232e200e03a78f8`.
- All five deterministic scenario-stratified folds improve TechnicalScore;
  fold deltas range from `+0.073480` to `+0.141378`.
- Production source has no dependency on `public_set`, ground truth, scenario
  labels, network services, or model APIs.

Observed public-set result:

| Metric | E1 | E2 | Delta |
| --- | ---: | ---: | ---: |
| Hit Rate@10 | 0.865000 | 0.985000 | +0.120000 |
| MRR | 0.522867 | 0.594141 | +0.071274 |
| MTTC | 4.320000 | 3.170000 | -1.150000 |
| Efficiency | 0.668000 | 0.783000 | +0.115000 |
| TechnicalScore | 0.722960 | 0.827342 | +0.104382 |

| Scenario | Hit Rate@10 | MRR | MTTC |
| --- | ---: | ---: | ---: |
| Buying | 0.987500 | 0.594182 | 2.687500 |
| Browsing | 1.000000 | 0.571905 | 2.837500 |
| Intent Override | 0.966667 | 0.663743 | 4.866667 |
| Boundary | 0.900000 | 0.562897 | 4.600000 |

Saved benchmark (`631` responses): startup `1.909806 s`, mean response latency
`19.325742 ms`, p95 `49.475683 ms`, p99 `66.936261 ms`, maximum
`86.831122 ms`, and peak RSS `312.219 MB`. Timing is environment-dependent.
E2 uses no model or API, makes no network calls, reports zero tokens, and has an
estimated model/API cost of `$0`.

### Limitations and interpretation

- These are development results on 200 released sessions, not an estimate of
  the hidden 800-session score.
- Boundary MRR decreases from `0.675000` in E1 to `0.562897` in E2 despite
  improved Boundary Hit Rate and MTTC. The Boundary sample has only 10 sessions.
- Parsing is deterministic and intentionally conservative. Unseen linguistic
  paraphrases can still be missed or assigned to an imperfect facet.
- Exclusion filtering uses normalized token-set containment with a local lexical
  negation guard, not semantic entailment; it can under-filter synonyms or
  over-filter ambiguous product copy.
- The anonymized profile is retained for later experiments but is not used in E2
  ranking.

A demonstrated three-turn correction session is recorded in
`docs/e2_demo_session.md`. Team names and contribution ownership are not present
in the supplied materials; the submitter must complete the contribution template
in that document before a final competition submission.

## E3 - Missingness-aware adaptive questions

Base commit: `5f31859fd4f9a5812750c620375c49ab2f4fef74` (E2).

Hypothesis: choosing the next clarification from the current candidate
uncertainty will expose useful target evidence sooner than a fixed question
schedule, lowering MTTC while preserving E2's Hit Rate and MRR.

Controlled changes from E2:

1. Extract shallow, deterministic facet evidence from public catalog metadata
   into a separate SQLite side table during agent startup.
2. After the displayed Top 10, collect up to 80 additional unseen BM25
   candidates without changing the displayed recommendation order.
3. For each specific question attribute, build an answer distribution that
   includes the explicit `<missing>` bucket.
4. Score answerability using observed coverage, normalized entropy, facet
   reliability, active-constraint downweighting, and expected rank gain under a
   TechnicalScore-shaped utility.
5. Make previously asked attributes ineligible, use `other` only below the
   specific-facet threshold, and stop asking at turn 10.
6. Record every selection reason and all per-facet statistics in the session's
   JSON-serializable evidence trace.

The first turn deliberately retains E2's `material` question as a conservative
guardrail. Duplicate prevention means the policy never repeats an attribute it
has already asked; a facet already present in the ledger remains eligible at a
0.60 weight because an additional same-facet detail can still be useful.

### Selection policy

Each candidate contributes its first normalized value for a facet or
`<missing>`. Let the candidate at rank `r` have prior weight `1 / sqrt(r)`. For a
non-missing answer, its hypothetical filtered rank is its position within the
same answer group. The per-candidate utility proxy is:

```text
U(rank, hit_turn) = 0                                      if rank > 10
U(rank, hit_turn) = 0.50 + 0.30 / rank
                    + 0.20 * clip((11 - hit_turn) / 10)    otherwise
```

Expected gain is the prior-weighted positive change from the old rank to the
hypothetical filtered rank on the next turn. The deterministic selection score
is:

```text
reliability * active_factor * observed_rate^1.5
            * (expected_gain + 0.025 * normalized_entropy)
```

The selected specific facet must score at least `0.006`. With fewer than eight
candidates, the policy uses the remaining deterministic fallback order. If no
specific facet clears the threshold, it uses `other` once. Tie-breaking follows
the declared attribute order and never depends on public labels.

Unchanged from E2:

- frozen catalog, public labels, evaluator, and scoring formula;
- typed constraint parsing, exclusion filtering, intent epochs, and unseen
  candidate exploration;
- SQLite FTS5 fields, tokenization, BM25 weights, query, and recommendation
  ordering;
- offline, standard-library-only execution;
- no fitting, training, neural model, LLM, network call, or hidden-label access;
- zero reported model tokens and estimated model/API cost of `$0`.

### Verification and results

Run:

```bash
python -m compileall -q starter tests tools
python -m unittest discover -v
python -m evaluator.local_evaluator \
  --output results/e3_adaptive_questions.json
python -m evaluator.local_evaluator \
  --output results/e3_adaptive_questions_repeat.json
sha256sum results/e3_adaptive_questions*.json
cmp results/e3_adaptive_questions.json \
  results/e3_adaptive_questions_repeat.json
python -m tools.fold_report \
  --baseline results/e2_structured_constraint_ledger.json \
  --candidate results/e3_adaptive_questions.json \
  --folds 5 --output results/e3_vs_e2_five_fold_report.json
python -m tools.benchmark_agent --output results/e3_benchmark.json
```

Verification evidence:

- `58/58` tests pass: all 47 E2 tests plus 11 E3 policy/integration tests.
- The two evaluator outputs are byte-identical with SHA-256
  `d3dd509754dd3152b761a42b3f494106fca14fe2a43baa0dbc896c50223187bc`.
- All 200 turn-one recommendation lists are identical to E2; both snapshots
  have SHA-256
  `20aa4ab3619a33d583541ebbcac030d307cda356ab7d4bad28fe28a53b3f6be3`
  under the documented sorted-JSON snapshot serialization.
- Production E3 source has no dependency on `public_set`, ground truth,
  scenario labels, evaluator modules, model APIs, or network services.

Observed public-set result:

| Metric | E2 | E3 | Delta |
| --- | ---: | ---: | ---: |
| Hit Rate@10 | 0.985000 | 0.985000 | 0.000000 |
| MRR | 0.594141 | 0.596099 | +0.001958 |
| MTTC | 3.170000 | 3.065000 | -0.105000 |
| Efficiency | 0.783000 | 0.793500 | +0.010500 |
| TechnicalScore | 0.827342 | 0.830030 | +0.002688 |

| Scenario | E2 HR | E3 HR | E2 MRR | E3 MRR | E2 MTTC | E3 MTTC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Buying | 0.987500 | 0.987500 | 0.594182 | 0.594182 | 2.687500 | 2.625000 |
| Browsing | 1.000000 | 1.000000 | 0.571905 | 0.571905 | 2.837500 | 2.787500 |
| Intent Override | 0.966667 | 0.966667 | 0.663743 | 0.664299 | 4.866667 | 4.600000 |
| Boundary | 0.900000 | 0.900000 | 0.562897 | 0.600397 | 4.600000 | 4.200000 |

Every scenario preserves Hit Rate and MRR and lowers MTTC. Across five
deterministic scenario-stratified folds, TechnicalScore deltas are `+0.005563`,
`+0.004500`, `+0.001750`, `0.000000`, and `+0.001625`: four improve and one is
neutral. Fold 3 has an MRR delta of `-0.012500`, offset by a `-0.275000` MTTC
delta; fold metrics are development stability checks, not fitted estimates.

Saved benchmark (`610` responses): startup `9.617874 s`, mean response latency
`35.018203 ms`, p95 `83.242458 ms`, p99 `101.071713 ms`, maximum
`134.601567 ms`, and peak RSS `328.359 MB`. Timing is environment-dependent.

### Limitations and interpretation

- The gain is small and measured on 200 released development sessions; it is
  not an estimate of the hidden 800-session score.
- The catalog facet extractor is deterministic and shallow. It can miss
  synonyms or treat a verbose feature as a distinct answer.
- The conservative first-turn material guardrail is intentionally not adaptive;
  candidate-aware selection begins after the first reply.
- Active facets are downweighted rather than forbidden because the simulator
  may disclose multiple useful values for one facet. Already asked attributes
  are always forbidden, so the agent never asks the same question twice.
- Building the catalog-only facet side table increases startup time relative to
  E2 (`9.62 s` versus `1.91 s`) and increases mean response latency (`35.0 ms`
  versus `19.3 ms`). It leaves the BM25 index and returned ranking unchanged.
