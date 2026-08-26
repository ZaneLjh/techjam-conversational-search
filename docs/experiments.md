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
