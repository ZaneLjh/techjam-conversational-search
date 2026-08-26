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
