# E2 Review Guide

## Preferred upload

Upload `TechJam_E2_review_bundle.zip`. It contains the implementation patch,
changed source, tests, reproducibility tools, documentation, and the small result
artifacts listed below. The frozen catalog is intentionally excluded.

## Output files for further review

The review bundle should contain:

- `TechJam_E2_implementation.patch`
- `starter/agent.py`
- `starter/constraints.py`
- `tests/test_constraint_ledger.py`
- `tests/test_stateful_agent.py`
- `tools/evaluate_variant.py`
- `tools/fold_report.py`
- `tools/benchmark_agent.py`
- `results/e1_stateful_lexical.json` (frozen comparison input)
- `results/e2_structured_constraint_ledger.json`
- `results/e2_structured_constraint_ledger_repeat.json`
- `results/e2a_ledger_only.json`
- `results/e2_five_fold_report.json`
- `results/e2_benchmark.json`
- `README.md`
- `docs/experiments.md`
- `docs/e2_demo_session.md`
- `docs/e2_review_guide.md`

If the reviewer accepts only individual files, upload the patch and all six
listed `results/` JSON files. The patch carries the source, tests, tools, and
docs.

Do not upload `.git/`, `.venv/`, `__pycache__/`, `.pyc` files, the 50,000-product
catalog, the participant PDFs, or local caches. The evaluator and public labels
are unchanged and do not need to be uploaded with an implementation-only review.

## Reproduction checklist

From a repository root that already contains `data/catalog.jsonl`:

```bash
python -m compileall -q starter tests tools
python -m unittest discover -v
python -m evaluator.local_evaluator \
  --output results/e2_structured_constraint_ledger.json
python -m evaluator.local_evaluator \
  --output results/e2_structured_constraint_ledger_repeat.json
sha256sum results/e2_structured_constraint_ledger*.json
python -m tools.evaluate_variant --disable-candidate-exploration \
  --output results/e2a_ledger_only.json
python -m tools.fold_report \
  --baseline results/e1_stateful_lexical.json \
  --candidate results/e2_structured_constraint_ledger.json \
  --folds 5 --output results/e2_five_fold_report.json
python -m tools.benchmark_agent --output results/e2_benchmark.json
git diff --check
git diff HEAD -- evaluator data/public_set.jsonl
```

Expected headline evidence:

- `47/47` tests pass.
- TechnicalScore `0.827342`, Hit Rate@10 `0.985000`, MRR `0.594141`,
  MTTC `3.170000`.
- Repeated evaluator files have SHA-256
  `af20738d40241258f0ae84a4ccc022d8c100ec52963004354232e200e03a78f8`.
- The evaluator and public label file have no diff.

Before a final competition submission, replace the team-contribution placeholders
in `docs/e2_demo_session.md` with the actual names and work ownership. That
information cannot be inferred from the participant kit.
