# ShopSIFT demonstrated public session

**DEMO_NOT_CAPTURED**

This file is intentionally a safety placeholder. Generate the real released-public transcript from the frozen E5 branch by following [`demo_session_guide.md`](demo_session_guide.md):

```bash
python -m tools.replay_public_session \
  --catalog data/catalog.jsonl \
  --dataset data/public_set.jsonl \
  --results "$HOME/shopsift_submission_evidence/e5_final_public_run1.json" \
  --sample-id public_0002 \
  --output-json "$HOME/shopsift_submission_evidence/e5_demo_public_session.json" \
  --output-markdown docs/demo_session.md
```

Do not commit this placeholder. The replay command replaces it with actual E5 output and validates `first_hit_turn` and `best_rank` against the recorded public-session summary.
