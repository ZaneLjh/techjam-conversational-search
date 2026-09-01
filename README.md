# ShopSIFT

![ShopSIFT — intent that survives change](docs/project_media/01_shopsift_cover.png)

ShopSIFT is a deterministic multi-turn shopping agent for the TikTok TechJam 2026 Conversational E-Commerce Search Challenge. It tracks changing constraints, asks structured clarification questions, retrieves candidates through complementary local routes, and applies guarded ordering to find the target product early and rank it highly.

> **Submitted configuration:** E5 guarded deterministic hybrid  
> **Runtime:** local, non-LLM, no external API, no network calls, zero model tokens  
> **E6/E7:** evaluated offline, failed or did not pass promotion, disabled in the submitted runtime

## Quick start

### Requirements

- Python 3.10 or later
- Python's standard-library `sqlite3` module with SQLite FTS5 support
- Git Bash, Linux, or macOS shell for the commands below
- no API key or non-obvious environment variable

The submitted E5 runtime uses the Python standard library and checked-in local assets. It does not require a hosted model or third-party API.

```bash
python -m pip install -r requirements.txt
```

The submitted `requirements.txt` is intentionally empty apart from explanatory comments, so this command performs no third-party installation.

### 1. Obtain the frozen catalog

Download the organizer-supplied file named exactly `catalog.jsonl.gz` from the Track 4 participant resources on the [TikTok TechJam 2026 Devpost page](https://tiktoktechjam2026.devpost.com/). Do not substitute another Amazon catalog dump. Verify it:

```bash
printf '%s *%s\n' \
  '07fd142631fd6b03e2b4d09988c3eb7d53720e9d57010c79db48eeaada50a8f8' \
  'catalog.jsonl.gz' \
  | sha256sum -c -
```

Place the decompressed catalog where the evaluator expects it:

```bash
gzip -dk catalog.jsonl.gz
mv catalog.jsonl data/catalog.jsonl
```

The catalog contains the frozen 50,000-product `Clothing_Shoes_and_Jewelry` retrieval and scoring space.

### 2. Verify the promoted E5 assets

```bash
printf '%s *%s\n%s *%s\n' \
  'dadffeabfe10e1a4c0dc3f727f0837c7de7015b9b0701c365525df95476edc2a' \
  'starter/assets/e5_intent_projection.jsonl.gz' \
  '65ea2d64383c4d94fde594fdfa3eb47863e5cfb758bb2127b4d27bfa65b3a4d2' \
  'starter/assets/e5_projection_manifest.json' \
  | sha256sum -c -
```

Expected output:

```text
starter/assets/e5_intent_projection.jsonl.gz: OK
starter/assets/e5_projection_manifest.json: OK
```

These assets are derived only from the participant-visible frozen catalog. They do not contain unreleased evaluation labels.

### 3. Run all tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```

### 4. Run the unmodified public evaluator

```bash
python -m evaluator.local_evaluator \
  --catalog data/catalog.jsonl \
  --dataset data/public_set.jsonl \
  --output results.json
```

The evaluator imports `Agent` from `starter/agent.py` and constructs it with the catalog path. Do not modify the evaluator when producing reported results.

## Required Agent interface

```python
class Agent:
    def __init__(self, catalog_path: str = "data/catalog.jsonl") -> None:
        ...

    def reset(self, session_id: str, user_profile: dict) -> None:
        ...

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        ...
```

`respond(...)` returns:

- `message`: customer-facing natural language;
- `ask_attribute`: one allowed structured attribute or `null`;
- `recommendations`: catalog-valid `parent_asin` values ordered best to worst; and
- optional `usage`: zero prompt and completion tokens for this non-LLM runtime.

The evaluator uses structured `ask_attribute`, not a question inferred from prose. Recommendation order determines rank; optional numeric scores are ignored. Duplicate and invalid identifiers are removed, and only the first 10 valid unique recommendations are scored.

The complete allowed-value contract is in [`docs/agent_api_contract.json`](docs/agent_api_contract.json).

## Architecture

![ShopSIFT E1–E5 architecture](docs/project_media/03_architecture.png)

ShopSIFT separates conversational state, high-recall candidate generation, and constraint-safe ordering:

1. **E1 — Stateful lexical retrieval:** retains visible conversational evidence across turns.
2. **E2 — Structured constraint ledger:** records active, avoided, superseded, unknown, and no-preference values; an override replaces only the affected preference.
3. **E3 — Adaptive clarification:** selects one evaluator-supported `ask_attribute` when another answer is expected to improve retrieval.
4. **E4 — Multi-route retrieval:** combines lexical, category, facet, current-turn, recovery, and catalog-intent evidence under constraint tiers.
5. **E4.5 — Catalog-intent projection:** maps visible clues into representations derived from the frozen public catalog.
6. **E5 — Guarded ordering:** allows deterministic semantic evidence to reorder only eligible display positions and otherwise preserves the exact predecessor ordering.
7. **Output validation:** emits stable, unique, catalog-valid ASINs in scoring order.

Conversation state is isolated by `session_id`; immutable catalog indexes and projection assets may be shared across sequential sessions.

### Submitted runtime configuration

```text
learned_or_fitted = false
projection_enabled = true
projection_rollout = false
quality_enabled = false
semantic_enabled = true
```

Here, semantic evidence is deterministic and catalog-derived. It does not call an LLM, external embedding API, cross-encoder, or hosted service.

## Evidence-gated development

| Stage | Main contribution | Final status |
|---|---|---|
| E1 | Stateful lexical retrieval | Retained |
| E2 | Structured constraint ledger and override handling | Retained |
| E3 | Adaptive structured clarification | Retained |
| E4/E4.1 | Multi-route retrieval, constraint tiers, compliance hardening | Retained |
| E4.5 | Catalog-derived intent projection | Retained |
| **E5** | Guarded deterministic semantic ordering | **Submitted configuration** |
| E6 | Shallow learned residual ranker | Failed promotion gate; disabled |
| E7 | Calibrated residual follow-up | Did not pass promotion; disabled |

E6/E7 code, learned artifacts, synthetic corpora, fold maps, and experiment controls are not loaded by the submitted E5 runtime and are not sources of official answers.

## Development results

These are released-public and synthetic **development results**, not results from the unreleased 800-session final evaluation.

### Released 200-session public development set

| System | HitRate@10 | MRR | MTTC | Efficiency | TechnicalScore |
|---|---:|---:|---:|---:|---:|
| Official starter | 0.125000 | 0.068034 | 9.810 | 0.1190 | 0.106710 |
| **ShopSIFT E5** | **1.000000** | **0.822490** | **2.400** | **0.8600** | **0.918747** |

Two complete E5 public evaluations produced byte-identical output. Relative to E4 on the same public set, E5 retained HitRate@10 of `1.000000`, increased MRR by `0.024292`, improved 10 sessions, tied 190, and recorded no paired regression.

### Product-group-disjoint synthetic development set

| System | Sessions | TechnicalScore |
|---|---:|---:|
| E4 comparator | 3,000 | 0.886730 |
| **ShopSIFT E5** | **3,000** | **0.908304** |

E5 was positive across all five grouped folds and all three fixed seeds used for promotion analysis. Synthetic sessions were used only for offline development and were never runtime target-answer data.

![Released public development results](docs/project_media/04_public_results.png)

```text
Efficiency = clip((11 - MTTC) / 10, 0, 1)

TechnicalScore =
    0.50 × HitRate@10
  + 0.30 × MRR
  + 0.20 × Efficiency
```

`TechnicalScore` is an objective input to Technical Execution, not the complete judging criterion.

## Runtime, tokens, network, and cost

Previously verified E5 measurements:

| Measure | Value |
|---|---:|
| Startup | 86.228 s |
| Public-evaluation response wall time | 70.206 s for 480 responses |
| Mean response latency | 146.123 ms |
| p95 response latency | 325.462 ms |
| p99 response latency | 496.590 ms |
| Maximum response latency | 1,058.373 ms |
| Network calls | 0 |
| Prompt/completion tokens | 0 / 0 |
| Estimated model/API cost | $0 |
| Peak RSS | Not recorded |

The `$0` estimate covers model/API charges only; ordinary local-compute and electricity costs are not priced. Hardware metadata and peak RSS were not retained in the cited benchmark, so neither is claimed.

## Data and privacy boundaries

ShopSIFT:

- recommends only `parent_asin` values from the frozen catalog;
- uses participant-visible catalog fields and catalog-derived local assets;
- does not modify or fabricate catalog identifiers;
- does not receive direct user identifiers, review text, timestamps, or raw purchase histories;
- uses the evaluator-generated `session_id` only for isolated transient state;
- treats the aggregate `user_profile` as bounded context that cannot override an explicit current constraint;
- does not use or reconstruct unreleased final labels; and
- makes no runtime network call.

See [DATA_ATTRIBUTION.md](DATA_ATTRIBUTION.md) for the organizer-provided Amazon Reviews 2023 attribution.

## Limitations

- Public and simulator-aligned synthetic performance does not guarantee performance on the unreleased final sessions.
- Repeated development against the 200-session public set creates selection-bias risk.
- A `parent_asin` identifies a product family; catalog metadata cannot guarantee currently available color/size variants.
- Aggregate-profile personalization is bounded and has not shown an independently promoted scoring gain.
- Recommendation explanations summarize available evidence but cannot prove incomplete metadata is exhaustive.
- E5's catalog-derived projection increases startup time.
- Peak memory consumption was not recorded in the cited benchmark.
- A tested learned residual ranker reduced held-out performance, so no learned ranker is enabled.
- ShopSIFT does not use product images, live inventory, transactions, or variant-level availability.

## Demonstration and report

- [Complete public multi-turn demonstration](docs/demo_session.md) — generated from the clean E5 branch during finalization; the final commit must not contain `DEMO_NOT_CAPTURED`
- [How to reproduce and capture the demonstration](docs/demo_session_guide.md)
- [Submission report](docs/submission_report.md)
- [Competition specification](docs/competition_specification.md)
- [Final evaluation FAQ](docs/final_evaluation_faq.md)
- [Submission rules](docs/submission_rules.md)

The demonstrated session uses only the released public set and shows visible user messages, the Agent's customer-facing message, structured `ask_attribute`, ordered recommendations, the visible intent change, and final target rank.

## Team contribution

**Jian Heng Lim** — designed and implemented ShopSIFT; developed the E1–E7 experimental sequence; ran public, synthetic, ablation, determinism, and performance evaluations; integrated organizer updates; and prepared the reproducible submission and documentation.

## Final-evaluation commitment

After the final package is released, evaluation will be run from the exact Git commit submitted before the deadline using the unmodified official evaluator. The Agent, prompts, indexes, projection assets, model configuration, and other solution components will remain frozen.

The generated `results.json`, per-session results, submitted commit hash, environment and execution details, commands, logs, latency, tokens, and cost evidence will be retained for organizer review. The immutable submitted commit and release tag are recorded on Devpost rather than embedded here, because this README is itself part of that commit.
