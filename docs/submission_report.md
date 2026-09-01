# ShopSIFT E5 Submission Report

## 1. Submission summary

ShopSIFT is a deterministic multi-turn product-retrieval agent for the frozen 50,000-product Amazon Reviews 2023 `Clothing_Shoes_and_Jewelry` catalog. It maintains conversational constraints, asks structured clarification questions, retrieves candidates through complementary routes, and applies guarded catalog-derived ordering without an LLM or external API.

The submitted configuration is the promoted **E5 guarded deterministic hybrid**. E6 and E7 experiments did not pass promotion and are disabled; neither is loaded by the submitted runtime.

The final immutable commit is recorded on Devpost after it is created. It is intentionally not embedded in this file because a file cannot contain the hash of the same commit that contains it.

## 2. Evaluator boundary

The unmodified evaluator imports `Agent` from `starter/agent.py`, constructs it with the catalog path, and calls:

```python
class Agent:
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

The response contains a customer-facing `message`, one allowed structured `ask_attribute` or `null`, and best-to-worst recommendations containing catalog-valid `parent_asin` values. Token usage is reported as zero for the submitted non-LLM path. Conversation state is isolated by evaluator-generated `session_id`; only immutable indexes are shared between sessions.

## 3. Method and architecture

### 3.1 Structured conversation state

The E2 constraint ledger records active requirements, avoided values, superseded preferences, unknown values, no-preference states, and visible evidence from the current and preceding turns. An explicit intent override replaces only the affected preference; unrelated active constraints remain in force.

The privacy-safe aggregate `user_profile` is bounded supporting context. It never overrides an explicit current-session requirement.

### 3.2 Adaptive clarification

E3 may ask one evaluator-supported attribute while returning recommendations in the same turn. The policy selects an attribute based on available evidence and expected value rather than following a fixed question order.

The natural-language question and structured `ask_attribute` are generated consistently because the simulator acts on `ask_attribute`; it does not infer the requested field from prose.

### 3.3 Multi-route candidate generation

E4 combines complementary routes:

- lexical matching;
- category and facet evidence;
- current-turn and retained conversational evidence;
- recovery routes for sparse or conflicting queries; and
- catalog-derived intent evidence.

The Agent receives no hidden Buying, Browsing, Intent Override, or Boundary label. Routing uses only inputs visible through the required Agent API.

### 3.4 Catalog-intent projection and guarded ordering

E4.5 projects visible clues into representations derived from the same frozen participant-visible catalog. E5 considers deterministic semantic evidence only after exact constraints and high-confidence locks are protected.

E5 may reorder eligible display positions. If a guard does not pass, it preserves the predecessor order instead of forcing a speculative change. Invalid and duplicate ASINs are removed, and recommendation order—not an optional numeric score—defines rank.

## 4. Model choice

The submitted E5 runtime is local, deterministic, and non-LLM:

```text
learned_or_fitted = false
projection_enabled = true
projection_rollout = false
quality_enabled = false
semantic_enabled = true
```

It does not use an external model API, hosted or local LLM, cross-encoder, E6 learned ranker, E7 calibrated residual path, synthetic targets at runtime, or unreleased evaluation labels.

E5 was selected because it provided the strongest accepted development evidence while retaining deterministic behavior, exact fallback, zero model-token use, and no network dependency. Increased model complexity was not promoted unless it passed predefined gates.

## 5. Experiment progression

| Stage | Main change | Decision |
|---|---|---|
| E1 | Stateful lexical retrieval | Deterministic baseline retained |
| E2 | Structured constraint ledger and intent-override handling | Retained |
| E3 | Value-of-information-inspired structured clarification | Retained |
| E4 | Multi-route retrieval and constraint tiers | Retained |
| E4.1 | Compliance hardening and tighter fusion guards | Incorporated where promoted |
| E4.5 | Catalog-derived intent projection | Retained |
| **E5** | Guarded deterministic ordering with predecessor fallback | **Promoted and submitted** |
| E6 | Shallow learned residual ranker | Failed promotion; disabled |
| E7 | Calibrated residual follow-up | Did not pass promotion; disabled |

The E6 gate reported `eligible=false` and `decision=keep_promoted_e5_as_default`. On its fresh group-disjoint synthetic evaluation, the tested learned reranker reduced TechnicalScore by `0.015252` relative to its E5 predecessor. E7 likewise did not earn promotion. These are offline development decisions, not claims about the unreleased final set.

## 6. Development evaluation

### 6.1 Released 200-session public set

| System | HitRate@10 | MRR | MTTC | Efficiency | TechnicalScore |
|---|---:|---:|---:|---:|---:|
| Official starter | 0.125000 | 0.068034 | 9.810 | 0.1190 | 0.106710 |
| **ShopSIFT E5** | **1.000000** | **0.822490** | **2.400** | **0.8600** | **0.918747** |

Two full E5 public runs were byte-identical. Relative to E4 on the same public set, E5 retained HitRate@10 at `1.000000`, improved MRR by `0.024292`, improved 10 sessions, tied 190, and recorded no paired regression.

The release procedure reruns the unmodified evaluator twice from the clean final branch. If those fresh results differ, the table and project media must be updated before submission rather than preserving stale values.

### 6.2 Product-group-disjoint synthetic development set

| System | Sessions | TechnicalScore |
|---|---:|---:|
| E4 comparator | 3,000 | 0.886730 |
| **ShopSIFT E5** | **3,000** | **0.908304** |

E5 was positive in all five grouped folds and all three fixed-seed evaluations used in promotion analysis. Synthetic sessions were used only for offline development, ablation, and promotion decisions. They are not loaded by the submitted Agent, contain no unreleased final labels, and are not a substitute for independent final evaluation.

### 6.3 Interpretation boundary

Every value in this section is released-public or synthetic development evidence. Nothing here claims or guarantees performance on the 800 unreleased final sessions.

`TechnicalScore` is an objective input to Technical Execution; it is not the complete Technical Execution assessment or the complete judging decision.

## 7. Runtime, token, network, and cost disclosure

Previously verified E5 measurements:

| Measure | E5 measurement |
|---|---:|
| Startup | 86.228 s |
| Public-evaluation response wall time | 70.206 s for 480 responses |
| Mean response latency | 146.123 ms |
| p95 response latency | 325.462 ms |
| p99 response latency | 496.590 ms |
| Maximum response latency | 1,058.373 ms |
| Network calls | 0 |
| Prompt tokens | 0 |
| Completion tokens | 0 |
| Estimated model/API cost | $0 |
| Peak RSS | Not recorded |

The `$0` figure covers model/API charges only. Local hardware, electricity, and engineering time are not priced. Hardware metadata and peak RSS were not retained with the cited benchmark, so neither is claimed. Startup is reported separately from response latency.

## 8. Data and artifact provenance

The scoring space is the organizer-provided frozen Amazon Reviews 2023 `Clothing_Shoes_and_Jewelry` catalog. Recommendations use only `parent_asin` values from this catalog.

```text
07fd142631fd6b03e2b4d09988c3eb7d53720e9d57010c79db48eeaada50a8f8  catalog.jsonl.gz
dadffeabfe10e1a4c0dc3f727f0837c7de7015b9b0701c365525df95476edc2a  starter/assets/e5_intent_projection.jsonl.gz
65ea2d64383c4d94fde594fdfa3eb47863e5cfb758bb2127b4d27bfa65b3a4d2  starter/assets/e5_projection_manifest.json
```

The E5 projection artifacts were derived only from participant-visible frozen catalog data. The project uses the organizer-provided evaluator, public sessions, and Amazon Reviews 2023 attribution documented in `DATA_ATTRIBUTION.md`. The runtime has no third-party Python package or external-service dependency.

Promoted E5 provenance:

```text
29d2a58  feat: promote E5 guarded hybrid as default
```

The final submission branch must contain that commit and organizer FAQ commit `9c9e7c9` or a later organizer commit containing it.

## 9. Fallback and failure behavior

1. Candidate generation and exact constraint checks run first.
2. Semantic ordering is considered only for eligible positions.
3. If its guard does not pass, the predecessor ordering is preserved.
4. Invalid and duplicate ASINs are removed before output.

There is no external service whose outage requires a network fallback. E6 and E7 are not fallback paths and are disabled. A missing or checksum-mismatched required E5 asset is treated as a release/setup failure; the release procedure stops rather than silently replacing or regenerating it.

## 10. Limitations

- Public and simulator-aligned synthetic performance does not guarantee performance on unreleased final sessions.
- Repeated development against 200 public sessions creates selection-bias risk.
- Synthetic sessions reproduce assumptions from the released evaluator and may not capture every independent-data failure mode.
- A `parent_asin` identifies a parent product, not guaranteed availability of every color/size SKU.
- Aggregate-profile personalization is bounded, has no independently promoted gain, and never overrides an active stated constraint.
- Explanations summarize available evidence but cannot prove incomplete catalog metadata is exhaustive.
- The learned E6 reranker reduced held-out performance and was not promoted; E7 likewise did not pass promotion.
- Initialization is materially longer than per-response latency.
- Peak memory use and original benchmark hardware metadata were not recorded.
- The system does not use images, live inventory, real transactions, or variant-level availability.
- No result in this report should be interpreted as performance on unreleased final data.

## 11. Reproduction

From the repository root after catalog placement and checksum verification:

```bash
python -m unittest discover -s tests -p "test_*.py"

python -m evaluator.local_evaluator \
  --catalog data/catalog.jsonl \
  --dataset data/public_set.jsonl \
  --output results.json
```

The submitted E5 runtime requires no API key and no non-obvious environment variable.

## 12. Team contribution

**Jian Heng Lim** — designed and implemented ShopSIFT; developed the E1–E7 experimental sequence; ran public, synthetic, ablation, determinism, and performance evaluations; integrated organizer updates; and prepared the reproducible submission and documentation.

## 13. Final-evaluation commitment

After the final package is released, the team will:

- use the exact repository commit submitted before the Devpost deadline;
- make no changes to the Agent, prompts, indexes, model configuration, sidecars, or other solution components;
- run the unmodified official evaluator;
- retain `results.json`, including per-session results;
- retain the submitted commit hash, environment details, dependency versions, exact command, timestamps, console logs, latency, token counts, and cost estimate; and
- provide supporting evidence to organizers if requested.

No final-evaluation score will be reported from a modified evaluator.
