# ShopSIFT Devpost submission copy

Prepared for the frozen E5 commit `9080b36572b494b060df07ca8dd9c38e8c801616`. Verify that the immutable commit and tag links are publicly accessible before submitting.

## Project name

**ShopSIFT**

## Elevator pitch

**176 characters, including spaces and punctuation:**

> ShopSIFT turns evolving shopping intent into ranked product recommendations through constraint tracking, adaptive questions, multi-route retrieval, and guarded local reranking.

If the live Devpost field unexpectedly enforces the standard 140-character tagline limit, use this **136-character fallback**:

> ShopSIFT tracks changing shopping intent, asks useful questions, and ranks catalog products with guarded, deterministic local retrieval.

## Project Story - About the project

### Inspiration

Shopping intent rarely arrives as one perfect query. A shopper may begin with a broad request, add a budget or use case, say they have no preference for one attribute, and later replace something they said earlier. Conventional search often treats each message in isolation. A conversational system can fail in the opposite direction by asking too many questions, forgetting hard constraints, or sounding helpful while ranking the wrong item.

We built **ShopSIFT** for a harder formulation: active exact-product retrieval under evolving constraints. The goal is not simply to generate a plausible answer. It is to identify the correct parent product as early as possible, rank it highly, and remain correct when the shopper's intent changes.

### What it does

ShopSIFT is a deterministic, multi-turn shopping agent for a frozen 50,000-product catalog. On every turn it can ask one structured clarification question, return ranked recommendations, or do both.

The agent:

- maintains active, avoided, superseded, unknown, and no-preference states in a structured constraint ledger;
- lets an explicit intent override replace only the affected preference while preserving unrelated constraints;
- changes retrieval emphasis according to the evidence available, rather than assuming access to a hidden Buying or Browsing label;
- combines lexical, category, facet, current-turn, recovery, and catalog-intent routes;
- asks an allowed attribute only when the expected information value justifies another question;
- projects visible clues into a catalog-derived intent space;
- protects exact constraints and high-confidence locks before reranking safer peer candidates; and
- validates and deduplicates catalog `parent_asin` values before returning them in scoring order.

ShopSIFT also accepts the provided privacy-safe aggregate profile as bounded context, but the shopper's current statements and hard constraints always take precedence.

### How we built it

We used an evidence-gated sequence of experiments rather than adding complexity all at once:

1. **E1 - Stateful lexical retrieval.** We established a fast deterministic baseline that carries visible conversational evidence across turns.
2. **E2 - Structured constraint ledger.** We made additions, conflicts, overrides, avoidance, unknowns, and no-preference states explicit.
3. **E3 - Adaptive clarification.** We added a value-of-information-inspired policy that returns a valid structured `ask_attribute` while still recommending products.
4. **E4 - Multi-route retrieval and constraint tiers.** Different evidence routes improve recall, while tiers stop weaker signals from violating stronger constraints.
5. **E4.5 - Catalog-intent projection.** Visible clues are projected into intent representations derived from the same frozen public catalog.
6. **E5 - Guarded ordering.** Deterministic semantic evidence reorders only eligible display positions, with exact predecessor fallback when a guard does not pass.

The submitted runtime is local and non-LLM. It uses immutable catalog-derived indexes and sidecars, isolated per-session state, deterministic stable tie-breaking, no external API, and no model tokens. Synthetic sessions and later learned-ranker experiments are offline evaluation tools only; they are never used as target-answer data at runtime.

The implementation follows the required Python interface:

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

The official objective combines retrieval success, rank, and speed of conversion:

$$
\text{TechnicalScore}
= 0.50\,\text{HitRate@10}
+ 0.30\,\text{MRR}
+ 0.20\,\text{Efficiency}.
$$

### Tools, APIs, assets, and libraries

We developed ShopSIFT in Python with Git and GitHub, and validated it with the repository's `unittest` suite and the unmodified organizer evaluator. The submitted E5 runtime makes no external API or network calls and uses no hosted model.

Its runtime data and assets are the organizer's frozen Amazon Reviews 2023 `Clothing_Shoes_and_Jewelry` catalog plus the E5 intent-projection sidecar and manifest derived only from that participant-visible catalog. The submitted runtime uses Python 3.10+ and the standard library, including `sqlite3` with FTS5; `requirements.txt` intentionally installs no third-party runtime package. Synthetic sessions and later learned-ranker experiments are offline development assets only and are not runtime sources of target answers.

### Results

On the released 200-session public development set, the frozen E5 configuration produced:

| System | HitRate@10 | MRR | MTTC | TechnicalScore |
|---|---:|---:|---:|---:|
| Official starter | 0.125 | 0.068034 | 9.810 | 0.106710 |
| **ShopSIFT E5** | **1.000** | **0.822490** | **2.400** | **0.918747** |

Public evaluator evidence is summarized in the [immutable submission report](https://github.com/ZaneLjh/techjam-conversational-search/blob/9080b36572b494b060df07ca8dd9c38e8c801616/docs/submission_report.md). The two byte-identical raw evaluator outputs are retained with the submission evidence and are available to organizers on request.

Relative to E4 on the same public set, E5 kept HitRate@10 at 1.000, improved MRR by 0.024292, improved 10 sessions, tied 190, and recorded no paired regression. Two complete E5 runs were byte-identical.

We also tested E5 on 3,000 product-group-disjoint synthetic sessions. Its TechnicalScore was 0.908304 versus 0.886730 for E4, with all 5 grouped folds and all 3 fixed seeds positive. The promotion evidence is summarized in the [immutable submission report](https://github.com/ZaneLjh/techjam-conversational-search/blob/9080b36572b494b060df07ca8dd9c38e8c801616/docs/submission_report.md#62-product-group-disjoint-synthetic-development-set), with underlying development records retained for organizer review. These sessions intentionally resemble the released deterministic task and are used only for development and promotion decisions.

These are **public and synthetic development results**, not a claim about the unreleased 800-session final evaluation.

### Challenges we faced

#### Preserving intent without preserving mistakes

A conversational memory system must retain useful context while forgetting exactly the right thing when the shopper changes their mind. Rebuilding intent from only the latest message lost constraints; accumulating every message created contradictions. The structured ledger gave each value an explicit status and made overrides testable.

#### Asking a useful question without delaying retrieval

Every clarification consumes a turn, yet a good answer can sharply improve rank. We treated questioning as a structured action tied to candidate uncertainty, and the agent still returns recommendations in the same turn.

#### Combining recall and precision safely

Broad routes recover products that a strict query may miss, but an unconstrained fusion can let weak semantic evidence overrule an exact requirement. We introduced constraint tiers, pinned evidence, recovery rules, and deterministic fallback before allowing reranking.

#### Improving the ranker without sacrificing hits

We tested a shallow learned residual ranker after E5. It reduced TechnicalScore by 0.015252 on a fresh group-disjoint synthetic corpus and produced more regressions than improvements, so it failed the promotion gate and was not enabled. This decision is documented in the [immutable submission report](https://github.com/ZaneLjh/techjam-conversational-search/blob/9080b36572b494b060df07ca8dd9c38e8c801616/docs/submission_report.md#5-experiment-progression), with the underlying gate output retained for organizer review. That negative result was valuable: model complexity is not progress unless it improves the metric while preserving reliable retrieval behavior.

#### Matching the evaluator exactly

The simulator responds to the structured `ask_attribute`, not to a question inferred from prose. Recommendation order is the ranking, optional numeric scores are ignored, and only the first 10 valid unique ASINs are scored. Interface validation and session isolation therefore became part of the retrieval design rather than last-minute packaging work.

### What we learned

The strongest lesson was that conversational product search is a joint state, retrieval, ranking, and decision problem. A better text matcher alone is insufficient.

We learned to:

- represent changing intent explicitly;
- align clarification with the evaluator's structured action space;
- separate high-recall candidate generation from constraint-safe ordering;
- treat public and synthetic results as development evidence, not private-set guarantees;
- use negative experiments to protect a strong predecessor; and
- favor a reproducible local system when added model complexity does not earn its operational cost.

### What's next

Future work could improve calibrated question-value estimation, evaluate richer residual ranking on genuinely independent data, and make evidence-based per-product explanations more complete without claiming unavailable variant-level details. The submission deliberately keeps the validated E5 core: deterministic, evaluator-compatible, zero-token, subsecond per-response latency after startup in the cited benchmark, and guarded against unsafe ranking changes.

## Built with

Add up to 25 tags. Recommended tags:

1. Python
2. Amazon Reviews 2023
3. Conversational AI
4. Natural Language Processing
5. Information Retrieval
6. Product Search
7. Recommender Systems
8. Search Ranking
9. BM25
10. Lexical Search
11. Hybrid Retrieval
12. Semantic Search
13. Intent Routing
14. Constraint Tracking
15. Dialogue Management
16. Adaptive Clarification
17. Constraint-Aware Search
18. Synthetic Evaluation
19. Offline Evaluation
20. Deterministic Systems
21. JSON
22. JSONL
23. gzip
24. Git
25. GitHub

Only add a library/model tag if it is actually present in the frozen E5 dependency manifest.

## Project Media - suggested images

Use four images in this order. A video is not included.

### 1. Cover image: “ShopSIFT - intent that survives change”

Create a clean 16:9 image with:

- project name and elevator pitch;
- a short flow: **Conversation -> Constraint ledger -> Multi-route retrieval -> Guarded ranking**; and
- a small footer: **Local | Deterministic | 0 model tokens**.

Avoid unsupported product imagery, TikTok trademarks used as project branding, or private-evaluation claims.

### 2. Complete multi-turn session

Use a horizontal sequence of three or four terminal/API screenshots from one actual public session:

1. broad initial request and first recommendations;
2. structured clarification with its `ask_attribute`;
3. user answer or intent override and updated active/superseded state; and
4. final ranked hit with turn and rank.

Show the natural-language response and structured output. Redact local paths, usernames, tokens, and unrelated machine details.

### 3. E1-E5 architecture

Show a top-down diagram:

**Visible message -> E2 constraint ledger -> E3 question policy -> E4 multi-route retrieval -> E4.5 intent projection -> E5 guarded ordering -> 10 valid unique ASINs**

Label synthetic generation and E6 as **offline evaluation only**, outside the runtime path.

### 4. Verified development evidence

Show a compact chart or table comparing the official starter and E5 public metrics. Label it prominently:

> Released 200-session public development set - not final evaluation

Optionally add a small test panel showing the final clean-clone test pass and frozen commit SHA.

## Additional information for judges and organizers

The details below apply to the frozen commit identified here. Public-development evaluator outputs are retained separately for organizer review rather than committed as runtime dependencies. The final-evaluation environment and outputs will be recorded with the final results as required by the organizer.

### Submission summary

- **Track:** Track 4 - TechJam Conversational E-Commerce Search Challenge
- **Project:** ShopSIFT
- **Submitted configuration:** E5 guarded deterministic hybrid
- **Public repository:** https://github.com/ZaneLjh/techjam-conversational-search
- **Frozen commit:** `9080b36572b494b060df07ca8dd9c38e8c801616`
- **Immutable commit link:** https://github.com/ZaneLjh/techjam-conversational-search/commit/9080b36572b494b060df07ca8dd9c38e8c801616
- **Final tag:** `techjam-2026-e5-final`
- **Immutable tag link:** https://github.com/ZaneLjh/techjam-conversational-search/tree/techjam-2026-e5-final
- **Agent entry point:** `starter/agent.py`
- **Required export:** `Agent`
- **Demonstration:** Project Media screenshot sequence plus [`docs/demo_session.md`](https://github.com/ZaneLjh/techjam-conversational-search/blob/9080b36572b494b060df07ca8dd9c38e8c801616/docs/demo_session.md)
- **Demo video:** Not included

The immutable commit and annotated tag identify the exact submission. Judges should evaluate that commit rather than a moving branch reference.

### Reproduction

From the repository root, after following the README's catalog and asset setup:

```bash
python -m unittest discover -s tests -p "test_*.py"

python -m evaluator.local_evaluator \
  --catalog data/catalog.jsonl \
  --dataset data/public_set.jsonl \
  --output results.json
```

- **Python requirement:** Python 3.10 or later
- **Runtime hardware:** CPU-only; no GPU is required; the exact final-evaluation OS, Python build, CPU, and RAM will be recorded with the final results
- **Dependencies:** Python standard library only, including `sqlite3` with SQLite FTS5 support; no third-party runtime package
- **Required environment variables:** None for the submitted E5 runtime
- **Network dependency:** None for the submitted E5 runtime

### Runtime configuration

```text
learned_or_fitted = false
projection_enabled = true
projection_rollout = false
quality_enabled = false
semantic_enabled = true
```

The E6 learned ranker and all E7 experiments are disabled. The synthetic corpora, fold maps, and training/evaluation tools are offline-only and are never loaded as sources of official answers.

### Development evidence

On the released 200-session public development set, E5 recorded:

- HitRate@10: `1.000000`
- MRR: `0.822490`
- MTTC: `2.400`
- Efficiency: `0.8600`
- TechnicalScore: `0.918747`

Two complete public runs were byte-identical. These are development results and are not presented as results on the unreleased 800-session final package.

- **Public-result evidence:** Two byte-identical full evaluator JSON outputs were produced during development and are retained outside the runtime repository for organizer review. The methodology, metrics, caveats, and reproduction command are documented in the frozen [`README.md`](https://github.com/ZaneLjh/techjam-conversational-search/blob/9080b36572b494b060df07ca8dd9c38e8c801616/README.md) and [`docs/submission_report.md`](https://github.com/ZaneLjh/techjam-conversational-search/blob/9080b36572b494b060df07ca8dd9c38e8c801616/docs/submission_report.md).

### Feasibility disclosure

Previously verified E5 measurements:

- startup: `86.228 s`;
- 480-response public run: `70.206 s`;
- mean latency: `146.123 ms`;
- p95 latency: `325.462 ms`;
- p99 latency: `496.590 ms`;
- maximum latency: `1,058.373 ms`;
- network calls: `0`;
- prompt/completion tokens: `0 / 0`; and
- estimated model/API cost: `$0` (local compute not priced).

The original benchmark's exact hardware metadata and peak RSS were not retained, so neither is claimed. The exact final-evaluation OS, Python build, CPU, RAM, execution commands, and result files will be recorded and retained with the final results.

### Data and asset disclosure

The runtime retrieval space is the organizer's frozen 50,000-product `Clothing_Shoes_and_Jewelry` catalog. Recommendations use only catalog-valid `parent_asin` values. The included indexes, intent projection, manifests, and sidecars are derived from the frozen participant-visible catalog. No unreleased labels, raw purchase histories, direct user identifiers, timestamps, or review text are used by the Agent. The evaluator-generated `session_id` is used only to isolate transient session state; `parent_asin` is the required product identifier.

Catalog archive SHA-256:

```text
07fd142631fd6b03e2b4d09988c3eb7d53720e9d57010c79db48eeaada50a8f8
```

Third-party/open-source assets and licences: the organizer-supplied frozen catalog, public sessions, evaluator, and Amazon Reviews 2023 attribution are documented in [`DATA_ATTRIBUTION.md`](https://github.com/ZaneLjh/techjam-conversational-search/blob/9080b36572b494b060df07ca8dd9c38e8c801616/DATA_ATTRIBUTION.md). The submitted runtime uses no external model, API, or third-party Python package. E5 sidecars are derived only from the participant-visible frozen catalog.

### Limitations

- Public and simulator-aligned synthetic performance does not guarantee performance on the unreleased final sessions.
- Parent ASIN metadata does not guarantee that every described color or size variant is currently purchasable.
- Safe aggregate-profile support is bounded; it has not shown an independently promoted scoring gain and never overrides active stated constraints.
- User-facing explanations are evidence summaries, not proof that incomplete catalog metadata is exhaustive.
- A tested shallow learned reranker reduced held-out synthetic performance, so it was not promoted.
- Peak memory use was not recorded in the cited benchmark.

### Final-evaluation commitment

After the final package is released, I will run the unmodified official evaluator from the exact frozen commit above without changing the Agent, prompts, indexes, model configuration, or other solution components. I will retain `results.json`, per-session results, the submitted commit hash, environment details, execution commands, and logs for organizer review.

### Team contributions and contact

- **Project lead:** Jian Heng Lim
- **Contributions:** Jian Heng Lim — system design; E1–E7 implementation and experiment progression; public, synthetic, ablation, determinism, and performance evaluation; organizer-update integration; reproducibility; documentation; and project media
- **Team representative:** Jian Heng Lim
- **Contact:** Through the registered Devpost team profile
