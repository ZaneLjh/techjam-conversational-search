# ShopSIFT demonstrated public session

- Source: released 200-session public development set
- Public sample: `public_0002`
- Scenario: Intent Override
- Agent configuration: frozen E5 guarded deterministic hybrid
- Final/private evaluation data used: No
- Label handling: The released public target was checked only after each Agent response and was never passed to reset() or respond().
- State annotation: Conversation-state annotation reconstructed from visible user messages; not an additional Agent input or internal debug export.

| Turn | Visible user message | Agent message | `ask_attribute` | Top three ranked ASINs | Visible state transition |
|---:|---|---|---|---|---|
| 1 | I'm looking for Accessories Belts. Buckle closure | Do you have a material preference? | `material` | B08S6Q2DM2, B078HG9KCY, B00N4CEGEW | Active: category: Accessories Belts, Buckle closure; superseded: — |
| 2 | For that, what matters is: leather; 100% Leather. | Which product feature matters most to you? | `feature` | B071X54486, B0C4HGYDMF, B07RQSN383 | Active: category: Accessories Belts, Buckle closure, 100% Leather, leather; superseded: — |
| 3 | Actually, ignore my earlier preference. What I need is: leather. | Do you have a color preference? | `color` | B0C4HGYDMF, B08FLYBP2J, B07RQSN383 | Active: category: Accessories Belts, 100% Leather, leather; superseded: Buckle closure |

- Outcome: public target first appeared on turn `3` at rank `6`.
- Validation: replayed `first_hit_turn` and `best_rank` matched the recorded public-session summary.
- Recommendation validation: ranking was normalized to catalog-valid, unique ASINs in scoring order.

> This demonstration is presentation evidence only. Official metrics come from the unmodified full public evaluator.
