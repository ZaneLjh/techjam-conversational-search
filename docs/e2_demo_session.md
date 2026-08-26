# E2 Demonstrated Multi-Turn Session

This deterministic demonstration uses the frozen catalog, `top_k=3`, and the
production `Agent`. It shows constraint accumulation followed by a punctuation-free
intent correction. Product identifiers are valid catalog `parent_asin` values.

## Turn 1

Customer: `I'm looking for shoes. I prefer red.`

Agent:

```json
{
  "message": "Do you have a material preference?",
  "ask_attribute": "material",
  "recommendations": [
    {"parent_asin": "B00Z4YM3D4"},
    {"parent_asin": "B09X7GRRRS"},
    {"parent_asin": "B01LY8HS06"}
  ],
  "usage": {"prompt_tokens": 0, "completion_tokens": 0}
}
```

Audit state: canonical query `shoes red`; intent epoch `0`.

## Turn 2

Customer: `For that, what matters is: waterproof.`

Agent:

```json
{
  "message": "Which product feature matters most to you?",
  "ask_attribute": "feature",
  "recommendations": [
    {"parent_asin": "B00R0RTO12"},
    {"parent_asin": "B078JDQMTL"},
    {"parent_asin": "B081SKLFXN"}
  ],
  "usage": {"prompt_tokens": 0, "completion_tokens": 0}
}
```

Audit state: canonical query `shoes waterproof red`; intent epoch `0`.

## Turn 3

Customer: `Instead of red I want blue.`

Agent:

```json
{
  "message": "Do you have a color preference?",
  "ask_attribute": "color",
  "recommendations": [
    {"parent_asin": "B08TG8P6HD"},
    {"parent_asin": "B008OK8D6Q"},
    {"parent_asin": "B07TKRQ98Y"}
  ],
  "usage": {"prompt_tokens": 0, "completion_tokens": 0}
}
```

Audit state: canonical query `shoes waterproof blue`; intent epoch `1`. The red
constraint remains in ledger history as superseded, while blue is active.

## Runtime and model disclosure

- Runtime: Python 3.10 or later.
- Dependencies: Python standard library only.
- Model/API: none.
- Network required: no.
- Reported model tokens: 0.
- Estimated model/API cost: `$0`.

## Team contribution disclosure — complete before final submission

The attached source materials do not identify team members, so names cannot be
filled in safely. Replace these placeholders before final submission:

- `[Team member name]`: `[architecture / implementation / evaluation / report contribution]`
- `[Team member name]`: `[architecture / implementation / evaluation / report contribution]`
