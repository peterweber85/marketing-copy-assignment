# Lodgify — Content Generation Pipeline

A vacation rental marketing copy generator built evaluation-first. The eval suite
is the primary deliverable; the generator exists to be measured.

---

## Approach

**Evaluation-first development (EDD)** means evals are written and validated before
the generator is built, so every prompt iteration is driven by metrics rather than
eyeballing outputs. The eval harness is [inspect-ai](https://inspect.aisi.org.uk/).

### Three evaluation dimensions

| Scorer | Type | What it catches |
|---|---|---|
| `grounding_scorer` | Rule-based, offline | Wrong bedroom/bathroom counts; invented amenity codes; null policies mentioned; fake social proof with zero reviews |
| `faithfulness_scorer` | LLM-as-judge (Sonnet) | Unsupported factual claims; embellishments; awards/attributes not in data |
| `quality_scorer` | LLM-as-judge (Sonnet) | Engagement, clarity, tone, specificity |

### The EDD arc (real, not staged)

1. **Evals reveal a problem:** `listing_eval` prompt v1 produces mean faithfulness
   **3.33/5** — the Sonnet judge flags embellishments like "saltwater-inspired pool",
   "full central heating" (data says "Heating"), "panoramic vistas from every room".
2. **Diagnosis:** the v1 prompt says "write compelling copy" without anti-embellishment
   rules. The model adds descriptive attributes it infers from context.
3. **Prompt fix:** v2 adds explicit rules: "describe amenities only with information
   available from the data — do not add qualifiers not in the structured fields."
4. **Evals confirm:** mean faithfulness improves to **3.83/5** (+15%). Individual
   fixes: Eixample Loft 3→5, Alpine Escape 3→4. Quality drops slightly (3.67→3.33)
   — the expected cost/quality tradeoff of a more constrained prompt.

The two prompt versions and their runs are committed in `logs/` so the before/after
is visible in `inspect view` without re-running anything.

### Stub validation (EDD discipline)

Before any real LLM call, the scorers were validated against hand-crafted stub
outputs for property 104 (the "trap": a studio whose owner headline falsely claims
two bedrooms):

| Stub | grounding | faithfulness | quality |
|---|---|---|---|
| Bad (invented Jacuzzi, wrong bedroom count, fake awards) | 0.50 | 1/5 | 2/5 |
| Good (correct studio description) | 1.00 | 2/5 | 3/5 |

This confirms the scorers discriminate good from bad copy before the real generator
is built — the AI equivalent of "watch the test fail first."

### Reliability

`reliability_eval` ran each of the 6 fixtures 3 times (18 total samples).
`grounding_scorer` held at **1.000** every run — structural checks on typed fields
are completely stable. `faithfulness_scorer` mean = **4.000** across all 18 runs.

### Model choices and cost/latency tradeoffs

- **Generation: Haiku 4.5** — fast, cheap ($1/$5 per MTok), sufficient for
  structured-output generation from a tight prompt.
- **Judge: Sonnet 4.6** — stronger reasoning for nuanced faithfulness assessment.
  LLM judging costs ~2× more than generation but is only called in eval, not production.
- **Prompt caching** is visible in the logs (Sonnet cache hits grow across runs in
  the same session). For production, caching the system prompt + property schema
  would reduce per-call judge cost by ~90%.

---

## Repository structure

```
lodgify/
  models.py        Pydantic input (PropertyInput) + output (ListingCopy) schemas
  amenities.py     Amenity-code → human-label map; CamelCase fallback
  ingest.py        HTML stripping, context normalization (no LLM)
  scorers.py       The three scorers — grounding, faithfulness, quality
  generator.py     Solver chain: ingest_solver + generate_solver (versioned prompts)
  data.py          Fixture loader (validates against PropertyInput on load)
fixtures/          6 property JSON files; each targets a specific eval dimension
logs/              Committed inspect-ai .eval logs (view offline with inspect view)
tests/
  test_pipeline.py 36 offline tests; covers DI, inheritance, mocking, scorers
evals.py           inspect-ai task definitions: stub_eval, listing_eval, reliability_eval
```

---

## How to run

**Setup (one time):**
```bash
uv sync
cp .env.example .env
# add ANTHROPIC_API_KEY to .env
```

**Run tests (offline, no API key needed):**
```bash
uv run pytest tests/ -v
```

**View committed eval results (no API key needed):**
```bash
uv run inspect view --log-dir logs/
```
Opens a browser UI. Each run shows: every property, the generated copy, per-scorer
verdicts, aggregate scores, and token usage. The two `listing_eval` runs labelled
v1 and v2 show the EDD before/after.

**Regenerate eval logs (requires API key in `.env`):**
```bash
# Validate scorers against stubs (EDD discipline)
uv run inspect eval evals.py@stub_eval --model anthropic/claude-sonnet-4-6 -T variant=bad
uv run inspect eval evals.py@stub_eval --model anthropic/claude-sonnet-4-6 -T variant=good

# The EDD arc — prompt v1 (problem) then v2 (fix)
uv run inspect eval evals.py@listing_eval --model anthropic/claude-haiku-4-5-20251001 -T prompt_version=v1
uv run inspect eval evals.py@listing_eval --model anthropic/claude-haiku-4-5-20251001 -T prompt_version=v2

# Reliability (3 epochs per property)
uv run inspect eval evals.py@reliability_eval --model anthropic/claude-haiku-4-5-20251001
```

### Reading the inspect view logs

In the browser UI (`inspect view`):
- **Left panel:** list of eval runs. The run name shows the task and timestamp.
- **Samples tab:** one row per property. Click to expand and see the generated copy
  and each scorer's verdict + explanation.
- **Scores column:** `grounding_scorer` is 0–1 (fraction of checks passed);
  `faithfulness_scorer` and `quality_scorer` are 1–5.
- **Usage tab:** token counts per model (Haiku for generation, Sonnet for judging).

Key runs to compare:
| Run | What to look for |
|---|---|
| `stub_eval variant=bad` | All three scorers should be low — confirms they catch failures |
| `stub_eval variant=good` | All three should be higher — confirms they pass good copy |
| `listing_eval prompt_v1` | faithfulness_scorer mean ~3.3; check property 102 (Eixample) and 106 (Alpine) |
| `listing_eval prompt_v2` | faithfulness_scorer improves to ~3.8; same properties now score higher |
| `reliability_eval` | 18 samples (6×3); grounding holds at 1.0 every run |

---

## How AI was used

Claude Code (claude-opus-4-8 and claude-sonnet-4-6) was used throughout:

- **Architecture design:** Claude explained inspect-ai's solver/scorer/task
  primitives and how they map onto the EDD requirements (Dataset → Solver chain →
  Scorers → committed `.eval` logs).
- **Code generation:** initial drafts of `models.py`, `scorers.py`, `generator.py`,
  `evals.py`, and `tests/test_pipeline.py` were generated by Claude and then
  reviewed and adjusted for correctness.
- **Fixture design:** Claude suggested the edge-case matrix (studio with misleading
  headline, zero-review property, null-policy property, HTML puffery) and the
  deliberate trap fixture design.
- **Debugging:** the stale `_SYSTEM_BASE` grounding instruction that prevented the
  EDD arc from manifesting was diagnosed via Claude by checking what context the
  model actually received. Similarly, the `ChatMessageSystem` API fix (vs. the
  rejected `config={"system": ...}` approach) was identified by Claude checking the
  inspect-ai source.
- **EDD iteration:** Claude read the per-sample faithfulness explanations from the
  eval log to diagnose the embellishment failure mode and design the anti-embellishment
  rule that became v2.

All design decisions (eval rubrics, scorer weights, prompt version strategy, fixture
edge cases) and all validation of the generated code were done by the human author.

---

## What I'd do with more time

- **Human eval:** recruit a small panel to score a sample of outputs on the same
  1–5 rubrics; calibrate the LLM judge against human scores and adjust the rubric
  wording where they diverge.
- **A/B testing:** deploy v1 and v2 to a shadow traffic split; measure real guest
  click-through and booking intent rather than proxy metrics.
- **Production monitoring:** track faithfulness and grounding on live model outputs
  using the same scorers, with an alert threshold (e.g. grounding < 0.95 → flag
  for human review).
- **Hallucination red-teaming:** adversarially design fixtures where the model is
  most likely to fabricate (very sparse data, misleading owner text in multiple
  languages, HTML with embedded fake stats).
- **Cost optimisation:** benchmark Haiku vs Sonnet on the eval suite; the eval logs
  show Haiku already scores 1.0 on grounding across all fixtures, so upgrading the
  generation model may not improve grounding and would cost 5× more.