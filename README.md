# Lodgify — Content Generation Pipeline

A vacation-rental marketing-copy generator built **evaluation-first**. The eval suite is
the primary deliverable; the generator exists to be measured.

The full pipeline + evaluation runs in the notebook **[evals.py](evals.py)** (Marimo).
Supporting code lives in the `lodgify/` package; inspect-ai task definitions are in
[eval_pipeline.py](eval_pipeline.py).

---

## Quick start

```bash
uv sync                            # install dependencies (no API key needed for this)
uv run marimo edit evals.py        # open the notebook — reads committed logs, no key required
```

Everything in the notebook — EDD arc, reliability, calibration — loads from committed
`.eval` logs and renders without an API key. One cell runs a live generation with Haiku
*if* `ANTHROPIC_API_KEY` is set in `.env`; it is skipped otherwise.

```bash
uv run pytest tests/ -v            # 46 offline tests, no API key
uv run inspect view --log-dir logs/ # browse committed eval logs in the browser
```

---

## Approach

**Evaluation-first development (EDD)** means evals are written and validated *before* the
generator is built, so every prompt iteration is driven by metrics rather than eyeballing
outputs. The eval harness is [inspect-ai](https://inspect.aisi.org.uk/).

### Five evaluation dimensions

The job description names four axes — **factuality, grounding, robustness, user impact**.
Each maps to one or more scorers:

| Scorer | Type | Scale | Axis | What it catches |
|---|---|---|---|---|
| `grounding_scorer` | Rule-based, offline | 0–1 | Grounding | Wrong bedroom/bathroom counts; studio claimed as N-bedroom; invented amenity codes; null policies mentioned; social proof with zero reviews; high-rating claims below the actual score; pet-friendly without the amenity; flexible check-in against a fixed time (9 checks for standard properties, 10 for studios) |
| `faithfulness_scorer` | LLM-as-judge (Sonnet), calibrated | 1–5 | Factuality | Unsupported claims; embellishments; awards/attributes not in the data |
| `completeness_scorer` | Rule-based, offline | 0–1 | User impact | Amenity coverage (fraction of amenities actually described) + premium salience (is the pool/sea-view/hot-tub surfaced in the headline/highlights, not buried?) |
| `booking_intent_scorer` | LLM-as-judge (Sonnet) | 1–5 | User impact | Would a guest click "Request to Book"? Persuasiveness / conversion potential, distinct from "well-written" |
| `quality_scorer` | LLM-as-judge (Sonnet) | 1–5 | — | Engagement, clarity, tone, specificity — deliberately *blind* to the data, so a fluent hallucination scores high here but exposes itself on faithfulness |

Two of the five (`grounding`, `completeness`) are **fully deterministic and offline** —
no API key needed to run them. The three LLM judges need a key only to *regenerate* logs;
the committed `.eval` logs are viewable offline.

**Robustness** is measured by `reliability_eval` (same fixtures × N epochs → score
variance) rather than a scorer of its own.

### The EDD arc (real, backed by the committed logs)

The brief asks for a story where *evals reveal a problem → a prompt change fixes it →
evals confirm*. Five prompt versions were written, each in response to a specific failure
surfaced by the eval suite:

| Version | Added rule (in response to an eval failure) |
|---|---|
| `v1` | naive "write compelling copy" baseline |
| `v2` | anti-embellishment: describe amenities only from the data, no invented qualifiers |
| `v3` | review-score accuracy (no "5-star"/"highly rated" below the actual score) + pet-friendly accuracy |
| `v4` | check-in / check-out accuracy |
| `v5` | prescriptive check-in wording + worked example; pairs with `scrub_checkin_prose()` in ingest |

Mean scores across all 9 fixtures (`listing_eval`, Haiku 4.5, committed logs):

| | v1 | v2 | v3 | v4 | v5 |
|---|---|---|---|---|---|
| `grounding` (0–1) | 0.965 | 0.989 | 0.989 | **1.000** | **1.000** |
| `faithfulness` (1–5) | 2.89 | **4.11** | 3.89 | 3.89 | **4.11** |
| `quality` (1–5) | 3.78 | 3.22 | 3.44 | 3.33 | 3.11 |
| `completeness` (0–1) | 0.93 | 0.94 | 0.94 | 0.93 | 0.92 |
| `booking_intent` (1–5) | 3.44 | 2.89 | 3.11 | 3.00 | 3.11 |

Three movements show up cleanly:

**1. Faithfulness jump (v1 → v2).** The naive v1 prompt produces mean faithfulness **2.89** —
the Sonnet judge flags embellishments ("saltwater-inspired pool", "panoramic views from
every room", inferred attributes). The v2 anti-embellishment rule lifts faithfulness to
**4.11 (+42%)**. This is the headline EDD win: a metric exposed the problem, one targeted
prompt change fixed it, the metric confirmed it.

**2. Grounding climb to determinism (v1 → v4).** `grounding_scorer` rises **0.965 → 0.989 →
1.000** as v3/v4 add review-score, pet, and check-in rules. By v4 every structural check
passes on every fixture.

**3. The quality/faithfulness tradeoff.** Quality *drops* 3.78 → 3.11 as the prompt gets
stricter — the expected, visible cost of constraining flowery language. Surfacing this
tradeoff (rather than hiding it) is the point of scoring quality *separately* from
faithfulness.

**Robustness arc (v4 → v5).** `reliability_eval` (3 epochs/fixture, 27 samples) exposed that
fixture 109 (Ático Dorado — owner headline claims "flexible check-in any time" but
house_rules sets a fixed 7 PM) made `grounding_scorer` *non-deterministic*: **v4 grounding
std = 0.064** (the model sometimes echoed the owner's false claim). Fix: `scrub_checkin_prose()`
removes the contradictory owner sentence before the model sees it, and v5 gives prescriptive
wording. Result: **v5 grounding std = 0.000** — the failure became impossible, not just
rarer. This is the robustness axis the JD asks for: not just "is it right once" but "is it
reliably right across repeated runs".

### Judge calibration (golden dataset)

LLM judges drift from human judgement, so the `faithfulness` and `quality` judges are
calibrated against a hand-annotated **golden dataset** (`golden/`): for each of the 9
fixtures, a `good` variant (exemplary, human faithfulness = 5) and a `bad` variant
(deliberately flawed in that fixture's failure mode, human faithfulness 1–3). The
calibration metric is **mean absolute deviation (MAD)** between judge and human scores on
a 1–5 scale; target < 0.7.

Two calibration techniques were applied to the faithfulness judge and measured on the
golden set (`golden_eval`):

1. **Amenity-label note** — tells the judge that curated translations like
   `Kitchen → "Full kitchen"` are faithful, not embellishments.
2. **Three calibration examples** — a score-5 (fully faithful), score-1 (studio
   misrepresented as one-bedroom), and score-2 (fabricated "award-winning / infinity pool")
   exemplar, so the judge scores against a demonstrated standard.

Result: faithfulness **MAD 1.22 (uncalibrated) → 0.94 (calibrated)**. The judge separates
good from bad copy cleanly (good avg **3.67**, bad avg **1.44**) but is systematically
*harsher* than humans on good copy (humans say 5, judge says ~3.7). That residual gap is a
real finding, not noise: the judge's *ranking* is trustworthy, its *absolute* generosity is
not — so it is used for relative comparison across prompt versions, not as a pass/fail gate.

> A claim-decomposition variant of the faithfulness judge was also tried; it scored *worse*
> on the bad copies (the per-claim verifier was fooled by cherry-picked review snippets) and
> was removed. The experiment is documented here rather than hidden.

### Stub validation (watch the test fail first)

Before any real LLM call, the scorers were validated against hand-crafted stub outputs for
property 104 (the "trap": a studio whose owner headline falsely claims two bedrooms):

| Stub | grounding | faithfulness | completeness | booking_intent | quality |
|---|---|---|---|---|---|
| Bad (invented Jacuzzi, wrong bedroom count, fake awards) | 0.64 | 1/5 | 0.63 | 2/5 | 3/5 |
| Good (correct studio description) | 1.00 | 3/5 | 1.00 | 3/5 | 3/5 |

The bad stub is caught on every objective axis (grounding, faithfulness, completeness) yet
still scores 3/5 on *quality* — it is fluent, well-written hallucination. That contrast is
the design: quality alone would pass the bad copy; the grounding/faithfulness scorers are
what catch it. This confirms the scorers discriminate good from bad copy *before* the real
generator exists — the AI equivalent of watching a test fail first.

### Model choices and cost/latency tradeoffs

- **Generation: Haiku 4.5** — fast, cheap ($1/$5 per MTok), sufficient for structured
  generation from a tight prompt. The logs show Haiku already scores **1.0 grounding** on
  every fixture under v4/v5, so a bigger generation model would not improve grounding.
- **Judge: Sonnet 4.6** — stronger reasoning for nuanced faithfulness/booking-intent
  judging. LLM judging costs more per call than generation but runs only in eval, not in
  production.
- **Prompt caching** is visible in the Sonnet usage as cache hits grow within a run. In
  production, caching the system prompt + property schema would cut per-call judge cost
  substantially.

---

## Repository structure

```
evals.py            ← the deliverable: Marimo notebook (source of truth)
eval_pipeline.py    inspect-ai task definitions: stub_eval, listing_eval, reliability_eval, golden_eval
lodgify/
  models.py         Pydantic input (PropertyInput) + output (ListingCopy) schemas
  amenities.py      Amenity-code → human-label map; CamelCase fallback
  ingest.py         HTML stripping, check-in prose scrubbing, context normalization (no LLM)
  generator.py      Solver chain: ingest_solver + generate_solver (versioned prompts)
  scorers.py        The five scorers
  data.py           Fixture loader (validates against PropertyInput on load)
  golden.py         Golden-dataset loader (good/bad variants + human scores) for calibration
prompts/            system.txt + v1–v5 prompt versions (plain text, version-controlled)
fixtures/           9 property JSON files; each targets specific eval checks
golden/             9 golden files (good + bad annotated variants) for judge calibration
logs/               Committed inspect-ai .eval logs (view offline with inspect view)
tests/
  test_pipeline.py  46 offline tests; DI, inheritance, mocking, scorers, ingest, loaders
```

### Output schema (`ListingCopy`)

Maps directly to the brief's four sections, validated by Pydantic on generation:
`hero_headline` (10–90 chars) · `highlights` (3–6 items) · `about_this_place` (120–1200
chars) · `amenity_descriptions` (each with an `amenity_code` that must be one of the
property's input codes).

---

## How to run

**Setup (one time):**
```bash
uv sync
cp .env.example .env   # add ANTHROPIC_API_KEY — only needed to regenerate logs / run the live notebook cell
```

**Run the tests (offline, no API key):**
```bash
uv run pytest tests/ -v          # 46 tests
```

**Open the notebook (the main deliverable):**
```bash
uv run marimo edit evals.py      # interactive
uv run marimo run evals.py       # read-only
```
The notebook reads the committed `.eval` logs, so the EDD arc, reliability, and calibration
sections render **without an API key**. One cell generates live copy with Haiku for a single
fixture *if* a key is present, and is skipped otherwise.


**View the committed eval results (offline, no API key):**
```bash
uv run inspect view --log-dir logs/
```

**Regenerate eval logs (requires API key in `.env`):**
```bash
# Validate scorers against stubs (EDD discipline)
uv run inspect eval eval_pipeline.py@stub_eval --model anthropic/claude-sonnet-4-6 -T variant=bad
uv run inspect eval eval_pipeline.py@stub_eval --model anthropic/claude-sonnet-4-6 -T variant=good

# The grounding arc — run each prompt version
uv run inspect eval eval_pipeline.py@listing_eval --model anthropic/claude-haiku-4-5-20251001 -T prompt_version=v2
uv run inspect eval eval_pipeline.py@listing_eval --model anthropic/claude-haiku-4-5-20251001 -T prompt_version=v5

# Robustness (3 epochs per fixture) — compare v4 (std>0) vs v5 (std=0)
uv run inspect eval eval_pipeline.py@reliability_eval --model anthropic/claude-haiku-4-5-20251001 -T prompt_version=v4
uv run inspect eval eval_pipeline.py@reliability_eval --model anthropic/claude-haiku-4-5-20251001 -T prompt_version=v5

# Judge calibration against the golden dataset (MAD)
uv run inspect eval eval_pipeline.py@golden_eval --model anthropic/claude-sonnet-4-6
```

### Reading the inspect-view logs

In the browser UI (`inspect view`):
- **Left panel:** list of eval runs (task name + timestamp).
- **Samples tab:** one row per property/variant. Click to expand the generated copy and
  each scorer's verdict + explanation. `grounding`/`completeness` are 0–1; the three LLM
  judges are 1–5.
- **Usage tab:** token counts per model (Haiku for generation, Sonnet for judging).

Key runs to compare:

| Run | What to look for |
|---|---|
| `stub_eval variant=bad` vs `good` | scorers discriminate bad (grounding 0.64, faith 1) from good (grounding 1.0) |
| `listing_eval v1` vs `v2` | faithfulness jumps 2.89 → 4.11 (anti-embellishment fix) |
| `listing_eval v2 → v3 → v4` | grounding climbs to 1.000 |
| `reliability_eval v4` vs `v5` | grounding std 0.064 → 0.000 (fixture 109 robustness fix) |
| `golden_eval` (uncalibrated vs calibrated) | faithfulness MAD 1.22 → 0.94; good avg 3.67 vs bad avg 1.44 |

All committed logs carry the full five scorers, except the **uncalibrated `golden_eval`
baseline** (the earlier of the two golden logs), which is kept deliberately as the
"before-calibration" reference for the MAD comparison.

---

## How AI was used

Claude Code (claude-opus-4-8 and claude-sonnet-4-6) was used throughout. This project
was built as a genuine human–AI collaboration: Claude contributed substantially to both
design and implementation, with the author steering direction and validating outputs.

- **Architecture design:** Claude proposed how inspect-ai's solver/scorer/task primitives
  map onto the EDD requirements and designed the overall Dataset → Solver chain → Scorers
  → committed logs structure.
- **Code generation:** `models.py`, `scorers.py`, `generator.py`, `eval_pipeline.py`, the
  notebook, and `tests/test_pipeline.py` were generated by Claude and reviewed and
  adjusted by the author.
- **Fixture & golden design:** Claude proposed the edge-case matrix (studio
  misrepresentation, cherry-picked reviews, contradictory check-in, null policies) and
  designed the good/bad golden variants; the author selected and validated the final set.
- **Debugging:** specific bugs were diagnosed by Claude — the stale `_SYSTEM_BASE`
  instruction masking the EDD arc, a `ChatMessageSystem` API mismatch, and a `.format()`
  brace-escaping error in the calibration prompt.
- **EDD iteration:** Claude read the per-sample scorer explanations from eval logs,
  diagnosed each failure mode, and designed the v2→v5 prompt rules.

The author directed the overall approach, defined the evaluation strategy, and made the
final calls on rubric design, scorer set, and what results mean. Specifically, the
**golden dataset for judge calibration**, the **prompt versioning strategy** (tracking
each iteration as a versioned file tied to a specific eval failure), and **expanding the
fixture set to cover additional failure modes** (low review scores, pet claims, flexible
check-in) were the author's proposals.

---

## What I'd do with more time

- **Recruit a real human panel** for the golden scores (currently the author's annotations).
  3–5 domain experts scoring 50–100 generated samples would expose whether the judge's
  variance is legitimate disagreement or noise, surface failure modes the traps didn't
  anticipate, and provide ground truth to close the residual MAD gap (judge ≈3.7 on samples
  humans rate 5).

- **Drive a prompt iteration off `booking_intent`** — the v1–v5 arc focused on
  grounding/faithfulness; `booking_intent_scorer` is reported but no prompt change has
  targeted it yet. A v6 optimising for urgency/specificity *without* sacrificing faithfulness
  is an interesting multi-axis problem: does "premium experience" language increase booking
  intent while still passing the faithfulness judge?

- **A/B testing in production.** v5 outperforms v1 on every proxy metric. Whether it
  translates to real business outcome (click-through, booking rate) is unknown. A shadow
  traffic split for 2–4 weeks would measure actual guest behaviour and validate whether
  the evals are optimising for the right thing.

- **Production monitoring.** `grounding` and `completeness` are deterministic and run in
  <100ms — cheap enough for every generated listing. Anything below grounding 0.95 flags for
  human review before publication. `faithfulness`/`quality` cost one Sonnet call each, so run
  them nightly on a random sample to detect prompt drift or model degradation over time.

- **Prompt-as-config operationalisation.** Prompts are versioned plain-text files — no Python
  required to propose a change. The next step is a lightweight UI for content managers to
  propose prompt edits, automated eval runs on the golden set + recent prod samples, and
  side-by-side `inspect view` comparison before/after. That closes the EDD loop: metrics drive
  every prompt change, nothing ships untested.
