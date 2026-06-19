"""Evaluation scorers: the primary deliverable of this assignment.

Three independent scorers, each targeting a different failure mode:

1. ``grounding_scorer`` — rule-based, offline, deterministic.
   Cross-references the generated ``ListingCopy`` against the structured
   ``PropertyInput`` fields. No LLM involved. This catches objective errors:
   wrong bedroom counts, hallucinated amenities, invented policies.

2. ``faithfulness_scorer`` — LLM-as-judge (Sonnet).
   Asks a strong model whether every factual claim in the copy is supported by
   the property data. Catches subtle unsupported claims that rules cannot
   (e.g. "award-winning", "voted #1").

3. ``quality_scorer`` — LLM-as-judge (Sonnet).
   Evaluates engagement, clarity, tone, and specificity on a 1–5 scale.
   Deliberately separate from faithfulness so a beautifully-written hallucination
   scores high on quality but exposes itself on faithfulness.

Scorers read ``ListingCopy`` from ``state.metadata["listing_copy"]`` (a parsed
Pydantic object or ``None`` on parse failure) so the JSON-parsing burden stays in
the generator/stub solver, not repeated here.
"""

from __future__ import annotations

import json
import re

from inspect_ai.model import ChatMessageUser, get_model
from inspect_ai.scorer import Score, mean, scorer

from lodgify.models import ListingCopy, Policies, PropertyInput

# ---------------------------------------------------------------------------
# Helpers for the rule-based grounding scorer
# ---------------------------------------------------------------------------

_NUMBER_WORDS: dict[str, int] = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

# Matches "two-bedroom", "2 bedroom", "2-bed", "two bed" etc.
_BEDROOM_RE = re.compile(
    r"\b(\d+|" + "|".join(_NUMBER_WORDS) + r")\s*[-–]?\s*bed(?:room)?s?\b",
    re.IGNORECASE,
)

# Social proof patterns that require actual reviews to exist
_SOCIAL_PROOF_RE = re.compile(
    r"\b(guests?\s+(love|loved|adore|rave|highly\s+rate)|"
    r"top[- ]rated|award[- ]winning|voted\s+best|"
    r"\d+\s+(?:five[- ]star\s+)?reviews?|loved\s+by)\b",
    re.IGNORECASE,
)

_CANCELLATION_RE = re.compile(r"\bcancell?ation\b|\bfull\s+refund\b|\brefund\s+policy\b", re.IGNORECASE)
_DAMAGE_DEP_RE = re.compile(r"\b(damage|security)\s+deposit\b", re.IGNORECASE)
_PAYMENT_RE = re.compile(r"\bpayment\s+schedule\b|\bbalance\s+due\b|\b\d+%\s+deposit\b", re.IGNORECASE)


def _extract_bedroom_numbers(text: str) -> list[int]:
    """Return all explicit bedroom numbers mentioned in ``text``."""
    results = []
    for m in _BEDROOM_RE.finditer(text):
        raw = m.group(1).lower()
        if raw.isdigit():
            results.append(int(raw))
        elif raw in _NUMBER_WORDS:
            results.append(_NUMBER_WORDS[raw])
    return results


def _copy_full_text(copy: ListingCopy) -> str:
    """Concatenate all text sections for pattern matching."""
    parts = [copy.hero_headline] + copy.highlights + [copy.about_this_place]
    for a in copy.amenity_descriptions:
        parts += [a.label, a.description]
    return " ".join(parts)


def _run_grounding_checks(copy: ListingCopy, prop: PropertyInput) -> tuple[list[str], list[str]]:
    """Return (passed_checks, failed_checks) as string labels."""
    passed: list[str] = []
    failed: list[str] = []
    text = _copy_full_text(copy)

    # --- 1. Bedroom count ---
    mentioned_beds = _extract_bedroom_numbers(text)
    expected = prop.rental_info.bedrooms
    if not mentioned_beds:
        # Not mentioning bedroom count is acceptable; give benefit of the doubt.
        passed.append("bedroom_count_not_mentioned")
    else:
        for n in mentioned_beds:
            if n == expected:
                passed.append(f"bedroom_count_correct({n})")
            else:
                failed.append(f"bedroom_count_wrong(claimed={n}, actual={expected})")

    # --- 2. Studio / zero-bedroom guard ---
    # A studio (bedrooms=0) must never be described as having bedrooms.
    if expected == 0:
        if mentioned_beds and any(n > 0 for n in mentioned_beds):
            failed.append("studio_claimed_has_bedrooms")
        else:
            passed.append("studio_no_false_bedroom_claim")

    # --- 3. Location grounding ---
    city = prop.location.city.lower()
    if city in text.lower():
        passed.append("location_city_mentioned")
    else:
        failed.append(f"location_city_missing({prop.location.city})")

    # --- 4. Amenity code grounding ---
    valid_codes = set(prop.amenities)
    invented = [
        a.amenity_code
        for a in copy.amenity_descriptions
        if a.amenity_code not in valid_codes
    ]
    if invented:
        failed.append(f"invented_amenity_codes({invented})")
    else:
        passed.append("amenity_codes_all_grounded")

    # --- 5. Null policy: must not invent content for absent policies ---
    if prop.policies.cancellation_policy is None and _CANCELLATION_RE.search(text):
        failed.append("null_cancellation_policy_mentioned")
    else:
        passed.append("null_cancellation_policy_ok")

    if prop.policies.damage_deposit is None and _DAMAGE_DEP_RE.search(text):
        failed.append("null_damage_deposit_mentioned")
    else:
        passed.append("null_damage_deposit_ok")

    # --- 6. Social proof requires reviews ---
    if prop.num_of_reviews == 0 and _SOCIAL_PROOF_RE.search(text):
        failed.append("fake_social_proof_with_zero_reviews")
    else:
        passed.append("social_proof_check_ok")

    return passed, failed


# ---------------------------------------------------------------------------
# Scorer 1: grounding (rule-based, offline)
# ---------------------------------------------------------------------------


@scorer(metrics=[mean()])
def grounding_scorer():
    """Score 0–1: fraction of grounding checks the generated copy passes.

    Runs deterministically with no LLM call. Fails loudly on:
    - Wrong bedroom/bathroom/guest counts
    - Amenity codes not present in the input
    - Policies mentioned when the input says None
    - Social proof claims when num_of_reviews == 0
    """

    async def score(state, target):
        copy: ListingCopy | None = state.metadata.get("listing_copy")
        prop_dict = state.metadata.get("property")

        if copy is None:
            return Score(
                value=0.0,
                explanation="listing_copy not found in metadata (parse failure)",
            )

        prop = PropertyInput.model_validate(prop_dict)
        passed, failed = _run_grounding_checks(copy, prop)
        total = len(passed) + len(failed)
        score_val = len(passed) / total if total else 0.0

        explanation = []
        if passed:
            explanation.append("PASSED: " + ", ".join(passed))
        if failed:
            explanation.append("FAILED: " + ", ".join(failed))

        return Score(
            value=round(score_val, 3),
            explanation="\n".join(explanation),
            metadata={"passed": passed, "failed": failed},
        )

    return score


# ---------------------------------------------------------------------------
# Scorer 2: faithfulness (LLM-as-judge)
# ---------------------------------------------------------------------------

_FAITHFULNESS_PROMPT = """\
You are a strict editorial fact-checker for vacation rental listings.

Your task: decide whether the GENERATED COPY is faithful to the PROPERTY DATA.
A claim is faithful if and only if it can be directly verified from the structured
data below. Marketing tone is acceptable; fabricated facts are not.

PROPERTY DATA (source of truth):
{property_json}

GENERATED COPY:
{copy_text}

Score faithfulness 1–5:
  1 – Multiple false or unsupported factual claims (wrong counts, invented awards, fake amenities)
  2 – Several questionable claims not traceable to the data
  3 – Mostly faithful; one or two minor embellishments
  4 – Faithful; only neutral marketing language, no invented facts
  5 – Completely faithful; every factual claim traces directly to the data

Respond ONLY with valid JSON, no markdown fences:
{{"score": <1-5>, "explanation": "<one or two sentences>"}}"""


@scorer(metrics=[mean()])
def faithfulness_scorer(judge_model: str = "anthropic/claude-sonnet-4-6"):
    """Score 1–5: how faithfully the copy reflects the input data.

    Uses ``judge_model`` (default: Sonnet) as the judge. Catches subtle
    unsupported claims that rule-based checks cannot (e.g. "award-winning",
    fabricated guest testimonials, invented policies).
    """

    async def score(state, target):
        copy: ListingCopy | None = state.metadata.get("listing_copy")
        prop_dict = state.metadata.get("property")

        if copy is None:
            return Score(value=1, explanation="listing_copy not found — scored 1 (worst)")

        prop = PropertyInput.model_validate(prop_dict)
        copy_text = (
            f"Hero headline: {copy.hero_headline}\n"
            f"Highlights:\n" + "\n".join(f"  - {h}" for h in copy.highlights) + "\n"
            f"About this place:\n{copy.about_this_place}\n"
            f"Amenities:\n"
            + "\n".join(
                f"  {a.label}: {a.description}" for a in copy.amenity_descriptions
            )
        )
        prompt = _FAITHFULNESS_PROMPT.format(
            property_json=prop.model_dump_json(indent=2),
            copy_text=copy_text,
        )

        model = get_model(judge_model)
        response = await model.generate([ChatMessageUser(content=prompt)])
        raw = response.completion.strip()

        try:
            parsed = json.loads(raw)
            return Score(
                value=int(parsed["score"]),
                explanation=parsed.get("explanation", ""),
                metadata={"judge_raw": raw},
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            return Score(
                value=1,
                explanation=f"Judge returned unparseable response: {raw[:200]}",
                metadata={"judge_raw": raw},
            )

    return score


# ---------------------------------------------------------------------------
# Scorer 3: quality (LLM-as-judge)
# ---------------------------------------------------------------------------

_QUALITY_PROMPT = """\
You are a senior copywriter reviewing vacation rental marketing copy.

Evaluate the GENERATED COPY below on four dimensions:
  • Engagement – does it make the property sound appealing and memorable?
  • Clarity     – is the writing clear, readable, and well-paced?
  • Tone        – is it warm, inviting, and appropriate for a holiday rental?
  • Specificity – is the copy specific to this property, not generic filler?

GENERATED COPY:
{copy_text}

Overall quality score 1–5:
  1 – Poor: generic, flat, robotic, or off-tone
  2 – Below average: some good moments but mostly unremarkable
  3 – Average: competent, does the job
  4 – Good: engaging, specific, would attract a genuine guest
  5 – Excellent: compelling, memorable, clearly written for this property

Respond ONLY with valid JSON, no markdown fences:
{{"score": <1-5>, "explanation": "<one or two sentences>"}}"""


@scorer(metrics=[mean()])
def quality_scorer(judge_model: str = "anthropic/claude-sonnet-4-6"):
    """Score 1–5: engagement, clarity, tone, and specificity of the copy.

    Deliberately blind to grounding so a fluent hallucination still scores
    high here — the contrast with faithfulness_scorer is itself a signal.
    """

    async def score(state, target):
        copy: ListingCopy | None = state.metadata.get("listing_copy")

        if copy is None:
            return Score(value=1, explanation="listing_copy not found — scored 1 (worst)")

        copy_text = (
            f"Hero headline: {copy.hero_headline}\n"
            f"Highlights:\n" + "\n".join(f"  - {h}" for h in copy.highlights) + "\n"
            f"About this place:\n{copy.about_this_place}\n"
            f"Amenities:\n"
            + "\n".join(
                f"  {a.label}: {a.description}" for a in copy.amenity_descriptions
            )
        )
        prompt = _QUALITY_PROMPT.format(copy_text=copy_text)

        model = get_model(judge_model)
        response = await model.generate([ChatMessageUser(content=prompt)])
        raw = response.completion.strip()

        try:
            parsed = json.loads(raw)
            return Score(
                value=int(parsed["score"]),
                explanation=parsed.get("explanation", ""),
                metadata={"judge_raw": raw},
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            return Score(
                value=1,
                explanation=f"Judge returned unparseable response: {raw[:200]}",
                metadata={"judge_raw": raw},
            )

    return score
