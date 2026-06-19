"""Evaluation scorers — five independent dimensions.

Each scorer targets a distinct quality axis so failures are diagnosable:

1. ``grounding_scorer`` — rule-based, offline, deterministic, 0–1.
   Structural correctness: bedroom counts, amenity codes, null policies, social
   proof without reviews, inflated ratings, pet/check-in claims.

2. ``faithfulness_scorer`` — LLM-as-judge (Sonnet), calibrated, 1–5.
   Factual accuracy: does every claim trace to the structured input? Calibrated
   with an amenity label note (labels like "Full kitchen" are faithful translations
   of internal codes, not embellishments) and 3 golden-dataset calibration examples.

3. ``completeness_scorer`` — rule-based, offline, deterministic, 0–1.
   Commercial coverage: does the copy describe enough of the property's amenities,
   and does it surface premium features (pool, hot tub, sea view) in the headline
   or highlights where a guest will see them first?

4. ``booking_intent_scorer`` — LLM-as-judge (Sonnet), 1–5.
   User impact: would a potential guest click "Request to Book" after reading this
   copy? Measures persuasiveness and conversion potential — distinct from quality
   (well-written) and faithfulness (accurate). A copy can be both accurate and
   well-written but still fail to create urgency for this specific property.

5. ``quality_scorer`` — LLM-as-judge (Sonnet), 1–5.
   Writing quality: engagement, clarity, tone, specificity. Deliberately blind to
   the property data so a fluent hallucination scores high here but exposes itself
   on faithfulness — the contrast between the two is itself a signal.

All scorers read ``ListingCopy`` from ``state.metadata["listing_copy"]`` (a parsed
Pydantic object or ``None`` on parse failure).
"""

from __future__ import annotations

import json
import re

from inspect_ai.model import ChatMessageUser, get_model
from inspect_ai.scorer import Score, mean, scorer, std

from lodgify.models import ListingCopy, PropertyInput

# ---------------------------------------------------------------------------
# Shared copy formatter
# ---------------------------------------------------------------------------


def _format_copy_text(copy: ListingCopy) -> str:
    """Format a ListingCopy as readable text for LLM judges."""
    parts = [
        f"Hero headline: {copy.hero_headline}",
        "Highlights:\n" + "\n".join(f"  - {h}" for h in copy.highlights),
        f"About this place:\n{copy.about_this_place}",
        "Amenities:\n" + "\n".join(
            f"  {a.label}: {a.description}" for a in copy.amenity_descriptions
        ),
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Helpers for the rule-based grounding scorer
# ---------------------------------------------------------------------------

_NUMBER_WORDS: dict[str, int] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

_BEDROOM_RE = re.compile(
    r"\b(\d+|" + "|".join(_NUMBER_WORDS) + r")\s*[-–]?\s*bed(?:room)?s?\b",
    re.IGNORECASE,
)

_SOCIAL_PROOF_RE = re.compile(
    r"\b(guests?\s+(love|loved|adore|rave|highly\s+rate)|"
    r"top[- ]rated|award[- ]winning|voted\s+best|"
    r"\d+\s+(?:five[- ]star\s+)?reviews?|loved\s+by)\b",
    re.IGNORECASE,
)

_CANCELLATION_RE = re.compile(
    r"\bcancell?ation\b|\bfull\s+refund\b|\brefund\s+policy\b", re.IGNORECASE
)
_DAMAGE_DEP_RE = re.compile(r"\b(damage|security)\s+deposit\b", re.IGNORECASE)

_HIGH_RATING_RE = re.compile(
    r"\b(5[- ]star|five[- ]star|top[- ]rated|highly[- ]rated|"
    r"consistently\s+(?:highly\s+)?(?:rated|praised|reviewed)|"
    r"glowing\s+reviews?|rave\s+reviews?|loved\s+by\s+(?:all\s+)?guests?)\b",
    re.IGNORECASE,
)

_PET_CLAIM_RE = re.compile(
    r"\b(pet[- ]friendly|dog[- ]friendly|cat[- ]friendly|"
    r"pets?\s+(?:welcome|allowed|friendly)|four[- ]legged\s+friends?\s+welcome)\b",
    re.IGNORECASE,
)

_FLEXIBLE_CHECKIN_RE = re.compile(
    r"\b(flexible\s+(?:check[- ]?in|arrival)|"
    r"24[- ]hour\s+check[- ]?in|check[- ]?in\s+any\s+time|arrive\s+any\s+time)\b",
    re.IGNORECASE,
)


def _extract_bedroom_numbers(text: str) -> list[int]:
    results = []
    for m in _BEDROOM_RE.finditer(text):
        raw = m.group(1).lower()
        if raw.isdigit():
            results.append(int(raw))
        elif raw in _NUMBER_WORDS:
            results.append(_NUMBER_WORDS[raw])
    return results


def _copy_full_text(copy: ListingCopy) -> str:
    parts = [copy.hero_headline] + copy.highlights + [copy.about_this_place]
    for a in copy.amenity_descriptions:
        parts += [a.label, a.description]
    return " ".join(parts)


def _run_grounding_checks(copy: ListingCopy, prop: PropertyInput) -> tuple[list[str], list[str]]:
    """Return (passed_checks, failed_checks) as string labels."""
    passed: list[str] = []
    failed: list[str] = []
    text = _copy_full_text(copy)

    mentioned_beds = _extract_bedroom_numbers(text)
    expected = prop.rental_info.bedrooms
    if not mentioned_beds:
        passed.append("bedroom_count_not_mentioned")
    else:
        for n in mentioned_beds:
            if n == expected:
                passed.append(f"bedroom_count_correct({n})")
            else:
                failed.append(f"bedroom_count_wrong(claimed={n}, actual={expected})")

    if expected == 0:
        if mentioned_beds and any(n > 0 for n in mentioned_beds):
            failed.append("studio_claimed_has_bedrooms")
        else:
            passed.append("studio_no_false_bedroom_claim")

    city = prop.location.city.lower()
    if city in text.lower():
        passed.append("location_city_mentioned")
    else:
        failed.append(f"location_city_missing({prop.location.city})")

    valid_codes = set(prop.amenities)
    invented = [a.amenity_code for a in copy.amenity_descriptions if a.amenity_code not in valid_codes]
    if invented:
        failed.append(f"invented_amenity_codes({invented})")
    else:
        passed.append("amenity_codes_all_grounded")

    if prop.policies.cancellation_policy is None and _CANCELLATION_RE.search(text):
        failed.append("null_cancellation_policy_mentioned")
    else:
        passed.append("null_cancellation_policy_ok")

    if prop.policies.damage_deposit is None and _DAMAGE_DEP_RE.search(text):
        failed.append("null_damage_deposit_mentioned")
    else:
        passed.append("null_damage_deposit_ok")

    if prop.num_of_reviews == 0 and _SOCIAL_PROOF_RE.search(text):
        failed.append("fake_social_proof_with_zero_reviews")
    else:
        passed.append("social_proof_check_ok")

    if _HIGH_RATING_RE.search(text) and prop.average_review_score < 4.0:
        failed.append(f"high_rating_claimed_with_score({prop.average_review_score:.1f})")
    else:
        passed.append("rating_claim_ok")

    if _PET_CLAIM_RE.search(text) and "PetFriendly" not in prop.amenities:
        failed.append("pet_friendly_claimed_without_amenity")
    else:
        passed.append("pet_claim_ok")

    _vague_times = {"flexible", "anytime", "any time", "24h", "24 hour", "whenever"}
    checkin_is_fixed = prop.house_rules.check_in_time.lower() not in _vague_times
    if checkin_is_fixed and _FLEXIBLE_CHECKIN_RE.search(text):
        failed.append(f"flexible_checkin_claimed_but_time_is({prop.house_rules.check_in_time!r})")
    else:
        passed.append("checkin_claim_ok")

    return passed, failed


# ---------------------------------------------------------------------------
# Scorer 1: grounding (rule-based, offline)
# ---------------------------------------------------------------------------


@scorer(metrics=[mean(), std()])
def grounding_scorer():
    """Score 0–1: fraction of grounding checks the generated copy passes."""

    async def score(state, target):
        copy: ListingCopy | None = state.metadata.get("listing_copy")
        prop_dict = state.metadata.get("property")

        if copy is None:
            return Score(value=0.0, explanation="listing_copy not found (parse failure)")

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
# Scorer 2: faithfulness (LLM-as-judge, calibrated)
# Options 1 + 2: amenity label note + calibration examples from golden dataset
# ---------------------------------------------------------------------------

_FAITHFULNESS_PROMPT = """\
You are a strict editorial fact-checker for vacation rental listings.

Your task: decide whether the GENERATED COPY is faithful to the PROPERTY DATA.
A claim is faithful if and only if it can be directly verified from the structured
data below. Marketing tone is acceptable; fabricated facts are not.

AMENITY LABEL NOTE (Option 1 — calibration fix):
The property data contains internal amenity codes (e.g. Kitchen, WiFi, HotTub,
PrivatePool). These are pre-translated to guest-facing labels before copy is
generated using a curated map:
  Kitchen        → "Full kitchen"          WiFi     → "Wi-Fi"
  HotTub         → "Hot tub"               PrivatePool → "Private pool"
  InternetBroadband → "High-speed internet"  Heating → "Heating"
These label translations are FAITHFUL — do not penalise them as embellishments.
Only flag claims that go BEYOND the label (e.g. "infinity pool" for PrivatePool,
"gourmet chef's kitchen" for Kitchen, "gigabit fibre" for WiFi).

CALIBRATION EXAMPLES (Option 2 — score by demonstrated standard):

--- Example A: score 5 (completely faithful) ---
Property data (key fields):
  rental_info: bedrooms=4, bathrooms=3, max_guests=8
  amenities: ["PrivatePool", "Kitchen", "SeaView", "BBQGrill", "Terrace"]
  location: Begur, Spain
  average_review_score: 4.96, num_of_reviews: 47

Copy:
  Hero: "Seafront Villa with Private Pool on the Costa Brava"
  Highlights: "Private pool with sea views", "4 bedrooms sleeping up to 8 guests",
    "Outdoor BBQ and shaded terrace"
  About: "...four bedrooms sleep eight guests across three bathrooms. A fully
    equipped kitchen and outdoor BBQ make it easy to host long group dinners..."
  Amenities: Full kitchen: "Fully equipped kitchen for group meals."
    Private pool: "A full-size private pool on the terrace, directly overlooking the sea."

{{"score": 5, "explanation": "All claims trace to structured fields. '4 bedrooms / 8 guests / 3 bathrooms' match rental_info exactly. 'Full kitchen' is the canonical label for the Kitchen code — faithful, not an embellishment. 'Private pool' matches PrivatePool with no added style qualifier. Every amenity described is in the amenities list."}}

--- Example B: score 1 (clear structural misrepresentation) ---
Property data (key fields):
  rental_info: bedrooms=0, max_guests=2   ← studio, zero separate bedrooms
  property_type: NormalApartment

Copy:
  Hero: "Charming One-Bedroom Apartment in Central Lisbon"
  Highlights: "Private bedroom and living area — ideal for couples"
  About: "This charming one-bedroom apartment..."

{{"score": 1, "explanation": "rental_info.bedrooms=0 means this is a studio with no separate bedroom. Describing it as a 'one-bedroom apartment' directly contradicts the structured data. This is not a marketing tone issue — it is a factual misrepresentation of the property type."}}

--- Example C: score 2 (fabricated attributes) ---
Property data (key fields):
  amenities: ["PrivatePool", "Kitchen", "SeaView"]
  (no award, no infinity design, no architect mentioned in any field)
  num_of_reviews: 47, average_review_score: 4.96

Copy:
  Hero: "Award-Winning Villa with Infinity Pool on the Costa Brava"
  About: "This award-winning villa...architect-designed infinity pool that dissolves
    into the sea...chef's kitchen..."

{{"score": 2, "explanation": "'Award-winning' appears in no structured field. 'Infinity pool' adds a design attribute not in the data — PrivatePool is listed but carries no style qualifier. 'Chef's kitchen' goes beyond the Kitchen label. Three distinct fabricated attributes."}}

---

PROPERTY DATA (source of truth):
{property_json}

GENERATED COPY:
{copy_text}

Score faithfulness 1–5:
  1 – Multiple false or unsupported factual claims
  2 – Several questionable claims not traceable to the data
  3 – Mostly faithful; one or two minor embellishments
  4 – Faithful; only neutral marketing language, no invented facts
  5 – Completely faithful; every factual claim traces directly to the data

Respond ONLY with valid JSON, no markdown fences:
{{"score": <1-5>, "explanation": "<one or two sentences>"}}"""


@scorer(metrics=[mean(), std()])
def faithfulness_scorer(judge_model: str = "anthropic/claude-sonnet-4-6"):
    """Score 1–5: how faithfully the copy reflects the input data.

    Calibrated via:
    - Option 1: amenity label note (prevents penalising curated translations)
    - Option 2: 3 calibration examples from the golden dataset
    """

    async def score(state, target):
        copy: ListingCopy | None = state.metadata.get("listing_copy")
        prop_dict = state.metadata.get("property")

        if copy is None:
            return Score(value=1, explanation="listing_copy not found — scored 1 (worst)")

        prop = PropertyInput.model_validate(prop_dict)
        prompt = _FAITHFULNESS_PROMPT.format(
            property_json=prop.model_dump_json(indent=2),
            copy_text=_format_copy_text(copy),
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


@scorer(metrics=[mean(), std()])
def quality_scorer(judge_model: str = "anthropic/claude-sonnet-4-6"):
    """Score 1–5: engagement, clarity, tone, and specificity of the copy.

    Deliberately blind to grounding so a fluent hallucination still scores
    high here — the contrast with faithfulness_scorer is itself a signal.
    """

    async def score(state, target):
        copy: ListingCopy | None = state.metadata.get("listing_copy")

        if copy is None:
            return Score(value=1, explanation="listing_copy not found — scored 1 (worst)")

        model = get_model(judge_model)
        response = await model.generate(
            [ChatMessageUser(content=_QUALITY_PROMPT.format(copy_text=_format_copy_text(copy)))]
        )
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
# Scorer 4: completeness (rule-based, offline)
# ---------------------------------------------------------------------------

# Amenity codes considered premium selling points that should appear
# in the headline or highlights (the first content a guest reads), not
# just buried in amenity_descriptions.
_PREMIUM_AMENITY_KEYWORDS: dict[str, list[str]] = {
    "PrivatePool":   ["pool"],
    "SwimmingPool":  ["pool"],
    "HotTub":        ["hot tub", "jacuzzi", "spa"],
    "SeaView":       ["sea view", "ocean view", "bay view", "sea views"],
    "MountainView":  ["mountain view", "mountain views", "alpine view"],
    "Terrace":       ["terrace", "roof terrace", "rooftop"],
    "Balcony":       ["balcony"],
    "Garden":        ["garden"],
    "Fireplace":     ["fireplace", "log fire", "wood fire", "wood-burning"],
    "BBQGrill":      ["bbq", "barbecue", "grill"],
}


@scorer(metrics=[mean(), std()])
def completeness_scorer():
    """Score 0–1: how completely the copy covers the property's amenities.

    Two deterministic sub-scores, averaged:

    amenity_coverage (0–1):
        Fraction of the property's amenity codes that have a corresponding entry
        in copy.amenity_descriptions. Catches cases where the model is technically
        grounded but skips half the property.

    premium_salience (0–1):
        Fraction of premium amenities (pool, hot tub, sea view, terrace, etc.) that
        appear explicitly in the hero_headline or highlights. A private pool buried
        only in amenity_descriptions is a missed commercial opportunity — guests
        decide from the top of the page. Properties with no premium amenities score
        1.0 on this dimension (nothing to check).
    """

    async def score(state, target):
        copy: ListingCopy | None = state.metadata.get("listing_copy")
        prop_dict = state.metadata.get("property")

        if copy is None:
            return Score(value=0.0, explanation="listing_copy not found")

        prop = PropertyInput.model_validate(prop_dict)
        all_codes = set(prop.amenities)

        # --- Component 1: amenity coverage ---
        described_codes = {a.amenity_code for a in copy.amenity_descriptions}
        covered = described_codes & all_codes
        coverage = len(covered) / len(all_codes) if all_codes else 1.0
        missing = all_codes - described_codes

        # --- Component 2: premium amenity salience ---
        premium_present = [c for c in prop.amenities if c in _PREMIUM_AMENITY_KEYWORDS]
        if premium_present:
            top_text = (copy.hero_headline + " " + " ".join(copy.highlights)).lower()
            salient = [
                code for code in premium_present
                if any(kw in top_text for kw in _PREMIUM_AMENITY_KEYWORDS[code])
            ]
            salience = len(salient) / len(premium_present)
            not_salient = [c for c in premium_present if c not in salient]
        else:
            salience = 1.0
            salient, not_salient = [], []

        combined = round((coverage + salience) / 2, 3)

        parts = [
            f"Amenity coverage: {coverage:.0%} "
            f"({len(covered)}/{len(all_codes)} amenities described"
            + (f"; missing: {sorted(missing)}" if missing else "") + ").",
            f"Premium salience: {salience:.0%} "
            f"({len(salient)}/{len(premium_present)} premium amenities in headline/highlights"
            + (f"; not surfaced: {not_salient}" if not_salient else "") + ").",
        ]

        return Score(
            value=combined,
            explanation=" ".join(parts),
            metadata={
                "amenity_coverage": round(coverage, 3),
                "premium_salience": round(salience, 3),
                "covered_codes": sorted(covered),
                "missing_codes": sorted(missing),
                "salient_premium": salient,
                "buried_premium": not_salient,
            },
        )

    return score


# ---------------------------------------------------------------------------
# Scorer 5: booking_intent (LLM-as-judge, user impact)
# ---------------------------------------------------------------------------

_BOOKING_INTENT_PROMPT = """\
You are a potential guest browsing vacation rental listings online.
You are considering a {property_type} in {location}, sleeping up to {max_guests} guests.

You have been shown the listing copy below. Based purely on what you read —
without knowing anything else about the property — decide how likely you are
to click "Request to Book" or add this listing to your shortlist.

Consider: does the copy make you excited about this specific property? Is it
compelling, specific, and trustworthy? Does it give you confidence the property
is worth booking, or does it feel generic and unconvincing?

LISTING COPY:
{copy_text}

Score booking intent 1–5:
  1 – Would scroll past immediately — generic, flat, or off-putting
  2 – Mildly interesting but not enough to act
  3 – Might look further — some appeal but not compelling enough to commit
  4 – Would click through — engaging and specific, makes me want to know more
  5 – Would book immediately — highly compelling, creates genuine urgency

Respond ONLY with valid JSON, no markdown fences:
{{"score": <1-5>, "explanation": "<one or two sentences about what drove the score>"}}"""


@scorer(metrics=[mean(), std()])
def booking_intent_scorer(judge_model: str = "anthropic/claude-sonnet-4-6"):
    """Score 1–5: how likely a potential guest is to click 'Request to Book'.

    Measures user impact / conversion potential — distinct from quality and
    faithfulness. A copy can be accurate and well-written but still fail to
    create urgency for this specific property (e.g. honest copy for a property
    with few amenities and mediocre reviews). Conversely, a compelling 5-star
    booking-intent score for a property with 0 reviews or null policies should
    trigger a faithfulness check — it may be overselling.

    The judge receives minimal property context (type, location, capacity) so it
    can calibrate expectations without being biased by the full amenity list.
    """

    async def score(state, target):
        copy: ListingCopy | None = state.metadata.get("listing_copy")
        prop_dict = state.metadata.get("property")

        if copy is None:
            return Score(value=1, explanation="listing_copy not found — scored 1 (worst)")

        prop = PropertyInput.model_validate(prop_dict)
        prompt = _BOOKING_INTENT_PROMPT.format(
            property_type=prop.property_type,
            location=f"{prop.location.city}, {prop.location.country}",
            max_guests=prop.rental_info.max_guests,
            copy_text=_format_copy_text(copy),
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
