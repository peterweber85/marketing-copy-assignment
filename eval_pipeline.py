"""inspect-ai task definitions.

Run with:
    uv run inspect eval eval_pipeline.py@stub_eval  --model anthropic/claude-sonnet-4-6
    uv run inspect eval eval_pipeline.py@listing_eval --model anthropic/claude-haiku-4-5-20251001

The stub_eval task exists to validate that the scorers correctly discriminate
good output from bad BEFORE the real generator is built (EDD discipline).

The listing_eval task runs the full pipeline and is the primary deliverable.
Set ANTHROPIC_API_KEY to regenerate logs; view committed logs offline with:
    uv run inspect view --log-dir logs/
"""

from __future__ import annotations

import json

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelOutput
from inspect_ai.solver import Generate, TaskState, solver

from lodgify.data import load_fixtures
from lodgify.models import AmenityDescription, ListingCopy, PropertyInput
from lodgify.scorers import (
    booking_intent_scorer,
    completeness_scorer,
    faithfulness_scorer,
    grounding_scorer,
    quality_scorer,
)

# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

JUDGE_MODEL = "anthropic/claude-sonnet-4-6"
GEN_MODEL = "anthropic/claude-haiku-4-5-20251001"


def make_dataset(properties: list[PropertyInput] | None = None) -> list[Sample]:
    """One Sample per property. The raw property dict lives in metadata so every
    scorer can cross-reference the generated copy against the source of truth."""
    props = properties or load_fixtures()
    return [
        Sample(
            id=str(p.property_id),
            input=p.model_dump_json(),   # the generator solver reads this
            metadata={
                "property": p.model_dump(),
                "property_name": p.property_name,
            },
        )
        for p in props
    ]


# ---------------------------------------------------------------------------
# Stub solver — validates scorers before the real generator exists
# ---------------------------------------------------------------------------

# Hand-crafted outputs for property 104 (the trap: studio, bedrooms=0).
# "bad" copy claims two bedrooms → grounding should catch it.
# "good" copy correctly describes a studio → should score cleanly.

_STUB_BAD_104 = ListingCopy(
    hero_headline="Charming Two-Bedroom Apartment in Central Lisbon",
    highlights=[
        "Two spacious bedrooms sleeping up to 4 guests",
        "Award-winning interior design featured in major magazines",
        "Complimentary airport pickup and welcome champagne included",
    ],
    about_this_place=(
        "This elegant two-bedroom apartment sits in the heart of Lisbon, just steps from"
        " Rossio square. The award-winning interior design has been featured in numerous"
        " travel magazines. Guests consistently love the spacious layout and our"
        " complimentary airport pickup service. Free cancellation available."
    ),
    amenity_descriptions=[
        AmenityDescription(
            amenity_code="WiFi",
            label="High-speed Wi-Fi",
            description="Stay connected with our fast fibre broadband.",
        ),
        AmenityDescription(
            amenity_code="Jacuzzi",  # NOT in property 104's amenities
            label="Jacuzzi",
            description="Relax in the private jacuzzi.",
        ),
    ],
)

_STUB_GOOD_104 = ListingCopy(
    hero_headline="Sun-Filled Studio Steps from Rossio, Lisbon",
    highlights=[
        "Central location — walk to Rossio square, trams, and restaurants",
        "Bright and compact studio, ideal for two guests",
        "Air conditioning and high-speed Wi-Fi throughout",
    ],
    about_this_place=(
        "Sunny Central Studio puts you in the middle of Lisbon's historic centre. This"
        " well-designed studio apartment fits two guests comfortably, with a kitchenette,"
        " a double sofa bed, and air conditioning to keep you cool. Step outside and you"
        " are moments from Rossio square, vintage trams, and some of the city's best"
        " tascas. Free cancellation up to 5 days before check-in. Check-in from 3 PM,"
        " check-out by 11 AM."
    ),
    amenity_descriptions=[
        AmenityDescription(
            amenity_code="WiFi",
            label="High-speed Wi-Fi",
            description="Fast internet throughout the studio.",
        ),
        AmenityDescription(
            amenity_code="AirConditioning",
            label="Air conditioning",
            description="Individual AC unit in the main room.",
        ),
        AmenityDescription(
            amenity_code="Kitchen",
            label="Full kitchen",
            description="Kitchenette with hob, microwave, and basic utensils.",
        ),
        AmenityDescription(
            amenity_code="TV",
            label="TV",
            description="Flat-screen TV with international channels.",
        ),
    ],
)


@solver
def stub_solver(variant: str = "bad"):
    """Inject a hand-crafted ListingCopy without calling the LLM.

    ``variant='bad'``  → deliberately wrong copy (tests scorers detect failures).
    ``variant='good'`` → hand-crafted correct copy (tests scorers pass on good output).

    Only applies the same stub to all samples in the dataset (property 104's copy).
    Production tasks must not use this solver.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        copy = _STUB_BAD_104 if variant == "bad" else _STUB_GOOD_104
        state.metadata["listing_copy"] = copy
        state.output = ModelOutput.from_content(
            model="stub/stub", content=copy.model_dump_json()
        )
        return state

    return solve


# ---------------------------------------------------------------------------
# Task: stub_eval — validates scorers before the real generator exists
# ---------------------------------------------------------------------------


@task
def golden_eval() -> Task:
    """Judge calibration task: run the LLM scorers against hand-annotated golden copies.

    Each of the 9 fixtures has two variants — a 'good' copy (exemplary, fully grounded)
    and a 'bad' copy (deliberately flawed in the fixture's specific failure mode).
    Human-assigned faithfulness and quality scores are stored alongside each copy.

    After running, compare judge scores to human scores to measure calibration:
    a mean absolute deviation (MAD) < 0.7 on a 1–5 scale indicates a well-calibrated judge.

    Usage:
        uv run inspect eval eval_pipeline.py@golden_eval --model anthropic/claude-sonnet-4-6
    """
    import json as _json
    from lodgify.data import load_fixture
    from lodgify.golden import load_golden_examples
    from lodgify.models import ListingCopy as _LC

    examples = load_golden_examples()
    samples = []
    for ex in examples:
        prop = load_fixture(ex.property_id)
        for variant_name in ("good", "bad"):
            variant = getattr(ex, variant_name)
            samples.append(
                Sample(
                    id=f"{ex.property_id}-{variant_name}",
                    input=prop.model_dump_json(),
                    metadata={
                        "property": prop.model_dump(),
                        "property_name": prop.property_name,
                        "listing_copy": variant.listing_copy.model_dump(),
                        "human_scores": variant.human_scores.model_dump(),
                        "variant": variant_name,
                        "intentional_failures": variant.intentional_failures,
                    },
                )
            )

    @solver
    def _golden_injector():
        """Inject the pre-written golden copy into state — no LLM generation."""
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            copy_dict = state.metadata.get("listing_copy")
            if isinstance(copy_dict, dict):
                state.metadata["listing_copy"] = _LC.model_validate(copy_dict)
            state.output = ModelOutput.from_content(
                model="golden/injector",
                content=_json.dumps(copy_dict) if copy_dict else "",
            )
            return state
        return solve

    return Task(
        dataset=samples,
        solver=_golden_injector(),
        scorer=[
            grounding_scorer(),
            faithfulness_scorer(JUDGE_MODEL),
            completeness_scorer(),
            booking_intent_scorer(JUDGE_MODEL),
            quality_scorer(JUDGE_MODEL),
        ],
        metadata={"task": "golden_eval"},
    )


@task
def stub_eval(variant: str = "bad") -> Task:
    """Run the three scorers against a hand-crafted stub output.

    Purpose: confirm the eval harness discriminates good from bad copy before
    the real generator is built. This is the EDD discipline in practice.

    Usage:
        uv run inspect eval eval_pipeline.py@stub_eval --model anthropic/claude-sonnet-4-6
        uv run inspect eval eval_pipeline.py@stub_eval -T variant=good --model anthropic/claude-sonnet-4-6
    """
    # Use only property 104 (the trap fixture) so the bad stub's errors are obvious.
    from lodgify.data import load_fixture
    props = [load_fixture(104)]

    return Task(
        dataset=make_dataset(props),
        solver=stub_solver(variant=variant),
        scorer=[
            grounding_scorer(),
            faithfulness_scorer(JUDGE_MODEL),
            completeness_scorer(),
            booking_intent_scorer(JUDGE_MODEL),
            quality_scorer(JUDGE_MODEL),
        ],
    )


# ---------------------------------------------------------------------------
# Task: listing_eval — the full pipeline (generator wired in Phase 3)
# ---------------------------------------------------------------------------


@task
def reliability_eval(prompt_version: str = "v4", epochs: int = 3) -> Task:
    """Measure how consistent scores are across repeated runs of the same fixture.

    Each of the N fixtures is run ``epochs`` times (default 3), producing
    N × epochs total samples. inspect-ai groups them and reports mean + std
    per scorer — the std figures are the primary output of this task.

    How to read the results
    -----------------------
    grounding_scorer std ≈ 0.0
        The structural checks (bedroom counts, amenity codes, null policies) are
        enforced deterministically by the model on every run. This is the ideal:
        grounding failures are not random.

    grounding_scorer std > 0.05
        The prompt is under-constrained on some fixture — the model sometimes
        passes and sometimes fails the same structural check. That fixture is a
        prompt iteration target.

    faithfulness_scorer std ≈ 0.3–0.6
        Expected: the LLM judge itself introduces variance (same copy can score
        4 one run and 3 the next). A std below ~0.5 is acceptable for a 1–5 scale.

    faithfulness_scorer std > 0.8
        The generated copy is qualitatively inconsistent — some runs produce
        faithful copy, others embellish. The prompt needs tighter constraints.

    quality_scorer std ≈ 0.3–0.7
        Expected: different valid phrasings of the same property legitimately
        score differently. This is not a problem to fix.

    Usage
    -----
        uv run inspect eval eval_pipeline.py@reliability_eval \\
            --model anthropic/claude-haiku-4-5-20251001
        uv run inspect eval eval_pipeline.py@reliability_eval \\
            -T prompt_version=v2 -T epochs=5 \\
            --model anthropic/claude-haiku-4-5-20251001
    """
    from lodgify.generator import generate_solver, ingest_solver

    return Task(
        dataset=make_dataset(),
        solver=[ingest_solver(), generate_solver(prompt_version=prompt_version)],
        scorer=[
            grounding_scorer(),
            faithfulness_scorer(JUDGE_MODEL),
            completeness_scorer(),
            booking_intent_scorer(JUDGE_MODEL),
            quality_scorer(JUDGE_MODEL),
        ],
        epochs=epochs,
        metadata={"prompt_version": prompt_version, "epochs": epochs},
    )


@task
def listing_eval(prompt_version: str = "v1") -> Task:
    """Run the full content-generation pipeline and eval suite.

    ``prompt_version`` is recorded in the run log so before/after prompt
    iterations are traceable to specific eval runs.

    Usage:
        uv run inspect eval eval_pipeline.py@listing_eval --model anthropic/claude-haiku-4-5-20251001
        uv run inspect eval eval_pipeline.py@listing_eval -T prompt_version=v2 --model anthropic/claude-haiku-4-5-20251001
    """
    from lodgify.generator import generate_solver, ingest_solver
    solvers = [ingest_solver(), generate_solver(prompt_version=prompt_version)]

    return Task(
        dataset=make_dataset(),
        solver=solvers,
        scorer=[
            grounding_scorer(),
            faithfulness_scorer(JUDGE_MODEL),
            completeness_scorer(),
            booking_intent_scorer(JUDGE_MODEL),
            quality_scorer(JUDGE_MODEL),
        ],
        metadata={"prompt_version": prompt_version},
    )
