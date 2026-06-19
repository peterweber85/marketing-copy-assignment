"""Generator: the two-solver chain that produces ListingCopy from a PropertyInput.

Pipeline stages:
    1. ingest_solver  — reads the raw JSON from state.input, builds a clean context
                        dict (HTML stripped, amenity codes translated), stores it in
                        state.metadata["context"]. No LLM.
    2. generate_solver — sends the context to Haiku via output_config structured
                        output (validated Pydantic ListingCopy), stores the parsed
                        object in state.metadata["listing_copy"] for scorers.

Prompt versioning is explicit: PROMPTS["v1"], PROMPTS["v2"], … are kept side by
side. The active version is recorded in state.metadata["prompt_version"] so every
inspect-ai run log is traceable to the exact prompt that produced it.

Dependency injection: generate_solver accepts a ``model`` parameter so tests can
swap in inspect-ai's ``mockllm/model`` without touching the rest of the solver.
"""

from __future__ import annotations

import json
import textwrap
from typing import Any

from inspect_ai.model import ChatMessageSystem, ChatMessageUser, ModelOutput, get_model
from inspect_ai.solver import Generate, TaskState, solver

from lodgify.ingest import build_context
from lodgify.models import ListingCopy, PropertyInput

# ---------------------------------------------------------------------------
# Prompt library (versioned — never edit in place, add v2, v3, …)
# ---------------------------------------------------------------------------

# Minimal system shared by all prompt versions. Does NOT contain bedroom/policy
# grounding rules — those are version-specific so v1 can genuinely fail the
# trap and v2 can demonstrably fix it (the EDD arc).
_SYSTEM_BASE = textwrap.dedent("""
    You are a professional copywriter for a vacation rental platform. Your job is
    to produce compelling marketing copy for property listings.
    Output must be valid JSON matching the requested schema exactly.
    Output JSON only — no markdown fences, no prose outside the JSON object.
""").strip()

PROMPTS: dict[str, str] = {
    # v1: naive prompt — instructs the model to use the owner's headline for
    # inspiration, which causes it to echo the misleading "2-Bedroom" claim
    # from property 104's owner_headline ("Spacious 2-Bedroom Apartment…")
    # despite bedrooms=0. The grounding_scorer catches this.
    "v1": textwrap.dedent("""
        Write marketing copy for the vacation rental property below.

        PROPERTY DATA (JSON):
        {context_json}

        Instructions:
        - Use the "owner_headline" as inspiration for the hero_headline — it
          captures what makes this property special.
        - Write compelling highlights and an "about this place" paragraph.
        - Add a short description for each amenity in the data.

        Produce a JSON object with these fields:
        - hero_headline: a short, punchy headline (10–90 chars)
        - highlights: 3–6 bullet strings — the property's best selling points
        - about_this_place: 120–1200 chars of flowing prose about the property
        - amenity_descriptions: list of objects with amenity_code, label, description
          (one per amenity in the data; max 300 chars per description)

        Output JSON only, no prose around it.
    """).strip(),

    # v2: adds explicit anti-embellishment grounding rules after the eval arc
    # revealed that v1 faithfulness mean = 3.33/5 because the model added
    # attributes not in the data (e.g. "saltwater-inspired pool", "full central
    # heating" when data just says "Heating", "panoramic vistas from every room").
    # Diagnosis: model elaborates amenity labels with its own assumptions.
    # Fix: require descriptions to stay within what the structured data states.
    "v2": textwrap.dedent("""
        Write marketing copy for the vacation rental property below.

        PROPERTY DATA (JSON):
        {context_json}

        GROUNDING RULES — apply these strictly:
        - Amenity descriptions: stay within what the data says. If the amenity
          label is "Kitchen", describe it as a kitchen — do NOT add "full",
          "gourmet", "chef's", or any qualifier not in the data. If the label
          is "Heating", do NOT say "central heating" or "underfloor heating".
        - Do NOT add attributes (material, style, size, view angle, quality)
          to any feature unless they are explicitly stated in the data.
        - Do NOT use the "owner_headline" or "about" text as a source for facts.
        - Only mention amenities that appear in the "amenities" list.
        - If a policy value is null, do not mention that policy.
        - If "reviews.count" is 0, do not claim social proof.

        Produce a JSON object with these fields:
        - hero_headline: a short, punchy headline (10–90 chars)
        - highlights: 3–6 bullet strings — the property's best selling points
        - about_this_place: 120–1200 chars of flowing prose about the property
        - amenity_descriptions: list of objects with amenity_code, label, description
          (one per amenity in the data; max 300 chars per description)

        Output JSON only, no prose around it.
    """).strip(),
}


# ---------------------------------------------------------------------------
# Solver 1: ingest (deterministic, no LLM)
# ---------------------------------------------------------------------------


@solver
def ingest_solver():
    """Parse + normalise the property fixture from state.input.

    Reads the raw JSON string placed there by the Dataset, strips HTML,
    translates amenity codes, and stores the result in state.metadata["context"]
    for the downstream generate_solver.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        prop = PropertyInput.model_validate_json(state.input_text)
        state.metadata["property"] = prop.model_dump()  # scorers read from here
        state.metadata["context"] = build_context(prop)
        return state

    return solve


# ---------------------------------------------------------------------------
# Solver 2: generate (calls Haiku via output_config structured output)
# ---------------------------------------------------------------------------


@solver
def generate_solver(
    prompt_version: str = "v1",
    model: str = "anthropic/claude-haiku-4-5-20251001",
):
    """Call the LLM and store a validated ListingCopy in state.metadata.

    The ``model`` parameter is the DI hook: tests inject
    ``mockllm/model`` here; production uses Haiku. inspect-ai resolves the
    model string via ``get_model()``.

    ``prompt_version`` selects from PROMPTS and is recorded in metadata so
    the run log is traceable to the exact prompt.
    """
    if prompt_version not in PROMPTS:
        raise ValueError(
            f"Unknown prompt_version {prompt_version!r}. "
            f"Available: {list(PROMPTS)}"
        )

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.metadata["prompt_version"] = prompt_version
        context: dict[str, Any] = state.metadata["context"]

        prompt = PROMPTS[prompt_version].format(
            context_json=json.dumps(context, indent=2, ensure_ascii=False)
        )

        llm = get_model(model)
        response = await llm.generate(
            [
                ChatMessageSystem(content=_SYSTEM_BASE),
                ChatMessageUser(content=prompt),
            ],
        )

        raw = response.completion.strip()

        # Strip markdown fences if the model wraps its JSON
        if raw.startswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(
                lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
            )

        try:
            copy = ListingCopy.model_validate_json(raw)
            state.metadata["listing_copy"] = copy
        except Exception as exc:
            state.metadata["listing_copy"] = None
            state.metadata["listing_copy_error"] = f"{type(exc).__name__}: {exc}"

        state.output = ModelOutput.from_content(model=model, content=raw)
        return state

    return solve
