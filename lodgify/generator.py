"""Generator: the two-solver chain that produces ListingCopy from a PropertyInput.

Pipeline stages:
    1. ingest_solver  — reads the raw JSON from state.input, builds a clean context
                        dict (HTML stripped, amenity codes translated), stores it in
                        state.metadata["context"]. No LLM.
    2. generate_solver — sends the context to Haiku, stores the validated
                        ListingCopy in state.metadata["listing_copy"] for scorers.

Prompts live in ``prompts/`` at the repo root — one plain-text file per version
(v1.txt, v2.txt, …). Adding a new version means creating a file there; no Python
changes required. The active version is recorded in state.metadata["prompt_version"]
so every inspect-ai run log is traceable to the exact prompt that produced it.

Dependency injection: generate_solver accepts a ``model`` parameter so tests can
swap in inspect-ai's ``mockllm/model`` without touching the rest of the solver.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from inspect_ai.model import ChatMessageSystem, ChatMessageUser, ModelOutput, get_model
from inspect_ai.solver import Generate, TaskState, solver

from lodgify.ingest import build_context
from lodgify.models import ListingCopy, PropertyInput

# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load_prompt(version: str) -> str:
    """Load a prompt template by version name (e.g. ``"v1"``).

    Reads ``prompts/<version>.txt`` relative to the repo root. Raises
    ``FileNotFoundError`` with a helpful message if the version doesn't exist.
    """
    path = PROMPTS_DIR / f"{version}.txt"
    if not path.exists():
        available = sorted(p.stem for p in PROMPTS_DIR.glob("v*.txt"))
        raise FileNotFoundError(
            f"No prompt file for version {version!r}. "
            f"Available: {available}. "
            f"Add prompts/{version}.txt to create a new version."
        )
    return path.read_text(encoding="utf-8")


def load_system_prompt() -> str:
    """Load the shared system prompt from ``prompts/system.txt``."""
    return (PROMPTS_DIR / "system.txt").read_text(encoding="utf-8").strip()


def available_versions() -> list[str]:
    """Return all prompt versions present in ``prompts/``, sorted."""
    return sorted(p.stem for p in PROMPTS_DIR.glob("v*.txt"))


# ---------------------------------------------------------------------------
# Solver 1: ingest (deterministic, no LLM)
# ---------------------------------------------------------------------------


@solver
def ingest_solver():
    """Parse and normalise the property fixture from state.input.

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
# Solver 2: generate (calls Haiku, validates output as ListingCopy)
# ---------------------------------------------------------------------------


@solver
def generate_solver(
    prompt_version: str = "latest",
    model: str = "anthropic/claude-haiku-4-5-20251001",
):
    """Call the LLM and store a validated ListingCopy in state.metadata.

    The ``model`` parameter is the DI hook: tests inject ``mockllm/model``
    here; production uses Haiku. inspect-ai resolves the model string via
    ``get_model()``.

    ``prompt_version`` names a file in ``prompts/`` (e.g. ``"v5"`` → reads
    ``prompts/v5.txt``). The special value ``"latest"`` resolves to the
    highest-numbered version present in ``prompts/``. The resolved version
    is recorded in metadata so the run log is traceable to the exact prompt.
    """
    # Resolve "latest" before loading so the resolved name is what gets
    # recorded in metadata — not the literal string "latest".
    if prompt_version == "latest":
        versions = available_versions()
        if not versions:
            raise FileNotFoundError("No prompt files found in prompts/")
        prompt_version = versions[-1]  # available_versions() returns sorted

    # Validate and load at construction time so a typo fails fast.
    prompt_template = load_prompt(prompt_version)
    system_prompt = load_system_prompt()

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.metadata["prompt_version"] = prompt_version
        context: dict[str, Any] = state.metadata["context"]

        prompt = prompt_template.format(
            context_json=json.dumps(context, indent=2, ensure_ascii=False)
        )

        llm = get_model(model)
        response = await llm.generate(
            [
                ChatMessageSystem(content=system_prompt),
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
