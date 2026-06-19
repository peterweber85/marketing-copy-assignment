"""Fixture loading.

Property fixtures live as JSON under ``fixtures/`` at the repo root. They are the
ground truth for the whole pipeline: the grounding scorers cross-reference generated
copy against these exact fields, so loading validates each file against
``PropertyInput`` and fails loudly on drift.
"""

from __future__ import annotations

from pathlib import Path

from lodgify.models import PropertyInput

# Repo-root-relative location of the fixtures, resolved from this file so it works
# regardless of the current working directory (notebook, pytest, or `inspect eval`).
FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


def load_fixtures(fixtures_dir: Path | None = None) -> list[PropertyInput]:
    """Load and validate every property fixture, sorted by ``property_id``."""
    directory = fixtures_dir or FIXTURES_DIR
    properties = [
        PropertyInput.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("*.json"))
    ]
    if not properties:
        raise FileNotFoundError(f"No property fixtures found in {directory}")
    return sorted(properties, key=lambda p: p.property_id)


def load_fixture(property_id: int, fixtures_dir: Path | None = None) -> PropertyInput:
    """Load a single fixture by ``property_id`` (handy for targeted tests)."""
    for prop in load_fixtures(fixtures_dir):
        if prop.property_id == property_id:
            return prop
    raise KeyError(f"No fixture with property_id={property_id}")
