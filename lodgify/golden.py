"""Golden dataset — hand-annotated (property, copy, human_scores) triples.

Purpose: calibrate the LLM judges (faithfulness_scorer, quality_scorer) by
measuring how well they agree with human-assigned scores on known-good and
known-bad copy. If mean absolute deviation between judge and human scores is
< 0.7 on a 1–5 scale, the judge is considered calibrated.

Two variants per fixture:
  good — exemplary, fully grounded copy; human faithfulness always 5.
  bad  — deliberately flawed copy targeting that fixture's specific failure
         mode; human faithfulness 1–3 depending on severity.

Quality scores reflect how well-written the copy is, independent of accuracy.
Bad copies are written to be fluent prose (quality 3–4) so the judge can
distinguish a beautifully-written hallucination from a poorly-written one.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from lodgify.models import AmenityDescription, ListingCopy

GOLDEN_DIR = Path(__file__).resolve().parent.parent / "golden"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class HumanScores(BaseModel):
    faithfulness: int  # 1–5, same scale as faithfulness_scorer
    quality: int       # 1–5, same scale as quality_scorer
    notes: str = ""


class GoldenVariant(BaseModel):
    listing_copy: ListingCopy
    human_scores: HumanScores
    intentional_failures: list[str] = []  # documents the flaw in bad variants


class GoldenExample(BaseModel):
    property_id: int
    good: GoldenVariant
    bad: GoldenVariant


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _parse_variant(v: dict) -> GoldenVariant:
    copy = ListingCopy(
        hero_headline=v["hero_headline"],
        highlights=v["highlights"],
        about_this_place=v["about_this_place"],
        amenity_descriptions=[
            AmenityDescription(**a) for a in v.get("amenity_descriptions", [])
        ],
    )
    scores = HumanScores(
        faithfulness=v["human_scores"]["faithfulness"],
        quality=v["human_scores"]["quality"],
        notes=v.get("human_notes", ""),
    )
    return GoldenVariant(
        listing_copy=copy,
        human_scores=scores,
        intentional_failures=v.get("intentional_failures", []),
    )


def _parse_golden(data: dict) -> GoldenExample:
    return GoldenExample(
        property_id=data["property_id"],
        good=_parse_variant(data["good"]),
        bad=_parse_variant(data["bad"]),
    )


def load_golden_examples(golden_dir: Path | None = None) -> list[GoldenExample]:
    """Load and validate every golden file, sorted by property_id."""
    directory = golden_dir or GOLDEN_DIR
    examples = [
        _parse_golden(json.loads(p.read_text(encoding="utf-8")))
        for p in sorted(directory.glob("*.json"))
    ]
    if not examples:
        raise FileNotFoundError(f"No golden files found in {directory}")
    return sorted(examples, key=lambda e: e.property_id)


def load_golden_example(property_id: int, golden_dir: Path | None = None) -> GoldenExample:
    """Load a single golden file by property_id."""
    for ex in load_golden_examples(golden_dir):
        if ex.property_id == property_id:
            return ex
    raise KeyError(f"No golden file for property_id={property_id}")
