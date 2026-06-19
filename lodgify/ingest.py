"""Data ingestion: normalise a raw ``PropertyInput`` into a clean LLM-ready context.

This is stage 1 of the pipeline. It is intentionally LLM-free; everything here is
deterministic and unit-testable without mocks.

Key jobs:
* Strip HTML from the owner-supplied ``description.description`` field.
* Translate amenity codes to human labels via the curated map.
* Surface null policies explicitly so the prompt can instruct the model to omit them.
* Cap review text to keep prompt size manageable.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

from lodgify.amenities import humanize
from lodgify.models import PropertyInput


class _HTMLStripper(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def get_text(self) -> str:
        return " ".join(self._parts)


def strip_html(html: str) -> str:
    """Remove HTML tags and collapse whitespace. Handles HTML entities."""
    stripper = _HTMLStripper()
    stripper.feed(html)
    return re.sub(r"\s+", " ", stripper.get_text()).strip()


def build_context(prop: PropertyInput, max_reviews: int = 5) -> dict:
    """Return a clean, structured dict the prompt builder can serialise directly.

    Values come exclusively from ``PropertyInput`` structured fields — no inference,
    no defaults invented. Null policies are surfaced as ``None`` so the prompt can
    tell the model to skip them rather than guess.
    """
    return {
        "name": prop.property_name,
        "type": prop.property_type,
        "location": {"city": prop.location.city, "country": prop.location.country},
        "capacity": {
            "bedrooms": prop.rental_info.bedrooms,
            "bathrooms": prop.rental_info.bathrooms,
            "max_guests": prop.rental_info.max_guests,
        },
        "amenities": [
            {"code": code, "label": humanize(code)} for code in prop.amenities
        ],
        "reviews": {
            "count": prop.num_of_reviews,
            "average_score": prop.average_review_score,
            "samples": prop.reviews[:max_reviews],
        },
        "owner_headline": prop.description.headline,
        "about": strip_html(prop.description.description),
        "policies": {
            "cancellation": prop.policies.cancellation_policy,
            "payment_schedule": prop.policies.payment_schedule,
            "damage_deposit": prop.policies.damage_deposit,
        },
        "house_rules": {
            "check_in": prop.house_rules.check_in_time,
            "check_out": prop.house_rules.check_out_time,
        },
    }
