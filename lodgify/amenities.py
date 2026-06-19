"""Amenity-code translation (part of data ingestion).

The input carries internal amenity *codes* like ``"InternetBroadband"``. Guests need
human labels like "High-speed internet". This mapping is the single source of truth for
that translation, so the model is never asked to guess what a code means — it receives
the label directly, and the grounding scorer checks output labels against it.

Unknown codes are not dropped silently (that would lose a real amenity) nor invented;
``humanize`` falls back to splitting the CamelCase code into words, and ``is_known``
lets callers flag codes worth adding to the table.
"""

from __future__ import annotations

import re

# Curated map of internal codes -> guest-facing labels. Extend as fixtures introduce
# new codes; keep labels factual (no marketing adjectives — those are the LLM's job).
AMENITY_LABELS: dict[str, str] = {
    "BathroomAndLaundry": "Bathroom and laundry",
    "DishWasher": "Dishwasher",
    "InternetBroadband": "High-speed internet",
    "WiFi": "Wi-Fi",
    "AirConditioning": "Air conditioning",
    "Heating": "Heating",
    "Kitchen": "Full kitchen",
    "FreeParking": "Free parking on premises",
    "SwimmingPool": "Swimming pool",
    "PrivatePool": "Private pool",
    "HotTub": "Hot tub",
    "Fireplace": "Fireplace",
    "WasherDryer": "Washer and dryer",
    "PetFriendly": "Pet friendly",
    "Elevator": "Elevator",
    "Balcony": "Balcony",
    "Terrace": "Terrace",
    "Garden": "Garden",
    "SeaView": "Sea view",
    "MountainView": "Mountain view",
    "Workspace": "Dedicated workspace",
    "TV": "TV",
    "Crib": "Crib",
    "Gym": "Gym",
    "BBQGrill": "BBQ grill",
    "Microwave": "Microwave",
    "CoffeeMaker": "Coffee maker",
    "SmokeAlarm": "Smoke alarm",
    "FirstAidKit": "First aid kit",
    "EVCharger": "EV charger",
}

_CAMEL_SPLIT = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def is_known(code: str) -> bool:
    """True if ``code`` is in the curated mapping."""
    return code in AMENITY_LABELS


def humanize(code: str) -> str:
    """Return a guest-facing label for an amenity ``code``.

    Falls back to a CamelCase split (``"OutdoorShower"`` -> ``"Outdoor shower"``) so an
    unmapped code still surfaces as a readable, faithful amenity rather than being lost.
    """
    if code in AMENITY_LABELS:
        return AMENITY_LABELS[code]
    words = _CAMEL_SPLIT.sub(" ", code).strip()
    if not words:
        return code
    return words[0].upper() + words[1:]


def humanize_all(codes: list[str]) -> dict[str, str]:
    """Map each input code to its label, preserving order via the returned dict."""
    return {code: humanize(code) for code in codes}
