"""Typed data contracts for the pipeline.

Two halves:

* ``PropertyInput`` mirrors the raw property record from the assignment spec. It is
  the *source of truth* that every generated claim must trace back to — the grounding
  scorers cross-reference output against these fields.
* ``ListingCopy`` is the structured output the LLM must produce. Field constraints here
  double as the first line of "output validation": malformed copy fails to parse before
  any scorer runs.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------------------
# Input: the raw property record (mirrors the assignment data spec)
# --------------------------------------------------------------------------------------


class Description(BaseModel):
    """Owner-supplied text. ``description`` may contain HTML and marketing puffery, so
    the ingest stage sanitizes it and downstream prompts must not treat it as fact."""

    name: str
    headline: str
    description: str  # may contain HTML


class RentalInfo(BaseModel):
    max_guests: int = Field(ge=1)
    bedrooms: int = Field(ge=0)  # studios legitimately have 0
    bathrooms: int = Field(ge=0)


class Location(BaseModel):
    city: str
    country: str
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


class Policies(BaseModel):
    """All fields nullable on purpose: a missing policy must never be invented in copy."""

    cancellation_policy: str | None = None
    payment_schedule: str | None = None
    damage_deposit: str | None = None


class HouseRules(BaseModel):
    check_in_time: str  # e.g. "3 PM"
    check_out_time: str  # e.g. "11 AM"


class PropertyInput(BaseModel):
    """One vacation-rental property: the full input to the pipeline."""

    model_config = ConfigDict(extra="forbid")

    property_id: int
    property_name: str
    property_type: str  # e.g. "NormalApartment", "Villa", "Cottage"
    description: Description
    amenities: list[str] = Field(default_factory=list)  # internal codes
    image_urls: list[str] = Field(default_factory=list)
    reviews: list[str] = Field(default_factory=list)  # full guest review texts
    num_of_reviews: int = Field(ge=0)
    average_review_score: float = Field(ge=0, le=5)
    rental_info: RentalInfo
    location: Location
    policies: Policies = Field(default_factory=Policies)
    house_rules: HouseRules


# --------------------------------------------------------------------------------------
# Output: the structured marketing copy the LLM must produce
# --------------------------------------------------------------------------------------


class AmenityDescription(BaseModel):
    """A single amenity rendered for guests. ``amenity_code`` ties the prose back to a
    specific input amenity so the grounding scorer can verify it was not invented."""

    model_config = ConfigDict(extra="forbid")

    amenity_code: str  # must be one of the property's input amenity codes
    label: str  # human-readable name shown to guests
    description: str = Field(min_length=1, max_length=300)


class ListingCopy(BaseModel):
    """The marketing copy for one listing page.

    The four sections map directly to the brief: hero headline, property highlights,
    "about this place", and amenities descriptions. Constraints keep the model honest
    about length/shape so validation catches structural failures early.
    """

    model_config = ConfigDict(extra="forbid")

    hero_headline: str = Field(min_length=10, max_length=90)
    highlights: list[str] = Field(min_length=3, max_length=6)
    about_this_place: str = Field(min_length=120, max_length=1200)
    amenity_descriptions: list[AmenityDescription] = Field(default_factory=list)
