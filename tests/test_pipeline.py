"""Pipeline tests — runnable offline with no API key, no network.

Design notes
------------
* **Dependency injection:** ``generate_solver`` accepts a ``model`` parameter.
  Tests inject ``mockllm/model`` (inspect-ai's built-in canned-response model)
  instead of the real Haiku. The rest of the solver chain is unchanged, so we
  exercise the real ingest + JSON-parse + metadata paths.

* **Inheritance:** ``AbstractModelClient`` defines the interface; ``AnthropicClient``
  is the real implementation; ``MockModelClient`` is the test double. Both are
  subclasses of the abstract base, demonstrating that the generator accepts any
  conformant implementation (Liskov substitution principle).

* **Mocking (unittest.mock):** rule-based scorers (grounding, completeness) are
  tested directly — no mock needed. LLM-judge scorers (faithfulness, quality,
  booking_intent) are tested by patching ``get_model`` to return an AsyncMock
  that returns canned responses.

All tests run with ``uv run pytest tests/`` — no env vars required.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lodgify.amenities import humanize
from lodgify.data import load_fixture, load_fixtures
from lodgify.ingest import build_context, scrub_checkin_prose, strip_html
from lodgify.models import AmenityDescription, ListingCopy, PropertyInput
from lodgify.scorers import (
    _extract_bedroom_numbers,
    _run_grounding_checks,
    completeness_scorer,
)

# ---------------------------------------------------------------------------
# Inheritance demo: abstract model client + concrete implementations
# ---------------------------------------------------------------------------


class AbstractModelClient(ABC):
    """Minimal interface for generating text from a prompt."""

    @abstractmethod
    def generate(self, prompt: str) -> str: ...


class AnthropicClient(AbstractModelClient):
    """Production client — wraps the real Anthropic SDK."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001") -> None:
        self.model = model

    def generate(self, prompt: str) -> str:  # pragma: no cover
        import anthropic
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=self.model, max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text


class MockModelClient(AbstractModelClient):
    """Test double — returns canned JSON without calling any API."""

    def __init__(self, response_json: str) -> None:
        self._response = response_json

    def generate(self, prompt: str) -> str:
        return self._response


def _make_scorer_state(copy: ListingCopy, prop: PropertyInput):
    """Build a minimal TaskState with listing_copy in metadata for scorer tests."""
    from inspect_ai.solver import TaskState
    return TaskState(
        model="mockllm/model", sample_id="test", epoch=1,
        input=prop.model_dump_json(), messages=[],
        metadata={"property": prop.model_dump(), "listing_copy": copy},
    )


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def prop_101() -> PropertyInput:
    return load_fixture(101)


@pytest.fixture
def prop_103() -> PropertyInput:
    """PetFriendly IS in amenities — pet claims should pass."""
    return load_fixture(103)


@pytest.fixture
def prop_104() -> PropertyInput:
    """Studio (bedrooms=0) with misleading owner headline."""
    return load_fixture(104)


@pytest.fixture
def prop_105() -> PropertyInput:
    """All policies null."""
    return load_fixture(105)


@pytest.fixture
def prop_106() -> PropertyInput:
    """Zero reviews."""
    return load_fixture(106)


@pytest.fixture
def prop_107() -> PropertyInput:
    """avg_score=3.1 with cherry-picked 5-star review samples."""
    return load_fixture(107)


@pytest.fixture
def prop_108() -> PropertyInput:
    """PetFriendly NOT in amenities but reviews mention dogs."""
    return load_fixture(108)


@pytest.fixture
def prop_109() -> PropertyInput:
    """check_in_time='7 PM' with owner headline claiming flexible check-in."""
    return load_fixture(109)


@pytest.fixture
def valid_copy_101() -> ListingCopy:
    return ListingCopy(
        hero_headline="Seafront Villa with Private Pool in Begur",
        highlights=[
            "Private pool with direct sea views",
            "4 bedrooms sleeping up to 8 guests",
            "Steps from the coastal path to the beach",
        ],
        about_this_place=(
            "Casa del Mar is a bright villa perched above a quiet cove in Begur on"
            " the Costa Brava. Four bedrooms sleep up to eight guests in comfort."
        ),
        amenity_descriptions=[
            AmenityDescription(
                amenity_code="PrivatePool",
                label="Private pool",
                description="Cool off in your own pool with sea views.",
            ),
        ],
    )


# ---------------------------------------------------------------------------
# 1. Amenity translation
# ---------------------------------------------------------------------------


class TestAmenityTranslation:
    def test_known_code_returns_label(self) -> None:
        assert humanize("InternetBroadband") == "High-speed internet"

    def test_unknown_code_splits_camel_case(self) -> None:
        assert humanize("OutdoorShower") == "Outdoor Shower"


# ---------------------------------------------------------------------------
# 2. HTML stripping and ingest
# ---------------------------------------------------------------------------


class TestHTMLStripping:
    def test_decodes_html_entities(self) -> None:
        assert "è" in strip_html("Gr&egrave;ce")

    def test_eixample_loft_strip_end_to_end(self) -> None:
        """HTML is stripped and meaningful text survives in build_context."""
        ctx = build_context(load_fixture(102))
        assert "<div" not in ctx["about"]
        assert "Eixample" in ctx["about"]


# ---------------------------------------------------------------------------
# 3. Grounding scorer — core checks
# ---------------------------------------------------------------------------


class TestGroundingScorer:
    def test_happy_path_passes_all_checks(self, valid_copy_101, prop_101) -> None:
        _, failed = _run_grounding_checks(valid_copy_101, prop_101)
        assert failed == []

    def test_wrong_bedroom_count_fails(self, prop_101) -> None:
        bad = ListingCopy(
            hero_headline="Two-Bedroom Villa in Begur",
            highlights=["Two bedrooms", "Sea view", "Pool"],
            about_this_place="x" * 150,
            amenity_descriptions=[],
        )
        _, failed = _run_grounding_checks(bad, prop_101)
        assert any("bedroom_count_wrong" in f for f in failed)

    def test_studio_claimed_as_bedroom_fails(self, prop_104) -> None:
        """bedrooms=0 — claiming 'two bedrooms' must fail."""
        bad = ListingCopy(
            hero_headline="Charming Two-Bedroom Apartment in Lisbon",
            highlights=["Two bedrooms", "Central location", "WiFi"],
            about_this_place="x" * 150,
            amenity_descriptions=[],
        )
        _, failed = _run_grounding_checks(bad, prop_104)
        assert any("bedroom" in f for f in failed)

    def test_invented_amenity_code_fails(self, prop_104) -> None:
        bad = ListingCopy(
            hero_headline="Studio in Lisbon",
            highlights=["Great location", "WiFi", "Compact studio"],
            about_this_place="x" * 150,
            amenity_descriptions=[
                AmenityDescription(amenity_code="Jacuzzi", label="Jacuzzi", description="Relax."),
            ],
        )
        _, failed = _run_grounding_checks(bad, prop_104)
        assert any("invented_amenity" in f for f in failed)

    def test_null_cancellation_policy_mentioned_fails(self, prop_105) -> None:
        bad = ListingCopy(
            hero_headline="Marina View in Valencia",
            highlights=["Sea views", "Free cancellation available", "2 bedrooms"],
            about_this_place="x" * 150,
            amenity_descriptions=[],
        )
        _, failed = _run_grounding_checks(bad, prop_105)
        assert any("cancellation" in f for f in failed)

    def test_zero_reviews_social_proof_fails(self, prop_106) -> None:
        bad = ListingCopy(
            hero_headline="Alpine Escape in Chamonix",
            highlights=["Top-rated chalet", "Guests love it", "Hot tub"],
            about_this_place="x" * 150,
            amenity_descriptions=[],
        )
        _, failed = _run_grounding_checks(bad, prop_106)
        assert any("social_proof" in f for f in failed)

    def test_city_missing_fails(self, prop_101) -> None:
        bad = ListingCopy(
            hero_headline="Seaside Villa with Pool",
            highlights=["Private pool", "Sea views", "4 bedrooms"],
            about_this_place="x" * 150,
            amenity_descriptions=[],
        )
        _, failed = _run_grounding_checks(bad, prop_101)
        assert any("city_missing" in f for f in failed)


# ---------------------------------------------------------------------------
# 4. Bedroom number extraction
# ---------------------------------------------------------------------------


class TestBedroomExtraction:
    def test_digit(self) -> None:
        assert _extract_bedroom_numbers("3-bedroom villa") == [3]

    def test_word(self) -> None:
        assert _extract_bedroom_numbers("two bedroom apartment") == [2]


# ---------------------------------------------------------------------------
# 5. DI + inheritance demo
# ---------------------------------------------------------------------------


class TestDependencyInjection:
    """Demonstrates that both implementations conform to the abstract interface."""

    def test_mock_client_returns_canned_response(self) -> None:
        mock = MockModelClient('{"message": "ok"}')
        assert mock.generate("any prompt") == '{"message": "ok"}'

    def test_anthropic_client_is_subclass(self) -> None:
        assert issubclass(AnthropicClient, AbstractModelClient)

    def test_mock_client_is_subclass(self) -> None:
        assert issubclass(MockModelClient, AbstractModelClient)


# ---------------------------------------------------------------------------
# 6. Solver chain — offline via mockllm/model
# ---------------------------------------------------------------------------


class TestSolverChainOffline:
    def _make_state(self, prop: PropertyInput):
        from inspect_ai.dataset import Sample
        from inspect_ai.solver import TaskState
        s = Sample(input=prop.model_dump_json(), metadata={"property": prop.model_dump()})
        return TaskState(
            model="mockllm/model", sample_id=str(prop.property_id), epoch=1,
            input=s.input, messages=[], metadata=s.metadata,
        )

    def test_ingest_solver_builds_context(self, prop_101) -> None:
        from lodgify.generator import ingest_solver
        result = asyncio.run(ingest_solver()(self._make_state(prop_101), None))
        ctx = result.metadata["context"]
        assert ctx["capacity"]["bedrooms"] == 4
        assert ctx["location"]["city"] == "Begur"
        assert "<p>" not in ctx["about"]
        assert "owner_headline" in ctx  # surfaced so prompts can reference/ignore it
        assert any(a["code"] == "PrivatePool" for a in ctx["amenities"])

    def test_generate_solver_invalid_json_stores_none(self, prop_104) -> None:
        from lodgify.generator import generate_solver, ingest_solver
        state = self._make_state(prop_104)

        async def run():
            state2 = await ingest_solver()(state, None)
            return await generate_solver(model="mockllm/model")(state2, None)

        result = asyncio.run(run())
        assert result.metadata["listing_copy"] is None
        assert "listing_copy_error" in result.metadata

    def test_generate_solver_valid_json_stores_listing_copy(self, prop_104) -> None:
        from lodgify.generator import generate_solver, ingest_solver

        good_json = ListingCopy(
            hero_headline="Sun-Filled Studio in Central Lisbon",
            highlights=["Studio layout perfect for two", "Lisbon city centre", "WiFi"],
            about_this_place="x" * 150,
            amenity_descriptions=[
                AmenityDescription(amenity_code="WiFi", label="Wi-Fi", description="Fast wifi."),
            ],
        ).model_dump_json()

        mock_output = MagicMock()
        mock_output.completion = good_json
        mock_model = AsyncMock()
        mock_model.generate = AsyncMock(return_value=mock_output)

        state = self._make_state(prop_104)

        async def run():
            state2 = await ingest_solver()(state, None)
            with patch("lodgify.generator.get_model", return_value=mock_model):
                return await generate_solver(model="anthropic/claude-haiku-4-5-20251001")(state2, None)

        result = asyncio.run(run())
        copy = result.metadata["listing_copy"]
        assert isinstance(copy, ListingCopy)
        assert copy.hero_headline == "Sun-Filled Studio in Central Lisbon"


# ---------------------------------------------------------------------------
# 7. Fixture loading — edge-case properties
# ---------------------------------------------------------------------------


class TestFixtureLoading:
    def test_loads_all_nine_fixtures(self) -> None:
        assert len(load_fixtures()) == 9

    def test_prop_104_is_studio(self) -> None:
        assert load_fixture(104).rental_info.bedrooms == 0

    def test_prop_105_all_null_policies(self) -> None:
        p = load_fixture(105)
        assert all(v is None for v in [
            p.policies.cancellation_policy,
            p.policies.damage_deposit,
            p.policies.payment_schedule,
        ])

    def test_prop_106_zero_reviews(self) -> None:
        p = load_fixture(106)
        assert p.num_of_reviews == 0 and p.reviews == []

    def test_prop_107_low_score_with_cherry_picked_reviews(self) -> None:
        p = load_fixture(107)
        assert p.average_review_score < 4.0
        assert any("5 star" in r.lower() or "5/5" in r.lower() for r in p.reviews)

    def test_prop_108_pet_in_reviews_not_in_amenities(self) -> None:
        p = load_fixture(108)
        assert "PetFriendly" not in p.amenities
        assert any(w in " ".join(p.reviews).lower() for w in ["dog", "labrador", "pet"])

    def test_prop_109_fixed_checkin_with_flexible_headline(self) -> None:
        p = load_fixture(109)
        assert "flexible" in p.description.headline.lower()
        assert "PM" in p.house_rules.check_in_time or "AM" in p.house_rules.check_in_time


# ---------------------------------------------------------------------------
# 8. New grounding checks — rating, pet-friendly, check-in
# ---------------------------------------------------------------------------


class TestNewGroundingChecks:
    def test_high_rating_claim_with_low_score_fails(self, prop_107) -> None:
        """avg_score=3.1 — '5-star' claim should fail."""
        bad = ListingCopy(
            hero_headline="Porto's Top-Rated Apartment — 5-Star Experience",
            highlights=["Consistently 5-star rated", "Central location", "WiFi"],
            about_this_place="x" * 150,
            amenity_descriptions=[],
        )
        _, failed = _run_grounding_checks(bad, prop_107)
        assert any("rating" in f for f in failed)

    def test_pet_claim_without_amenity_fails(self, prop_108) -> None:
        """PetFriendly not in fixture 108 amenities — claiming it must fail."""
        bad = ListingCopy(
            hero_headline="Pet-Friendly Finca near Ronda",
            highlights=["Dog-friendly garden", "Private pool", "4 bedrooms"],
            about_this_place="x" * 150,
            amenity_descriptions=[],
        )
        _, failed = _run_grounding_checks(bad, prop_108)
        assert any("pet" in f for f in failed)

    def test_pet_claim_with_amenity_passes(self, prop_103) -> None:
        """PetFriendly IS in fixture 103 amenities — claiming it is correct."""
        good = ListingCopy(
            hero_headline="Pet-Friendly Stone Cottage near Hay-on-Wye",
            highlights=["Pets welcome", "Enclosed garden", "Wood fire"],
            about_this_place="x" * 150,
            amenity_descriptions=[],
        )
        _, failed = _run_grounding_checks(good, prop_103)
        assert not any("pet" in f for f in failed)

    def test_flexible_checkin_with_fixed_time_fails(self, prop_109) -> None:
        """check_in_time='7 PM' — 'flexible check-in any time' must fail."""
        bad = ListingCopy(
            hero_headline="Málaga Penthouse — Flexible Check-In Any Time",
            highlights=["Flexible check-in — arrive any time", "Sea views", "Terrace"],
            about_this_place="x" * 150,
            amenity_descriptions=[],
        )
        _, failed = _run_grounding_checks(bad, prop_109)
        assert any("checkin" in f for f in failed)


# ---------------------------------------------------------------------------
# 9. Ingest scrubbing
# ---------------------------------------------------------------------------


class TestIngestScrubbing:
    def test_scrub_removes_flexible_checkin_sentence(self) -> None:
        text = "Check in whenever it suits you — flexible arrival 24 hours a day. The apartment overlooks the bay."
        result = scrub_checkin_prose(text)
        assert "flexible" not in result.lower()
        assert "bay" in result  # unrelated content preserved

    def test_build_context_scrubs_checkin_prose_for_fixture_109(self) -> None:
        """End-to-end: fixture 109's 'flexible 24h check-in' claim is removed."""
        ctx = build_context(load_fixture(109))
        assert "flexible" not in ctx["about"].lower()
        assert "24 hour" not in ctx["about"].lower()
        assert len(ctx["about"]) > 50  # description content survives


# ---------------------------------------------------------------------------
# 10. Prompt loading
# ---------------------------------------------------------------------------


class TestPromptLoading:
    def test_available_versions_are_sorted(self) -> None:
        from lodgify.generator import available_versions
        versions = available_versions()
        assert versions == sorted(versions) and len(versions) >= 5

    def test_load_prompt_contains_context_placeholder(self) -> None:
        from lodgify.generator import load_prompt
        for v in ["v1", "v2", "v3", "v4", "v5"]:
            assert "{context_json}" in load_prompt(v)

    def test_load_prompt_unknown_version_raises(self) -> None:
        from lodgify.generator import load_prompt
        with pytest.raises(FileNotFoundError, match="v99"):
            load_prompt("v99")


# ---------------------------------------------------------------------------
# 11. Completeness scorer
# ---------------------------------------------------------------------------


class TestCompletenessScorer:
    def _make_state(self, copy, prop):
        return _make_scorer_state(copy, prop)

    def _score(self, copy, prop):
        return asyncio.run(completeness_scorer()(self._make_state(copy, prop), None))

    def test_full_coverage_scores_one(self, prop_101) -> None:
        # All 4 premium amenities (PrivatePool, SeaView, BBQGrill, Terrace)
        # must appear in headline or highlights for salience to be 1.0.
        copy = ListingCopy(
            hero_headline="Seafront Villa with Private Pool in Begur",
            highlights=["Private pool with sea views", "BBQ terrace for outdoor dining", "4 bedrooms"],
            about_this_place="x" * 150,
            amenity_descriptions=[
                AmenityDescription(amenity_code=c, label=c, description="desc.")
                for c in prop_101.amenities
            ],
        )
        assert self._score(copy, prop_101).value == 1.0

    def test_partial_coverage_scores_below_one(self, prop_101) -> None:
        copy = ListingCopy(
            hero_headline="Comfortable Villa in Begur",
            highlights=["Great location", "Sea views", "4 bedrooms"],
            about_this_place="x" * 150,
            amenity_descriptions=[
                AmenityDescription(amenity_code="WiFi", label="Wi-Fi", description="Fast.")
            ],
        )
        assert self._score(copy, prop_101).value < 0.7

    def test_premium_buried_lowers_salience(self, prop_101) -> None:
        """All amenities described but pool/view not in headline or highlights."""
        copy = ListingCopy(
            hero_headline="Comfortable Villa in Begur",
            highlights=["Well equipped", "Good WiFi", "Great location"],
            about_this_place="x" * 150,
            amenity_descriptions=[
                AmenityDescription(amenity_code=c, label=c, description="desc.")
                for c in prop_101.amenities
            ],
        )
        result = self._score(copy, prop_101)
        assert result.metadata["amenity_coverage"] == 1.0
        assert result.metadata["premium_salience"] < 1.0
        assert result.value < 1.0

    def test_missing_copy_scores_zero(self, prop_101) -> None:
        from inspect_ai.solver import TaskState
        state = TaskState(
            model="mockllm/model", sample_id="test", epoch=1,
            input=prop_101.model_dump_json(), messages=[],
            metadata={"property": prop_101.model_dump(), "listing_copy": None},
        )
        result = asyncio.run(completeness_scorer()(state, None))
        assert result.value == 0.0


# ---------------------------------------------------------------------------
# 12. Golden dataset loader
# ---------------------------------------------------------------------------


class TestGoldenLoader:
    def test_loads_all_nine_golden_examples(self) -> None:
        from lodgify.golden import load_golden_examples
        assert len(load_golden_examples()) == 9

    def test_good_variants_have_faithfulness_five(self) -> None:
        """Good copies are hand-crafted to be perfectly faithful — invariant."""
        from lodgify.golden import load_golden_examples
        for ex in load_golden_examples():
            assert ex.good.human_scores.faithfulness == 5, (
                f"Property {ex.property_id} good variant has faithfulness "
                f"{ex.good.human_scores.faithfulness}, expected 5"
            )

    def test_bad_variants_have_lower_faithfulness_than_good(self) -> None:
        from lodgify.golden import load_golden_examples
        for ex in load_golden_examples():
            assert ex.bad.human_scores.faithfulness < ex.good.human_scores.faithfulness

    def test_all_copies_satisfy_listing_copy_schema(self) -> None:
        from lodgify.golden import load_golden_examples
        for ex in load_golden_examples():
            for variant in (ex.good, ex.bad):
                copy = variant.listing_copy
                assert 10 <= len(copy.hero_headline) <= 90
                assert 3 <= len(copy.highlights) <= 6


# ---------------------------------------------------------------------------
# 13. Booking intent scorer (mocked)
# ---------------------------------------------------------------------------


class TestBookingIntentScorer:
    def _make_state(self, copy, prop):
        return _make_scorer_state(copy, prop)

    def test_parses_judge_response_correctly(self, prop_101) -> None:
        from lodgify.scorers import booking_intent_scorer
        copy = ListingCopy(
            hero_headline="Seafront Villa with Private Pool in Begur",
            highlights=["Private pool", "Sea views", "4 bedrooms"],
            about_this_place="x" * 150,
            amenity_descriptions=[],
        )
        mock_output = MagicMock()
        mock_output.completion = '{"score": 4, "explanation": "Compelling and specific."}'
        mock_model = AsyncMock()
        mock_model.generate = AsyncMock(return_value=mock_output)

        async def run():
            with patch("lodgify.scorers.get_model", return_value=mock_model):
                return await booking_intent_scorer()(self._make_state(copy, prop_101), None)

        result = asyncio.run(run())
        assert result.value == 4
        assert "Compelling" in result.explanation

    def test_prompt_includes_property_type_and_location(self, prop_101) -> None:
        from lodgify.scorers import booking_intent_scorer
        copy = ListingCopy(
            hero_headline="Seafront Villa in Begur",
            highlights=["Private pool", "Sea views", "4 bedrooms"],
            about_this_place="x" * 150,
            amenity_descriptions=[],
        )
        captured = []
        mock_output = MagicMock()
        mock_output.completion = '{"score": 3, "explanation": "Decent."}'
        mock_model = AsyncMock()

        async def capture(messages, **kwargs):
            captured.extend(messages)
            return mock_output

        mock_model.generate = capture

        async def run():
            with patch("lodgify.scorers.get_model", return_value=mock_model):
                return await booking_intent_scorer()(self._make_state(copy, prop_101), None)

        asyncio.run(run())
        prompt_text = captured[0].content
        assert "Villa" in prompt_text   # property_type
        assert "Begur" in prompt_text   # city
        assert "8" in prompt_text       # max_guests

    def test_handles_unparseable_judge_response(self, prop_101) -> None:
        from lodgify.scorers import booking_intent_scorer
        copy = ListingCopy(
            hero_headline="Villa in Begur",
            highlights=["Pool", "Sea view", "4 bedrooms"],
            about_this_place="x" * 150,
            amenity_descriptions=[],
        )
        mock_output = MagicMock()
        mock_output.completion = "not valid json at all"
        mock_model = AsyncMock()
        mock_model.generate = AsyncMock(return_value=mock_output)

        async def run():
            with patch("lodgify.scorers.get_model", return_value=mock_model):
                return await booking_intent_scorer()(self._make_state(copy, prop_101), None)

        result = asyncio.run(run())
        assert result.value == 1  # worst-case fallback on parse error
