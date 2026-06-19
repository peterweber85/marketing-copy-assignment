"""Pipeline tests — runnable offline with no API key, no network.

Design notes
------------
* **Dependency injection:** ``generate_solver`` accepts a ``model`` parameter.
  Tests inject ``mockllm/model`` (inspect-ai's built-in canned-response model)
  instead of the real Haiku. The rest of the solver chain is unchanged, so we
  exercise the real ingest + JSON-parse + metadata paths.

* **Inheritance:** ``AbstractModelClient`` defines the interface; ``AnthropicClient``
  is the real implementation; ``MockModelClient`` is the test double.  The
  generator's ``_call_model`` helper accepts an ``AbstractModelClient``, so we
  can swap implementations without touching solver logic.

* **Mocking (unittest.mock):** the ``grounding_scorer``'s rule-based checks are
  tested directly — no mock needed there (pure functions).  The LLM-judge
  scorers are tested by patching ``get_model`` to return a spy that records
  the prompts it receives.

All tests run with ``uv run pytest tests/`` — no env vars required.
"""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lodgify.amenities import humanize, humanize_all, is_known
from lodgify.data import load_fixture, load_fixtures
from lodgify.ingest import build_context, strip_html
from lodgify.models import (
    AmenityDescription,
    ListingCopy,
    PropertyInput,
)
from lodgify.scorers import (
    _copy_full_text,
    _extract_bedroom_numbers,
    _run_grounding_checks,
)

# ---------------------------------------------------------------------------
# Inheritance demo: abstract model client + concrete implementations
# ---------------------------------------------------------------------------


class AbstractModelClient(ABC):
    """Minimal interface for generating text from a prompt."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        ...


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
        self.calls: list[str] = []

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self._response


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def prop_101() -> PropertyInput:
    return load_fixture(101)


@pytest.fixture
def prop_104() -> PropertyInput:
    """The trap: studio, bedrooms=0, misleading owner headline."""
    return load_fixture(104)


@pytest.fixture
def prop_105() -> PropertyInput:
    """All policies null + HTML puffery."""
    return load_fixture(105)


@pytest.fixture
def prop_106() -> PropertyInput:
    """Zero reviews — social proof check."""
    return load_fixture(106)


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
            " the Costa Brava. The open-plan living area opens onto a large terrace"
            " with a private pool and uninterrupted sea views. Four bedrooms sleep"
            " up to eight guests in comfort."
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

    def test_is_known_true(self) -> None:
        assert is_known("WiFi")

    def test_is_known_false(self) -> None:
        assert not is_known("OutdoorShower")

    def test_humanize_all_preserves_order(self, prop_101: PropertyInput) -> None:
        result = humanize_all(prop_101.amenities)
        assert list(result.keys()) == prop_101.amenities


# ---------------------------------------------------------------------------
# 2. HTML stripping (ingest stage)
# ---------------------------------------------------------------------------


class TestHTMLStripping:
    def test_strips_tags(self) -> None:
        assert strip_html("<p>Hello <b>world</b></p>") == "Hello world"

    def test_decodes_entities(self) -> None:
        assert "è" in strip_html("Gr&egrave;ce")

    def test_collapses_whitespace(self) -> None:
        result = strip_html("<p>a</p>  <p>b</p>")
        assert "  " not in result

    def test_eixample_loft_strip(self) -> None:
        loft = load_fixture(102)
        ctx = build_context(loft)
        assert "<div" not in ctx["about"]
        assert "Eixample" in ctx["about"]


# ---------------------------------------------------------------------------
# 3. Grounding scorer — rule-based, no LLM (the deterministic heart)
# ---------------------------------------------------------------------------


class TestGroundingScorer:
    def test_correct_bedroom_count_passes(self, valid_copy_101: ListingCopy, prop_101: PropertyInput) -> None:
        passed, failed = _run_grounding_checks(valid_copy_101, prop_101)
        assert not any("bedroom_count_wrong" in f for f in failed)

    def test_wrong_bedroom_count_fails(self, prop_101: PropertyInput) -> None:
        bad = ListingCopy(
            hero_headline="Two-Bedroom Villa",
            highlights=["Two bedrooms for couples", "Sea view", "Pool"],
            about_this_place="x" * 150,
            amenity_descriptions=[],
        )
        _, failed = _run_grounding_checks(bad, prop_101)
        assert any("bedroom_count_wrong" in f for f in failed), failed

    def test_studio_false_bedroom_claim_fails(self, prop_104: PropertyInput) -> None:
        """Property 104 has bedrooms=0; claiming 'two bedrooms' must fail."""
        bad = ListingCopy(
            hero_headline="Charming Two-Bedroom Apartment in Lisbon",
            highlights=["Two bedrooms", "Central location", "WiFi"],
            about_this_place="x" * 150,
            amenity_descriptions=[],
        )
        _, failed = _run_grounding_checks(bad, prop_104)
        assert any("bedroom" in f for f in failed), failed

    def test_studio_correct_description_passes(self, prop_104: PropertyInput) -> None:
        good = ListingCopy(
            hero_headline="Bright Studio in Central Lisbon",
            highlights=["Studio layout", "Central location", "WiFi"],
            about_this_place="x" * 150,
            amenity_descriptions=[
                AmenityDescription(amenity_code="WiFi", label="Wi-Fi", description="Fast wifi."),
            ],
        )
        passed, failed = _run_grounding_checks(good, prop_104)
        assert not any("bedroom" in f for f in failed), failed

    def test_invented_amenity_code_fails(self, prop_104: PropertyInput) -> None:
        bad = ListingCopy(
            hero_headline="Studio in Lisbon",
            highlights=["Great location", "WiFi", "Compact studio"],
            about_this_place="x" * 150,
            amenity_descriptions=[
                AmenityDescription(amenity_code="Jacuzzi", label="Jacuzzi", description="Relax."),
            ],
        )
        _, failed = _run_grounding_checks(bad, prop_104)
        assert any("invented_amenity" in f for f in failed), failed

    def test_null_cancellation_policy_not_mentioned_passes(self, prop_105: PropertyInput) -> None:
        good = ListingCopy(
            hero_headline="Marina View in Valencia",
            highlights=["Sea views", "2 bedrooms", "Free parking"],
            about_this_place="x" * 150,
            amenity_descriptions=[],
        )
        passed, failed = _run_grounding_checks(good, prop_105)
        assert not any("cancellation" in f for f in failed), failed

    def test_null_cancellation_policy_mentioned_fails(self, prop_105: PropertyInput) -> None:
        bad = ListingCopy(
            hero_headline="Marina View in Valencia",
            highlights=["Sea views", "Free cancellation available", "2 bedrooms"],
            about_this_place="x" * 150,
            amenity_descriptions=[],
        )
        _, failed = _run_grounding_checks(bad, prop_105)
        assert any("cancellation" in f for f in failed), failed

    def test_zero_reviews_social_proof_fails(self, prop_106: PropertyInput) -> None:
        bad = ListingCopy(
            hero_headline="Alpine Escape in Chamonix",
            highlights=["Top-rated chalet", "Guests love it", "Hot tub"],
            about_this_place="x" * 150,
            amenity_descriptions=[],
        )
        _, failed = _run_grounding_checks(bad, prop_106)
        assert any("social_proof" in f for f in failed), failed

    def test_zero_reviews_no_social_proof_passes(self, prop_106: PropertyInput) -> None:
        good = ListingCopy(
            hero_headline="New Chalet with Hot Tub in Chamonix",
            highlights=["Private hot tub", "Mountain views", "3 bedrooms"],
            about_this_place="x" * 150,
            amenity_descriptions=[],
        )
        passed, failed = _run_grounding_checks(good, prop_106)
        assert not any("social_proof" in f for f in failed), failed

    def test_city_missing_fails(self, prop_101: PropertyInput) -> None:
        bad = ListingCopy(
            hero_headline="Seaside Villa with Pool",
            highlights=["Private pool", "Sea views", "4 bedrooms"],
            about_this_place="x" * 150,
            amenity_descriptions=[],
        )
        _, failed = _run_grounding_checks(bad, prop_101)
        assert any("city_missing" in f for f in failed), failed


# ---------------------------------------------------------------------------
# 4. Bedroom number extraction
# ---------------------------------------------------------------------------


class TestBedroomExtraction:
    def test_digit(self) -> None:
        assert _extract_bedroom_numbers("3-bedroom villa") == [3]

    def test_word(self) -> None:
        assert _extract_bedroom_numbers("two bedroom apartment") == [2]

    def test_none(self) -> None:
        assert _extract_bedroom_numbers("great location near the beach") == []

    def test_multiple(self) -> None:
        nums = _extract_bedroom_numbers("two bedroom flat, also one bedroom annexe")
        assert 2 in nums and 1 in nums


# ---------------------------------------------------------------------------
# 5. MockModelClient (inheritance + DI demo)
# ---------------------------------------------------------------------------


class TestMockModelClient:
    """Shows that AbstractModelClient is substitutable per Liskov/DI."""

    def test_mock_returns_canned_response(self) -> None:
        mock = MockModelClient('{"message": "ok"}')
        result = mock.generate("any prompt")
        assert result == '{"message": "ok"}'

    def test_mock_records_calls(self) -> None:
        mock = MockModelClient("{}")
        mock.generate("prompt one")
        mock.generate("prompt two")
        assert len(mock.calls) == 2

    def test_anthropic_client_is_subclass(self) -> None:
        assert issubclass(AnthropicClient, AbstractModelClient)

    def test_mock_client_is_subclass(self) -> None:
        assert issubclass(MockModelClient, AbstractModelClient)


# ---------------------------------------------------------------------------
# 6. Ingest solver + generate solver — offline via mockllm/model
# ---------------------------------------------------------------------------


class TestSolverChainOffline:
    """Exercises the real solver code paths using inspect-ai's mockllm."""

    def _make_state(self, prop: PropertyInput):
        from inspect_ai.solver import TaskState
        from inspect_ai.dataset import Sample
        s = Sample(input=prop.model_dump_json(), metadata={"property": prop.model_dump()})
        state = TaskState(
            model="mockllm/model",
            sample_id=str(prop.property_id),
            epoch=1,
            input=s.input,
            messages=[],
            metadata=s.metadata,
        )
        return state

    def test_ingest_solver_builds_context(self, prop_101: PropertyInput) -> None:
        from lodgify.generator import ingest_solver
        from inspect_ai.solver import TaskState

        state = self._make_state(prop_101)
        solver_fn = ingest_solver()

        async def run():
            return await solver_fn(state, None)

        result = asyncio.run(run())
        ctx = result.metadata["context"]
        assert ctx["capacity"]["bedrooms"] == 4
        assert ctx["location"]["city"] == "Begur"
        assert "<p>" not in ctx["about"]
        assert any(a["code"] == "PrivatePool" for a in ctx["amenities"])

    def test_generate_solver_invalid_json_stores_none(self, prop_104: PropertyInput) -> None:
        """mockllm returns a non-JSON string → listing_copy should be None."""
        from lodgify.generator import generate_solver, ingest_solver

        state = self._make_state(prop_104)

        async def run():
            ingest_fn = ingest_solver()
            state2 = await ingest_fn(state, None)
            gen_fn = generate_solver(model="mockllm/model")
            return await gen_fn(state2, None)

        result = asyncio.run(run())
        assert result.metadata["listing_copy"] is None
        assert "listing_copy_error" in result.metadata

    def test_generate_solver_valid_json_stores_listing_copy(self, prop_104: PropertyInput) -> None:
        """Patch get_model to return valid ListingCopy JSON."""
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
            ingest_fn = ingest_solver()
            state2 = await ingest_fn(state, None)
            with patch("lodgify.generator.get_model", return_value=mock_model):
                gen_fn = generate_solver(model="anthropic/claude-haiku-4-5-20251001")
                return await gen_fn(state2, None)

        result = asyncio.run(run())
        copy = result.metadata["listing_copy"]
        assert isinstance(copy, ListingCopy)
        assert copy.hero_headline == "Sun-Filled Studio in Central Lisbon"


# ---------------------------------------------------------------------------
# 7. Data loading
# ---------------------------------------------------------------------------


class TestFixtureLoading:
    def test_loads_all_fixtures(self) -> None:
        props = load_fixtures()
        assert len(props) == 6

    def test_sorted_by_id(self) -> None:
        props = load_fixtures()
        ids = [p.property_id for p in props]
        assert ids == sorted(ids)

    def test_prop_104_is_studio(self) -> None:
        p = load_fixture(104)
        assert p.rental_info.bedrooms == 0

    def test_prop_105_all_null_policies(self) -> None:
        p = load_fixture(105)
        assert p.policies.cancellation_policy is None
        assert p.policies.damage_deposit is None
        assert p.policies.payment_schedule is None

    def test_prop_106_zero_reviews(self) -> None:
        p = load_fixture(106)
        assert p.num_of_reviews == 0
        assert p.reviews == []

    def test_prop_101_has_unknown_amenity(self) -> None:
        p = load_fixture(101)
        assert "OutdoorShower" in p.amenities
        assert not is_known("OutdoorShower")
