import json
from types import SimpleNamespace

import httpx
import pytest

from app.providers.base import (
    ProviderOutputError,
    ProviderUnavailable,
    RawCitation,
    RetrievedChunk,
)
from app.providers.ollama import OllamaProvider

SETTINGS = SimpleNamespace(ollama_url="http://ollama", ollama_model="qwen3:8b", ollama_think=False)
CHUNKS = [
    RetrievedChunk(1, "Leave Policy", "2.1", "Annual Leave", "25 days.", 0.8),
    RetrievedChunk(2, "Expense Policy", "3.2", "Mileage", "45 pence.", 0.7),
]


def reply(content):
    return httpx.Response(200, json={"message": {"content": content}})


def good_reply(chunk_ref="S1", quote="25 days."):
    return reply(json.dumps({"answer": "A", "citations": [{"chunk_ref": chunk_ref, "quote": quote}]}))


def provider(handler, settings=SETTINGS):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OllamaProvider(settings, client=client)


def test_sends_schema_and_think_setting():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return good_reply()

    provider(handler).answer("q", CHUNKS)
    assert set(bodies[0]["format"]["properties"]) == {"covered", "answer", "citations"}
    assert bodies[0]["think"] is False
    assert bodies[0]["stream"] is False


def test_think_setting_can_be_enabled():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return good_reply()

    settings = SimpleNamespace(**{**vars(SETTINGS), "ollama_think": True})
    provider(handler, settings).answer("q", CHUNKS)
    assert bodies[0]["think"] is True


def test_labels_sections_S1_S2():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return good_reply()

    provider(handler).answer("how much?", CHUNKS)
    user = bodies[0]["messages"][-1]["content"]
    assert "[S1]" in user and "[S2]" in user
    assert "Leave Policy" in user and "3.2 Mileage" in user and "how much?" in user


def test_maps_chunk_ref_to_index():
    result = provider(lambda r: good_reply("S2", "x")).answer("q", CHUNKS)
    assert result.answer == "A"
    assert result.citations == [RawCitation(1, "x")]


def test_bad_chunk_ref_becomes_unknown_index():
    result = provider(lambda r: good_reply("nope", "x")).answer("q", CHUNKS)
    assert result.citations == [RawCitation(-1, "x")]


def test_invalid_json_retries_once_then_succeeds():
    calls = []

    def handler(request):
        calls.append(1)
        return reply("not json") if len(calls) == 1 else good_reply()

    assert provider(handler).answer("q", CHUNKS).answer == "A"
    assert len(calls) == 2


def test_invalid_json_twice_raises():
    with pytest.raises(ProviderOutputError):
        provider(lambda r: reply("not json")).answer("q", CHUNKS)


def test_wrong_shape_counts_as_invalid():
    with pytest.raises(ProviderOutputError):
        provider(lambda r: reply('{"answer": 3}')).answer("q", CHUNKS)


def test_connection_error_raises_unavailable():
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(ProviderUnavailable, match="ollama serve"):
        provider(handler).answer("q", CHUNKS)


def test_missing_model_raises_unavailable():
    with pytest.raises(ProviderUnavailable, match="ollama pull qwen3:8b"):
        provider(lambda r: httpx.Response(404, json={"error": "x"})).answer("q", CHUNKS)


def test_default_client_allows_slow_model_load():
    assert OllamaProvider(SETTINGS).client.timeout.read == 180


def test_check_reports_missing_model():
    tags = {"models": [{"name": "qwen2.5:7b"}]}
    with pytest.raises(ProviderUnavailable, match="ollama pull qwen3:8b"):
        provider(lambda r: httpx.Response(200, json=tags)).check()


def test_check_passes_when_model_present():
    tags = {"models": [{"name": "qwen3:8b"}]}
    provider(lambda r: httpx.Response(200, json=tags)).check()


def test_reminder_about_labels_comes_after_the_question():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return good_reply()

    provider(handler).answer("how much?", CHUNKS)
    user = bodies[0]["messages"][-1]["content"]
    assert user.index("how much?") < user.index("Do not mention labels")


def test_chunk_ref_can_name_document_and_section():
    result = provider(lambda r: good_reply("Expense Policy, section 3.2 Mileage", "x")).answer("q", CHUNKS)
    assert result.citations == [RawCitation(1, "x")]


def test_chunk_ref_with_wrong_section_number_is_unknown():
    result = provider(lambda r: good_reply("Expense Policy, section 9.9 Mileage", "x")).answer("q", CHUNKS)
    assert result.citations == [RawCitation(-1, "x")]


def test_schema_asks_whether_policies_cover_the_question():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return good_reply()

    provider(handler).answer("q", CHUNKS)
    schema = bodies[0]["format"]
    assert "covered" in schema["properties"] and "covered" in schema["required"]


def test_reads_covered_flag():
    content = json.dumps({"covered": False, "answer": "Not covered.", "citations": []})
    assert provider(lambda r: reply(content)).answer("q", CHUNKS).covered is False


def test_covered_defaults_to_true_when_missing():
    assert provider(lambda r: good_reply()).answer("q", CHUNKS).covered is True


def test_unresolved_ref_is_found_by_its_quote():
    result = provider(lambda r: good_reply("Expense Policy", "45 pence.")).answer("q", CHUNKS)
    assert result.citations == [RawCitation(1, "45 pence.")]


def test_unresolved_ref_with_quote_in_no_section_stays_unknown():
    result = provider(lambda r: good_reply("Expense Policy", "99 pence.")).answer("q", CHUNKS)
    assert result.citations == [RawCitation(-1, "99 pence.")]


def test_read_timeout_says_the_model_is_slow():
    def handler(request):
        raise httpx.ReadTimeout("slow")

    with pytest.raises(ProviderUnavailable, match="took too long"):
        provider(handler).answer("q", CHUNKS)
