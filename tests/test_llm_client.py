import json

import pytest
import requests
from google.genai import types

from src.agent.llm_client import LLMBackendError, OllamaClient
from src.schemas import Intent, IntentClassification


class FakeHTTPResponse:
    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json_data = json_data or {}
        self.text = text

    def json(self):
        return self._json_data


def make_client(monkeypatch, response=None, raises=None):
    captured = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        if raises:
            raise raises
        return response

    monkeypatch.setattr(requests, "post", fake_post)
    return OllamaClient("http://ollama-host:11434", "gemma4:e4b"), captured


def test_schema_request_body_shape(monkeypatch):
    client, captured = make_client(
        monkeypatch,
        response=FakeHTTPResponse(
            json_data={
                "message": {"content": '{"intent":"faq","confidence":0.9}'},
                "prompt_eval_count": 12,
                "eval_count": 6,
            }
        ),
    )
    client.models.generate_content(
        model="gemini-2.5-flash",  # should be ignored
        contents="how many vacation days?",
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=IntentClassification,
            temperature=0.0,
        ),
    )
    body = captured["json"]
    assert body["model"] == "gemma4:e4b"  # not the passed-in gemini name
    assert body["stream"] is False
    assert body["format"] == IntentClassification.model_json_schema()
    assert body["messages"] == [{"role": "user", "content": "how many vacation days?"}]
    assert captured["url"] == "http://ollama-host:11434/api/chat"


def test_schema_response_parses_valid_json(monkeypatch):
    client, _ = make_client(
        monkeypatch,
        response=FakeHTTPResponse(
            json_data={
                "message": {"content": '{"intent":"faq","confidence":0.9}'},
                "prompt_eval_count": 12,
                "eval_count": 6,
            }
        ),
    )
    result = client.models.generate_content(
        model="x",
        contents="q",
        config=types.GenerateContentConfig(response_schema=IntentClassification),
    )
    assert result.parsed == IntentClassification(intent=Intent.FAQ, confidence=0.9)


def test_schema_response_invalid_json_fails_closed(monkeypatch):
    client, _ = make_client(
        monkeypatch,
        response=FakeHTTPResponse(
            json_data={"message": {"content": "not json at all"}, "prompt_eval_count": 1, "eval_count": 1}
        ),
    )
    result = client.models.generate_content(
        model="x", contents="q", config=types.GenerateContentConfig(response_schema=IntentClassification)
    )
    assert result.parsed is None  # no exception raised


def test_tool_request_body_shape(monkeypatch):
    client, captured = make_client(
        monkeypatch,
        response=FakeHTTPResponse(
            json_data={"message": {"content": "hi"}, "prompt_eval_count": 1, "eval_count": 1}
        ),
    )
    tool = types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="search_kb",
                description="search the kb",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "question": types.Schema(type="STRING"),
                        "category": types.Schema(type="STRING", enum=["leave", "payroll"]),
                        "parties": types.Schema(type="ARRAY", items=types.Schema(type="STRING")),
                    },
                    required=["question"],
                ),
            )
        ]
    )
    client.models.generate_content(
        model="x",
        contents="hello",
        config=types.GenerateContentConfig(system_instruction="be helpful", tools=[tool]),
    )
    body = captured["json"]
    assert body["messages"][0] == {"role": "system", "content": "be helpful"}
    assert body["messages"][1] == {"role": "user", "content": "hello"}
    fn = body["tools"][0]["function"]
    assert fn["name"] == "search_kb"
    assert fn["parameters"]["required"] == ["question"]
    assert fn["parameters"]["properties"]["category"]["enum"] == ["leave", "payroll"]
    assert fn["parameters"]["properties"]["parties"]["items"] == {"type": "string"}


def test_tool_call_response_translated_to_function_call_parts(monkeypatch):
    client, _ = make_client(
        monkeypatch,
        response=FakeHTTPResponse(
            json_data={
                "message": {
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "search_kb", "arguments": {"question": "leave policy"}}}
                    ],
                },
                "prompt_eval_count": 5,
                "eval_count": 3,
            }
        ),
    )
    result = client.models.generate_content(model="x", contents="q", config=types.GenerateContentConfig())
    parts = result.candidates[0].content.parts
    assert len(parts) == 1
    assert parts[0].function_call.name == "search_kb"
    assert dict(parts[0].function_call.args) == {"question": "leave policy"}


def test_plain_text_response_has_no_function_calls(monkeypatch):
    client, _ = make_client(
        monkeypatch,
        response=FakeHTTPResponse(
            json_data={"message": {"content": "here's your answer"}, "prompt_eval_count": 5, "eval_count": 3}
        ),
    )
    result = client.models.generate_content(model="x", contents="q", config=types.GenerateContentConfig())
    assert result.text == "here's your answer"
    parts = result.candidates[0].content.parts
    assert all(p.function_call is None for p in parts)


def test_round_trip_function_response_becomes_tool_message(monkeypatch):
    """The critical case: a Content(role="user", parts=[function_response]) must
    become a "tool" message despite its role being "user", not "assistant"."""
    client, captured = make_client(
        monkeypatch,
        response=FakeHTTPResponse(
            json_data={
                "message": {
                    "content": "",
                    "tool_calls": [{"function": {"name": "search_kb", "arguments": {"question": "x"}}}],
                },
                "prompt_eval_count": 1,
                "eval_count": 1,
            }
        ),
    )
    first = client.models.generate_content(
        model="x", contents="how many leave days?", config=types.GenerateContentConfig()
    )
    model_content = first.candidates[0].content

    contents = [
        types.Content(role="user", parts=[types.Part(text="how many leave days?")]),
        model_content,
        types.Content(
            role="user",
            parts=[types.Part.from_function_response(name="search_kb", response={"answer": "15 days"})],
        ),
    ]
    client.models.generate_content(model="x", contents=contents, config=types.GenerateContentConfig())

    messages = captured["json"]["messages"]
    assert messages[1] == {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": "search_kb", "arguments": {"question": "x"}}}],
    }
    assert messages[2] == {"role": "tool", "content": json.dumps({"answer": "15 days"})}


def test_usage_mapping(monkeypatch):
    client, _ = make_client(
        monkeypatch,
        response=FakeHTTPResponse(
            json_data={"message": {"content": "hi"}, "prompt_eval_count": 40, "eval_count": 10}
        ),
    )
    result = client.models.generate_content(model="x", contents="q", config=types.GenerateContentConfig())
    assert result.usage_metadata.prompt_token_count == 40
    assert result.usage_metadata.candidates_token_count == 10
    assert result.usage_metadata.total_token_count == 50


def test_usage_mapping_missing_keys_default_to_zero(monkeypatch):
    client, _ = make_client(monkeypatch, response=FakeHTTPResponse(json_data={"message": {"content": "hi"}}))
    result = client.models.generate_content(model="x", contents="q", config=types.GenerateContentConfig())
    assert result.usage_metadata.total_token_count == 0


def test_connection_error_raises_llm_backend_error(monkeypatch):
    client, _ = make_client(monkeypatch, raises=requests.exceptions.ConnectionError("refused"))
    with pytest.raises(LLMBackendError) as exc_info:
        client.models.generate_content(model="x", contents="q", config=types.GenerateContentConfig())
    assert exc_info.value.code == 503


def test_non_200_raises_llm_backend_error(monkeypatch):
    client, _ = make_client(monkeypatch, response=FakeHTTPResponse(status_code=500, text="server error"))
    with pytest.raises(LLMBackendError) as exc_info:
        client.models.generate_content(model="x", contents="q", config=types.GenerateContentConfig())
    assert exc_info.value.code == 500
