"""Ollama backend adapter (Module 7/8: provider-agnostic LLM client).

OllamaClient mimics google-genai's client.models.generate_content(model,
contents, config) call signature and response shape (.text, .parsed,
.candidates[0].content.parts[*].function_call, .usage_metadata) so that
router.py, orchestrator.py, and tools.py need no changes beyond swapping
which client factory they call — they already read Gemini-shaped objects,
this adapter conforms to that shape rather than the other way around.

Wire format reference (Ollama /api/chat, non-streamed):
- request: {"model", "messages": [...], "stream": false,
  "format": <json-schema, optional>, "tools": <openai-style, optional>,
  "options": {"temperature": ...}}
- assistant tool call: {"role": "assistant", "content": "",
  "tool_calls": [{"function": {"name": ..., "arguments": <dict>}}]}
  (arguments is already a parsed dict in Ollama, unlike OpenAI's JSON string)
- tool result: {"role": "tool", "content": <json string>} — no tool_call_id,
  Ollama doesn't use correlation IDs
- token counts are top-level on the response: prompt_eval_count, eval_count
  (not nested under a "usage" key)
"""

import json

import requests
from google.genai import types
from pydantic import ValidationError

_OLLAMA_CONNECT_TIMEOUT = 5
_OLLAMA_READ_TIMEOUT = 120


class LLMBackendError(Exception):
    """Raised on any Ollama connectivity/HTTP failure. Carries .code so
    callers that already handle google.genai.errors.APIError (which also
    exposes .code) can catch both with one except clause."""

    def __init__(self, message: str, code: int = 503):
        super().__init__(message)
        self.code = code


class _UsageMetadata:
    def __init__(self, prompt_tokens: int, completion_tokens: int):
        self.prompt_token_count = prompt_tokens
        self.candidates_token_count = completion_tokens
        self.total_token_count = prompt_tokens + completion_tokens


class _Candidate:
    def __init__(self, content: types.Content):
        self.content = content


class _OllamaResponse:
    def __init__(self, message: dict, usage: _UsageMetadata, response_schema=None):
        self.usage_metadata = usage
        self.text = message.get("content") or None
        self.parsed = self._parse(message, response_schema)
        self.candidates = [_Candidate(self._build_content(message))]

    @staticmethod
    def _parse(message: dict, response_schema):
        if response_schema is None:
            return None
        content = message.get("content")
        if not content:
            return None
        try:
            return response_schema.model_validate_json(content)
        except (ValidationError, ValueError):  # fail closed, never raise
            return None

    @staticmethod
    def _build_content(message: dict) -> types.Content:
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            parts = [
                types.Part(
                    function_call=types.FunctionCall(
                        name=call["function"]["name"],
                        args=call["function"].get("arguments") or {},
                    )
                )
                for call in tool_calls
            ]
            return types.Content(role="model", parts=parts)
        return types.Content(role="model", parts=[types.Part(text=message.get("content") or "")])


class _OllamaModels:
    def __init__(self, base_url: str, model: str):
        self._base_url = base_url.rstrip("/")
        self._model = model

    def generate_content(self, model, contents, config) -> _OllamaResponse:
        # `model` (whatever the caller resolved config.ACTIVE_CHAT_MODEL to) is
        # ignored — this client always serves whichever model it was constructed
        # with, from OLLAMA_CHAT_MODEL.
        messages = _contents_to_messages(contents, getattr(config, "system_instruction", None))
        body = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": getattr(config, "temperature", None) or 0.0},
        }
        response_schema = getattr(config, "response_schema", None)
        if response_schema is not None:
            body["format"] = response_schema.model_json_schema()
        tools = getattr(config, "tools", None)
        if tools:
            body["tools"] = _tools_to_ollama(tools)

        try:
            resp = requests.post(
                f"{self._base_url}/api/chat",
                json=body,
                timeout=(_OLLAMA_CONNECT_TIMEOUT, _OLLAMA_READ_TIMEOUT),
            )
        except requests.exceptions.RequestException as exc:
            raise LLMBackendError(f"Ollama request failed: {exc}", code=503) from exc

        if resp.status_code != 200:
            raise LLMBackendError(
                f"Ollama returned HTTP {resp.status_code}: {resp.text[:300]}",
                code=resp.status_code,
            )

        data = resp.json()
        usage = _UsageMetadata(
            prompt_tokens=data.get("prompt_eval_count", 0) or 0,
            completion_tokens=data.get("eval_count", 0) or 0,
        )
        return _OllamaResponse(data.get("message", {}), usage, response_schema=response_schema)


class OllamaClient:
    """Drop-in stand-in for google.genai.Client, scoped to what this project
    calls: client.models.generate_content(model, contents, config)."""

    def __init__(self, base_url: str, model: str):
        self.model = model
        self.models = _OllamaModels(base_url, model)


def _contents_to_messages(contents, system_instruction: str | None) -> list[dict]:
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})

    if isinstance(contents, str):
        messages.append({"role": "user", "content": contents})
        return messages

    for content in contents:
        parts = list(content.parts or [])
        function_responses = [p for p in parts if getattr(p, "function_response", None)]
        function_calls = [p for p in parts if getattr(p, "function_call", None)]

        if function_responses:
            for part in function_responses:
                fr = part.function_response
                messages.append({"role": "tool", "content": json.dumps(fr.response)})
            continue

        if function_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": part.function_call.name,
                                "arguments": dict(part.function_call.args or {}),
                            }
                        }
                        for part in function_calls
                    ],
                }
            )
            continue

        text = "".join(p.text or "" for p in parts if getattr(p, "text", None))
        role = "assistant" if content.role == "model" else "user"
        messages.append({"role": role, "content": text})

    return messages


def _tools_to_ollama(tools: list) -> list[dict]:
    declarations = []
    for tool in tools:
        declarations.extend(tool.function_declarations or [])
    return [
        {
            "type": "function",
            "function": {
                "name": fd.name,
                "description": fd.description or "",
                "parameters": _schema_to_dict(fd.parameters),
            },
        }
        for fd in declarations
    ]


def _schema_to_dict(schema) -> dict:
    if schema is None:
        return {"type": "object", "properties": {}}
    type_map = {"OBJECT": "object", "STRING": "string", "ARRAY": "array", "NUMBER": "number", "BOOLEAN": "boolean"}
    result: dict = {"type": type_map.get(schema.type, "object")}
    if getattr(schema, "description", None):
        result["description"] = schema.description
    if getattr(schema, "enum", None):
        result["enum"] = list(schema.enum)
    if getattr(schema, "properties", None):
        result["properties"] = {name: _schema_to_dict(prop) for name, prop in schema.properties.items()}
    if getattr(schema, "required", None):
        result["required"] = list(schema.required)
    if getattr(schema, "items", None):
        result["items"] = _schema_to_dict(schema.items)
    return result
