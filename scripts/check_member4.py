"""Quick readiness check for Member 4 services.

Run after starting the API/UI/MLflow services:

    python scripts/check_member4.py

Optional:

    python scripts/check_member4.py --api-url http://localhost:8000 --mlflow-url http://localhost:5000
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import requests


def _print_header(title: str) -> None:
    print(f"\n== {title} ==")


def _get_json(url: str, timeout: int = 10) -> tuple[bool, dict[str, Any] | str]:
    try:
        response = requests.get(url, timeout=timeout)
        if not response.ok:
            return False, f"HTTP {response.status_code}: {response.text[:300]}"
        try:
            return True, response.json()
        except ValueError:
            return True, response.text[:300]
    except requests.RequestException as exc:
        return False, str(exc)


def _post_json(url: str, payload: dict[str, Any], timeout: int = 90) -> tuple[bool, dict[str, Any] | str]:
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        if not response.ok:
            return False, f"HTTP {response.status_code}: {response.text[:300]}"
        return True, response.json()
    except requests.RequestException as exc:
        return False, str(exc)
    except ValueError as exc:
        return False, f"Invalid JSON response: {exc}"


def _status_line(name: str, ok: bool, detail: str = "") -> None:
    marker = "PASS" if ok else "FAIL"
    suffix = f" - {detail}" if detail else ""
    print(f"[{marker}] {name}{suffix}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check API, UI, and MLflow readiness.")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--ui-url", default="http://localhost:8501")
    parser.add_argument("--mlflow-url", default="http://localhost:5000")
    parser.add_argument(
        "--skip-chat",
        action="store_true",
        help="Skip POST /chat. Useful when GEMINI_API_KEY or Chroma is not ready.",
    )
    args = parser.parse_args()

    api_url = args.api_url.rstrip("/")
    ui_url = args.ui_url.rstrip("/")
    mlflow_url = args.mlflow_url.rstrip("/")

    _print_header("Useful URLs")
    print(f"API docs: {api_url}/docs")
    print(f"API health: {api_url}/health")
    print(f"Streamlit UI: {ui_url}")
    print(f"MLflow: {mlflow_url}")

    _print_header("API")
    ok, health = _get_json(f"{api_url}/health")
    _status_line("GET /health", ok)
    if ok:
        print(json.dumps(health, indent=2))
    else:
        print(health)

    ok, openapi = _get_json(f"{api_url}/openapi.json")
    endpoint_count = len(openapi.get("paths", {})) if isinstance(openapi, dict) else 0
    _status_line("OpenAPI schema", ok, f"{endpoint_count} paths" if ok else str(openapi))

    if not args.skip_chat:
        ok, chat = _post_json(
            f"{api_url}/chat",
            {"session_id": "member4-check", "message": "What is the vacation leave policy?"},
        )
        if ok and isinstance(chat, dict):
            detail = f"{len(chat.get('citations', []))} citations, {len(chat.get('sources', []))} sources"
            _status_line("POST /chat", True, detail)
            print(chat.get("reply", "")[:500])
        else:
            _status_line("POST /chat", False, str(chat))

    _print_header("MLflow")
    ok, mlflow_health = _get_json(f"{mlflow_url}/health")
    if ok:
        _status_line("MLflow /health", True)
    else:
        ok, mlflow_home = _get_json(mlflow_url)
        _status_line("MLflow home", ok, "" if ok else str(mlflow_home))

    _print_header("UI")
    ok, ui_home = _get_json(ui_url)
    _status_line("Streamlit reachable", ok, "" if ok else str(ui_home))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
