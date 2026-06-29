"""Check whether configured service credentials work.

This script validates connectivity/authentication for the main services used by
this project. It reports PASS/FAIL per service instead of raising immediately.

Usage:
    uv run python scripts/check_api_keys.py
    uv run python scripts/check_api_keys.py --env .env
    uv run python scripts/check_api_keys.py --skip-blob
    uv run python scripts/check_api_keys.py --skip-openai --skip-search
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import hmac
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Callable, Iterable

import redis
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

try:
    from supabase import create_client

    HAS_SUPABASE = True
except ImportError:
    create_client = None
    HAS_SUPABASE = False


@dataclass
class CheckResult:
    name: str
    ok: bool
    message: str


def load_env(env_path: str | None) -> None:
    if env_path:
        load_dotenv(env_path, override=True)
    else:
        load_dotenv(override=True)


def get_env(name: str, required: bool = True) -> str | None:
    value = os.getenv(name)
    if required and not value:
        raise ValueError(f"Missing required env var: {name}")
    return value


def print_result(result: CheckResult) -> None:
    status = "PASS" if result.ok else "FAIL"
    print(f"[{status}] {result.name}: {result.message}")


def request_json(
    url: str,
    headers: dict[str, str] | None = None,
    method: str = "GET",
) -> tuple[int, str]:
    req = urllib.request.Request(url, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            body = response.read().decode("utf-8", errors="replace")
            return response.status, body
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return exc.code, body
    except urllib.error.URLError as exc:
        return 0, str(exc)


def check_azure_openai() -> CheckResult:
    endpoint = get_env("AZURE_OPENAI_ENDPOINT")
    api_key = get_env("AZURE_OPENAI_API_KEY")
    deployment = get_env("AZURE_OPENAI_DEPLOYMENT_NAME")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

    try:
        model = AzureChatOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            azure_deployment=deployment,
            api_version=api_version,
            temperature=0,
        )
        response = model.invoke("Reply with a single word: pong")
        text = getattr(response, "content", str(response))
        return CheckResult(
            name="Azure OpenAI",
            ok=True,
            message=f"Connected successfully. Response: {text[:80]!r}",
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(name="Azure OpenAI", ok=False, message=str(exc))


def check_azure_search() -> CheckResult:
    endpoint = get_env("AZURE_SEARCH_ENDPOINT")
    api_key = get_env("AZURE_SEARCH_API_KEY")
    try:
        client = SearchIndexClient(endpoint=endpoint, credential=AzureKeyCredential(api_key))
        index_names = list(client.list_index_names())
        return CheckResult(
            name="Azure AI Search",
            ok=True,
            message=f"Connected successfully. Found {len(index_names)} index(es).",
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(name="Azure AI Search", ok=False, message=str(exc))


def _storage_authorization_header(
    account_name: str,
    account_key: str,
    verb: str,
    path: str,
    query: str,
    x_ms_date: str,
) -> str:
    canonicalized_headers = f"x-ms-date:{x_ms_date}\nx-ms-version:2023-11-03"
    canonicalized_resource = f"/{account_name}{path}"
    if query:
        query_params = urllib.parse.parse_qs(query, keep_blank_values=True)
        for key in sorted(query_params):
            values = ",".join(sorted(v.lower() for v in query_params[key]))
            canonicalized_resource += f"\n{key.lower()}:{values}"

    string_to_sign = (
        f"{verb}\n"
        "\n"
        "\n"
        "\n"
        "\n"
        "\n"
        "\n"
        "\n"
        "\n"
        "\n"
        "\n"
        "\n"
        f"{canonicalized_headers}\n"
        f"{canonicalized_resource}"
    )

    decoded_key = base64.b64decode(account_key)
    signature = base64.b64encode(
        hmac.new(decoded_key, string_to_sign.encode("utf-8"), hashlib.sha256).digest()
    ).decode("utf-8")

    return f"SharedKey {account_name}:{signature}"


def check_azure_blob_storage() -> CheckResult:
    account_name = get_env("AZURE_STORAGE_ACCOUNT_NAME")
    account_key = get_env("AZURE_STORAGE_ACCOUNT_KEY")
    container = os.getenv("AZURE_STORAGE_CONTAINER_NAME", "")

    path = "/"
    query = "comp=list"
    url = f"https://{account_name}.blob.core.windows.net/{path.lstrip('/')}?{query}"
    x_ms_date = dt.datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S GMT")
    headers = {
        "x-ms-date": x_ms_date,
        "x-ms-version": "2023-11-03",
        "Authorization": _storage_authorization_header(
            account_name,
            account_key,
            "GET",
            path,
            query,
            x_ms_date,
        ),
    }

    status, body = request_json(url, headers=headers)
    if status == 200:
        extra = f" Container={container!r}." if container else ""
        return CheckResult(
            name="Azure Blob Storage", ok=True, message=f"Connected successfully.{extra}"
        )
    return CheckResult(name="Azure Blob Storage", ok=False, message=f"HTTP {status}: {body[:200]}")


def check_document_intelligence() -> CheckResult:
    endpoint = get_env("AZURE_DOC_INTELLIGENCE_ENDPOINT")
    api_key = get_env("AZURE_DOC_INTELLIGENCE_KEY")
    candidates = [
        "/documentintelligence/info?api-version=2024-02-29-preview",
        "/documentmodels?api-version=2024-02-29-preview",
        "/formrecognizer/info?api-version=2023-07-31",
        "/formrecognizer/documentModels?api-version=2023-07-31",
    ]
    headers = {"Ocp-Apim-Subscription-Key": api_key}

    for candidate in candidates:
        status, body = request_json(endpoint.rstrip("/") + candidate, headers=headers)
        if status in {200, 401, 403}:
            return CheckResult(
                name="Azure Document Intelligence",
                ok=status == 200,
                message=(
                    "Connected successfully."
                    if status == 200
                    else f"Auth/endpoint returned HTTP {status}: {body[:160]}"
                ),
            )

    return CheckResult(
        name="Azure Document Intelligence",
        ok=False,
        message="Could not reach a known Document Intelligence endpoint path.",
    )


def check_redis() -> CheckResult:
    url = get_env("REDIS_URL")
    try:
        client = redis.Redis.from_url(url, socket_connect_timeout=5, socket_timeout=5)
        pong = client.ping()
        return CheckResult(name="Redis", ok=bool(pong), message="PING succeeded.")
    except Exception as exc:  # noqa: BLE001
        return CheckResult(name="Redis", ok=False, message=str(exc))


def check_supabase() -> CheckResult:
    if not HAS_SUPABASE:
        return CheckResult(
            name="Supabase", ok=True, message="Skipped (supabase client package not installed)."
        )
    url = get_env("SUPABASE_URL")
    key = get_env("SUPABASE_KEY")
    try:
        _client = create_client(url, key)
        auth_url = url.rstrip("/") + "/auth/v1/settings"
        headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
        }
        status, body = request_json(auth_url, headers=headers)
        if status == 200:
            return CheckResult(name="Supabase", ok=True, message="Connected successfully.")
        return CheckResult(name="Supabase", ok=False, message=f"HTTP {status}: {body[:160]}")
    except Exception as exc:  # noqa: BLE001
        return CheckResult(name="Supabase", ok=False, message=str(exc))


def _langfuse_basic_auth(public_key: str, secret_key: str) -> str:
    token = base64.b64encode(f"{public_key}:{secret_key}".encode("utf-8")).decode("utf-8")
    return f"Basic {token}"


def check_langfuse() -> CheckResult:
    host = get_env("LANGFUSE_HOST")
    public_key = get_env("LANGFUSE_PUBLIC_KEY")
    secret_key = get_env("LANGFUSE_SECRET_KEY")

    candidates = [
        "/api/public/health",
        "/api/public/ingestion/health",
        "/api/public/auth/health",
    ]
    headers = {
        "Authorization": _langfuse_basic_auth(public_key, secret_key),
        "Content-Type": "application/json",
    }

    for candidate in candidates:
        status, body = request_json(host.rstrip("/") + candidate, headers=headers)
        if status in {200, 204, 401, 403}:
            return CheckResult(
                name="Langfuse",
                ok=status in {200, 204},
                message=(
                    "Connected successfully."
                    if status in {200, 204}
                    else f"Auth/endpoint returned HTTP {status}: {body[:160]}"
                ),
            )

    return CheckResult(
        name="Langfuse", ok=False, message="Could not reach a known Langfuse health endpoint."
    )


def run_checks(skip_blob: bool) -> list[CheckResult]:
    checks: list[Callable[[], CheckResult]] = [
        check_azure_openai,
        check_azure_search,
        check_document_intelligence,
        check_redis,
        check_supabase,
        check_langfuse,
    ]
    if not skip_blob:
        checks.append(check_azure_blob_storage)

    results: list[CheckResult] = []
    for check in checks:
        try:
            result = check()
        except Exception as exc:  # noqa: BLE001
            result = CheckResult(name=check.__name__, ok=False, message=str(exc))
        results.append(result)
    return results


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate configured API keys and service access.")
    parser.add_argument("--env", default=None, help="Path to .env file (default: auto-discover)")
    parser.add_argument("--skip-blob", action="store_true", help="Skip Azure Blob Storage test")
    args = parser.parse_args(list(argv) if argv is not None else None)

    load_env(args.env)

    results = run_checks(skip_blob=args.skip_blob)
    print("\nService check summary\n---------------------")
    for result in results:
        print_result(result)

    failures = [r for r in results if not r.ok]
    if failures:
        print(f"\n{len(failures)} check(s) failed.")
        return 1

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
