"""Optionally route τ-bench's model calls through a Portkey gateway.

Put these in bench/.env (git-ignored):

    PORTKEY_BASE_URL=...          # the gateway URL
    PORTKEY_API_KEY=...
    PORTKEY_CONFIG=...            # Portkey config id, if the gateway needs one
    CF_ACCESS_CLIENT_ID=...       # only if the gateway sits behind Cloudflare Access
    CF_ACCESS_CLIENT_SECRET=...
    SSL_CERT_FILE=...             # only if the gateway uses a private certificate authority

Then name models ``portkey/<model>``, e.g. ``--agent-llm portkey/gpt-4.1-2025-04-14``.
τ-bench's own LLM judge (a bare ``gpt-...`` model name) is routed the same way. The
credentials are added to each request here, so they never end up in τ-bench's saved
run settings or logs.
"""

from __future__ import annotations

import os
from typing import Any

import litellm
import tau2.utils.llm_utils as llm_utils

PREFIX = "portkey/"


def enabled() -> bool:
    return bool(os.environ.get("PORTKEY_BASE_URL") and os.environ.get("PORTKEY_API_KEY"))


def _gateway() -> dict[str, Any]:
    base = os.environ["PORTKEY_BASE_URL"].rstrip("/")
    headers = {"x-portkey-api-key": os.environ["PORTKEY_API_KEY"]}
    if config := os.environ.get("PORTKEY_CONFIG"):
        headers["x-portkey-config"] = config
    if client_id := os.environ.get("CF_ACCESS_CLIENT_ID"):
        headers["CF-Access-Client-Id"] = client_id
        headers["CF-Access-Client-Secret"] = os.environ["CF_ACCESS_CLIENT_SECRET"]
    return {
        "api_base": base if base.endswith("/v1") else f"{base}/v1",
        "api_key": "portkey",  # the gateway authenticates with the headers above
        "extra_headers": headers,
    }


def install() -> None:
    """Wrap τ-bench's model call so ``portkey/`` (and bare ``gpt-``) models use the gateway."""
    if not enabled():
        return
    if cert := os.environ.get("SSL_CERT_FILE"):
        litellm.ssl_verify = cert
    original = llm_utils.completion

    def completion(*args: Any, model: str, **kwargs: Any) -> Any:
        if model.startswith(PREFIX) or model.startswith("gpt-"):
            name = model.removeprefix(PREFIX)
            return original(*args, model=f"openai/{name}", **kwargs, **_gateway())
        return original(*args, model=model, **kwargs)

    llm_utils.completion = completion
