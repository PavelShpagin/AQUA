#!/usr/bin/env python3
"""
Minimal Red Sparta check: call LLM Proxy with an O3 backend.

Requirements on Red Sparta:
- API_TOKEN available in env (or OPENAI_API_KEY if proxy accepts it)
- INTERNAL_HTTP_PROXY is set by environment (optional)

Usage:
  python test/sparta_o3_proxy.py | cat
"""

import os
import json
import time
import requests

PROD_LLM_PROXY = "http://clapi.prod-cheetah.grammarlyaws.com/api/v0/llm-proxy"

# One of the O3 backends exposed in proxy inventory. If this doesn't work in a
# given cluster, try the others manually: openai_direct_o3_priority, _scale_tier.
O3_BACKEND = os.getenv("O3_PROXY_BACKEND", "openai_direct_o3").strip()

PROMPT = "Say 'Hello' and nothing else."


def main():
    api_token = (os.getenv("API_TOKEN", "").strip() or os.getenv("OPENAI_API_KEY", "").strip())
    if not api_token:
        raise SystemExit("API_TOKEN not set")

    payload = {
        "tracking_id": "",
        "messages": [
            {"role": "system", "content": "You are an expert GEC evaluator."},
            {"role": "user", "content": PROMPT, "cache_control": {"type": "ephemeral"}}
        ],
        "llm_backend": O3_BACKEND,
        "metadata": {"json": "{}"},
        "tags": {"test": "sparta_o3_proxy"},
        "api_token": api_token,
        "generation_parameters": {"json": json.dumps({"temperature": 0.0})}
    }

    proxies = None
    if os.getenv("INTERNAL_HTTP_PROXY"):
        proxies = {"http": os.getenv("INTERNAL_HTTP_PROXY")}

    t0 = time.time()
    r = requests.post(PROD_LLM_PROXY, json=payload, proxies=proxies, timeout=30)
    dt = time.time() - t0

    # Try to parse JSON even if non-200 for debugging
    try:
        data = r.json()
    except Exception:
        data = {"raw": (r.text or "")[:200]}

    if r.status_code == 200:
        # Red Sparta may return chunked format
        if isinstance(data, dict) and "chunk" in data:
            text = (data.get("chunk", {}).get("text") or "").strip()
        else:
            # Fallback fields
            text = (data.get("response", {}).get("text") or "").strip()
        print(f"status=OK time={dt:.2f}s backend={O3_BACKEND}\n{text}")
        return 0
    else:
        print(f"status=HTTP_{r.status_code} time={dt:.2f}s backend={O3_BACKEND}\n{json.dumps(data)[:300]}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


