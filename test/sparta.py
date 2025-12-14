#!/usr/bin/env python3
"""
Test script to verify backend functionality on Red Sparta.
Tests various model backends through the LLM Proxy.
"""

import os
import sys
import time
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.llm.backends import call_model, is_red_sparta

# Test configurations
TEST_BACKENDS = [
    # OpenAI models
    ('gpt-4.1', 'OpenAI 4.1'),
    ('gpt-4.1-nano', 'OpenAI 4.1 Nano'),
    ('gpt-4o', 'GPT-4o'),
    ('gpt-4o-mini', 'GPT-4o Mini'),
    
    # GPT-5 models (new)
    ('gpt-5-mini', 'GPT-5 Mini'),
    ('gpt-5-nano', 'GPT-5 Nano'),
    
    # Gemini models
    ('gemini-2.0-flash-lite', 'Gemini 2.0 Flash Lite'),
    ('gas_gemini20_flash_lite', 'Gemini Flash Lite (GAS)'),
    
    # Bedrock models
    ('llama3.2-1b', 'Llama 3.2 1B'),
    ('llama3.2-3b', 'Llama 3.2 3B'),
    ('nova-lite', 'AWS Nova Lite'),
    
    # Other models
    ('hermes-3b', 'Hermes 3B'),
    ('mistral-7b', 'Mistral 7B'),
    
    # O-series
    ('o3', 'O3'),
    ('o4-mini', 'O4 Mini'),
]

TEST_PROMPT = "Say 'Hello' and nothing else."


def _proxy_call_simple_o3(prompt: str, backend: str, api_token: str) -> Tuple[bool, str]:
    """Direct LLM Proxy call for O3, mirroring sparta_o3_proxy.py behavior."""
    import json as _json
    import requests as _req
    url = "http://clapi.prod-cheetah.grammarlyaws.com/api/v0/llm-proxy"
    payload = {
        "tracking_id": "",
        "messages": [
            {"role": "system", "content": "You are an expert GEC evaluator."},
            {"role": "user", "content": prompt, "cache_control": {"type": "ephemeral"}},
        ],
        "llm_backend": backend,
        "metadata": {"json": "{}"},
        "tags": {"test": "sparta_o3_proxy_inline"},
        "api_token": api_token,
    }
    # Skip generation_parameters for O-series backends
    if not backend.startswith("openai_direct_o3"):
        payload["generation_parameters"] = {"json": _json.dumps({"temperature": 0.0})}
    proxies = {"http": os.getenv("INTERNAL_HTTP_PROXY")} if os.getenv("INTERNAL_HTTP_PROXY") else None
    r = _req.post(url, json=payload, proxies=proxies, timeout=30)
    if r.status_code != 200:
        try:
            data = r.json()
        except Exception:
            data = {"raw": (r.text or "")[:200]}
        return False, str(data)[:300]
    try:
        data = r.json()
    except Exception:
        return False, (r.text or "")[:200]
    txt = ""
    if isinstance(data, dict):
        if "chunk" in data:
            txt = (data.get("chunk", {}).get("text") or "").strip()
        if not txt:
            txt = (data.get("response", {}).get("text") or "").strip()
        if not txt:
            # Standard OpenAI-like shape
            try:
                ch = (data.get("choices") or [{}])[0]
                txt = (ch.get("message", {}) or {}).get("content") or ""
                if isinstance(txt, list):
                    txt = "\n".join([str(p.get("text", "")) for p in txt if isinstance(p, dict)])
                txt = (txt or "").strip()
            except Exception:
                txt = ""
        if not txt:
            raw = data.get("content", "")
            if isinstance(raw, list):
                txt = "\n".join([str(p.get("text", "")) for p in raw if isinstance(p, dict)])
            elif isinstance(raw, str):
                txt = raw
            txt = (txt or "").strip()
    return (len(txt) > 0), txt or "empty"


def test_backend(backend: str, name: str, api_token: str) -> Dict:
    """Test a single backend."""
    print(f"\nTesting {name} ({backend})...")
    
    start_time = time.time()
    try:
        # On Red Sparta, O3 must be addressed via proxy backend name
        backend_to_use = backend
        if is_red_sparta() and backend == 'o3':
            backend_to_use = os.getenv('O3_PROXY_BACKEND', 'openai_direct_o3')

        # For O3 on Sparta, directly hit the proxy like the standalone script
        if is_red_sparta() and backend == 'o3':
            ok, text = _proxy_call_simple_o3(TEST_PROMPT, backend_to_use, api_token)
            elapsed = time.time() - start_time
            if ok and text:
                print(f"  ✓ SUCCESS in {elapsed:.2f}s")
                print(f"    Response: {text[:100]}")
                return {
                    'backend': backend_to_use,
                    'name': name,
                    'status': 'SUCCESS',
                    'time': elapsed,
                    'response': text[:100],
                    'tokens': 0
                }
            else:
                print(f"  ✗ FAILED in {elapsed:.2f}s")
                print(f"    Error: {text[:100] if text else 'No error message'}")
                return {
                    'backend': backend_to_use,
                    'name': name,
                    'status': 'FAILED',
                    'time': elapsed,
                    'error': text[:100] if text else 'Unknown'
                }

        success, content, tokens = call_model(
            TEST_PROMPT,
            backend_to_use,
            api_token,
            temperature_override=0.0
        )
        elapsed = time.time() - start_time
        
        if success:
            # Check if response is reasonable
            if content and len(content) > 0:
                print(f"  ✓ SUCCESS in {elapsed:.2f}s")
                print(f"    Response: {content[:100]}")
                print(f"    Tokens: {tokens.get('total_tokens', 'N/A')}")
                return {
                    'backend': backend_to_use,
                    'name': name,
                    'status': 'SUCCESS',
                    'time': elapsed,
                    'response': content[:100],
                    'tokens': tokens.get('total_tokens', 0)
                }
            else:
                print(f"  ✗ EMPTY RESPONSE in {elapsed:.2f}s")
                return {
                    'backend': backend_to_use,
                    'name': name,
                    'status': 'EMPTY',
                    'time': elapsed
                }
        else:
            print(f"  ✗ FAILED in {elapsed:.2f}s")
            print(f"    Error: {content[:100] if content else 'No error message'}")
            return {
                'backend': backend_to_use,
                'name': name,
                'status': 'FAILED',
                'time': elapsed,
                'error': content[:100] if content else 'Unknown'
            }
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"  ✗ EXCEPTION in {elapsed:.2f}s")
        print(f"    Error: {str(e)[:100]}")
        return {
            'backend': backend,
            'name': name,
            'status': 'EXCEPTION',
            'time': elapsed,
            'error': str(e)[:100]
        }


def main():
    """Run backend tests."""
    print("=" * 60)
    print("BACKEND FUNCTIONALITY TEST")
    print("=" * 60)
    
    # Check environment
    print(f"\nEnvironment: {'Red Sparta' if is_red_sparta() else 'Local'}")
    
    # Get API token
    api_token = os.getenv('API_TOKEN', '') or os.getenv('OPENAI_API_KEY', '')
    if not api_token:
        print("\n⚠️  WARNING: No API_TOKEN or OPENAI_API_KEY found")
        print("Some backends may fail without proper authentication")
    
    # Test each backend
    results = []
    for backend, name in TEST_BACKENDS:
        result = test_backend(backend, name, api_token)
        results.append(result)
        time.sleep(0.5)  # Small delay between tests
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    working = [r for r in results if r['status'] == 'SUCCESS']
    failed = [r for r in results if r['status'] in ['FAILED', 'EXCEPTION']]
    empty = [r for r in results if r['status'] == 'EMPTY']
    
    print(f"\n✓ Working: {len(working)}/{len(results)}")
    for r in working:
        print(f"  - {r['name']} ({r['backend']}): {r['time']:.2f}s")
    
    if empty:
        print(f"\n⚠ Empty responses: {len(empty)}")
        for r in empty:
            print(f"  - {r['name']} ({r['backend']})")
    
    if failed:
        print(f"\n✗ Failed: {len(failed)}")
        for r in failed:
            print(f"  - {r['name']} ({r['backend']}): {r.get('error', 'Unknown error')}")
    
    # Recommendations
    if working:
        fastest = min(working, key=lambda x: x['time'])
        print(f"\n🚀 Fastest working backend: {fastest['name']} ({fastest['time']:.2f}s)")
        
        if is_red_sparta():
            print("\n📝 Recommended backends for Red Sparta:")
            for r in sorted(working, key=lambda x: x['time'])[:3]:
                print(f"  - {r['backend']}: {r['time']:.2f}s")
    else:
        print("\n⚠️  No working backends found!")
        if is_red_sparta():
            print("Check your Red Sparta proxy configuration and API_TOKEN")
        else:
            print("Check your API keys and network connectivity")
    
    return len(working) > 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
