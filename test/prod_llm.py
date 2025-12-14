#!/usr/bin/env python3
"""Test production and QA LLM endpoints to verify they're working."""

import os
import sys
import json
import time
import requests
from pathlib import Path
from typing import Dict, Tuple, Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            if '=' in line and not line.startswith('#'):
                k, v = line.strip().split('=', 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"\''))

# Endpoints
PROD_LLM_PROXY = "http://clapi.prod-cheetah.grammarlyaws.com/api/v0/llm-proxy"
PROD_TRANSPARENT = "http://clapi.prod-cheetah.grammarlyaws.com/transparent/openai/v1"
QA_LLM_PROXY = "http://clapi.qa-text-processing.grammarlyaws.com/api/v0/llm-proxy"
QA_TRANSPARENT = "http://clapi.qa-text-processing.grammarlyaws.com/transparent/openai/v1"

# Test models
TEST_MODELS = {
    "llm_proxy": [
        "openai_direct_chat_gpt4o_mini",
        "openai_direct_gpt41_nano",
        "gas_gemini20_flash_lite"
    ],
    "transparent": [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4.1-nano",
        "gpt-4.1",
        "o4-mini",
        "o3-2025-04-16"
    ],
    "openai_direct": [
        "gpt-5-nano",
        "gpt-5-mini"
    ]
}

# Simple test prompt
TEST_PROMPT = "Is 'I have a book' grammatically correct? Answer Yes or No."

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'


def test_llm_proxy(endpoint: str, model: str, api_token: str) -> Tuple[bool, str, float]:
    """Test LLM Proxy endpoint with specific model."""
    start_time = time.time()
    
    payload = {
        "tracking_id": "",
        "messages": [
            {
                "role": "system",
                "content": "You are an expert GEC evaluator."
            },
            {
                "role": "user",
                "content": TEST_PROMPT,
                "cache_control": {"type": "ephemeral"}
            }
        ],
        "llm_backend": model,
        "metadata": {"json": "{}"},
        "tags": {"test": "prod_llm"},
        "api_token": api_token,
        "generation_parameters": {"json": json.dumps({"temperature": 0.0})}
    }
    
    try:
        # Use proxy if on production network
        proxies = None
        if "prod" in endpoint and os.getenv("INTERNAL_HTTP_PROXY"):
            proxies = {"http": os.getenv("INTERNAL_HTTP_PROXY")}
        
        response = requests.post(
            endpoint,
            json=payload,
            proxies=proxies,
            timeout=30
        )
        
        # Check if response is JSON
        try:
            response.json()
        except json.JSONDecodeError:
            elapsed = time.time() - start_time
            # HTML response likely means authentication page
            if "<html" in response.text.lower():
                return False, "HTML response - likely authentication required or wrong URL", elapsed
            else:
                return False, f"Non-JSON response: {response.text[:200]}", elapsed
        
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            if "chunk" in data and "text" in data["chunk"]:
                return True, data["chunk"]["text"][:100], elapsed
            elif "response" in data and "text" in data["response"]:
                return True, data["response"]["text"][:100], elapsed
            elif "error" in data:
                # Handle API error responses
                error_msg = data.get("error", {}).get("message", str(data["error"]))
                return False, f"API Error: {error_msg}", elapsed
            else:
                return False, f"Unexpected response format: {json.dumps(data)[:200]}", elapsed
        else:
            return False, f"HTTP {response.status_code}: {response.text[:200]}", elapsed
            
    except requests.RequestException as e:
        elapsed = time.time() - start_time
        return False, f"Request error: {str(e)}", elapsed
    except Exception as e:
        elapsed = time.time() - start_time
        return False, f"Unexpected error: {str(e)}", elapsed


def test_transparent_api(base_url: str, model: str, api_token: str) -> Tuple[bool, str, float]:
    """Test Transparent API endpoint with specific model."""
    start_time = time.time()
    
    url = f"{base_url}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "X-LLM-Proxy-Calling-Service": "gec-test"  # Required header for transparent API
    }
    # Authorization header is optional for default envs; include only if a real token is set
    if api_token and api_token != "dummy_token_for_connectivity_test":
        headers["Authorization"] = f"Bearer {api_token}"
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an expert GEC evaluator."},
            {"role": "user", "content": TEST_PROMPT}
        ]
    }
    # Avoid sending temperature for reasoning/o-series models
    if not (model.startswith("o3") or model.startswith("o1") or model.startswith("o4")):
        payload["temperature"] = 0.0
    
    try:
        # Use proxy if on production network
        proxies = None
        if "prod" in base_url and os.getenv("INTERNAL_HTTP_PROXY"):
            proxies = {"http": os.getenv("INTERNAL_HTTP_PROXY")}
        
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            proxies=proxies,
            timeout=30
        )
        
        # Check if response is JSON
        try:
            response.json()
        except json.JSONDecodeError:
            elapsed = time.time() - start_time
            # HTML response likely means authentication page
            if "<html" in response.text.lower():
                return False, "HTML response - likely authentication required or wrong URL", elapsed
            else:
                return False, f"Non-JSON response: {response.text[:200]}", elapsed
        
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            if "choices" in data and data["choices"]:
                content = data["choices"][0]["message"]["content"]
                return True, content[:100], elapsed
            elif "error" in data:
                # Handle API error responses
                error_msg = data.get("error", {}).get("message", str(data["error"]))
                return False, f"API Error: {error_msg}", elapsed
            else:
                return False, f"Unexpected response format: {json.dumps(data)[:200]}", elapsed
        else:
            return False, f"HTTP {response.status_code}: {response.text[:200]}", elapsed
            
    except requests.RequestException as e:
        elapsed = time.time() - start_time
        return False, f"Request error: {str(e)}", elapsed
    except Exception as e:
        elapsed = time.time() - start_time
        return False, f"Unexpected error: {str(e)}", elapsed


def test_openai_direct(model: str, api_key: str) -> Tuple[bool, str, float]:
    """Test OpenAI Direct API with specific model."""
    start_time = time.time()
    
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # GPT-5 models don't support temperature parameter yet
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an expert GEC evaluator."},
            {"role": "user", "content": TEST_PROMPT}
        ]
    }
    
    # Add temperature for models that support it
    if "gpt-5" not in model.lower():
        payload["temperature"] = 0.0
    
    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            if "choices" in data and data["choices"]:
                content = data["choices"][0]["message"]["content"]
                return True, content[:100], elapsed
            elif "error" in data:
                # Handle API error responses
                error_msg = data.get("error", {}).get("message", str(data["error"]))
                return False, f"API Error: {error_msg}", elapsed
            else:
                return False, f"Unexpected response format: {json.dumps(data)[:200]}", elapsed
        else:
            return False, f"HTTP {response.status_code}: {response.text[:200]}", elapsed
            
    except requests.RequestException as e:
        elapsed = time.time() - start_time
        return False, f"Request error: {str(e)}", elapsed
    except Exception as e:
        elapsed = time.time() - start_time
        return False, f"Unexpected error: {str(e)}", elapsed


def check_connectivity(url: str) -> Tuple[bool, str]:
    """Quick connectivity check for an endpoint."""
    try:
        response = requests.get(url, timeout=5)
        if response.status_code < 500:
            return True, f"Reachable (HTTP {response.status_code})"
        else:
            return False, f"Server error (HTTP {response.status_code})"
    except requests.exceptions.Timeout:
        return False, "Timeout"
    except requests.exceptions.ConnectionError:
        return False, "Connection failed"
    except Exception as e:
        return False, str(e)[:50]

def main():
    """Run all tests and print results."""
    # Get API credentials
    api_token = os.getenv("API_TOKEN")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not api_token:
        print(f"{Colors.YELLOW}Warning: API_TOKEN not found in environment{Colors.RESET}")
        print("Some tests will be skipped or may fail.")
        print("\nTo set up:")
        print("1. Create a .env file in the project root")
        print("2. Add: API_TOKEN=your_api_token_here")
        print()
        # Continue with limited testing
        api_token = "dummy_token_for_connectivity_test"
    
    print("=" * 80)
    print("LLM Endpoint Tests")
    print("=" * 80)
    print()
    
    results = []
    
    # Test LLM Proxy endpoints
    print("Testing LLM Proxy Endpoints...")
    print("-" * 40)
    
    for endpoint_name, endpoint_url in [("PROD", PROD_LLM_PROXY), ("QA", QA_LLM_PROXY)]:
        print(f"\n{endpoint_name} LLM Proxy: {endpoint_url}")
        for model in TEST_MODELS["llm_proxy"]:
            success, response, elapsed = test_llm_proxy(endpoint_url, model, api_token)
            status = f"{Colors.GREEN}✓{Colors.RESET}" if success else f"{Colors.RED}✗{Colors.RESET}"
            print(f"  {status} {model:30} [{elapsed:.2f}s]")
            if not success:
                print(f"     {Colors.RED}Error:{Colors.RESET} {response}")
            else:
                print(f"     {Colors.BLUE}Response:{Colors.RESET} {response}")
            results.append((f"{endpoint_name} LLM Proxy", model, success, elapsed))
    
    # Test Transparent API endpoints
    print("\n" + "=" * 80)
    print("Testing Transparent API Endpoints...")
    print("-" * 40)
    
    for endpoint_name, base_url in [("PROD", PROD_TRANSPARENT), ("QA", QA_TRANSPARENT)]:
        print(f"\n{endpoint_name} Transparent API: {base_url}")
        for model in TEST_MODELS["transparent"]:
            success, response, elapsed = test_transparent_api(base_url, model, api_token)
            status = f"{Colors.GREEN}✓{Colors.RESET}" if success else f"{Colors.RED}✗{Colors.RESET}"
            print(f"  {status} {model:30} [{elapsed:.2f}s]")
            if not success:
                print(f"     {Colors.RED}Error:{Colors.RESET} {response}")
            else:
                print(f"     {Colors.BLUE}Response:{Colors.RESET} {response}")
            results.append((f"{endpoint_name} Transparent", model, success, elapsed))
    
    # Test OpenAI Direct (if key available)
    if openai_key:
        print("\n" + "=" * 80)
        print("Testing OpenAI Direct API...")
        print("-" * 40)
        
        for model in TEST_MODELS["openai_direct"]:
            success, response, elapsed = test_openai_direct(model, openai_key)
            status = f"{Colors.GREEN}✓{Colors.RESET}" if success else f"{Colors.RED}✗{Colors.RESET}"
            print(f"  {status} {model:30} [{elapsed:.2f}s]")
            if not success:
                print(f"     {Colors.RED}Error:{Colors.RESET} {response}")
            else:
                print(f"     {Colors.BLUE}Response:{Colors.RESET} {response}")
            results.append(("OpenAI Direct", model, success, elapsed))
    else:
        print("\nSkipping OpenAI Direct tests (OPENAI_API_KEY not found)")
    
    # Connectivity Check
    print("\n" + "=" * 80)
    print("Connectivity Check")
    print("=" * 80)
    
    endpoints = [
        ("PROD LLM Proxy", PROD_LLM_PROXY),
        ("QA LLM Proxy", QA_LLM_PROXY),
        ("PROD Transparent", PROD_TRANSPARENT),
        ("QA Transparent", QA_TRANSPARENT),
        ("OpenAI API", "https://api.openai.com")
    ]
    
    for name, url in endpoints:
        reachable, msg = check_connectivity(url)
        status = f"{Colors.GREEN}✓{Colors.RESET}" if reachable else f"{Colors.RED}✗{Colors.RESET}"
        print(f"{status} {name:20} {msg}")
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    total = len(results)
    passed = sum(1 for _, _, success, _ in results if success)
    failed = total - passed
    avg_time = sum(elapsed for _, _, _, elapsed in results) / total if total > 0 else 0
    
    print(f"\nTotal Tests: {total}")
    print(f"Passed: {Colors.GREEN}{passed}{Colors.RESET}")
    print(f"Failed: {Colors.RED}{failed}{Colors.RESET}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    print(f"Average Response Time: {avg_time:.2f}s")
    
    # Working endpoints summary
    working_endpoints = set()
    for endpoint, model, success, _ in results:
        if success:
            working_endpoints.add(endpoint)
    
    if working_endpoints:
        print(f"\n{Colors.GREEN}Working Endpoints:{Colors.RESET}")
        for endpoint in sorted(working_endpoints):
            print(f"  - {endpoint}")
    
    # List failed tests
    if failed > 0:
        print(f"\n{Colors.RED}Failed Tests:{Colors.RESET}")
        for endpoint, model, success, _ in results:
            if not success:
                print(f"  - {endpoint}: {model}")
        
        # Provide troubleshooting tips
        print(f"\n{Colors.YELLOW}Troubleshooting Tips:{Colors.RESET}")
        
        # Check for specific error patterns
        auth_errors = [r for r in results if not r[2] and "authentication" in str(r)]
        token_errors = [r for r in results if not r[2] and "token" in str(r).lower()]
        
        if auth_errors:
            print("- PROD endpoints require proper authentication or VPN access")
        if token_errors:
            print("- Ensure API_TOKEN is valid and has proper permissions")
        if not openai_key and any("OpenAI" in r[0] for r in results if not r[2]):
            print("- Set OPENAI_API_KEY for OpenAI Direct API tests")
    
    print("\n" + "=" * 80)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
