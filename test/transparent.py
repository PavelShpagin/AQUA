#!/usr/bin/env python3
"""
Simple Transparent API Test Script

Tests OpenAI models through the Grammarly Transparent API.
If you can access the proxy endpoint, this should work seamlessly.
"""

import os
import json
from openai import OpenAI

# Set proxy environment
os.environ["HTTP_PROXY"] = os.environ.get("INTERNAL_HTTP_PROXY", "")

# Load API token from environment
API_TOKEN = os.getenv('API_TOKEN', '')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')

# Models to test
TEST_MODELS = [
    "gpt-4o",
    "gpt-4o-mini", 
    "o1-preview-2024-09-12",
    "o3-mini-2025-01-31",
    "o4-mini-2025-04-16",
    "o3-2025-04-16"
]

def get_client():
    """Get OpenAI client configured for transparent API"""
    return OpenAI(
        api_key=OPENAI_API_KEY or "dummy-key",
        base_url='http://clapi.qa-text-processing.grammarlyaws.com/transparent/openai/v1',
        default_headers={'X-LLM-Proxy-Calling-Service': 'test-script'}
    )

def test_model(client, model_name):
    """Test a single model with a simple prompt"""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Hello, respond with a simple greeting."}],
            timeout=30
        )
        
        if response.choices and response.choices[0].message.content:
            content = response.choices[0].message.content.strip()
            return True, content[:80] + ("..." if len(content) > 80 else "")
        else:
            return False, "Empty response"
            
    except Exception as e:
        error_msg = str(e)
        if "does not exist" in error_msg:
            return False, "Model not available"
        elif "rate limit" in error_msg.lower():
            return False, "Rate limited" 
        elif "timeout" in error_msg.lower():
            return False, "Request timeout"
        elif "unauthorized" in error_msg.lower():
            return False, "Authentication error"
        else:
            return False, f"Error: {error_msg[:60]}..."

def main():
    """Test all supported models"""
    print("Testing Transparent API Models")
    print("Endpoint: http://clapi.qa-text-processing.grammarlyaws.com/transparent/openai/v1")
    print("=" * 70)
    
    if not OPENAI_API_KEY:
        print("Warning: OPENAI_API_KEY not found in environment variables")
        print("Set OPENAI_API_KEY in your .env file or environment")
        print()
    
    try:
        client = get_client()
    except Exception as e:
        print(f"Failed to create client: {e}")
        return
    
    working_models = []
    
    for i, model in enumerate(TEST_MODELS, 1):
        print(f"[{i}/{len(TEST_MODELS)}] Testing {model}...")
        
        success, response = test_model(client, model)
        
        if success:
            print(f"  ✓ Working: {response}")
            working_models.append(model)
        else:
            print(f"  ✗ Failed: {response}")
        print()
    
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if working_models:
        print(f"✓ Working models ({len(working_models)}):")
        for model in working_models:
            print(f"  - {model}")
        print(f"\nRecommended model: {working_models[-1]}")
    else:
        print("✗ No working models found")
        print("Check your network connection and API credentials")
    
    success_rate = len(working_models) / len(TEST_MODELS) * 100
    print(f"\nOverall: {len(working_models)}/{len(TEST_MODELS)} models working ({success_rate:.1f}% success)")

if __name__ == "__main__":
    main()
