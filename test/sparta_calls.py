#!/usr/bin/env python3
"""
Test script for verifying LLM backend calls work on both local and Red Sparta environments.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.llm.backends import call_model, is_red_sparta

# Test prompt
TEST_PROMPT = "Complete this sentence in 5 words or less: The sky is"

# List of backends to test
BACKENDS_TO_TEST = [
    # Gemini models
    ('gas_gemini20_flash_lite', 'Gemini 2.0 Flash Lite (via LLM Proxy)'),
    ('gemini-2.0-flash-lite', 'Gemini 2.0 Flash Lite (direct/dplane)'),
    
    # GPT models via transparent proxy
    ('gpt-4o-mini', 'GPT-4o-mini (transparent proxy)'),
    ('gpt-4o', 'GPT-4o (transparent proxy)'),
    
    # O-series models
    ('o3-mini', 'O3-mini (transparent proxy)'),
    ('o1-mini', 'O1-mini (transparent proxy)'),
    
    # Claude (if available)
    ('claude-3-haiku', 'Claude 3 Haiku (via LLM Proxy)'),
]

def test_backend(backend_name, description):
    """Test a single backend."""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"Backend: {backend_name}")
    print(f"Environment: {'Red Sparta' if is_red_sparta() else 'Local'}")
    print('-'*60)
    
    # Skip certain tests if not in actual Red Sparta
    if is_red_sparta() and not os.getenv('API_TOKEN'):
        print("⚠️  Skipped: No API_TOKEN set for Red Sparta")
        return None
    
    try:
        # Get API key based on backend
        if backend_name.startswith('gpt') or backend_name.startswith('o'):
            api_key = os.getenv('API_TOKEN', os.getenv('OPENAI_API_KEY', ''))
            if not api_key:
                print("❌ No API_TOKEN or OPENAI_API_KEY found")
                return False
        elif 'gemini' in backend_name:
            # Gemini via dplane doesn't need a key, via direct needs GEMINI_API_KEY
            api_key = os.getenv('GEMINI_API_KEY', 'dummy')
        elif 'claude' in backend_name:
            api_key = os.getenv('API_TOKEN', os.getenv('ANTHROPIC_API_KEY', ''))
            if not api_key:
                print("❌ No API_TOKEN or ANTHROPIC_API_KEY found")
                return False
        else:
            api_key = os.getenv('API_TOKEN', '')
        
        # Make the call
        success, content, token_usage = call_model(TEST_PROMPT, backend_name, api_key)
        
        if success:
            print(f"✅ Success!")
            print(f"Response: {content[:100]}...")
            print(f"Tokens: {token_usage}")
            return True
        else:
            print(f"❌ Failed")
            print(f"Error: {content}")
            return False
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def main():
    """Run all backend tests."""
    print("="*60)
    print("LLM Backend Test Suite")
    print("="*60)
    
    # Check environment
    print(f"\nEnvironment Detection:")
    print(f"- Red Sparta: {is_red_sparta()}")
    print(f"- INTERNAL_HTTP_PROXY: {os.getenv('INTERNAL_HTTP_PROXY', 'Not set')}")
    print(f"- API_TOKEN: {'Set' if os.getenv('API_TOKEN') else 'Not set'}")
    print(f"- OPENAI_API_KEY: {'Set' if os.getenv('OPENAI_API_KEY') else 'Not set'}")
    
    # Load endpoints
    from utils.llm.backends import LLM_PROXY_ENDPOINT, TRANSPARENT_OPENAI_BASE
    print(f"\nConfigured Endpoints:")
    print(f"- LLM Proxy: {LLM_PROXY_ENDPOINT}")
    print(f"- Transparent OpenAI: {TRANSPARENT_OPENAI_BASE}")
    
    # Run tests
    results = {}
    for backend, description in BACKENDS_TO_TEST:
        success = test_backend(backend, description)
        results[backend] = success
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for backend, description in BACKENDS_TO_TEST:
        status = "✅" if results.get(backend, False) else "❌"
        print(f"{status} {description}: {backend}")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\nPassed: {passed}/{total}")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    exit(main())
