#!/usr/bin/env python3
"""Simple test for LLM proxy integration"""

import requests
import json
import os

# Load environment variables
for env_file in ['../.env', '.env']:
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key.strip()] = value.strip().strip('"\'')
        break


def call_llm_proxy(source, target):
    """Call LLM proxy to evaluate grammar correction"""
    api_token = os.getenv('API_TOKEN')
    if not api_token:
        return None
    
    prompt = f"Compare these texts:\nOriginal: {source}\nCorrected: {target}\nIs this a valid correction? Answer Yes or No."
    
    data = {
        "messages": [{"role": "user", "content": prompt}],
        "llm_backend": "openai_direct_gpt41_mini",
        "api_token": api_token,
        "moderation": {"state": "disabled"},
        "tags": {"calling_service": "gec-test"},
        "generation_parameters": {
            "json": json.dumps({"temperature": 0.0, "max_tokens": 50})
        }
    }

    try:
        response = requests.post(
            "http://clapi.qa-text-processing.grammarlyaws.com/api/v0/llm-proxy",
            json=data, 
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if "chunk" in result and "text" in result["chunk"]:
                return result["chunk"]["text"]
            elif "response" in result and "text" in result["response"]:
                return result["response"]["text"]
                
    except Exception as e:
        print(f"LLM proxy call failed: {e}")
    
    return None


def test_llm_proxy():
    """Test LLM proxy with grammar correction example"""
    source = "I has a book"
    target = "I have a book"
    
    # Check if API token is available
    if not os.getenv('API_TOKEN'):
        print("No API token found")
        return False
    
    result = call_llm_proxy(source, target)
    print(f"LLM proxy response: {result}")
    
    return result is not None


if __name__ == "__main__":
    success = test_llm_proxy()
    print(f"Test {'passed' if success else 'failed'}")