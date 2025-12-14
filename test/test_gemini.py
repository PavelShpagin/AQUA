#!/usr/bin/env python3
"""Simple test for Gemini model integration"""

import json
import subprocess
import tempfile
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


def call_gemini(prompt):
    """Call Gemini API via cURL"""
    data = {
        "model": "gas_gemini20_flash_lite",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 100
    }

    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            temp_file = f.name

        result = subprocess.run([
            'curl', '-X', 'POST', '--max-time', '30',
            '-H', 'Accept: application/json',
            '-H', 'Content-Type: application/json',
            '-H', 'X-LLM-Proxy-Calling-Service: test@grammarly.com',
            '-d', f'@{temp_file}',
            'https://apigw.dplane.ppgr.io/clapi/api/v1/chat/completions'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0 and result.stdout.strip():
            resp = json.loads(result.stdout)
            if "choices" in resp:
                return resp["choices"][0]["message"]["content"].strip()
                
    except Exception as e:
        print(f"Gemini call failed: {e}")
    
    return None


def test_gemini():
    """Test Gemini with simple math question"""
    # Check cURL availability
    try:
        subprocess.run(['curl', '--version'], capture_output=True, check=True)
    except:
        print("cURL not available")
        return False
    
    result = call_gemini("What is 2+2? Answer briefly.")
    print(f"Gemini response: {result}")
    
    return result is not None


if __name__ == "__main__":
    success = test_gemini()
    print(f"Test {'passed' if success else 'failed'}")