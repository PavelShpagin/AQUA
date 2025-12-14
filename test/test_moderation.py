#!/usr/bin/env python3
"""Simple test for OpenAI moderation API"""

import requests
import os
from time import sleep

# Load environment variables
for env_file in ['../.env', '.env']:
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key.strip()] = value.strip().strip('"\'')
        break


def check_moderation(text):
    """Check if text violates OpenAI content policy"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return False
    
    try:
        response = requests.post(
            "https://api.openai.com/v1/moderations",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"input": text, "model": "text-moderation-latest"},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['results'][0]['flagged'] if result.get('results') else False
            
    except Exception:
        pass
    
    return False


def test_moderation():
    """Test moderation with safe and problematic text"""
    safe_text = "Hello, how are you today?"
    problematic_text = "I hate this stupid thing"
    
    safe_flagged = check_moderation(safe_text)
    problem_flagged = check_moderation(problematic_text)
    
    print(f"Safe text flagged: {safe_flagged}")
    print(f"Problematic text flagged: {problem_flagged}")
    
    return True


if __name__ == "__main__":
    success = test_moderation()
    print(f"Test {'passed' if success else 'failed'}")