import os
import requests
import json

# Load environment variables from .env at project root if present
from pathlib import Path
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            if '=' in line and not line.startswith('#'):
                k, v = line.strip().split('=', 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"\''))
                
def test_gpt5_nano():
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_TOKEN")
    if not api_key:
        print("No OpenAI API key found in environment.")
        return False

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-5-nano",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ],
    }

    try:
        r = requests.post(url, headers=headers, json=data, timeout=30)
        if r.status_code == 200:
            resp = r.json()
            print("gpt-5-nano response:", resp["choices"][0]["message"]["content"].strip())
            return True
        else:
            print(f"OpenAI API error: {r.status_code} {r.text}")
            return False
    except Exception as e:
        print(f"Request failed: {e}")
        return False

if __name__ == "__main__":
    ok = test_gpt5_nano()
    print("Test passed" if ok else "Test failed")
