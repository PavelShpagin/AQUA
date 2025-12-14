#!/usr/bin/env python3
"""
OpenAI Moderation wrapper. Returns True if content is flagged/blocked.
Requires OPENAI_API_KEY to be set.
"""

import os
from typing import Tuple

import requests


def _load_env_key_from_dotenv() -> None:
    if 'OPENAI_API_KEY' in os.environ:
        return
    for env_file in ('.env', '../.env'):
        if os.path.exists(env_file):
            try:
                with open(env_file, 'r') as f:
                    for line in f:
                        if '=' in line and not line.strip().startswith('#'):
                            k, v = line.strip().split('=', 1)
                            k = k.strip(); v = v.strip().strip('"\'')
                            os.environ.setdefault(k, v)
            except Exception:
                pass
            
            break

# Alias OPENAI_API_TOKEN -> OPENAI_API_KEY for compatibility
if 'OPENAI_API_KEY' not in os.environ and 'OPENAI_API_TOKEN' in os.environ:
    os.environ['OPENAI_API_KEY'] = os.environ['OPENAI_API_TOKEN']


def check_openai_moderation(text: str) -> Tuple[bool, str]:
    _load_env_key_from_dotenv()
    api_key = os.getenv('OPENAI_API_KEY', '')
    if not api_key:
        return False, ''
    try:
        r = requests.post(
            'https://api.openai.com/v1/moderations',
            headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'},
            json={'model': 'text-moderation-latest', 'input': text},
            timeout=30,
        )
        if r.status_code != 200:
            return False, ''
        data = r.json()
        res = data.get('results', [{}])[0]
        flagged = bool(res.get('flagged', False))
        # Also consider category booleans explicitly
        cats = res.get('categories', {}) or {}
        if any(bool(cats.get(k, False)) for k in cats.keys()):
            flagged = True
        return flagged, ('flagged' if flagged else '')
    except Exception:
        return False, ''


