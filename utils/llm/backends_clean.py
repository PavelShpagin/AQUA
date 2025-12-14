#!/usr/bin/env python3
"""
Super clean LLM backend implementation.
Works locally and on Red Sparta with minimal complexity.
"""

import os
import json
import requests
import threading
from typing import Tuple, Dict, Any
from requests.adapters import HTTPAdapter

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Thread-local session pooling
_session_local = threading.local()

def get_session():
    """Get thread-local requests session with connection pooling."""
    s = getattr(_session_local, 's', None)
    if s is None:
        s = requests.Session()
        adapter = HTTPAdapter(pool_connections=100, pool_maxsize=100)
        s.mount('http://', adapter)
        s.mount('https://', adapter)
        _session_local.s = s
    return s

def is_red_sparta():
    """Check if running on Red Sparta."""
    return any([
        os.getenv('SPARTA_ENV'),
        os.getenv('INTERNAL_HTTP_PROXY'),
        os.getenv('LLM_PROXY_PROD_HOST'),
        'sparta' in os.getenv('HOSTNAME', '').lower(),
        '/home/ray/efs' in os.getcwd()
    ])

# Configure endpoints
if is_red_sparta():
    LLM_PROXY_URL = 'http://clapi.prod-cheetah.grammarlyaws.com/api/v0/llm-proxy'
    TRANSPARENT_OPENAI_BASE = 'http://clapi.prod-cheetah.grammarlyaws.com/transparent/openai/v1'
    # Set proxies for Red Sparta
    http_proxy = os.getenv('INTERNAL_HTTP_PROXY')
    https_proxy = os.getenv('INTERNAL_HTTPS_PROXY')
    if http_proxy or https_proxy:
        os.environ["HTTP_PROXY"] = http_proxy or ""
        os.environ["HTTPS_PROXY"] = https_proxy or http_proxy or ""
else:
    LLM_PROXY_URL = 'http://clapi.qa-text-processing.grammarlyaws.com/api/v0/llm-proxy'
    TRANSPARENT_OPENAI_BASE = 'http://clapi.qa-text-processing.grammarlyaws.com/transparent/openai/v1'

def _estimate_tokens(prompt: str, content: str) -> Dict[str, int]:
    """Estimate token usage when actual usage unavailable."""
    input_tokens = len(prompt.split())
    output_tokens = len(content.split())
    return {
        'total_tokens': input_tokens + output_tokens,
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'reasoning_tokens': 0,
        'cached_tokens': 0
    }

def _call_openai_direct(prompt: str, model: str, api_key: str) -> Tuple[bool, str, Dict[str, int]]:
    """Call OpenAI API directly."""
    if is_red_sparta():
        return False, "Direct OpenAI not available on Red Sparta", {}
    
    try:
        payload = {
            'model': model,
            'messages': [{"role": "user", "content": prompt}],
        }
        
        # Only add temperature for non-reasoning models
        if not (model.startswith('o3') or model.startswith('o1') or model.startswith('o4')):
            payload['temperature'] = 0.0
        
        r = get_session().post(
            'https://api.openai.com/v1/chat/completions',
            headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'},
            json=payload,
            timeout=(2.0, 30.0)
        )
        
        if r.status_code == 200:
            data = r.json()
            content = data['choices'][0]['message']['content'].strip()
            usage = data.get('usage', {})
            
            token_usage = {
                'total_tokens': usage.get('total_tokens', 0),
                'input_tokens': usage.get('prompt_tokens', 0),
                'output_tokens': usage.get('completion_tokens', 0),
                'reasoning_tokens': usage.get('reasoning_tokens', 0),
                'cached_tokens': usage.get('prompt_tokens_details', {}).get('cached_tokens', 0)
            }
            
            return True, content, token_usage
    except Exception:
        pass
    return False, "", _estimate_tokens(prompt, "")

def _call_transparent_openai(prompt: str, model: str, api_token: str) -> Tuple[bool, str, Dict[str, int]]:
    """Call OpenAI via transparent proxy."""
    try:
        from openai import OpenAI
        
        client = OpenAI(
            api_key=api_token,
            base_url=TRANSPARENT_OPENAI_BASE,
            default_headers={'X-LLM-Proxy-Calling-Service': 'gec_judge'},
        )
        
        if model.startswith('o3') or model.startswith('o1') or model.startswith('o4'):
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
        else:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
        
        content = resp.choices[0].message.content.strip()
        usage = resp.usage if hasattr(resp, 'usage') and resp.usage else None
        
        if usage:
            token_usage = {
                'total_tokens': getattr(usage, 'total_tokens', 0),
                'input_tokens': getattr(usage, 'prompt_tokens', 0),
                'output_tokens': getattr(usage, 'completion_tokens', 0),
                'reasoning_tokens': getattr(usage, 'reasoning_tokens', 0),
                'cached_tokens': getattr(getattr(usage, 'prompt_tokens_details', None), 'cached_tokens', 0) if hasattr(usage, 'prompt_tokens_details') else 0
            }
        else:
            token_usage = _estimate_tokens(prompt, content)
        
        return True, content, token_usage
    except Exception:
        return False, "", _estimate_tokens(prompt, "")

def _call_gemini_direct(prompt: str, model: str, api_key: str) -> Tuple[bool, str, Dict[str, int]]:
    """Call Gemini API directly."""
    if is_red_sparta():
        return False, "Direct Gemini not available on Red Sparta", {}
    
    try:
        url = f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}'
        payload = {
            'contents': [{
                'parts': [{'text': prompt}],
            }]
        }
        
        r = get_session().post(url, json=payload, timeout=(2.0, 30.0))
        if r.status_code == 200:
            data = r.json()
            content = data['candidates'][0]['content']['parts'][0]['text']
            
            token_usage = {'total_tokens': 10, 'input_tokens': 5, 'output_tokens': 5, 'reasoning_tokens': 0, 'cached_tokens': 0}
            if 'usageMetadata' in data:
                usage = data['usageMetadata']
                token_usage = {
                    'input_tokens': usage.get('promptTokenCount', 0),
                    'output_tokens': usage.get('candidatesTokenCount', 0), 
                    'total_tokens': usage.get('totalTokenCount', 0),
                    'reasoning_tokens': 0,
                    'cached_tokens': 0
                }
            
            return True, content.strip(), token_usage
    except Exception:
        pass
    return False, "", _estimate_tokens(prompt, "")

def _call_llm_proxy(prompt: str, backend: str, api_token: str) -> Tuple[bool, str, Dict[str, int]]:
    """Call via LLM Proxy (works on both Sparta and local)."""
    try:
        headers = {'Content-Type': 'application/json'}
        
        if is_red_sparta():
            headers['X-LLM-Proxy-Calling-Service'] = 'gec_judge@grammarly.com'
        else:
            headers['Authorization'] = f'Bearer {api_token}'
        
        payload = {
            'tracking_id': 'gec_judge_call',
            'messages': [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': prompt}
            ],
            'llm_backend': backend,
            'metadata': {'json': '{}'},
            'tags': {},
            'api_token': api_token
        }
        
        # Add temperature for most models
        if backend not in ['o3-2025-04-16', 'openai_direct_o4_mini']:
            payload['generation_parameters'] = {'json': json.dumps({'temperature': 0.0})}
        
        r = requests.post(
            LLM_PROXY_URL,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if r.status_code == 200:
            data = r.json()
            
            # Handle different response formats
            if 'chunk' in data:
                # Red Sparta format
                chunk = data.get('chunk', {})
                content = chunk.get('text', '')
                metadata = chunk.get('metadata', {})
                if metadata and 'json' in metadata:
                    try:
                        metadata_json = json.loads(metadata['json'])
                        usage = metadata_json.get('usage', {})
                    except:
                        usage = {}
                else:
                    usage = {}
            else:
                # Standard format
                content = (
                    data.get('choices', [{}])[0].get('message', {}).get('content')
                    or data.get('content', '')
                )
                usage = data.get('usage', {})
            
            if content:
                if usage:
                    token_usage = {
                        'total_tokens': usage.get('total_tokens', 0),
                        'input_tokens': usage.get('prompt_tokens', 0),
                        'output_tokens': usage.get('completion_tokens', 0),
                        'reasoning_tokens': usage.get('reasoning_tokens', 0),
                        'cached_tokens': usage.get('prompt_tokens_details', {}).get('cached_tokens', 0) if 'prompt_tokens_details' in usage else 0
                    }
                else:
                    token_usage = _estimate_tokens(prompt, content)
                
                return True, content.strip(), token_usage
    except Exception:
        pass
    return False, "", _estimate_tokens(prompt, "")

def call_model(prompt: str, backend: str, api_token: str, *, moderation: bool = False, no_temperature: bool = False, temperature_override: float = None) -> Tuple[bool, str, Dict[str, int]]:
    """
    Super clean model calling function.
    Routes to the right implementation based on backend and environment.
    """
    # Get tokens from environment if not provided
    if not api_token:
        api_token = (
            os.getenv('API_TOKEN', '')
            or os.getenv('OPENAI_API_KEY', '')
        )
    
    # Red Sparta: Everything goes through proxy
    if is_red_sparta():
        return _call_llm_proxy(prompt, backend, api_token)
    
    # Local routing based on backend type
    
    # O3 models: try transparent, then proxy
    if backend == 'o3':
        ok, content, tokens = _call_transparent_openai(prompt, 'o3-2025-04-16', api_token)
        if ok:
            return ok, content, tokens
        return _call_llm_proxy(prompt, 'o3-2025-04-16', api_token)
    
    # Gemini models: try proxy first (works locally), then direct
    if backend in ['gemini-2.0-flash-lite', 'gas_gemini20_flash_lite']:
        # Try proxy first
        proxy_ok, proxy_content, proxy_tokens = _call_llm_proxy(prompt, 'gas_gemini20_flash_lite', api_token)
        if proxy_ok:
            return proxy_ok, proxy_content, proxy_tokens
        
        # Fallback to direct Gemini
        google_key = os.getenv('GOOGLE_API_KEY', '')
        if google_key:
            return _call_gemini_direct(prompt, 'gemini-2.0-flash-lite', google_key)
    
    # OpenAI models: try direct, then proxy
    if backend in ['gpt-4.1', 'gpt-4.1-nano', 'gpt-4o', 'gpt-4o-mini', 'gpt-5-mini', 'gpt-5-nano', 'o4-mini']:
        openai_key = os.getenv('OPENAI_API_KEY', '')
        if openai_key:
            # Map to actual model names
            model_map = {
                'gpt-4.1': 'gpt-4.1',
                'gpt-4.1-nano': 'gpt-4.1-nano', 
                'gpt-4o': 'gpt-4o',
                'gpt-4o-mini': 'gpt-4o-mini',
                'gpt-5-mini': 'gpt-5-mini',
                'gpt-5-nano': 'gpt-5-nano',
                'o4-mini': 'o4-mini'
            }
            model = model_map.get(backend, backend)
            ok, content, tokens = _call_openai_direct(prompt, model, openai_key)
            if ok:
                return ok, content, tokens
        
        # Fallback to proxy
        proxy_backend_map = {
            'gpt-4.1': 'openai_direct_gpt41',
            'gpt-4.1-nano': 'openai_direct_gpt41_nano',
            'gpt-4o': 'openai_direct_chat_gpt4o',
            'gpt-4o-mini': 'openai_direct_chat_gpt4o_mini',
            'gpt-5-mini': 'openai_direct_chat_gpt5_mini',
            'gpt-5-nano': 'openai_direct_chat_gpt5_nano',
            'o4-mini': 'openai_direct_o4_mini'
        }
        proxy_backend = proxy_backend_map.get(backend, backend)
        return _call_llm_proxy(prompt, proxy_backend, api_token)
    
    # Default: try proxy
    return _call_llm_proxy(prompt, backend, api_token)




