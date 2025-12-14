#!/usr/bin/env python3
"""
Unified backend router for all judges.

Supports friendly backend names and maps them to concrete API calls.

Return signature: (success: bool, content: str, token_usage: dict)
where token_usage contains: {
    'total_tokens': int,
    'input_tokens': int, 
    'output_tokens': int,
    'reasoning_tokens': int,
    'cached_tokens': int
}
"""

import os
import time
import json
import requests
import threading
from requests.adapters import HTTPAdapter

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # If python-dotenv is not installed, skip
    pass
from typing import Tuple, Dict, Any

# Session pooling for TCP/TLS reuse (10x speedup for bulk requests)
_session_local = threading.local()

def get_session():
    """Get thread-local requests session with connection pooling."""
    s = getattr(_session_local, 's', None)
    if s is None:
        s = requests.Session()
        # Tune pool size to concurrency; 100 is good for most bulk tasks
        adapter = HTTPAdapter(pool_connections=100, pool_maxsize=100)
        s.mount('http://', adapter)
        s.mount('https://', adapter)
        _session_local.s = s
    return s


def get_proxies():
    """Get proxy configuration for Red Sparta environment."""
    if is_red_sparta():
        http_proxy = os.getenv('INTERNAL_HTTP_PROXY')
        https_proxy = os.getenv('INTERNAL_HTTPS_PROXY')
        # Only return proxies if they actually exist
        if http_proxy or https_proxy:
            return {
                'http': http_proxy,
                'https': https_proxy or http_proxy  # Fallback to http if https not set
            }
    return None


def is_red_sparta():
    """Check if running in Red Sparta environment."""
    # Red Sparta indicators: SPARTA_ENV, INTERNAL_HTTP_PROXY, LLM_PROXY_PROD_HOST, or specific hostnames
    return any([
        os.getenv('SPARTA_ENV'),
        os.getenv('INTERNAL_HTTP_PROXY'),
        os.getenv('LLM_PROXY_PROD_HOST'),  # Legacy compatibility
        'sparta' in os.getenv('HOSTNAME', '').lower(),
        'red' in os.getenv('ZONE', '').lower(),
        '/home/ray/efs' in os.getcwd()  # Red Sparta EFS path indicator
    ])

# Auto-set LLM_PROXY_PROD_HOST for legacy compatibility
if is_red_sparta() and not os.getenv('LLM_PROXY_PROD_HOST'):
    os.environ['LLM_PROXY_PROD_HOST'] = '1'

# Configure endpoints based on environment
if is_red_sparta():
    # Red Sparta: Use prod-cheetah (allowlisted)
    LLM_PROXY_ENDPOINT = os.getenv('LLM_PROXY_URL', 'http://clapi.prod-cheetah.grammarlyaws.com/api/v0/llm-proxy')
    TRANSPARENT_OPENAI_BASE = 'http://clapi.prod-cheetah.grammarlyaws.com/transparent/openai/v1'
    BATCH_API_ENDPOINT = 'http://clapi.prod-cheetah.grammarlyaws.com/transparent/openai/v1/files'
    # Set internal proxies for all HTTP requests in Red Sparta
    os.environ["HTTP_PROXY"] = os.environ.get("INTERNAL_HTTP_PROXY", "")
    os.environ["HTTPS_PROXY"] = os.environ.get("INTERNAL_HTTPS_PROXY", "")
else:
    # Local: Use QA endpoint (faster for development)
    LLM_PROXY_ENDPOINT = os.getenv('LLM_PROXY_URL', 'http://clapi.qa-text-processing.grammarlyaws.com/api/v0/llm-proxy')
    TRANSPARENT_OPENAI_BASE = 'http://clapi.qa-text-processing.grammarlyaws.com/transparent/openai/v1'
    BATCH_API_ENDPOINT = 'http://clapi.qa-text-processing.grammarlyaws.com/transparent/openai/v1/files'


# Ensure OpenAI key variable compatibility
if 'OPENAI_API_KEY' not in os.environ and 'OPENAI_API_TOKEN' in os.environ:
    os.environ['OPENAI_API_KEY'] = os.environ['OPENAI_API_TOKEN']


def _estimate_tokens(prompt: str, content: str) -> Dict[str, int]:
    """Estimate token usage (fallback when actual usage unavailable)"""
    input_tokens = len(prompt.split())
    output_tokens = len(content.split())
    return {
        'total_tokens': input_tokens + output_tokens,
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'reasoning_tokens': 0,
        'cached_tokens': 0
    }


def map_backend_alias(backend: str) -> str:
    b = backend.strip().lower()
    alias = {
        # Transparent API aliases
        'o4-mini': 'openai_direct_o4_mini',  # Use Red Sparta working name
        'o3': 'o3-2025-04-16',
        # Gemini - small/fast models
        'gemini-2.0-flash-lite': 'gemini-2.0-flash-lite',
        'gas_gemini20_flash_lite': 'gas_gemini20_flash_lite',
        'gas_gemini20_flash': 'gas_gemini20_flash',
        'gas_gemma3_27b': 'gas_gemma3_27b',  # Gemma 3 27B - smaller than Gemini
        # Bedrock Llama aliases - small models
        'llama3.2-1b': 'aws_bedrock_llama32_1b',  # Ultra small 1B model
        'llama3.2-3b': 'aws_bedrock_llama32_3b',  # Small 3B model  
        'llama3.2-11b': 'aws_bedrock_llama32_11b',
        'llama3.3-70b': 'aws_bedrock_llama33_70b',
        # Bedrock Nova - AWS's small models
        'nova-lite': 'aws_bedrock_nova_lite',  # Very fast/cheap AWS model
        # Hermes small models (3B)
        'hermes-3b': 'hermes_3b_semisynt1m_tn20_2025_03_07',
        # OpenAI direct - Updated with Red Sparta working names
        'gpt-4.1': 'openai_direct_gpt41',
        'gpt-4.1-nano': 'openai_direct_gpt41_nano',  # Fixed: use correct Red Sparta name
        'gpt-4o-mini': 'openai_direct_chat_gpt4o_mini',  # Fixed: use Red Sparta name
        'gpt-4o': 'openai_direct_chat_gpt4o',  # Keep as-is for now
        # GPT-5 models (new)
        'gpt-5-mini': 'openai_direct_chat_gpt5_mini',
        'gpt-5-nano': 'openai_direct_chat_gpt5_nano',
        'gpt5-mini': 'openai_direct_chat_gpt5_mini',  # Alternative alias
        'gpt5-nano': 'openai_direct_chat_gpt5_nano',  # Alternative alias
        # Search preview model
        'gpt-4o-mini-search': 'openai_direct_gpt4o_mini_search_preview',
        # Mistral 7B
        'mistral-7b': 'mistral_7b_trt_llm',
    }
    return alias.get(b, backend)


def _call_transparent_openai(prompt: str, backend: str, api_token: str) -> Tuple[bool, str, Dict[str, int]]:
    try:
        from openai import OpenAI
    except Exception:
        return False, "", {'total_tokens': 0, 'input_tokens': 0, 'output_tokens': 0, 'reasoning_tokens': 0, 'cached_tokens': 0}
    # Optional calling-service header (if provided by env)
    calling_service = os.getenv('CALLING_SERVICE', '') or 'gec_judge'
    
    client = OpenAI(
        api_key=api_token,
        base_url=TRANSPARENT_OPENAI_BASE,
        default_headers={'X-LLM-Proxy-Calling-Service': calling_service},
    )
    try:
        if backend.startswith('o3') or backend.startswith('o1') or backend.startswith('o4'):
            # Remove explicit completion cap to allow model defaults (no artificial truncation)
            resp = client.chat.completions.create(
                model=backend,
                messages=[{"role": "user", "content": prompt}],
            )
        else:
            resp = client.chat.completions.create(
                model=backend,
                messages=[{"role": "user", "content": prompt}],
                temperature=1.0,
            )
        content = resp.choices[0].message.content.strip()
        
        # Extract actual token usage from response if available
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
        return False, "", {'total_tokens': 0, 'input_tokens': 0, 'output_tokens': 0, 'reasoning_tokens': 0, 'cached_tokens': 0}


def _call_openai_direct(prompt: str, model: str, api_key: str) -> Tuple[bool, str, Dict[str, int]]:
    """Call OpenAI API directly (requires external internet access, won't work on Red Sparta)."""
    if is_red_sparta():
        # On Red Sparta, can't access external OpenAI - fall back to transparent proxy
        return _call_transparent_openai(prompt, model, api_key)
    
    try:
        # Build request payload - o3/o4 models don't support temperature
        payload = {
            'model': model,
            'messages': [{"role": "user", "content": prompt}],
        }
        
        # Only add temperature for non-reasoning models
        if not (model.startswith('o3') or model.startswith('o1') or model.startswith('o4')):
            payload['temperature'] = 1.0
        
        r = get_session().post(
            'https://api.openai.com/v1/chat/completions',
            headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'},
            json=payload,
            timeout=(2.0, 45.0),  # (connect, read)
        )
        if r.status_code == 200:
            data = r.json()
            content = data['choices'][0]['message']['content'].strip()
            
            # Extract token usage from response
            usage = data.get('usage', {})
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
            
            return True, content, token_usage
        else:
            # Non-200 status code
            try:
                error_data = r.json()
                error_msg = error_data.get('error', {}).get('message', f'HTTP {r.status_code}')
            except:
                error_msg = f'HTTP {r.status_code}: {r.text[:100]}'
            return False, error_msg, {'total_tokens': 0, 'input_tokens': 0, 'output_tokens': 0, 'reasoning_tokens': 0, 'cached_tokens': 0}
    except Exception as e:
        return False, f"OpenAI direct API error: {str(e)}", {'total_tokens': 0, 'input_tokens': 0, 'output_tokens': 0, 'reasoning_tokens': 0, 'cached_tokens': 0}


def _call_gemini(prompt: str, model: str, api_key: str) -> Tuple[bool, str, Dict[str, int]]:
    """Call Gemini API directly (requires external internet access, won't work on Red Sparta)."""
    if is_red_sparta():
        # On Red Sparta, can't access external Google API - use dplane instead
        return _call_llm_proxy(prompt, model, api_key, no_temperature=False, temperature_override=None)
    
    # Google Generative Language API v1beta
    url = f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}'
    payload = {
        'contents': [{
            'parts': [{'text': prompt}],
        }]
    }
    try:
        r = get_session().post(url, json=payload, timeout=(2.0, 12.0))  # Fail fast
        if r.status_code == 200:
            data = r.json()
            content = data['candidates'][0]['content']['parts'][0]['text']
            
            # Extract token usage if available
            usage = data.get('usageMetadata', {})
            if usage:
                token_usage = {
                    'total_tokens': usage.get('totalTokenCount', 0),
                    'input_tokens': usage.get('promptTokenCount', 0),
                    'output_tokens': usage.get('candidatesTokenCount', 0),
                    'reasoning_tokens': 0,  # Gemini doesn't typically have separate reasoning tokens
                    'cached_tokens': usage.get('cachedContentTokenCount', 0)
                }
            else:
                token_usage = _estimate_tokens(prompt, content)
            
            return True, content, token_usage
    except Exception:
        pass
    return False, "", {'total_tokens': 0, 'input_tokens': 0, 'output_tokens': 0, 'reasoning_tokens': 0, 'cached_tokens': 0}


def _call_gemini_dplane(prompt: str, model: str) -> Tuple[bool, str, Dict[str, int]]:
    """Call Gemini via dplane (local) or LLM Proxy (Red Sparta)."""
    # On Red Sparta, dplane is external - use LLM Proxy instead
    if is_red_sparta():
        api_token = os.getenv('API_TOKEN', '')
        if not api_token:
            return False, "No API_TOKEN for Red Sparta", {'total_tokens': 0, 'input_tokens': 0, 'output_tokens': 0, 'reasoning_tokens': 0, 'cached_tokens': 0}
        return _call_llm_proxy(prompt, 'gas_gemini20_flash_lite', api_token, no_temperature=False, temperature_override=None)
    
    # Local environment - use dplane directly
    try:
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'X-LLM-Proxy-Calling-Service': 'test@grammarly.com'
        }
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 1.0
        }
        
        r = get_session().post(
            'https://apigw.dplane.ppgr.io/clapi/api/v1/chat/completions',
            headers=headers,
            json=payload,
            timeout=(2.0, 10.0)  # Fail fast for bulk processing
        )
        
        if r.status_code == 200:
            data = r.json()
            if "choices" in data and data["choices"] and len(data["choices"]) > 0:
                choice = data["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    content = choice["message"]["content"].strip()
                    
                    # Extract token usage if available
                    usage = data.get('usage', {})
                    if usage:
                        token_usage = {
                            'total_tokens': usage.get('total_tokens', 0),
                            'input_tokens': usage.get('prompt_tokens', 0),
                            'output_tokens': usage.get('completion_tokens', 0),
                            'reasoning_tokens': 0,  # Gemini doesn't typically have separate reasoning tokens
                            'cached_tokens': 0
                        }
                    else:
                        token_usage = _estimate_tokens(prompt, content)
                    
                    return True, content, token_usage
                else:
                    print(f"Gemini dplane: Missing content in message: {choice}")
            else:
                print(f"Gemini dplane: No valid choices in response: {data}")
        else:
            print(f"Gemini dplane Error: Status {r.status_code}")
            try:
                error_data = r.json()
                print(f"Error response: {error_data}")
            except:
                print(f"Error text: {r.text[:200]}")
                
    except Exception as e:
        print(f"Gemini dplane Exception: {e}")
    
    return False, "", {'total_tokens': 0, 'input_tokens': 0, 'output_tokens': 0, 'reasoning_tokens': 0, 'cached_tokens': 0}


def _call_gemini_direct(prompt: str, model: str, api_key: str) -> Tuple[bool, str, Dict[str, int]]:
    """
    Call Gemini models directly via Google's generative AI API.
    Ultra-fast fallback for when proxy/dplane fails.
    """
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": 4096,
                "topP": 1.0,
                "topK": 32
            }
        }
        
        response = get_session().post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=(1.5, 8.0)  # Aggressive timeouts for speed
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'candidates' in result and result['candidates']:
                candidate = result['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    content = candidate['content']['parts'][0].get('text', '')
                    
                    # Extract token usage
                    token_usage = {'total_tokens': 0, 'input_tokens': 0, 'output_tokens': 0, 'reasoning_tokens': 0, 'cached_tokens': 0}
                    if 'usageMetadata' in result:
                        usage = result['usageMetadata']
                        token_usage['input_tokens'] = usage.get('promptTokenCount', 0)
                        token_usage['output_tokens'] = usage.get('candidatesTokenCount', 0) 
                        token_usage['total_tokens'] = usage.get('totalTokenCount', 0)
                    
                    return True, content.strip(), token_usage
        
        return False, f"Gemini direct API error: {response.status_code}", {'total_tokens': 0, 'input_tokens': 0, 'output_tokens': 0, 'reasoning_tokens': 0, 'cached_tokens': 0}
        
    except Exception as e:
        return False, f"Gemini direct API exception: {str(e)[:100]}", {'total_tokens': 0, 'input_tokens': 0, 'output_tokens': 0, 'reasoning_tokens': 0, 'cached_tokens': 0}


def _call_llm_proxy(prompt: str, backend: str, api_token: str, no_temperature: bool = False, *, temperature_override: float = None) -> Tuple[bool, str, Dict[str, int]]:
    try:
        # Debug mode
        debug_mode = os.getenv('LLM_DEBUG') == '1'

        # Build headers based on environment
        headers: Dict[str, str] = {'Content-Type': 'application/json'}
        
        if is_red_sparta():
            # Red Sparta requires specific headers and payload format
            calling_service = os.getenv('CALLING_SERVICE', 'gec_judge@grammarly.com')
            headers['X-LLM-Proxy-Calling-Service'] = calling_service
        else:
            # Local environment - standard auth header
            headers['Authorization'] = f'Bearer {api_token}'
        
        # Check if this model supports generation_parameters
        # GPT-5 series, O-series models might not support it
        models_without_gen_params = [
            'openai_direct_chat_gpt5_mini',
            'openai_direct_chat_gpt5_nano', 
            'openai_direct_chat_gpt5',
            'openai_direct_o4_mini',
            'o3-2025-04-16',
            'o4-mini-2025-04-16'
            'openai_direct_gpt4o_mini_search_preview'
        ]
        
        payload = {
            'tracking_id': 'gec_judge_call',
            'messages': [
                {
                    'role': 'system',
                    'content': 'You are a helpful assistant.'
                },
                {
                    'role': 'user', 
                    'content': prompt
                }
            ],
            'llm_backend': backend,
            'metadata': {'json': '{}'},
            'tags': {},  # Empty tags like in working script
            'api_token': api_token
        }
        
        # Provide max_tokens and temperature (unless suppressed) for compatibility with proxy expectations
        if backend not in models_without_gen_params:
            generation_params = {}
            if not no_temperature:
                generation_params['temperature'] = (
                    temperature_override if temperature_override is not None else 1.0
                )
            payload['generation_parameters'] = {'json': json.dumps(generation_params)}
        
        try:
            if debug_mode:
                print(f"[LLM_DEBUG] POST {LLM_PROXY_ENDPOINT} backend={backend}")
                print(f"[LLM_DEBUG] Payload keys: {list(payload.keys())}")
                if 'generation_parameters' in payload:
                    print(f"[LLM_DEBUG] Generation params: {payload['generation_parameters']['json']}")
                else:
                    print(f"[LLM_DEBUG] No generation_parameters for model {backend}")
                if is_red_sparta():
                    proxies = get_proxies()
                    if proxies:
                        print(f"[LLM_DEBUG] Red Sparta mode - using proxies: {proxies}")
                    else:
                        print(f"[LLM_DEBUG] Red Sparta mode - no proxies configured")
            
            # Use proxies on Red Sparta
            if is_red_sparta():
                proxies = get_proxies()
                # Match the working snippet exactly - use proxies if available
                if proxies:
                    # Red Sparta with proxies (actual cluster)
                    r = requests.post(
                        LLM_PROXY_ENDPOINT,
                        headers=headers,  # Include headers with X-LLM-Proxy-Calling-Service
                        json=payload,
                        proxies=proxies,
                        timeout=30
                    )
                else:
                    # Red Sparta without proxies (might be local test or direct access)
                    r = requests.post(
                        LLM_PROXY_ENDPOINT,
                        headers=headers,  # Include headers with X-LLM-Proxy-Calling-Service
                        json=payload,
                        timeout=30
                    )
            else:
                # Session pooling for local
                r = get_session().post(
                    LLM_PROXY_ENDPOINT,
                    headers=headers,
                    json=payload,
                    timeout=(2.0, 30.0)
                )
        except requests.exceptions.Timeout:
            print(f"LLM Proxy timeout after 14s for {backend}")
            return False, "Request timeout", {'total_tokens': 0, 'input_tokens': 0, 'output_tokens': 0, 'reasoning_tokens': 0, 'cached_tokens': 0}
        except requests.exceptions.ConnectionError as e:
            print(f"LLM Proxy connection error for {backend}: {str(e)[:100]}")
            return False, "Connection error", {'total_tokens': 0, 'input_tokens': 0, 'output_tokens': 0, 'reasoning_tokens': 0, 'cached_tokens': 0}
        if r.status_code >= 400:
            print(f"LLM Proxy Error: Status {r.status_code}, Backend: {backend}")
            try:
                error_data = r.json()
                print(f"Error response: {error_data}")
                error_msg = str(error_data)
            except:
                print(f"Error text: {r.text[:200]}")
                error_msg = r.text[:200]
            return False, error_msg, {'total_tokens': 0, 'input_tokens': 0, 'output_tokens': 0, 'reasoning_tokens': 0, 'cached_tokens': 0}
        if r.status_code == 200:
            # Check if we got HTML (Cloudflare block) instead of JSON
            if 'text/html' in r.headers.get('Content-Type', ''):
                print(f"LLM Proxy blocked by Cloudflare/gateway - are you on the right network?")
                return False, "Blocked by network gateway", {'total_tokens': 0, 'input_tokens': 0, 'output_tokens': 0, 'reasoning_tokens': 0, 'cached_tokens': 0}
            
            data = r.json()
            
            # Handle LLM Proxy response format (chunk/metadata structure from working script)
            if 'chunk' in data:
                # Red Sparta format - extract text from chunk
                chunk = data.get('chunk', {})
                content = chunk.get('text', '')
                
                # Try to parse JSON content if it looks like JSON
                if content and content.strip().startswith('{'):
                    try:
                        content_json = json.loads(content)
                        # Extract actual response text if it's structured
                        if 'response' in content_json:
                            content = content_json['response']
                        elif 'text' in content_json:
                            content = content_json['text']
                        elif 'content' in content_json:
                            content = content_json['content']
                        # For classification responses
                        elif 'classification' in content_json:
                            content = json.dumps(content_json)  # Keep as JSON for parsing
                    except:
                        pass  # Keep original content if not valid JSON
                
                # Try to parse the metadata for usage info
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
                # Extract token usage from response (handle both formats)
                if usage:
                    # Check for cached tokens in prompt_tokens_details (standard OpenAI format)
                    prompt_details = usage.get('prompt_tokens_details', {})
                    cached_tokens = prompt_details.get('cached_tokens', 0) if prompt_details else 0
                    
                    token_usage = {
                        'total_tokens': usage.get('total_tokens', 0),
                        'input_tokens': usage.get('prompt_tokens', 0),
                        'output_tokens': usage.get('completion_tokens', 0),
                        'reasoning_tokens': usage.get('reasoning_tokens', 0),
                        'cached_tokens': cached_tokens
                    }
                else:
                    token_usage = _estimate_tokens(prompt, content)
                
                return True, content.strip(), token_usage
    except Exception:
        pass
    return False, "", {'total_tokens': 0, 'input_tokens': 0, 'output_tokens': 0, 'reasoning_tokens': 0, 'cached_tokens': 0}


def call_model(prompt: str, backend: str, api_token: str, *, moderation: bool = False, no_temperature: bool = False, temperature_override: float = None) -> Tuple[bool, str, Dict[str, int]]:
    """
    Route LLM call based on backend.
    Red Sparta: Everything goes through LLM Proxy.
    Local: Smart routing based on backend type.
    """
    debug_mode = os.getenv('LLM_DEBUG') == '1'
    if debug_mode:
        print(f"[LLM_DEBUG] call_model: backend={backend}, api_token={'***' if api_token else 'None'}, is_red_sparta={is_red_sparta()}")
        print(f"[LLM_DEBUG] Prompt preview: {prompt[:100]}...")
    
    original_backend = backend
    b = map_backend_alias(backend)

    # Strict routing for o4-mini via Transparent OpenAI with exact model name (no fallbacks)
    if original_backend == 'o4-mini':
        ok, content, tokens = _call_transparent_openai(prompt, 'o4-mini-2025-04-16', api_token)
        return ok, content, tokens

    # Red Sparta: Everything through LLM Proxy
    if is_red_sparta():
        if debug_mode:
            print(f"[LLM_DEBUG] Red Sparta: routing {original_backend} -> {b} via LLM Proxy")
        return _call_llm_proxy(prompt, b, api_token, no_temperature, temperature_override=temperature_override)

    # Transparent OpenAI: use transparent path, then proxy. No remap to other models.
    if b in ['o3-2025-04-16']:
        # Use transparent API for o3 models
        ok, content, tokens = _call_transparent_openai(prompt, b, api_token)
        if ok:
            return ok, content, tokens
        proxy_ok, proxy_content, proxy_tokens = _call_llm_proxy(prompt, b, api_token, no_temperature, temperature_override=temperature_override)
        return proxy_ok, proxy_content, proxy_tokens

    # Gemini family
    if b in ['gemini-2.0-flash-lite', 'gas_gemini20_flash_lite']:
        gem_ok, gem_content, gem_tokens = _call_gemini_dplane(prompt, 'gas_gemini20_flash_lite')
        if gem_ok:
            return gem_ok, gem_content, gem_tokens
        key = os.getenv('GOOGLE_API_KEY', '')
        if key:
            return _call_gemini_direct(prompt, 'gemini-2.0-flash-lite', key)
        return False, '', {'total_tokens': 0, 'input_tokens': 0, 'output_tokens': 0, 'reasoning_tokens': 0, 'cached_tokens': 0}

    # OpenAI 4.1 / 4.1-nano / GPT-5: try proxy, then transparent, then direct with provided key. No 4o fallback.
    if b in ['openai_direct_gpt41', 'openai_direct_gpt41_nano', 'openai_direct_o4_mini',
             'openai_direct_chat_gpt5_mini', 'openai_direct_chat_gpt5_nano', 'gpt-5-mini', 'gpt-5-nano',
             'openai_direct_gpt4o_mini_search_preview', 'openai_direct_chat_gpt4o_mini', 'openai_direct_chat_gpt4o', '']:
        proxy_ok, proxy_content, proxy_tokens = _call_llm_proxy(prompt, b, api_token, no_temperature, temperature_override=temperature_override)
        if proxy_ok:
            return proxy_ok, proxy_content, proxy_tokens
        # Direct OpenAI with provided OPENAI_API_KEY only (no alt model)
        oai = os.getenv('OPENAI_API_KEY', '')
        if oai:
            return _call_openai_direct(prompt, b, oai)
        # If neither worked
        return False, '', {'total_tokens': 0, 'input_tokens': 0, 'output_tokens': 0, 'reasoning_tokens': 0, 'cached_tokens': 0}

    # Default: call via proxy using mapped backend
    return _call_llm_proxy(prompt, b, api_token, no_temperature, temperature_override=temperature_override)


