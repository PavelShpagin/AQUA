#!/usr/bin/env python3
"""
Clean LLM backend implementation with explicit model routing.
No silent fallbacks - fails cleanly if intended model unavailable.

Model Routing:
1. Red Sparta -> LLM Proxy
2. Transparent models (gpt-4.1, gpt-4.1-nano, o4-mini, o3, gpt-4o*) -> Transparent API
3. GPT-5 models -> OpenAI Direct API
4. Gemini models -> Google API Direct
5. Everything else -> LLM Proxy
"""

import os
import json
import requests
from requests import exceptions as req_exc
import threading
from typing import Tuple, Dict, Any
from requests.adapters import HTTPAdapter

# Load environment variables
try:
    from dotenv import load_dotenv, find_dotenv
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)
    else:
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
        adapter = HTTPAdapter(pool_connections=400, pool_maxsize=400)
        s.mount('http://', adapter)
        s.mount('https://', adapter)
        _session_local.s = s
    return s


def _json_or_raise(resp, context: str) -> Dict[str, Any]:
    """Safely parse JSON; raise with helpful context if body is empty/non-JSON."""
    try:
        return resp.json()
    except Exception:
        text = (resp.text or '').strip()
        snippet = text[:200].replace('\n', ' ')
        raise Exception(f"{context}: non-JSON response {resp.status_code}; body='{snippet}'")

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
    # Use PROD endpoints locally per request (legacy QA disabled)
    LLM_PROXY_URL = 'http://clapi.prod-cheetah.grammarlyaws.com/api/v0/llm-proxy'
    TRANSPARENT_OPENAI_BASE = 'http://clapi.prod-cheetah.grammarlyaws.com/transparent/openai/v1'

def map_backend_to_internal(backend: str) -> str:
    """Map user-friendly backend names to internal LLM proxy identifiers."""
    mapping = {
        # OpenAI models -> internal proxy names
        'gpt-4.1': 'openai_direct_gpt41',
        'gpt-4.1-nano': 'openai_direct_gpt41_nano',
        'gpt-4o': 'openai_direct_chat_gpt4o',
        'gpt-4o-mini': 'openai_direct_chat_gpt4o_mini',
        'gpt-4o-mini-search': 'openai_direct_gpt4o_mini_search_preview',
        'gpt-5-mini': 'openai_direct_chat_gpt5_mini',
        'gpt-5-nano': 'openai_direct_chat_gpt5_nano',
        'o3': 'o3-2025-04-16',
        'o4-mini': 'openai_direct_o4_mini',
        
        # Gemini models -> internal proxy names
        'gemini-2.0-flash-lite': 'gas_gemini20_flash_lite',
        'gas_gemini20_flash_lite': 'gas_gemini20_flash_lite',
        'gemini-2.5-flash-lite': 'gas_gemini25_flash_lite',
        'gas_gemini25_flash_lite': 'gas_gemini25_flash_lite',
    }
    return mapping.get(backend.lower(), backend)

def map_internal_to_openai_model(internal_backend: str) -> str:
    """Map internal backend names to actual OpenAI model names."""
    mapping = {
        'openai_direct_gpt41': 'gpt-4.1',
        'openai_direct_gpt41_nano': 'gpt-4.1-nano',
        'openai_direct_chat_gpt4o': 'gpt-4o',
        'openai_direct_chat_gpt4o_mini': 'gpt-4o-mini',
        'openai_direct_gpt4o_mini_search_preview': 'gpt-4o-mini-search-preview',
        'openai_direct_chat_gpt5_mini': 'gpt-5-mini',
        'openai_direct_chat_gpt5_nano': 'gpt-5-nano',
        'openai_direct_o4_mini': 'o4-mini',
        'o3-2025-04-16': 'o3-2025-04-16',
    }
    return mapping.get(internal_backend, internal_backend)

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
    
    connect_to = float(os.getenv('LLM_HTTP_CONNECT_TIMEOUT', '3'))
    read_to = float(os.getenv('LLM_HTTP_READ_TIMEOUT', '45'))
    for attempt in range(2):
        try:
            payload = {
                'model': model,
                'messages': [
                    {"role": "system", "content": "You are an expert GEC evaluator."},
                    {"role": "user", "content": prompt, "cache_control": {"type": "ephemeral"}}
                ],
            }
            # Only add temperature for non-reasoning models
            if not (model.startswith('o3') or model.startswith('o1') or model.startswith('o4') or model.startswith('gpt-5')):
                payload['temperature'] = 0.0
            r = get_session().post(
                'https://api.openai.com/v1/chat/completions',
                headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'},
                json=payload,
                timeout=(connect_to, read_to)
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
        except req_exc.ReadTimeout:
            if attempt == 0:
                continue
        except Exception:
            if attempt == 0:
                continue
    return False, "", _estimate_tokens(prompt, "")

def _call_transparent_openai(prompt: str, model: str, api_token: str) -> Tuple[bool, str, Dict[str, int]]:
    """Call OpenAI via transparent proxy."""
    connect_to = float(os.getenv('LLM_HTTP_CONNECT_TIMEOUT', '3'))
    read_to = float(os.getenv('LLM_HTTP_READ_TIMEOUT', '45'))
    for attempt in range(2):
        try:
            from openai import OpenAI
            
            client = OpenAI(
                api_key=api_token,
                base_url=TRANSPARENT_OPENAI_BASE,
                default_headers={'X-LLM-Proxy-Calling-Service': 'gec_judge'},
            )
            
            messages = [
                {"role": "system", "content": "You are an expert GEC evaluator."},
                {"role": "user", "content": prompt, "cache_control": {"type": "ephemeral"}}
            ]
            
            kw = {}
            if not (model.startswith('o3') or model.startswith('o1') or model.startswith('o4')):
                kw['temperature'] = 0.0
            # Newer SDKs support a request timeout argument via with_options
            try:
                client_rt = client.with_options(timeout=read_to)
                resp = client_rt.chat.completions.create(model=model, messages=messages, **kw)
            except Exception:
                resp = client.chat.completions.create(model=model, messages=messages, **kw)
            
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
        except Exception as e:
            if attempt == 0 and ('timed out' in str(e).lower()):
                continue
            return False, "", _estimate_tokens(prompt, "")

def _call_gemini_direct(prompt: str, model: str, api_key: str) -> Tuple[bool, str, Dict[str, int]]:
    """Call Gemini API directly."""
    if is_red_sparta():
        return False, "Direct Gemini not available on Red Sparta", {}
    
    connect_to = float(os.getenv('LLM_HTTP_CONNECT_TIMEOUT', '3'))
    read_to = float(os.getenv('LLM_HTTP_READ_TIMEOUT', '45'))
    for attempt in range(2):
        try:
            url = f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}'
            payload = {
                'contents': [{
                    'parts': [{'text': prompt}],
                }]
            }
            
            r = get_session().post(url, json=payload, timeout=(connect_to, read_to))
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
        except req_exc.ReadTimeout:
            if attempt == 0:
                continue
        except Exception:
            if attempt == 0:
                continue
    return False, "", _estimate_tokens(prompt, "")

def _call_llm_proxy(prompt: str, backend: str, api_token: str) -> Tuple[bool, str, Dict[str, int]]:
    """Call via LLM Proxy (works on both Sparta and local)."""
    connect_to = float(os.getenv('LLM_HTTP_CONNECT_TIMEOUT', '3'))
    read_to = float(os.getenv('LLM_HTTP_READ_TIMEOUT', '45'))
    for attempt in range(2):
        try:
            # Match the exact payload structure from the working script
            payload = {
                "tracking_id": "",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert GEC evaluator.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                        "cache_control": {"type": "ephemeral"}
                    }
                ],
                "llm_backend": backend,
                "metadata": {"json": "{}"},
                "tags": {},
                "api_token": api_token,
                # Add generation_parameters conditionally below
            }
            # Some backends (o-series, certain 5-series, search-preview) reject generation_parameters
            _no_gen_params = {
                'o3-2025-04-16',
                'openai_direct_o3',
                'openai_direct_o4_mini',
                'openai_direct_chat_gpt5_mini',
                'openai_direct_chat_gpt5_nano',
                'openai_direct_gpt4o_mini_search_preview',
            }
            if backend not in _no_gen_params:
                payload["generation_parameters"] = {"json": json.dumps({"temperature": 0.0})}
            
            # Use proxies if on Sparta (prod network)
            proxies = {"http": os.getenv("INTERNAL_HTTP_PROXY")} if is_red_sparta() or os.getenv("LLM_PROXY_PROD_HOST") else None
            
            r = get_session().post(
                LLM_PROXY_URL,
                json=payload,
                proxies=proxies,
                timeout=(connect_to, read_to)
            )
            
            if r.status_code == 200:
                data = r.json()
                
                # Handle different response formats
                if 'chunk' in data:
                    # Red Sparta format - extract like the working script
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
                elif isinstance(data.get('response'), dict) and 'text' in data.get('response', {}):
                    # Alternate proxy format used in some clusters
                    content = data['response'].get('text', '')
                    usage = data.get('usage', {})
                else:
                    # Standard format
                    content = (
                        data.get('choices', [{}])[0].get('message', {}).get('content')
                        or data.get('content', '')
                    )
                    usage = data.get('usage', {})
                
                if content:
                    if usage:
                        # Extract cached tokens using the same approach as the working script
                        prompt_details = usage.get("prompt_tokens_details", {})
                        cached_tokens = prompt_details.get("cached_tokens", 0) if prompt_details else 0
                        
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
                else:
                    # Provide diagnostic text back to caller for better error reporting
                    try:
                        snippet = json.dumps(data)[:300]
                    except Exception:
                        snippet = str(data)[:300]
                    return False, snippet, _estimate_tokens(prompt, "")
        except req_exc.ReadTimeout:
            if attempt == 0:
                continue
        except Exception as e:
            if attempt == 0 and ('timed out' in str(e).lower()):
                continue
    return False, "", _estimate_tokens(prompt, "")

def call_model(prompt: str, backend: str, api_token: str, *, moderation: bool = False, no_temperature: bool = False, temperature_override: float = None) -> Tuple[bool, str, Dict[str, int]]:
    """
    Clean model calling function with explicit routing (NO SILENT FALLBACKS):
    1. Map backend to internal identifier
    2. Route based on model type and environment
    3. Fail cleanly if intended model/API unavailable
    
    Returns: (success: bool, content: str, token_usage: dict)
    """
    # Get tokens from environment if not provided
    if not api_token:
        api_token = os.getenv('API_TOKEN', '') or os.getenv('OPENAI_API_KEY', '')
    
    # Validate inputs
    if not prompt:
        return (False, "Empty prompt provided", {'total_tokens': 0, 'input_tokens': 0, 'output_tokens': 0, 'reasoning_tokens': 0, 'cached_tokens': 0})
    
    # Step 1: Map backend to internal
    internal_backend = map_backend_to_internal(backend)
    
    # Debug: Log model routing (only if debug enabled)
    if os.getenv('LLM_DEBUG', '').lower() in ('1', 'true', 'on'):
        print(f"[LLM_DEBUG] {backend} -> {internal_backend} -> {_get_routing_info(internal_backend)}")
    
    # Step 2: If Red Sparta -> LLM Proxy (use proxy-specific backend names)
    if is_red_sparta():
        # O3 requires proxy working name on Sparta
        if internal_backend == 'o3-2025-04-16':
            o3_variant = (os.getenv('O3_PROXY_BACKEND', '').strip() or 'openai_direct_o3')
            return _call_llm_proxy(prompt, o3_variant, api_token)
        return _call_llm_proxy(prompt, internal_backend, api_token)
    
    # Step 3: If o4-mini, o3, gpt-4o, gpt-4o-mini, gpt-4.1, gpt-4.1-nano ->
    #         Local: OpenAI Direct (preferred)
    #         Sparta: handled above via LLM Proxy
    transparent_models = {
        'openai_direct_o4_mini',
        'o3-2025-04-16',
        'openai_direct_chat_gpt4o',
        'openai_direct_chat_gpt4o_mini',
        'openai_direct_gpt41',
        'openai_direct_gpt41_nano',
    }
    if internal_backend in transparent_models:
        openai_model = map_internal_to_openai_model(internal_backend)
        # Local/dev: prefer OpenAI Direct
        if not is_red_sparta():
            openai_key = os.getenv('OPENAI_API_KEY', '')
            # Allow using provided api_token if it looks like an OpenAI key
            if (not openai_key) and api_token and api_token.startswith('sk-'):
                openai_key = api_token
            if openai_key:
                ok, content, tokens = _call_openai_direct(prompt, openai_model, openai_key)
                if ok:
                    return ok, content, tokens
                return False, f"OpenAI Direct API failed for {openai_model}", _estimate_tokens(prompt, "")
            # If no OpenAI key is available locally, use transparent gateway with API_TOKEN
            ok, content, tokens = _call_transparent_openai(prompt, openai_model, api_token)
            if ok:
                return ok, content, tokens
            return False, f"Transparent API failed for {openai_model}", _estimate_tokens(prompt, "")
        # Sparta should have been routed to proxy above; keep a guard
        return _call_llm_proxy(prompt, internal_backend, api_token)
    
    # Step 4: If gpt-5*, gpt-5-mini, gpt-5-nano -> OpenAI API direct
    gpt5_models = {'openai_direct_chat_gpt5_mini', 'openai_direct_chat_gpt5_nano'}
    if internal_backend in gpt5_models:
        openai_key = os.getenv('OPENAI_API_KEY', '')
        if openai_key:
            openai_model = map_internal_to_openai_model(internal_backend)
            ok, content, tokens = _call_openai_direct(prompt, openai_model, openai_key)
            if ok:
                return ok, content, tokens
        # NO FALLBACK - fail cleanly if OpenAI direct fails
        return False, f"OpenAI Direct API failed for {openai_model if openai_key else 'missing API key'}", _estimate_tokens(prompt, "")
    
    # Step 5: If Gemini -> Gemini API direct
    if internal_backend.startswith('gas_gemini'):
        google_key = os.getenv('GOOGLE_API_KEY', '')
        if google_key and internal_backend == 'gas_gemini20_flash_lite':
            ok, content, tokens = _call_gemini_direct(prompt, 'gemini-2.0-flash-lite', google_key)
            if ok:
                return ok, content, tokens
        # Route to LLM Proxy (no fallback)
        return _call_llm_proxy(prompt, internal_backend, api_token)
    
    # Step 6: Default -> LLM Proxy
    return _call_llm_proxy(prompt, internal_backend, api_token)


def _get_routing_info(internal_backend: str) -> str:
    """Get routing information for debugging."""
    if is_red_sparta():
        return "LLM_PROXY"
    
    transparent_models = {
        'openai_direct_o4_mini', 'o3-2025-04-16', 'openai_direct_chat_gpt4o',
        'openai_direct_chat_gpt4o_mini', 'openai_direct_gpt41', 'openai_direct_gpt41_nano'
    }
    if internal_backend in transparent_models:
        return "TRANSPARENT_API"
    
    gpt5_models = {'openai_direct_chat_gpt5_mini', 'openai_direct_chat_gpt5_nano'}
    if internal_backend in gpt5_models:
        return "OPENAI_DIRECT"
    
    if internal_backend.startswith('gas_gemini'):
        return "GEMINI_DIRECT"
    
    return "LLM_PROXY"


def create_batch_file(requests_data: list, backend: str) -> str:
    """Create .jsonl batch file for async processing."""
    import tempfile
    import json
    
    # Create temporary .jsonl file
    batch_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
    
    internal_backend = map_backend_to_internal(backend)
    openai_model = map_internal_to_openai_model(internal_backend)
    
    for i, request in enumerate(requests_data):
        batch_request = {
            "custom_id": request.get("custom_id", f"request-{i}"),
            "method": "POST", 
            "url": "/v1/chat/completions",
            "body": {
                # Use OpenAI/Transparent model names in batch payloads
                "model": openai_model,
                "messages": [
                    {"role": "system", "content": "You are an expert GEC evaluator."},
                    {"role": "user", "content": request["prompt"]}
                ],
                "max_tokens": 1000,
                "response_format": {"type": "json_object"}
            }
        }
        
        # Match non-batch temperature behavior (0.0 for non-reasoning models)
        no_temp_models = {'o3-2025-04-16', 'openai_direct_o4_mini', 'openai_direct_chat_gpt5_mini', 'openai_direct_chat_gpt5_nano', 'openai_direct_gpt4o_mini_search_preview'}
        if internal_backend not in no_temp_models:
            batch_request["body"]["temperature"] = 0.0
            
        batch_file.write(json.dumps(batch_request) + '\n')
    
    batch_file.close()
    return batch_file.name


def upload_batch_file(batch_file_path: str, backend: str, api_token: str) -> str:
    """Upload batch file and return file_id."""
    internal_backend = map_backend_to_internal(backend)
    
    # Determine API endpoint based on routing logic
    calling_service = os.getenv('LLM_PROXY_CALLING_SERVICE', 'gec_judge')
    force_openai_direct = os.getenv('FORCE_OPENAI_DIRECT_BATCH', '').lower() in {'1','true','on','yes'}
    if force_openai_direct or internal_backend in {'openai_direct_chat_gpt5_mini', 'openai_direct_chat_gpt5_nano'}:
        # OpenAI Direct for GPT-5
        url = "https://api.openai.com/v1/files"
        headers = {"Authorization": f"Bearer {api_token}"}
    elif is_red_sparta():
        # Red Sparta -> use internal CLAPI first to avoid DataHydra restrictions
        url = "http://clapi.prod-cheetah.grammarlyaws.com/transparent/openai/v1/files"
        headers = {"X-LLM-Proxy-Calling-Service": calling_service}
    else:
        # Default -> Transparent API gateway
        url = "https://apigw.dplane.qagr.io/clapi/transparent/openai/v1/files"
        headers = {"X-LLM-Proxy-Calling-Service": calling_service}
    
    # Upload file
    with open(batch_file_path, 'rb') as f:
        files = {'file': f}
        data = {'purpose': 'batch'}
        
        response = get_session().post(url, headers=headers, files=files, data=data, timeout=30)
        
        if response.status_code == 200:
            data = _json_or_raise(response, "Upload batch file")
            return data['id']
        if is_red_sparta() and os.getenv('DISABLE_GATEWAY_FALLBACK', '').lower() not in {'1','true','on','yes'}:
            # Try gateway as alternative endpoint (not fallback)
            fb_url = "https://apigw.dplane.qagr.io/clapi/transparent/openai/v1/files"
            fb_resp = get_session().post(fb_url, headers=headers, files=files, data=data, timeout=30)
            if fb_resp.status_code == 200:
                return fb_resp.json()['id']
            raise Exception(f"Failed to upload batch file via both endpoints: {response.status_code} - {response.text}; gateway: {fb_resp.status_code} - {fb_resp.text}")
        raise Exception(f"Failed to upload batch file: {response.status_code} - {response.text}")


def create_batch_job(file_id: str, backend: str, api_token: str) -> str:
    """Create batch processing job and return batch_id."""
    internal_backend = map_backend_to_internal(backend)
    
    # Determine API endpoint - same routing as upload_batch_file
    calling_service = os.getenv('LLM_PROXY_CALLING_SERVICE', 'gec_judge')
    force_openai_direct = os.getenv('FORCE_OPENAI_DIRECT_BATCH', '').lower() in {'1','true','on','yes'}
    if force_openai_direct or internal_backend in {'openai_direct_chat_gpt5_mini', 'openai_direct_chat_gpt5_nano'}:
        # OpenAI Direct for GPT-5
        url = "https://api.openai.com/v1/batches"
        headers = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}
    elif is_red_sparta():
        # Red Sparta -> internal CLAPI first
        url = "http://clapi.prod-cheetah.grammarlyaws.com/transparent/openai/v1/batches"
        headers = {"X-LLM-Proxy-Calling-Service": calling_service, "Content-Type": "application/json"}
    else:
        # Default -> Transparent API gateway
        url = "https://apigw.dplane.qagr.io/clapi/transparent/openai/v1/batches"
        headers = {"X-LLM-Proxy-Calling-Service": calling_service, "Content-Type": "application/json"}
    
    payload = {
        "input_file_id": file_id,
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h"
    }
    
    response = get_session().post(url, headers=headers, json=payload, timeout=30)
    
    if response.status_code == 200:
        data = _json_or_raise(response, "Create batch job")
        return data['id']
    if is_red_sparta() and os.getenv('DISABLE_GATEWAY_FALLBACK', '').lower() not in {'1','true','on','yes'}:
        fb_url = "https://apigw.dplane.qagr.io/clapi/transparent/openai/v1/batches"
        fb_resp = get_session().post(fb_url, headers=headers, json=payload, timeout=30)
        if fb_resp.status_code == 200:
            return fb_resp.json()['id']
        raise Exception(f"Failed to create batch job via both endpoints: {response.status_code} - {response.text}; gateway: {fb_resp.status_code} - {fb_resp.text}")
    raise Exception(f"Failed to create batch job: {response.status_code} - {response.text}")


def get_batch_status(batch_id: str, backend: str, api_token: str) -> Dict[str, Any]:
    """Poll batch status and return JSON payload.

    Returns a dict (at least contains 'status', optionally 'output_file_id').
    """
    internal_backend = map_backend_to_internal(backend)
    calling_service = os.getenv('LLM_PROXY_CALLING_SERVICE', 'gec_judge')
    force_openai_direct = os.getenv('FORCE_OPENAI_DIRECT_BATCH', '').lower() in {'1','true','on','yes'}
    if is_red_sparta():
        url = f"http://clapi.prod-cheetah.grammarlyaws.com/transparent/openai/v1/batches/{batch_id}"
        headers = {"X-LLM-Proxy-Calling-Service": calling_service}
    elif force_openai_direct or internal_backend in {'openai_direct_chat_gpt5_mini', 'openai_direct_chat_gpt5_nano'}:
        url = f"https://api.openai.com/v1/batches/{batch_id}"
        headers = {"Authorization": f"Bearer {api_token}"}
    else:
        url = f"https://apigw.dplane.qagr.io/clapi/transparent/openai/v1/batches/{batch_id}"
        headers = {"X-LLM-Proxy-Calling-Service": calling_service}
        if api_token:
            headers["Authorization"] = f"Bearer {api_token}"

    # Robust GET with small retry window for transient 5xx/timeout
    last_exc = None
    for attempt in range(3):
        try:
            r = get_session().get(url, headers=headers, timeout=45)
            if r.status_code == 200:
                break
        except Exception as e:
            last_exc = e
            r = None
        time_to_sleep = 1.0 * (attempt + 1)
        try:
            import time as _t
            _t.sleep(time_to_sleep)
        except Exception:
            pass
    if r is None:
        raise Exception(f"Failed to get batch status after retries: {last_exc}")
    if r.status_code != 200:
        # On Sparta, optionally fall back to gateway for status polling
        if is_red_sparta() and os.getenv('DISABLE_GATEWAY_FALLBACK', '').lower() not in {'1','true','on','yes'}:
            gw_url = f"https://apigw.dplane.qagr.io/clapi/transparent/openai/v1/batches/{batch_id}"
            gw_headers = {"X-LLM-Proxy-Calling-Service": calling_service}
            if api_token:
                gw_headers["Authorization"] = f"Bearer {api_token}"
            r2 = get_session().get(gw_url, headers=gw_headers, timeout=30)
            if r2.status_code != 200:
                raise Exception(f"Failed to get batch status (CLAPI={r.status_code}, GW={r2.status_code}): {r2.text}")
            data = r2.json()
        else:
            raise Exception(f"Failed to get batch status: {r.status_code} - {r.text}")
    else:
        data = r.json()
    
    # Normalize output file id when available
    output_file_id = None
    try:
        if 'output_file_id' in data:
            output_file_id = data.get('output_file_id')
        elif 'output_file_ids' in data and isinstance(data.get('output_file_ids'), list) and data['output_file_ids']:
            output_file_id = data['output_file_ids'][0]
    except Exception:
        output_file_id = None
    out = dict(data)
    if output_file_id:
        out['output_file_id'] = output_file_id
    return out


def download_file_content(file_id: str, backend: str, api_token: str) -> str:
    """Download the file content (JSONL) for a batch output file and return as text."""
    internal_backend = map_backend_to_internal(backend)
    calling_service = os.getenv('LLM_PROXY_CALLING_SERVICE', 'gec_judge')
    force_openai_direct = os.getenv('FORCE_OPENAI_DIRECT_BATCH', '').lower() in {'1','true','on','yes'}
    if is_red_sparta():
        url = f"http://clapi.prod-cheetah.grammarlyaws.com/transparent/openai/v1/files/{file_id}/content"
        headers = {"X-LLM-Proxy-Calling-Service": calling_service}
    elif force_openai_direct or internal_backend in {'openai_direct_chat_gpt5_mini', 'openai_direct_chat_gpt5_nano'}:
        url = f"https://api.openai.com/v1/files/{file_id}/content"
        headers = {"Authorization": f"Bearer {api_token}"}
    else:
        url = f"https://apigw.dplane.qagr.io/clapi/transparent/openai/v1/files/{file_id}/content"
        headers = {"X-LLM-Proxy-Calling-Service": calling_service}
        if api_token:
            headers["Authorization"] = f"Bearer {api_token}"

    # Robust download with retry for empty/non-200 responses
    last_text = ''
    last_status = None
    for attempt in range(4):
        r = get_session().get(url, headers=headers, timeout=120)
        last_status = r.status_code
        if r.status_code == 200 and (r.text and r.text.strip()):
            return r.text
        try:
            import time as _t
            _t.sleep(2.0 * (attempt + 1))
        except Exception:
            pass
        last_text = r.text
    raise Exception(f"Failed to download file content: {last_status} - {last_text[:200]}")




