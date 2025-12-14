#!/usr/bin/env python3
"""
Modular judge utilities.

Implements unified LLM calling (transparent API and direct) and modular pipeline
based on specialized prompts: NONSENSE, MEANING_CHANGE, RELATIVE_REWARD/REWARD,
and fallback comprehensive TPFP prompt.

This module intentionally accepts prompts as arguments; it does not define prompts.
"""

import os
import time
import requests
from typing import Tuple, Optional
from utils.llm.backends import call_model
try:
    from utils.llm.settings import get_llm_temperature
except Exception:
    def get_llm_temperature(default: float = 1.0) -> float:
        import os
        value = os.getenv("LLM_TEMPERATURE") or os.getenv("TEMPERATURE")
        try:
            return float(value) if value is not None else float(default)
        except Exception:
            return float(default)


TRANSPARENT_API_BACKENDS = [
    'gpt-4o', 'gpt-4o-mini',
    'o1-preview-2024-09-12', 'o3-mini-2025-01-31', 'o4-mini-2025-04-16', 'o3-2025-04-16'
]


def _call_transparent_api(prompt: str, backend: str, api_token: str, max_retries: int = 10, max_delay: int = 60) -> Tuple[bool, str, int]:
    try:
        from openai import OpenAI
    except Exception:
        return False, "", 0

    base_url = os.getenv('LLM_PROXY_BASE_URL', 'http://clapi.qa-text-processing.grammarlyaws.com/transparent/openai/v1')
    calling_service = os.getenv('CALLING_SERVICE', 'gec_judge')

    client = OpenAI(
        api_key=api_token,
        base_url=base_url,
        default_headers={'X-LLM-Proxy-Calling-Service': calling_service},
    )

    for attempt in range(max_retries):
        try:
            if backend.startswith('o3') or backend.startswith('o1'):
                resp = client.chat.completions.create(
                    model=backend,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=1000,
                )
            else:
                resp = client.chat.completions.create(
                    model=backend,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=get_llm_temperature(),
                    max_tokens=1000,
                )
            content = resp.choices[0].message.content.strip()
            tokens = len(prompt.split()) + len(content.split())
            return True, content, tokens
        except Exception:
            if attempt == max_retries - 1:
                break
            time.sleep(min(2 ** attempt, max_delay))
    return False, "", 0


def _call_openai_direct(prompt: str, backend: str, api_token: str, max_retries: int = 5) -> Tuple[bool, str, int]:
    url = 'https://api.openai.com/v1/chat/completions'
    headers = {'Authorization': f'Bearer {api_token}', 'Content-Type': 'application/json'}
    payload = {
        'model': backend,
        'messages': [{"role": "user", "content": prompt}],
        'temperature': get_llm_temperature(),
        'max_tokens': 1000,
    }
    for attempt in range(max_retries):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=60)
            if r.status_code == 200:
                content = r.json()['choices'][0]['message']['content'].strip()
                tokens = len(prompt.split()) + len(content.split())
                return True, content, tokens
        except Exception:
            pass
        time.sleep(1 + attempt)
    return False, "", 0


def unified_llm_call(prompt: str, backend: str, api_token: str, *, moderation: bool = False) -> Tuple[bool, str, int]:
    # Delegate to unified backends router
    success, content, token_dict = call_model(prompt, backend, api_token, moderation=moderation)
    # Convert token dict to total for backward compatibility
    total_tokens = token_dict.get('total_tokens', 0) if isinstance(token_dict, dict) else 0
    return success, content, total_tokens


def _demo_relative_reward(s: str, t: str) -> int:
    # crude heuristic: minor improvement => +1, degrade => -1, else 0
    if s == t:
        return 0
    # shorter edit distance => improvements
    return 1 if len(t) <= len(s) else -1


def classify_modular(
    s: str,
    t: str,
    backend: str,
    api_token: Optional[str],
    *,
    language_label: str,
    prompts,
    demo: bool = False,
    moderation: bool = False,
    return_debug: bool = False,
) -> Tuple[str, str, Optional[dict]]:
    """
    Full modular pipeline implementation per docs/general.md algorithm.
    
    Models:
    - Nonsense detector: Score from -1 to 3 for nonsense level
    - Meaning-change model: 0-4 scale for meaning alteration  
    - Reward model: -3 to +3 quality improvement score
    
    Algorithm from docs/general.md:
    1. Check FP1: nonsense ≥ 2 OR meaning change ≥ 2
    2. Check FP2: nonsense ≥ 0 OR meaning change ≥ 2 OR quality < 0
    3. Check FP3: 0 < quality < 1
    4. TP: quality ≥ 1
    """
    if not api_token:
        if return_debug:
            return 'Error', 'API token required for modular pipeline', {}
        return 'Error', 'API token required for modular pipeline'

    reasoning_parts = []
    debug_info = {
        'nonsense_score': None,
        'meaning_change_score': None,
        'quality_score': None,
        'modular_scores': {}
    }
    
    # === RUN ALL MODELS IN PARALLEL ===
    # Following docs/general.md: "each 'subcheck' of each FP1, FP2, FP3 check is independent and thus can be easily parallelized"
    
    # 1) Nonsense detection on corrected text
    nonsense_prompt = prompts.NONSENSE_PROMPT.format(t)
    ok_nons, resp_nons, _ = unified_llm_call(nonsense_prompt, backend, api_token, moderation=moderation)
    nonsense_score = 0
    if ok_nons:
        import re
        # Handle both old binary format and new numeric format
        score_match = re.search(r'SCORE:\s*([-]?[0-3])', resp_nons)
        if score_match:
            nonsense_score = int(score_match.group(1))
        elif 'answer: yes' in resp_nons.lower():
            nonsense_score = 2  # Map binary yes to medium nonsense
        elif 'answer: no' in resp_nons.lower():
            nonsense_score = 0  # Map binary no to neutral
        
        debug_info['nonsense_score'] = nonsense_score
        debug_info['modular_scores']['nonsense'] = nonsense_score
        reasoning_parts.append(f"Nonsense: {nonsense_score}")
    else:
        reasoning_parts.append("Nonsense: FAILED")
        debug_info['modular_scores']['nonsense'] = 'FAILED'

    # 2) Meaning change check
    meaning_prompt = prompts.MEANING_CHANGE_PROMPT.format(s, t)
    ok_mc, resp_mc, _ = unified_llm_call(meaning_prompt, backend, api_token, moderation=moderation)
    meaning_score = 0
    if ok_mc:
        import re
        severity_match = re.search(r'SEVERITY:\s*([0-4])', resp_mc)
        if severity_match:
            meaning_score = int(severity_match.group(1))
            debug_info['meaning_change_score'] = meaning_score
            debug_info['modular_scores']['meaning_change'] = meaning_score
        reasoning_parts.append(f"Meaning change: {meaning_score}")
    else:
        reasoning_parts.append("Meaning change: FAILED")
        debug_info['modular_scores']['meaning_change'] = 'FAILED'

    # 3) Quality/Reward assessment
    reward_prompt = prompts.RELATIVE_REWARD_PROMPT.format(s, t)
    ok_rr, resp_rr, _ = unified_llm_call(reward_prompt, backend, api_token, moderation=moderation)
    quality_score = 0
    if ok_rr:
        import re
        improvement_match = re.search(r'IMPROVEMENT:\s*\[?([-+]?[0-3])\]?', resp_rr)
        if improvement_match:
            quality_score = int(improvement_match.group(1))
            debug_info['quality_score'] = quality_score
            debug_info['modular_scores']['quality'] = quality_score
        reasoning_parts.append(f"Quality: {quality_score}")
    else:
        reasoning_parts.append("Quality: FAILED")
        debug_info['modular_scores']['quality'] = 'FAILED'

    # === APPLY CLASSIFICATION ALGORITHM FROM docs/general.md ===
    # Cascading checks: FP1 -> FP2 -> FP3 -> TP
    
    # Check FP1: nonsense ≥ 2 OR meaning change ≥ 2
    if nonsense_score >= 2:
        label = 'FP1'
        reason = f"Major nonsense detected (score: {nonsense_score}). {' | '.join(reasoning_parts)}"
    elif meaning_score >= 2:
        label = 'FP1' 
        reason = f"Significant meaning change (score: {meaning_score}). {' | '.join(reasoning_parts)}"
    
    # Check FP2: quality < 0 (incorrect correction)
    elif quality_score < 0:
        label = 'FP2'
        reason = f"Quality degradation (score: {quality_score}). {' | '.join(reasoning_parts)}"
    
    # Check FP3: 0 ≤ quality < 1 (preferential/minimal improvement)
    elif 0 <= quality_score < 1:
        label = 'FP3'
        if quality_score == 0:
            reason = f"No improvement - preferential change (quality: 0). {' | '.join(reasoning_parts)}"
        else:
            reason = f"Minimal improvement (quality: {quality_score}). {' | '.join(reasoning_parts)}"
    
    # TP: quality ≥ 1 (meaningful improvement)
    else:
        label = 'TP'
        reason = f"Valid correction (quality: {quality_score}). {' | '.join(reasoning_parts)}"
    
    debug_info['final_label'] = label
    
    if return_debug:
        return label, reason, debug_info
    return label, reason


