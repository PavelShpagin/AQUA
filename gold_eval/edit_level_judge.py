import argparse
import json
import os
import sys
import csv
from collections import Counter
import pandas as pd
import requests
from tqdm import tqdm
import re
from typing import Dict, List, Tuple, Optional
import concurrent.futures
from time import sleep
import random
from datetime import datetime

# Set up path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Backend configurations
BACKEND_CONFIGS = {
    "gpt-4o": {
        "url": "https://api.openai.com/v1/chat/completions",
        "model": "gpt-4o",
        "headers": lambda api_key: {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    },
    "gpt-4o-mini": {
        "url": "https://api.openai.com/v1/chat/completions", 
        "model": "gpt-4o-mini",
        "headers": lambda api_key: {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    }
}

# Transparent API supported backends
TRANSPARENT_API_BACKENDS = [
    'gpt-4o', 'gpt-4o-mini', 
    'o1-preview-2024-09-12',
    'o3-mini-2025-01-31', 'o3-2025-04-16',
    'o4-mini-2025-04-16'
]

# Add support for more backends as needed
LANGUAGE_MAP = {
    "en": "English",
    "de": "German", 
    "ua": "Ukrainian",
    "es": "Spanish"
}

EDIT_LEVEL_JUDGE_PROMPT = """You are an expert GEC (Grammatical Error Correction) judge specializing in LANGUAGE_PLACEHOLDER. Your task is to evaluate edits within a LANGUAGE_PLACEHOLDER sentence and determine if there are any missed errors.

**Important**: Apply LANGUAGE_PLACEHOLDER-specific grammatical rules, morphology, syntax, and linguistic conventions. Consider language-specific error patterns, case systems, agreement rules, and orthographic standards.

Following the FP Severity Guidelines, classify each edit into:
- **TP**: Correct GEC — Minimal corrections that resolve grammatical mistakes without changing meaning, adding errors, or preferential improvements
- **FP3**: Optional edit — No errors introduced; stylistic/preferential suggestion or over-correction/redundancy on already correct text
- **FP2**: Bad edit--medium — Introduces grammatical errors or minor meaning changes; straightforwardly incorrect edits
- **FP1**: Bad edit--severe — Major meaning changes, nonsense, offensive/sensitive content, or breaking the sentence structure

## Guidelines

### Holistic assessment (context + interaction among edits)
- Classify each edit with respect to the full Original → Corrected sentence and the interaction with other edits.
- If an edit only “works” because another edit compensates for it, do not mark it TP in isolation; reflect the dependency and penalize the offending/edit-causing inconsistency (often FP2/FP1).
- If an edit looks odd alone but is necessary and correct given companion edits (e.g., agreement after a head change), consider it TP.
- Prefer minimal, semantically preserving transformations. Do not reward compensatory edit pairs that cancel each other or produce net stylistic drift (often FP3 or FP2).
- Prioritize global grammatical consistency (agreement, case, word order) and meaning preservation over local token-level preferences.
 - When in doubt between FP1 and FP2, prefer FP2 unless the suggestion is extreme nonsense or breaks structural integrity (then FP1).
 - Structural integrity means that quotes, brackets, capitalization, etc., remain proper; breaking them is FP1.

### FP1 - Critical Error
- Suggestions triggering sensitivity issues (changing pronouns, cultural references inappropriately)
- Major meaning changes that alter key information or interpretation
- Nonsensical suggestions that break structural integrity
- Examples: "crimea=>crime", "2-3/10=>3/10" (alters medical scale), introducing random formatting

### FP2 - Medium Error  
- Ungrammatical suggestions that don't pose severe reputational risk
- Introduces grammatical errors or minor meaning changes
- Examples: "genetic=>genetically" (incorrect adverb usage), "didn't=>would not" (changes meaning slightly)

### FP3 - Minor Error
- Stylistic or preferential suggestions on already correct text
- Redundancy or over-correction without fixing actual errors
- Examples: Oxford comma preferences, "doesn't=>does not", "I=>I just" (unnecessary addition)

### TP - Correct GEC
- Fixes clear grammatical mistakes (spelling, agreement, syntax, case, morphology)
- Minimal corrections that improve accuracy without changing meaning
- Apply LANGUAGE_PLACEHOLDER-specific grammar rules (case systems, verb conjugations, gender agreement, etc.)

## Task Instructions

1. **Analyze and classify each edit holistically** — use the full-sentence context and the interaction among edits when assigning TP/FP3/FP2/FP1.
2. **Consider language-specific features**:
   - For inflected languages: Case, gender, number agreement
   - For Slavic languages (Ukrainian): Complex case systems, aspect, palatalization
   - For Germanic languages (German): Cases, compound words, word order
   - For Romance languages (Spanish): Gender agreement, verb conjugations, subjunctive
   - Orthographic and diacritical mark corrections
3. **Check for missed errors** — ONLY if there are clear, obvious grammatical errors remaining in the final corrected sentence. Be conservative; do not flag stylistic preferences or debatable issues as missed errors.
4. **Identify writing type** from: Academic/educational/research, Customer service/support, Internal company announcements, Legal, Medical, Official business/company communications, Personal/narrative, Professional correspondence, Promotional/advertising/marketing, Proposals/reports, Review/critique/feedback, Technical documentation/support, Other

## Output Format

Return your response in this EXACT JSON format:

{
  "writing_type": "[Academic/educational/research, Customer service/support, Internal company announcements, Legal, Medical, Official business/company communications, Personal/narrative, Professional correspondence, Promotional/advertising/marketing, Proposals/reports, Review/critique/feedback, Technical documentation/support, Other]",
  "edits": {
    "{original1=>corrected1}": "TP|FP1|FP2|FP3",
    "{original2=>corrected2}": "TP|FP1|FP2|FP3"
  },
  "missed_error": true|false,
  "reasoning": "Your comprehensive reasoning for the entire sentence (explain dependencies among edits if relevant)"
}

## Examples

**Example 1:**
Original: "I have just recieved the letter"
Corrected: "I have just received the letter"  
Alignment: "I have just {recieved=>received} the letter"

Output:
{
  "writing_type": "Personal/narrative",
  "edits": {
    "{recieved=>received}": "TP"
  },
  "missed_error": false,
  "reasoning": "Single spelling correction without introducing errors. The edit correctly fixes 'recieved' to 'received'."
}

**Example 2:**
Original: "We invited Alice, Bob and Charlie"
Corrected: "We invited Alice, Bob, and Charlie" 
Alignment: "We invited Alice, Bob{=>,} and Charlie"

Output:
{
  "writing_type": "Other",
  "edits": {
    "{=>,}": "FP3"
  },
  "missed_error": false,
  "reasoning": "Stylistic change to already correct text. The Oxford comma is a matter of preference, not a grammatical requirement."
}

**Example 3 (No changes):**
Original: "The cat is sleeping on the mat"
Corrected: "The cat is sleeping on the mat"
Alignment: "The cat is sleeping on the mat"

Output:
{
  "writing_type": "Personal/narrative",
  "edits": {},
  "missed_error": false,
  "reasoning": "No corrections needed, sentence is already correct."
}

Now analyze:

Language: LANGUAGE_PLACEHOLDER
Original: ORIGINAL_PLACEHOLDER
Corrected: CORRECTED_PLACEHOLDER
Alignment: ALIGNMENT_PLACEHOLDER

Output:"""

def extract_edits_from_alignment(alignment: str) -> List[str]:
    """Extract individual edits from alignment string"""
    if not alignment or not isinstance(alignment, str):
        return []
        
    # Pattern to match {original=>corrected} or {original=>corrected:::type}
    pattern = r'\{([^}]+)\}'
    matches = re.findall(pattern, alignment)
    
    edits = []
    for match in matches:
        # Remove type information if present
        edit_part = match.split(':::')[0]
        edits.append(edit_part)
    
    return edits

def compute_final_label(src: str, tgt: str, missed_error: bool, edits_with_classifications: List[Dict]) -> str:
    """
    Compute final label based on improved logic rules.
    
    Logic:
    1. If src == tgt and no missed_error → TN
    2. If src == tgt and missed_error → FN  
    3. If src != tgt:
       - Find worst edit classification among actual edits
       - If missed_error=true AND worst edit is TP → FN
       - If missed_error=true AND worst edit is FP → use worst FP (FP1 > FP2 > FP3)
       - If missed_error=false → use worst edit classification
    
    Example: missed error + [TP, TP, FP3, FP2] → FP2 (not FN)
    """
    # Rule 1: If src == tgt and no missed_error, then TN
    if src == tgt and not missed_error:
        return "TN"
    
    # Rule 2: If src == tgt and missed_error, then FN
    if src == tgt and missed_error:
        return "FN"
    
    # Rule 3: If src != tgt, find worst edit classification
    if not edits_with_classifications:
        return "ERROR"  # No edits found
    
    # Priority: FP1 > FP2 > FP3 > TP
    label_priority = {"FP1": 4, "FP2": 3, "FP3": 2, "TP": 1, "ERROR": 0}
    
    worst_label = "TP"  # Default
    for edit_info in edits_with_classifications:
        classification = edit_info.get("classification", "ERROR")
        if label_priority.get(classification, 0) > label_priority.get(worst_label, 0):
            worst_label = classification
    
    # Rule 4: Handle missed errors with actual edits
    if missed_error:
        # If there are missed errors AND the worst actual edit is FP, use the FP
        if worst_label in ["FP1", "FP2", "FP3"]:
            return worst_label
        # If there are missed errors AND all actual edits are TP, then FN
        elif worst_label == "TP":
            return "FN"
    
    # Rule 5: No missed errors, use worst edit classification
    return worst_label

def format_annotated_alignment(alignment: str, edits_with_classifications: List[Dict]) -> str:
    """Order-based annotation: first edit gets first classification, etc."""
    return create_annotated_alignment(alignment, edits_with_classifications, human_readable=False)

def format_annotated_alignment_human(alignment: str, edits_with_classifications: List[Dict]) -> str:
    """Order-based human-readable annotation."""
    return create_annotated_alignment(alignment, edits_with_classifications, human_readable=True)

def call_transparent_api(prompt: str, backend: str, api_key: str, max_retries: int = 25) -> Optional[str]:
    """Call LLM using transparent API for supported backends"""
    try:
        from openai import OpenAI
    except ImportError:
        print("OpenAI library not available. Please install it for transparent API support.")
        return None
    
    # Use transparent API endpoint
    client = OpenAI(
        api_key=api_key,
        base_url='http://clapi.qa-text-processing.grammarlyaws.com/transparent/openai/v1',
        default_headers={'X-LLM-Proxy-Calling-Service': 'kostiantyn.omelianchuk@grammarly.com'}
    )

    max_delay = 600  # Maximum delay for server errors
    
    for attempt in range(max_retries):
        try:
            # Progressive delay for retries (except first attempt)
            if attempt > 0:
                base_delay = min(3 ** attempt, max_delay)
                jitter = 0.5  # Fixed jitter
                delay = base_delay + jitter
                sleep(delay)

            # Make API call - use minimal parameters like the working implementation
            completion = client.chat.completions.create(
                model=backend,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Validate response structure
            if not hasattr(completion, 'choices') or completion.choices is None or len(completion.choices) == 0:
                if attempt == max_retries - 1:
                    print(f"No choices in transparent API response after {max_retries} attempts")
                    return None
                continue
            
            if not hasattr(completion.choices[0], 'message') or not completion.choices[0].message.content:
                if attempt == max_retries - 1:
                    print(f"No message content in transparent API response after {max_retries} attempts")
                    return None
                continue

            content = completion.choices[0].message.content
            return content.strip()
                
        except Exception as e:
            error_msg = str(e)
            
            # Handle different error types with appropriate delays
            if "rate limit" in error_msg.lower() and attempt < max_retries - 1:
                continue
            elif "server error" in error_msg.lower() or "500" in error_msg:
                if attempt < max_retries - 1:
                    sleep(30 + (attempt * 20))  # Longer delay for server errors
                    continue
                else:
                    print(f"Transparent API server error after {max_retries} attempts: {error_msg}")
                    return None
            elif attempt == max_retries - 1:
                print(f"Transparent API failed after {max_retries} attempts: {error_msg}")
                return None
            continue

    print(f"All transparent API retries failed after {max_retries} attempts")
    return None

def call_llm(prompt: str, backend: str, api_key: str, max_retries: int = 25) -> Optional[str]:
    """Call LLM backend with retry logic"""
    # Use transparent API for supported backends
    if backend in TRANSPARENT_API_BACKENDS:
        return call_transparent_api(prompt, backend, api_key, max_retries)
    
    # Use direct OpenAI API for standard backends
    if backend not in BACKEND_CONFIGS:
        raise ValueError(f"Unsupported backend: {backend}")
    
    config = BACKEND_CONFIGS[backend]
    
    payload = {
        "model": config["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 2000
    }
    
    headers = config["headers"](api_key)
    
    for attempt in range(max_retries):
        try:
            response = requests.post(config["url"], json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed after {max_retries} attempts: {e}")
                return None
            sleep(2 ** attempt)  # Exponential backoff
    
    return None

def create_annotated_alignment(alignment: str, edits_list: List[Dict], human_readable: bool = False) -> str:
    """Create annotated alignment by index alignment (first-to-first)."""
    if not alignment or not edits_list:
        return alignment

    import re
    import re

    # Find all {edit} in order
    matches = list(re.finditer(r'\{([^}]+)\}', alignment))
    if not matches:
        return alignment

    # Collect classifications in order
    cls_list: List[str] = [ed.get('classification', 'TP') for ed in edits_list]

    # Rebuild string with first-to-first mapping
    out = []
    last = 0
    for i, m in enumerate(matches):
        out.append(alignment[last:m.start()])
        inner = m.group(1)
        if i < len(cls_list):
            lbl = cls_list[i]
            if human_readable:
                human = {'TP': 'True', 'FP3': 'Minor', 'FP2': 'Medium', 'FP1': 'Critical'}.get(lbl, lbl)
                out.append('{' + inner + ':' + human + '}')
            else:
                out.append('{' + inner + ':::' + lbl + '}')
        else:
            # No available label; keep original edit without :::ERROR
            out.append(m.group(0))
        last = m.end()
    out.append(alignment[last:])
    return ''.join(out)

def extract_data_from_malformed_json(response: str) -> Optional[Dict]:
    """Extract data from malformed JSON using regex patterns"""
    try:
        # Extract writing_type
        writing_type_match = re.search(r'"writing_type"\s*:\s*"([^"]+)"', response)
        writing_type = writing_type_match.group(1) if writing_type_match else "Other"
        
        # Extract missed_error
        missed_error_match = re.search(r'"missed_error"\s*:\s*(true|false)', response)
        missed_error = missed_error_match.group(1) == 'true' if missed_error_match else False
        
        # missed_error_details field removed
        
        # Extract reasoning
        reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]+(?:"[^"]*"[^"]*)*)"', response, re.DOTALL)
        reasoning = reasoning_match.group(1) if reasoning_match else "Unable to extract reasoning"
        
        # Extract edits using multiple strategies
        edits = []
        
        # Strategy 1: Find lines that look like edit entries
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            # Look for patterns like: "something": "TP" or {something=>other}: "FP2"
            patterns = [
                r'"(\{[^}]*\})"\s*:\s*"(TP|FP[123])"',
                r'"(\{[^}]*\})"\s*:\s*"(TP|FP[123])',
                r'(\{[^}]*\})\s*:\s*"(TP|FP[123])"',
                r'(\{[^}]*\})\s*:\s*(TP|FP[123])',
                r'"([^"]*=>+[^"]*)"\s*:\s*"(TP|FP[123])"',
                r'([^:]*=>+[^:]*)\s*:\s*"(TP|FP[123])"',
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, line)
                for match in matches:
                    edit_text, classification = match
                    edit_text = edit_text.strip().strip('"\'')
                    
                    # Unescape JSON-escaped characters to match original alignment format
                    edit_text = edit_text.replace('\\"', '"').replace('\\\\', '\\')
                    
                    # Ensure proper formatting
                    if '=>' in edit_text and not edit_text.startswith('{'):
                        edit_text = '{' + edit_text + '}'
                    
                    edits.append({
                        "edit": edit_text,
                        "classification": classification,
                        "reasoning": f"Classified as {classification}"
                    })
        
        # Remove duplicates
        seen = set()
        unique_edits = []
        for edit in edits:
            key = (edit['edit'], edit['classification'])
            if key not in seen:
                seen.add(key)
                unique_edits.append(edit)
        edits = unique_edits
        
        return {
            "writing_type": writing_type,
            "edits": edits,
            "missed_error": missed_error,
            "reasoning": reasoning,
            "labeled_alignment": ""
        }
        
    except Exception as e:
        print(f"Fallback extraction error: {e}")
        return None

def parse_new_json_response(response: str) -> Optional[Dict]:
    """Parse new JSON response format from LLM"""
    if not response:
        return None
        
    try:
        # Clean response and try to parse JSON
        cleaned_response = response.strip()
        
        # Handle code blocks if present
        if cleaned_response.startswith('```json'):
            cleaned_response = cleaned_response[7:]
        elif cleaned_response.startswith('```'):
            cleaned_response = cleaned_response[3:]
        if cleaned_response.endswith('```'):
            cleaned_response = cleaned_response[:-3]
        cleaned_response = cleaned_response.strip()
        
        # Aggressive JSON cleaning
        # Remove comments (// text)
        cleaned_response = re.sub(r'//.*?(?=\n|$)', '', cleaned_response, flags=re.MULTILINE)
        
        # Remove trailing commas
        cleaned_response = re.sub(r',(\s*[}\]])', r'\1', cleaned_response)
        
        # Fix common malformed patterns
        # Fix unescaped quotes in values
        cleaned_response = re.sub(r'": "([^"]*)"([^"]*)"([^"]*)"', r'": "\1\\\"\2\\\"\3"', cleaned_response)
        
        # Fix malformed keys with quotes
        cleaned_response = re.sub(r'"(\{[^}]*)"([^}]*)"([^}]*\})"\s*:', r'"\1\\\"\2\\\"\3":', cleaned_response)
        
        # Fix single quotes to double quotes
        cleaned_response = re.sub(r"'([^']*)'", r'"\1"', cleaned_response)
        
        # Fix missing quotes around keys
        cleaned_response = re.sub(r'(\{[^}]*\})\s*:', r'"\1":', cleaned_response)
        
        # Parse JSON
        data = json.loads(cleaned_response)
        
        # Validate required fields
        required_fields = ['writing_type', 'edits', 'missed_error', 'reasoning']
        if not all(field in data for field in required_fields):
            print(f"Missing required fields in JSON response: {[f for f in required_fields if f not in data]}")
            return None
        
        # Convert edits format to our internal format
        edits = []
        for edit_text, classification in data.get('edits', {}).items():
            # Unescape JSON-escaped characters to match original alignment format
            unescaped_edit = edit_text.replace('\\"', '"').replace('\\\\', '\\')
            edits.append({
                "edit": unescaped_edit,
                "classification": classification,
                "reasoning": f"Classified as {classification}"
            })
        
        # missed_error_details field removed
        
        return {
            "writing_type": data['writing_type'],
            "edits": edits,
            "missed_error": data['missed_error'],
            "reasoning": data['reasoning'],
            "labeled_alignment": ""  # Not used in new format
        }
        
    except json.JSONDecodeError as e:
        # Use robust extraction instead of trying to fix JSON
        return extract_data_from_malformed_json(response)
    except Exception as e:
        print(f"Error parsing new JSON response: {e}")
        print(f"Response: {response[:300]}...")
        return None

def parse_json_response(response: str) -> Optional[Dict]:
    """Parse JSON response from LLM, handling common formatting issues"""
    if not response:
        return None
        
    # Clean up the response
    response = response.strip()
    
    # Try to find JSON block
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Look for JSON object - find the outermost braces
        start_idx = response.find('{')
        if start_idx == -1:
            print(f"No JSON object found in response: {response[:200]}...")
            return None
            
        # Find the matching closing brace
        brace_count = 0
        end_idx = -1
        for i in range(start_idx, len(response)):
            if response[i] == '{':
                brace_count += 1
            elif response[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break
        
        if end_idx == -1:
            # Truncated JSON - try to parse anyway, might be partial but parseable
            json_str = response[start_idx:]
        else:
            json_str = response[start_idx:end_idx]
    
    # Try parsing the JSON
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        # Try to fix common JSON issues
        try:
            # Remove trailing commas
            fixed_json = re.sub(r',(\s*[}\]])', r'\1', json_str)
            
            # Fix common problematic patterns in edit fields
            # Fix malformed edit fields like {"edit": ,=>. ", ...}
            fixed_json = re.sub(r'"edit":\s*([^",{}\[\]]+)(?=,|\s*")', r'"edit": "\1"', fixed_json)
            
            # Fix unescaped quotes in edit values
            # Pattern: "edit": "some=>value with "quotes"" should be "edit": "some=>value with \"quotes\""
            def fix_quotes_in_edits(match):
                field_content = match.group(1)
                # Escape internal quotes
                escaped_content = field_content.replace('"', '\\"')
                return f'"edit": "{escaped_content}"'
            
            # Apply quote fixing to edit fields
            fixed_json = re.sub(r'"edit":\s*"([^"]*(?:"[^"]*)*)"', fix_quotes_in_edits, fixed_json)
            
            # Fix edit fields that have unquoted values with special chars
            # Look for patterns like "edit": value=>other,
            fixed_json = re.sub(r'"edit":\s*([^",]+=>?[^",]*),', r'"edit": "\1",', fixed_json)
            
            # Try parsing the fixed version
            return json.loads(fixed_json)
        except json.JSONDecodeError:
            # If still failing, try more aggressive fixes
            try:
                # Last resort: try to extract just the key components we need
                writing_type_match = re.search(r'"writing_type":\s*"([^"]*)"', json_str)
                missed_error_match = re.search(r'"missed_error":\s*(true|false)', json_str)
                
                if writing_type_match and missed_error_match:
                    # Create a minimal valid response
                    return {
                        "writing_type": writing_type_match.group(1),
                        "edits": [],  # Default to empty edits if parsing fails
                        "missed_error": missed_error_match.group(1) == 'true',
                        "missed_error_details": None,
                        "reasoning": "JSON parsing partially failed, using minimal response"
                    }
            except:
                pass
                
            print(f"JSON parse error: {e}")
            print(f"Failed JSON: {json_str[:500]}...")
            return None

def process_sample(sample: Dict, language: str, backend: str, api_key: str, demo_mode: bool = False) -> Dict:
    """Process a single sample through the judge"""
    # Convert to strings and handle NaN values
    original = str(sample['src']) if pd.notna(sample['src']) else ""
    corrected = str(sample['tgt']) if pd.notna(sample['tgt']) else ""
    alignment = str(sample['aligned']) if pd.notna(sample['aligned']) else ""
    
    # Skip processing if any required field is empty
    if not original or not corrected or not alignment:
        return {
            'idx': sample.get('idx', 0),
            'src': original,
            'tgt': corrected,
            'aligned': alignment,
            'num_edits': 0,
            'llm_edits': alignment,
            'llm_edits_h': alignment,
            'llm_missed_error': None,
            'llm_writing_type': None,
            'llm_reasoning': f"Missing required fields: src='{original}', tgt='{corrected}', alignment='{alignment}'",
            'llm_label': 'ERROR'
        }
    
    # Extract edits
    edits = extract_edits_from_alignment(alignment)
    num_edits = len(edits)
    
    # Create prompt by replacing placeholders
    # Minimal sanitization to avoid JSON escape issues in LLM output (e.g., backslashes)
    original_prompt = original.replace('\\', '\\\\')
    corrected_prompt = corrected.replace('\\', '\\\\')
    alignment_prompt = alignment.replace('\\', '\\\\')
    # Build prompt with sanitized strings
    prompt = EDIT_LEVEL_JUDGE_PROMPT.replace("LANGUAGE_PLACEHOLDER", LANGUAGE_MAP.get(language, language))
    prompt = prompt.replace("ORIGINAL_PLACEHOLDER", original_prompt)
    prompt = prompt.replace("CORRECTED_PLACEHOLDER", corrected_prompt)
    prompt = prompt.replace("ALIGNMENT_PLACEHOLDER", alignment_prompt)
    
    # Call LLM or use demo response
    if demo_mode:
        response = create_demo_response(original, corrected, alignment, edits)
        # Demo mode uses new JSON format
        parsed_response = parse_new_json_response(response) if response else None
    else:
        response = call_llm(prompt, backend, api_key)
        # Try new JSON format first, then fall back to old formats (no re-calls)
        parsed_response = None
        if response:
            parsed_response = parse_new_json_response(response)
            if not parsed_response:
                parsed_response = parse_json_response(response)
    
    # Extract data from parsed response
    if parsed_response:
        llm_edits_list = parsed_response.get('edits', [])
        llm_missed_error = parsed_response.get('missed_error', False)
        llm_writing_type = parsed_response.get('writing_type', 'Other')
        llm_reasoning = parsed_response.get('reasoning', '')

        # Compute final label using logic rules
        llm_label = compute_final_label(original, corrected, llm_missed_error, llm_edits_list)
        
        # Format annotated alignment string from edits
        llm_edits = create_annotated_alignment(alignment, llm_edits_list, human_readable=False)
        llm_edits_h = create_annotated_alignment(alignment, llm_edits_list, human_readable=True)
    else:
        # Handle parsing failure
        llm_edits_list = []
        llm_missed_error = None
        llm_writing_type = None
        llm_reasoning = f"Failed to parse LLM response: {response[:100]}..." if response else "No LLM response"
        llm_label = 'ERROR'
        llm_edits = alignment  # Use original alignment
        llm_edits_h = alignment  # Use original alignment
    
    # Prepare result for CSV output - only specified columns
    result = {
        'idx': sample.get('idx', 0),
        'src': original,
        'tgt': corrected,
        'aligned': alignment,
        'num_edits': num_edits,
        'llm_edits': llm_edits,
        'llm_edits_h': llm_edits_h,
        'llm_missed_error': llm_missed_error,
        'llm_writing_type': llm_writing_type,
        'llm_reasoning': llm_reasoning,
        'llm_label': llm_label
    }
    
    return result

def create_demo_response(original: str, corrected: str, alignment: str, edits: List[str]) -> str:
    """Create a demo response for testing without API calls"""
    
    # Simple heuristic classification for demo
    edit_classifications = []
    for edit in edits:
        if "=>" in edit:
            before, after = edit.split("=>", 1)
            
            # Demo logic - very simplified
            if before.strip() == "" or after.strip() == "":
                # Insertion or deletion
                classification = "TP"
                reasoning = "Insertion/deletion correction"
            elif len(before) > len(after) + 5 or len(after) > len(before) + 5:
                # Major change
                classification = "FP2" 
                reasoning = "Significant text change"
            elif before.lower() == after.lower():
                # Case change
                classification = "TP"
                reasoning = "Capitalization correction"
            else:
                # Regular change
                classification = "TP"
                reasoning = "Standard correction"
                
            edit_classifications.append({
                "edit": edit,
                "classification": classification,
                "reasoning": reasoning
            })
    
    # Demo sentence-level logic (using improved classification)
    missed_error = False  # Demo mode rarely has missed errors
    
    if original == corrected and not missed_error:
        sentence_level = "TN"
    elif original == corrected and missed_error:
        sentence_level = "FN"
    elif not edit_classifications:
        # No edits found but src != tgt - this shouldn't happen in good data
        sentence_level = "ERROR"
    else:
        # Apply improved logic: find worst edit classification
        priorities = {"FP1": 4, "FP2": 3, "FP3": 2, "TP": 1}
        worst_label = max([ec["classification"] for ec in edit_classifications], 
                         key=lambda x: priorities.get(x, 0))
        
        # If missed_error=true AND worst edit is FP, use the FP
        if missed_error and worst_label in ["FP1", "FP2", "FP3"]:
            sentence_level = worst_label
        # If missed_error=true AND all edits are TP, then FN
        elif missed_error and worst_label == "TP":
            sentence_level = "FN"
        # Otherwise use worst edit classification
        else:
            sentence_level = worst_label
    
    # Create edits dict for new JSON format
    edits_dict = {}
    for edit_class in edit_classifications:
        edit = edit_class["edit"]
        classification = edit_class["classification"]
        edits_dict[f"{{{edit}}}"] = classification
    
    # Return new JSON format
    demo_response = {
        "writing_type": "Personal/narrative",
        "edits": edits_dict,
        "missed_error": missed_error,
        "reasoning": f"Demo classification based on {len(edits)} edit(s)"
    }
    
    return json.dumps(demo_response, indent=2)

def process_data(data_file: str, language: str, backend: str, api_key: str, sample_size: Optional[int] = None, demo_mode: bool = False, n_parallel_jobs: int = 10) -> List[Dict]:
    """Process the data file"""
    # Read data
    df = pd.read_csv(data_file)
    
    # Process ALL samples (both with and without changes)
    # This is important for getting TNs (True Negatives) where src==tgt and no missed errors
    df_to_process = df.copy()
    
    if sample_size:
        df_to_process = df_to_process.sample(n=min(sample_size, len(df_to_process)))
    
    # Show breakdown of samples
    changes_count = len(df_to_process[df_to_process['src'] != df_to_process['tgt']])
    no_changes_count = len(df_to_process[df_to_process['src'] == df_to_process['tgt']])
    
    print(f"Processing {len(df_to_process)} samples total:")
    print(f"  - {changes_count} samples with changes (src != tgt)")
    print(f"  - {no_changes_count} samples without changes (src == tgt)")
    
    if demo_mode:
        print("Running in DEMO mode (no API calls)")
        print(f"Using {n_parallel_jobs} parallel workers")
    else:
        print(f"Using {n_parallel_jobs} parallel workers")
    
    # Process samples in parallel
    results = []
    
    if demo_mode or len(df_to_process) == 1:
        # Sequential processing for demo mode or single sample
        for _, row in tqdm(df_to_process.iterrows(), total=len(df_to_process)):
            result = process_sample(row.to_dict(), language, backend, api_key, demo_mode)
            results.append(result)
    else:
        # Parallel processing for real API calls
        max_workers = min(n_parallel_jobs, 50)  # Cap at 50 for API stability
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_sample = {
                executor.submit(process_sample, row.to_dict(), language, backend, api_key, demo_mode): row
                for _, row in df_to_process.iterrows()
            }
            
            # Collect results as they complete
            for future in tqdm(concurrent.futures.as_completed(future_to_sample), total=len(df_to_process), desc="Processing samples"):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    # Handle individual sample failures
                    sample_data = future_to_sample[future]
                    print(f"Error processing sample {sample_data.get('idx', 'unknown')}: {e}")
                    # Add error result
                    error_result = {
                        'idx': sample_data.get('idx', 0),
                        'src': sample_data.get('src', ''),
                        'tgt': sample_data.get('tgt', ''),
                        'alignment': sample_data.get('alignment', ''),
                        'extracted_edits': [],
                        'llm_response': None,
                        'parsed_response': None,
                        'sentence_level_label': 'ERROR',
                        'missed_error': None,
                        'writing_type': None,
                        'reasoning': f"Processing error: {str(e)}"
                    }
                    results.append(error_result)
    
    return results

def calculate_statistics(results: List[Dict]) -> Dict:
    """Calculate label distribution statistics"""
    labels = [r['llm_label'] for r in results if r['llm_label'] != 'ERROR']
    
    total = len(labels)
    if total == 0:
        return {}
    
    counter = Counter(labels)
    
    stats = {}
    for label in ['TP', 'FP3', 'FP2', 'FP1', 'TN', 'FN']:
        count = counter.get(label, 0)
        percentage = (count / total) * 100 if total > 0 else 0
        stats[label] = {'count': count, 'percentage': percentage}
    
    return stats

def generate_examples_file(results: List[Dict], examples_file: str, language: str, backend: str) -> None:
    """Generate examples file with 3 examples of each category"""
    
    # Group results by label
    label_examples = {}
    for result in results:
        label = result.get('llm_label', 'ERROR')
        if label not in label_examples:
            label_examples[label] = []
        label_examples[label].append(result)
    
    # Generate examples text
    with open(examples_file, 'w', encoding='utf-8') as f:
        f.write(f"Edit-Level GEC Judge Examples\n")
        f.write(f"Language: {language}\n")
        f.write(f"Backend: {backend}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        
        # Process each label in order
        for label in ['TP', 'FP3', 'FP2', 'FP1', 'TN', 'FN']:
            if label in label_examples and label_examples[label]:
                f.write(f"{label} Examples:\n")
                f.write("-" * 20 + "\n")
                
                # Get up to 3 examples
                examples = label_examples[label][:3]
                
                for i, example in enumerate(examples, 1):
                    f.write(f"\nExample {i}:\n")
                    f.write(f"Original:  {example.get('src', 'N/A')}\n")
                    f.write(f"Corrected: {example.get('tgt', 'N/A')}\n")
                    f.write(f"Alignment: {example.get('aligned', 'N/A')}\n")
                    
                    # Add edit-level details if available
                    parsed_response = example.get('parsed_response')
                    if parsed_response and 'edits' in parsed_response:
                        f.write(f"Edits: {len(parsed_response['edits'])} edit(s)\n")
                        for edit in parsed_response['edits'][:2]:  # Show max 2 edits
                            f.write(f"  - {edit.get('edit', 'N/A')}: {edit.get('classification', 'N/A')} ({edit.get('reasoning', 'N/A')})\n")
                    
                    if parsed_response and 'reasoning' in parsed_response:
                        reasoning = parsed_response['reasoning'][:150] + "..." if len(parsed_response['reasoning']) > 150 else parsed_response['reasoning']
                        f.write(f"Reasoning: {reasoning}\n")
                    
                    if parsed_response and 'missed_error' in parsed_response:
                        f.write(f"Missed Error: {parsed_response['missed_error']}\n")
                                # missed_error_details field removed
                
                f.write("\n" + "=" * 60 + "\n\n")
            else:
                f.write(f"{label} Examples:\n")
                f.write("-" * 20 + "\n")
                f.write("No examples found.\n\n")
                f.write("=" * 60 + "\n\n")
        
        # Add summary statistics
        f.write("Summary Statistics:\n")
        f.write("-" * 20 + "\n")
        total_samples = len([r for r in results if r.get('llm_label') != 'ERROR'])
        for label in ['TP', 'FP3', 'FP2', 'FP1', 'TN', 'FN']:
            count = len(label_examples.get(label, []))
            percentage = (count / total_samples * 100) if total_samples > 0 else 0
            f.write(f"{label}: {count} ({percentage:.1f}%)\n")

def main():
    parser = argparse.ArgumentParser(description="Edit-level GEC Judge")
    parser.add_argument('--data', type=str, required=True, help='Path to CSV data file')
    parser.add_argument('--lang', type=str, choices=['en', 'de', 'ua', 'es'], required=True, help='Language code')
    # Combine standard and transparent API backends
    all_backends = list(BACKEND_CONFIGS.keys()) + [b for b in TRANSPARENT_API_BACKENDS if b not in BACKEND_CONFIGS.keys()]
    parser.add_argument('--backend', type=str, choices=all_backends, default='gpt-4o-mini', help='LLM backend')
    parser.add_argument('--sample', type=int, help='Number of samples to process (default: all)')
    parser.add_argument('--api_key', type=str, help='API key for the backend')
    parser.add_argument('--output', type=str, help='Output file path (default: auto-generated)')
    parser.add_argument('--demo', action='store_true', help='Run in demo mode without API calls')
    parser.add_argument('--n_parallel_jobs', type=int, default=10, help='Number of parallel workers (default: 10)')
    
    args = parser.parse_args()
    
    # Get API key from environment if not provided (not needed in demo mode)
    api_key = args.api_key or os.getenv('OPENAI_API_KEY')
    if not api_key and not args.demo:
        print("Error: API key required. Set OPENAI_API_KEY environment variable or use --api_key")
        print("For testing without API key, use --demo flag")
        return
    
    # Process data
    results = process_data(args.data, args.lang, args.backend, api_key or "demo", args.sample, args.demo, args.n_parallel_jobs)
    
    # Calculate statistics
    stats = calculate_statistics(results)
    
    # Print statistics
    print("\n" + "="*50)
    print("LABEL DISTRIBUTION")
    print("="*50)
    
    total_processed = len([r for r in results if r['llm_label'] != 'ERROR'])
    total_errors = len([r for r in results if r['llm_label'] == 'ERROR'])
    
    print(f"Successfully processed: {total_processed}")
    print(f"Processing errors: {total_errors}")
    print()
    
    for label in ['TP', 'FP3', 'FP2', 'FP1', 'TN', 'FN']:
        if label in stats:
            count = stats[label]['count']
            percentage = stats[label]['percentage']
            print(f"{label}: {count:4d} ({percentage:5.1f}%)")
    
    # Create output directory
    output_dir = "data/results/processed_edits"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results as CSV
    if args.output:
        output_file = args.output
    else:
        base_name = os.path.splitext(os.path.basename(args.data))[0]
        demo_suffix = "_demo" if args.demo else ""
        output_file = os.path.join(output_dir, f"edit_judge_{base_name}_{args.lang}_{args.backend}{demo_suffix}.csv")
    
    # Define CSV columns as requested  
    csv_columns = ['idx', 'src', 'tgt', 'aligned', 'num_edits', 'llm_edits', 'llm_edits_h', 'llm_missed_error', 'llm_writing_type', 'llm_reasoning', 'llm_label']
    
    # Create DataFrame from results
    csv_data = []
    for result in results:
        csv_row = {col: result.get(col, '') for col in csv_columns}
        csv_data.append(csv_row)
    
    df_output = pd.DataFrame(csv_data)
    df_output.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"\nResults saved to: {output_file}")
    
    # Generate examples file
    examples_file = os.path.join(output_dir, f"examples_{base_name}_{args.lang}_{args.backend}{demo_suffix}.txt")
    generate_examples_file(results, examples_file, args.lang, args.backend)
    print(f"Examples saved to: {examples_file}")

if __name__ == "__main__":
    main()