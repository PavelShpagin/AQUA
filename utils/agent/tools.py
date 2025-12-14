#!/usr/bin/env python3
"""
Agent tools for GEC judging per docs/general.md.

Tools available:
- Language rulebooks with grammar rules (RAG over language conventions)
- Guidelines and knowledge base
- Web search for domain knowledge
- Meaning-change tool
- Nonsense detector
- Reward/quality tool
"""

import os
import re
import json
from typing import Dict, Any, Optional, Tuple
from utils.llm.backends import call_model


def nonsense_detector_tool(text: str, backend: str, api_token: str, prompt_template: str) -> Dict[str, Any]:
    """
    Nonsense detector tool: Returns YES if text introduces nonsense, NO otherwise.
    Uses judge-specific prompt template.
    """
    prompt = prompt_template.format(text=text)
    
    ok, response, _tokens = call_model(prompt, backend, api_token)
    if not ok:
        return {"result": "UNKNOWN", "reasoning": "Tool call failed"}
    
    resp_lower = response.strip().lower()
    is_nonsense = 'answer: yes' in resp_lower
    
    return {
        "result": "YES" if is_nonsense else "NO",
        "reasoning": response.strip(),
        "confidence": "HIGH" if 'answer:' in resp_lower else "LOW"
    }


def meaning_change_tool(original: str, suggested: str, backend: str, api_token: str, prompt_template: str) -> Dict[str, Any]:
    """
    Meaning-change detection tool: Returns severity level 0-4.
    Uses judge-specific prompt template.
    """
    prompt = prompt_template.format(original=original, suggested=suggested)
    
    ok, response, _tokens = call_model(prompt, backend, api_token)
    if not ok:
        return {"severity": 0, "reasoning": "Tool call failed", "confidence": "LOW"}
    
    # Extract severity level
    severity = 0
    severity_match = re.search(r'SEVERITY:\s*([0-4])', response)
    if severity_match:
        severity = int(severity_match.group(1))
    
    return {
        "severity": severity,
        "reasoning": response.strip(),
        "confidence": "HIGH" if severity_match else "LOW"
    }


def reward_quality_tool(original: str, suggested: str, backend: str, api_token: str, prompt_template: str) -> Dict[str, Any]:
    """
    Reward/quality assessment tool: Returns relative quality score.
    Uses judge-specific prompt template.
    """
    prompt = prompt_template.format(original=original, suggested=suggested)
    
    ok, response, _tokens = call_model(prompt, backend, api_token)
    if not ok:
        return {"score": 0, "reasoning": "Tool call failed", "confidence": "LOW"}
    
    # Extract score
    score = 0
    score_match = re.search(r'SCORE:\s*([-+]?[0-3])', response)
    if score_match:
        score = int(score_match.group(1))
    
    return {
        "score": score,
        "reasoning": response.strip(),
        "confidence": "HIGH" if score_match else "LOW"
    }


def grammar_rag_tool(query: str, language: str, backend: str, api_token: str) -> Dict[str, Any]:
    """
    Advanced Grammar RAG tool: Retrieves specific grammar rules using ChromaDB only.
    
    Args:
        query: Specific grammar rule query (e.g., "comma usage with adjectives", "subject-verb agreement")
        language: Target language for grammar rules
        backend: LLM backend (unused, kept for compatibility)
        api_token: API token (unused, kept for compatibility)
    
    Returns:
        Dict with grammar rule, examples, confidence, and metadata
    """
    try:
        from utils.rag.advanced_grammar_rag import AdvancedGrammarRAG
        
        # Initialize ChromaDB-only RAG system
        rag = AdvancedGrammarRAG(
            backend="chromadb",
            embedding_provider="sentence_transformer",
            language=language
        )
        
        # Search for rules
        results = rag.search_rules(query, k=3)
        
        if not results or len(results) == 0:
            return {
                "rule": f"No grammar rules found for query: '{query}' in {language}",
                "source": "ChromaDB Grammar Database",
                "confidence": "LOW",
                "query_matched": query,
                "error": "No matching rules in database"
            }
        
        best_result = results[0]
        
        # Extract rule information from document
        doc_text = best_result.get('document', '')
        metadata = best_result.get('metadata', {})
        
        # Parse rule name and description
        if ':' in doc_text:
            rule_name, description = doc_text.split(':', 1)
            rule_name = rule_name.strip()
            description = description.strip()
        else:
            rule_name = best_result.get('rule_name', 'Grammar Rule')
            description = doc_text
        
        # Extract examples if present
        examples = []
        if 'Examples:' in description:
            desc_part, examples_part = description.split('Examples:', 1)
            description = desc_part.strip()
            if 'Keywords:' in examples_part:
                examples_part, _ = examples_part.split('Keywords:', 1)
            examples = [ex.strip() for ex in examples_part.split(';') if ex.strip()]
        
        # Format comprehensive rule text
        rule_text = f"**{rule_name}**: {description}"
        if examples:
            rule_text += f"\n\n**Examples**: {'; '.join(examples)}"
        
        # Add metadata if available
        category = metadata.get('category', '')
        severity = metadata.get('severity', '')
        if category or severity:
            rule_text += f"\n\n**Category**: {category.title()}"
            if severity:
                rule_text += f" | **Severity**: {severity.title()}"
        
        return {
            "rule": rule_text,
            "source": "ChromaDB Grammar Database",
            "confidence": "HIGH",
            "query_matched": query,
            "relevance_score": best_result.get('distance', 0),
            "category": category,
            "severity": severity,
            "rule_id": metadata.get('rule_id', ''),
            "total_results": len(results)
        }
        
    except Exception as e:
        return {
            "rule": f"Grammar RAG system error: {str(e)}",
            "source": "Error",
            "confidence": "LOW",
            "query_matched": query,
            "error": str(e)
        }


def web_search_tool(query: str, backend: str, api_token: str) -> Dict[str, Any]:
    """
    Web search tool for domain knowledge using openai_direct_gpt4o_mini_search_preview.
    """
    search_backend = "openai_direct_gpt4o_mini_search_preview"
    
    search_prompt = f"""Search for information about: {query}

Focus on:
- Grammar rules and conventions
- Language usage guidelines  
- Domain-specific writing standards
- Authoritative linguistic resources

Provide a concise summary of relevant findings."""
    
    ok, response, _ = call_model(search_prompt, search_backend, api_token, no_temperature=True)
    if not ok:
        return {
            "results": [],
            "summary": f"Search failed for: {query}",
            "confidence": "LOW"
        }
    
    return {
        "results": [response.strip()],
        "summary": f"Web search results for: {query}",
        "confidence": "HIGH",
        "raw_response": response.strip()
    }


def comprehensive_analysis_tool(original: str, suggested: str, language: str, backend: str, api_token: str, prompt_template: str) -> Dict[str, Any]:
    """
    Comprehensive analysis tool using TNFN model call.
    Uses judge-specific prompt template for TNFN classification.
    """
    # For comprehensive analysis, we use TNFN classification on the original text
    # This helps determine if the text needs correction at all
    try:
        from judges.tnfn.prompts import TNFN_PROMPT
        from utils.judge import build_numbered_prompt, parse_tnfn_label, get_language_label
        
        language_label = get_language_label(language)
        tnfn_prompt = build_numbered_prompt(TNFN_PROMPT, language_label, original, "")
        
        ok, response, _ = call_model(tnfn_prompt, backend, api_token)
        if not ok:
            return {"classification": "ERROR", "reason": "TNFN call failed", "confidence": "LOW"}
        
        tnfn_label = parse_tnfn_label(response)
        
        # If TN (no correction needed), and we have a suggestion different from original,
        # this suggests the correction might be unnecessary (FP3 or worse)
        # If FN (correction needed), and we have a good suggestion, this supports TP
        if original == suggested:
            classification = "TN" if tnfn_label == "TN" else "FN"
        else:
            # We have a correction suggestion
            if tnfn_label == "TN":
                # Original was correct, but we're suggesting changes -> likely FP
                classification = "FP3"  # Assume minor unless proven otherwise
            else:
                # Original needed correction, and we have a suggestion -> likely TP
                classification = "TP"   # Assume good unless proven otherwise
        
        return {
            "classification": classification,
            "reason": f"TNFN analysis: {tnfn_label}. {response.strip()}",
            "tnfn_result": tnfn_label,
            "confidence": "HIGH"
        }
        
    except Exception as e:
        # Fallback to using the provided prompt template for TPFP analysis
        prompt = prompt_template.format(original=original, suggested=suggested, language=language)
    
    ok, response, _tokens = call_model(prompt, backend, api_token)
    if not ok:
        return {"classification": "ERROR", "reason": "Tool call failed", "confidence": "LOW"}
    
    try:
        # Try to parse JSON response
        js_text = response.strip()
        if js_text.startswith('```json'):
            js_text = js_text[js_text.find('{'): js_text.rfind('}')+1]
        elif js_text.startswith('```'):
            lines = js_text.split('\n')
            js_text = '\n'.join([l for l in lines if not l.startswith('```')])
        
        data = json.loads(js_text)
        classification = data.get('classification', '').strip().upper()
        
        return {
            "classification": classification if classification in ['TP', 'FP1', 'FP2', 'FP3'] else "ERROR",
            "reason": data.get('reason', ''),
            "type_of_writing": data.get('type_of_writing', ''),
            "confidence": "HIGH"
        }
    except Exception:
        # Fallback regex parsing
        match = re.search(r'"classification"\s*:\s*"(TP|FP[123])"', response, re.IGNORECASE)
        if match:
            return {
                "classification": match.group(1).upper(),
                "reason": response.strip(),
                "type_of_writing": "",
                "confidence": "MEDIUM"
            }
        
        return {
            "classification": "ERROR",
            "reason": response.strip(),
            "type_of_writing": "",
            "confidence": "LOW"
        }


# Tool registry for agent
AVAILABLE_TOOLS = {
    "grammar_rag": grammar_rag_tool,
}
