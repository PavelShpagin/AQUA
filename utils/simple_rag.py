#!/usr/bin/env python3
"""
Simple RAG system for Spanish grammar rules without external dependencies.
"""

import json
import os
from typing import List, Dict, Any, Optional


def load_spanish_rules() -> Dict[str, Any]:
    """Load Spanish grammar rules from JSON file."""
    rules_file = "data/rag/spanish_simple_rules.json"
    if not os.path.exists(rules_file):
        return {"rules": []}
    
    try:
        with open(rules_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {"rules": []}


def query_spanish_rules(edit_text: str, original: str = "", corrected: str = "") -> List[Dict[str, Any]]:
    """Query Spanish grammar rules for relevant patterns."""
    rules_data = load_spanish_rules()
    rules = rules_data.get("rules", [])
    
    relevant_rules = []
    query_text = f"{edit_text} {original} {corrected}".lower()
    
    for rule in rules:
        # Check if any patterns match
        patterns = rule.get("patterns", [])
        examples = rule.get("examples", [])
        
        # Simple pattern matching
        rule_relevant = False
        
        # Check patterns
        for pattern in patterns:
            if "->" in pattern:
                orig_part, corr_part = pattern.split("->", 1)
                if orig_part.lower() in query_text or corr_part.lower() in query_text:
                    rule_relevant = True
                    break
        
        # Check examples
        if not rule_relevant:
            for example in examples:
                if any(word.lower() in query_text for word in example.split()):
                    rule_relevant = True
                    break
        
        # Check category keywords
        if not rule_relevant:
            category_keywords = {
                "accent_marks": ["accent", "más", "sí", "tú", "él", "está", "cómo"],
                "gender_agreement": ["gender", "la", "el", "una", "un"],
                "number_agreement": ["number", "plural", "singular", "estudiantes", "días"],
                "contractions": ["contraction", "al", "del", "a el", "de el"],
                "verb_conjugation": ["verb", "conjugation", "hablo", "hablas", "habla"],
                "stylistic_synonyms": ["email", "correo", "app", "aplicación", "link", "enlace"],
                "critical_meaning": ["meaning", "no", "sí", "mañana", "noche"]
            }
            
            category = rule.get("category", "")
            keywords = category_keywords.get(category, [])
            if any(keyword in query_text for keyword in keywords):
                rule_relevant = True
        
        if rule_relevant:
            relevant_rules.append(rule)
    
    return relevant_rules


def get_rule_classification_hint(edit_text: str, original: str = "", corrected: str = "") -> Dict[str, Any]:
    """Get classification hint based on Spanish grammar rules."""
    rules = query_spanish_rules(edit_text, original, corrected)
    
    if not rules:
        return {"classification": None, "confidence": 0.0, "reasoning": "No matching rules"}
    
    # Analyze rule categories to suggest classification
    categories = [rule.get("category", "") for rule in rules]
    
    if "critical_meaning" in categories:
        return {
            "classification": "FP1",
            "confidence": 0.9,
            "reasoning": f"Critical meaning change detected. Matching rules: {[r['id'] for r in rules]}"
        }
    
    if any(cat in categories for cat in ["accent_marks", "gender_agreement", "number_agreement", "contractions", "verb_conjugation"]):
        return {
            "classification": "TP",
            "confidence": 0.8,
            "reasoning": f"Grammar correction detected. Matching rules: {[r['id'] for r in rules]}"
        }
    
    if "stylistic_synonyms" in categories:
        return {
            "classification": "FP3",
            "confidence": 0.7,
            "reasoning": f"Stylistic change detected. Matching rules: {[r['id'] for r in rules]}"
        }
    
    return {
        "classification": "TP",
        "confidence": 0.3,
        "reasoning": f"Uncertain classification. Matching rules: {[r['id'] for r in rules]}"
    }


def format_rules_for_prompt(rules: List[Dict[str, Any]]) -> str:
    """Format rules for inclusion in LLM prompt."""
    if not rules:
        return "No specific Spanish grammar rules found for this edit."
    
    formatted = "**Relevant Spanish Grammar Rules:**\n"
    for rule in rules[:3]:  # Limit to top 3 rules
        formatted += f"- **{rule.get('category', 'Unknown').replace('_', ' ').title()}** ({rule.get('id', '')}): {rule.get('description', '')}\n"
        if rule.get('examples'):
            formatted += f"  Examples: {', '.join(rule['examples'][:3])}\n"
    
    return formatted

