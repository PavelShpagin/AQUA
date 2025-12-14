#!/usr/bin/env python3
"""
Enhanced agent tools with advanced chunked RAG and direct rule querying.
"""

import os
import re
import json
from typing import Dict, Any, Optional, List
from utils.rag.advanced_chunked_rag import AdvancedChunkedRAG

def enhanced_grammar_rag_tool(query: str, language: str, backend: str, api_token: str) -> Dict[str, Any]:
    """
    Enhanced Grammar RAG tool with chunked rulebooks and intelligent querying.
    
    Args:
        query: Specific grammar rule query or error description
        language: Target language for grammar rules
        backend: LLM backend (unused, kept for compatibility)
        api_token: API token (unused, kept for compatibility)
    
    Returns:
        Dict with comprehensive grammar rule information
    """
    try:
        # Initialize chunked RAG system
        rag = AdvancedChunkedRAG(language=language.lower())
        
        # Use smart query to get comprehensive information
        result = rag.smart_query(query)
        
        if "error" in result:
            return {
                "rule": f"No grammar rules found for query: '{query}' in {language}",
                "source": "Enhanced Chunked Grammar Database",
                "confidence": "LOW",
                "query_matched": query,
                "error": result["error"]
            }
        
        return {
            "rule": result["rule"],
            "source": "Enhanced Chunked Grammar Database", 
            "confidence": result["confidence"],
            "query_matched": query,
            "relevance_score": result.get("relevance_score", 0),
            "rule_name": result.get("rule_name", ""),
            "chunk_type": result.get("chunk_type", ""),
            "total_chunks": result.get("total_chunks", 0),
            "query_type": "enhanced_chunked"
        }
        
    except Exception as e:
        return {
            "rule": f"Enhanced Grammar RAG system error: {str(e)}",
            "source": "Error",
            "confidence": "LOW", 
            "query_matched": query,
            "error": str(e)
        }

def direct_rule_query_tool(rule_name: str, language: str, backend: str, api_token: str) -> Dict[str, Any]:
    """
    Direct rule query tool - get comprehensive information about a specific rule.
    
    Args:
        rule_name: Exact name of the grammar rule
        language: Target language
        backend: LLM backend (unused)
        api_token: API token (unused)
    
    Returns:
        Dict with complete rule information including definition, examples, errors
    """
    try:
        # Initialize chunked RAG system
        rag = AdvancedChunkedRAG(language=language.lower())
        
        # Get comprehensive rule information
        rule_info = rag.get_comprehensive_rule_info(rule_name)
        
        if "error" in rule_info:
            return {
                "rule": f"Rule '{rule_name}' not found in {language} grammar database",
                "source": "Direct Rule Query",
                "confidence": "LOW",
                "query_matched": rule_name,
                "error": rule_info["error"]
            }
        
        # Format comprehensive response
        rule_text = f"**{rule_info['rule_name']}**"
        
        if "definition" in rule_info:
            rule_text += f"\n\n**Definition**: {rule_info['definition'][0]}"
        
        if "examples" in rule_info:
            examples = "; ".join(rule_info['examples'])
            rule_text += f"\n\n**Correct Examples**: {examples}"
        
        if "counter_examples" in rule_info:
            errors = "; ".join(rule_info['counter_examples'])
            rule_text += f"\n\n**Common Errors**: {errors}"
        
        if "usage" in rule_info:
            usage = "; ".join(rule_info['usage'])
            rule_text += f"\n\n**Usage Notes**: {usage}"
        
        return {
            "rule": rule_text,
            "source": "Direct Rule Query Database",
            "confidence": "HIGH",
            "query_matched": rule_name,
            "rule_name": rule_info['rule_name'],
            "total_chunks": rule_info['total_chunks'],
            "language": rule_info['language'],
            "query_type": "direct_rule"
        }
        
    except Exception as e:
        return {
            "rule": f"Direct rule query error: {str(e)}",
            "source": "Error",
            "confidence": "LOW",
            "query_matched": rule_name,
            "error": str(e)
        }

def category_search_tool(category: str, language: str, backend: str, api_token: str) -> Dict[str, Any]:
    """
    Category search tool - find all rules in a specific grammar category.
    
    Args:
        category: Grammar category (e.g., "grammar", "orthography", "punctuation")
        language: Target language
        backend: LLM backend (unused)
        api_token: API token (unused)
    
    Returns:
        Dict with rules from the specified category
    """
    try:
        # Initialize chunked RAG system
        rag = AdvancedChunkedRAG(language=language.lower())
        
        # Search by category
        results = rag.search_by_category(category)
        
        if not results:
            return {
                "rule": f"No rules found in category '{category}' for {language}",
                "source": "Category Search",
                "confidence": "LOW",
                "query_matched": category,
                "error": "No rules in category"
            }
        
        # Group results by rule name
        rules_by_name = {}
        for result in results:
            rule_name = result['metadata']['rule_name']
            if rule_name not in rules_by_name:
                rules_by_name[rule_name] = []
            rules_by_name[rule_name].append(result)
        
        # Format response
        rule_text = f"**{category.title()} Rules in {language.title()}**\n\n"
        
        for rule_name, rule_chunks in rules_by_name.items():
            rule_text += f"• **{rule_name}**: "
            
            # Find definition chunk
            definition_chunk = next((c for c in rule_chunks if c['metadata']['chunk_type'] == 'definition'), None)
            if definition_chunk:
                # Extract first sentence of definition
                definition = definition_chunk['document'].split('.')[0] + '.'
                rule_text += definition
            else:
                rule_text += "Grammar rule"
            
            rule_text += "\n"
        
        return {
            "rule": rule_text,
            "source": "Category Search Database",
            "confidence": "HIGH",
            "query_matched": category,
            "total_rules": len(rules_by_name),
            "total_chunks": len(results),
            "query_type": "category_search"
        }
        
    except Exception as e:
        return {
            "rule": f"Category search error: {str(e)}",
            "source": "Error",
            "confidence": "LOW",
            "query_matched": category,
            "error": str(e)
        }

def error_focused_search_tool(error_description: str, language: str, backend: str, api_token: str) -> Dict[str, Any]:
    """
    Error-focused search tool - find rules based on error descriptions.
    
    Args:
        error_description: Description of the grammatical error
        language: Target language
        backend: LLM backend (unused)
        api_token: API token (unused)
    
    Returns:
        Dict with rules relevant to the error type
    """
    try:
        # Initialize chunked RAG system
        rag = AdvancedChunkedRAG(language=language.lower())
        
        # Focus search on counter-examples and definitions
        results = rag.search_by_query(
            error_description, 
            chunk_types=["counter_examples", "definition", "examples"]
        )
        
        if not results:
            return {
                "rule": f"No rules found for error: '{error_description}' in {language}",
                "source": "Error-Focused Search",
                "confidence": "LOW",
                "query_matched": error_description,
                "error": "No matching error patterns"
            }
        
        # Get the most relevant rule
        best_result = results[0]
        rule_name = best_result['metadata']['rule_name']
        
        # Get comprehensive information about this rule
        rule_info = rag.get_comprehensive_rule_info(rule_name)
        
        # Format response focusing on error correction
        rule_text = f"**Error Type**: {error_description}\n\n"
        rule_text += f"**Relevant Rule**: {rule_name}\n\n"
        
        if "definition" in rule_info:
            rule_text += f"**Rule**: {rule_info['definition'][0]}\n\n"
        
        if "counter_examples" in rule_info:
            errors = "; ".join(rule_info['counter_examples'])
            rule_text += f"**Common Errors**: {errors}\n\n"
        
        if "examples" in rule_info:
            examples = "; ".join(rule_info['examples'])
            rule_text += f"**Correct Forms**: {examples}"
        
        return {
            "rule": rule_text,
            "source": "Error-Focused Search Database",
            "confidence": "HIGH" if best_result['relevance_score'] > 0.7 else "MEDIUM",
            "query_matched": error_description,
            "relevance_score": best_result['relevance_score'],
            "rule_name": rule_name,
            "query_type": "error_focused"
        }
        
    except Exception as e:
        return {
            "rule": f"Error-focused search error: {str(e)}",
            "source": "Error",
            "confidence": "LOW",
            "query_matched": error_description,
            "error": str(e)
        }

# Enhanced tool registry with multiple specialized tools
ENHANCED_TOOLS = {
    "enhanced_grammar_rag": enhanced_grammar_rag_tool,
    "direct_rule_query": direct_rule_query_tool,
    "category_search": category_search_tool,
    "error_focused_search": error_focused_search_tool
}

# For backward compatibility, use enhanced tool as default
AVAILABLE_TOOLS = {
    "grammar_rag": enhanced_grammar_rag_tool
}



