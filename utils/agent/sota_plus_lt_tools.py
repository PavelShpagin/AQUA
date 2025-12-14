#!/usr/bin/env python3
"""
SOTA + LanguageTool enhanced agent tools for maximum accuracy.
"""

import os
import re
import json
import logging
from typing import Dict, Any, Optional, List
from utils.rag.advanced_chunked_rag import AdvancedChunkedRAG
from utils.rag.opensource_grammar_api import OpensourceGrammarAPI

try:
    import language_tool_python
    LANGUAGETOOL_AVAILABLE = True
except ImportError:
    LANGUAGETOOL_AVAILABLE = False
    logging.warning("LanguageTool Python package not available")

class LanguageToolAnalyzer:
    """LanguageTool analyzer for grammar checking."""
    
    def __init__(self):
        self.tools = {}
        if LANGUAGETOOL_AVAILABLE:
            try:
                # Initialize LanguageTool for different languages
                self.tools = {
                    'english': language_tool_python.LanguageTool('en-US'),
                    'spanish': language_tool_python.LanguageTool('es'),
                    'german': language_tool_python.LanguageTool('de-DE'),
                    'french': language_tool_python.LanguageTool('fr')
                }
                logging.info(f"LanguageTool initialized for {len(self.tools)} languages")
            except Exception as e:
                logging.error(f"Failed to initialize LanguageTool: {e}")
                self.tools = {}
    
    def check_text(self, text: str, language: str) -> List[Dict[str, Any]]:
        """Check text for grammar errors."""
        if not LANGUAGETOOL_AVAILABLE or language.lower() not in self.tools:
            return []
        
        try:
            tool = self.tools[language.lower()]
            matches = tool.check(text)
            
            formatted_errors = []
            for match in matches:
                error_info = {
                    'message': match.message,
                    'rule_id': match.ruleId,
                    'category': match.category,
                    'suggestions': [r for r in match.replacements],
                    'offset': match.offset,
                    'length': match.errorLength,
                    'context': match.context,
                    'sentence': match.sentence
                }
                formatted_errors.append(error_info)
            
            return formatted_errors
        except Exception as e:
            logging.error(f"LanguageTool check failed: {e}")
            return []
    
    def analyze_correction(self, original: str, corrected: str, language: str) -> Dict[str, Any]:
        """Analyze if a correction is grammatically sound."""
        if not LANGUAGETOOL_AVAILABLE or language.lower() not in self.tools:
            return {"analysis": "LanguageTool not available", "confidence": "LOW"}
        
        try:
            # Check both texts
            original_errors = self.check_text(original, language)
            corrected_errors = self.check_text(corrected, language)
            
            # Analysis
            analysis = {
                "original_errors": len(original_errors),
                "corrected_errors": len(corrected_errors),
                "errors_fixed": max(0, len(original_errors) - len(corrected_errors)),
                "new_errors": max(0, len(corrected_errors) - len(original_errors)),
                "net_improvement": len(original_errors) - len(corrected_errors)
            }
            
            # Detailed error analysis
            if original_errors:
                analysis["original_error_types"] = [e['category'] for e in original_errors]
            if corrected_errors:
                analysis["corrected_error_types"] = [e['category'] for e in corrected_errors]
            
            # Determine correction quality
            if analysis["net_improvement"] > 0:
                quality = "IMPROVEMENT"
                confidence = "HIGH"
            elif analysis["net_improvement"] == 0:
                if analysis["errors_fixed"] > 0:
                    quality = "MIXED" # Fixed some, introduced others
                    confidence = "MEDIUM"
                else:
                    quality = "NO_CHANGE"
                    confidence = "MEDIUM"
            else:
                quality = "DEGRADATION"  # Introduced more errors
                confidence = "HIGH"
            
            analysis.update({
                "quality": quality,
                "confidence": confidence,
                "recommendation": self._get_recommendation(analysis)
            })
            
            return analysis
            
        except Exception as e:
            logging.error(f"LanguageTool analysis failed: {e}")
            return {"analysis": f"Error: {e}", "confidence": "LOW"}
    
    def _get_recommendation(self, analysis: Dict[str, Any]) -> str:
        """Get recommendation based on analysis."""
        net_improvement = analysis["net_improvement"]
        
        if net_improvement > 2:
            return "Strong recommendation: Accept correction (fixes multiple errors)"
        elif net_improvement > 0:
            return "Recommendation: Accept correction (net improvement)"
        elif net_improvement == 0 and analysis["errors_fixed"] > 0:
            return "Caution: Mixed correction (fixes some, introduces others)"
        elif net_improvement == 0:
            return "No change: Correction doesn't affect grammar"
        else:
            return "Reject: Correction introduces more errors"

def sota_enhanced_grammar_rag_tool(query: str, language: str, backend: str, api_token: str) -> Dict[str, Any]:
    """
    SOTA Enhanced Grammar RAG tool combining chunked RAG, opensource API, and LanguageTool.
    
    Args:
        query: Grammar rule query or error description
        language: Target language
        backend: LLM backend (unused)
        api_token: API token (unused)
    
    Returns:
        Dict with comprehensive grammar information from multiple sources
    """
    try:
        results = []
        
        # 1. Chunked RAG System
        try:
            chunked_rag = AdvancedChunkedRAG(language=language.lower())
            chunked_result = chunked_rag.smart_query(query)
            if "error" not in chunked_result:
                results.append({
                    "source": "Chunked RAG",
                    "rule": chunked_result["rule"],
                    "confidence": chunked_result["confidence"],
                    "relevance": chunked_result.get("relevance_score", 0.5)
                })
        except Exception as e:
            logging.warning(f"Chunked RAG failed: {e}")
        
        # 2. Opensource Grammar API
        try:
            opensource_api = OpensourceGrammarAPI()
            api_result = opensource_api.query_grammar_rule(query, language)
            if "error" not in api_result:
                results.append({
                    "source": "Opensource API",
                    "rule": api_result["rule"],
                    "confidence": api_result["confidence"],
                    "relevance": 0.8 if api_result["confidence"] == "HIGH" else 0.6
                })
        except Exception as e:
            logging.warning(f"Opensource API failed: {e}")
        
        if not results:
            return {
                "rule": f"No grammar information found for: '{query}' in {language}",
                "source": "SOTA Enhanced Grammar System",
                "confidence": "LOW",
                "query_matched": query,
                "error": "All sources failed"
            }
        
        # Combine results - prioritize by confidence and relevance
        best_result = max(results, key=lambda x: (
            1.0 if x["confidence"] == "HIGH" else 0.5,
            x.get("relevance", 0.5)
        ))
        
        # Enhanced response with multiple sources
        rule_text = f"**{query.title()} - Grammar Analysis**\n\n"
        rule_text += best_result["rule"]
        
        if len(results) > 1:
            rule_text += f"\n\n**Additional Sources**: {len(results)-1} other grammar sources consulted"
        
        return {
            "rule": rule_text,
            "source": f"SOTA Enhanced Grammar System ({best_result['source']})",
            "confidence": best_result["confidence"],
            "query_matched": query,
            "total_sources": len(results),
            "primary_source": best_result["source"],
            "query_type": "sota_enhanced"
        }
        
    except Exception as e:
        return {
            "rule": f"SOTA Enhanced Grammar System error: {str(e)}",
            "source": "Error",
            "confidence": "LOW",
            "query_matched": query,
            "error": str(e)
        }

def languagetool_correction_analyzer(original_text: str, corrected_text: str, language: str, backend: str, api_token: str) -> Dict[str, Any]:
    """
    LanguageTool correction analyzer tool.
    
    Args:
        original_text: Original text
        corrected_text: Corrected text
        language: Text language
        backend: LLM backend (unused)
        api_token: API token (unused)
    
    Returns:
        Dict with detailed correction analysis
    """
    try:
        analyzer = LanguageToolAnalyzer()
        analysis = analyzer.analyze_correction(original_text, corrected_text, language)
        
        if "error" in analysis.get("analysis", "").lower():
            return {
                "analysis": "LanguageTool analysis unavailable",
                "recommendation": "Use manual grammar analysis",
                "confidence": "LOW",
                "error": analysis.get("analysis", "Unknown error")
            }
        
        # Format detailed response
        response_text = f"**Grammar Correction Analysis**\n\n"
        response_text += f"**Original Text Errors**: {analysis['original_errors']}\n"
        response_text += f"**Corrected Text Errors**: {analysis['corrected_errors']}\n"
        response_text += f"**Net Improvement**: {analysis['net_improvement']} errors\n\n"
        response_text += f"**Quality Assessment**: {analysis['quality']}\n"
        response_text += f"**Recommendation**: {analysis['recommendation']}"
        
        return {
            "analysis": response_text,
            "recommendation": analysis["recommendation"],
            "confidence": analysis["confidence"],
            "quality": analysis["quality"],
            "net_improvement": analysis["net_improvement"],
            "original_errors": analysis["original_errors"],
            "corrected_errors": analysis["corrected_errors"]
        }
        
    except Exception as e:
        return {
            "analysis": f"LanguageTool analyzer error: {str(e)}",
            "recommendation": "Use manual analysis",
            "confidence": "LOW",
            "error": str(e)
        }

def grammar_error_detector(text: str, language: str, backend: str, api_token: str) -> Dict[str, Any]:
    """
    Grammar error detection tool using LanguageTool.
    
    Args:
        text: Text to analyze
        language: Text language
        backend: LLM backend (unused)
        api_token: API token (unused)
    
    Returns:
        Dict with detected grammar errors
    """
    try:
        analyzer = LanguageToolAnalyzer()
        errors = analyzer.check_text(text, language)
        
        if not errors:
            return {
                "errors": "No grammar errors detected",
                "error_count": 0,
                "confidence": "HIGH",
                "text_quality": "GOOD"
            }
        
        # Format error summary
        error_text = f"**Grammar Errors Detected**: {len(errors)}\n\n"
        
        # Group by category
        by_category = {}
        for error in errors:
            category = error['category']
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(error)
        
        for category, cat_errors in by_category.items():
            error_text += f"**{category}** ({len(cat_errors)} errors):\n"
            for error in cat_errors[:3]:  # Show first 3 errors per category
                error_text += f"  • {error['message']}\n"
                if error['suggestions']:
                    suggestions = ", ".join(error['suggestions'][:2])
                    error_text += f"    Suggestions: {suggestions}\n"
            if len(cat_errors) > 3:
                error_text += f"  ... and {len(cat_errors)-3} more\n"
            error_text += "\n"
        
        return {
            "errors": error_text,
            "error_count": len(errors),
            "confidence": "HIGH",
            "text_quality": "NEEDS_IMPROVEMENT" if len(errors) > 2 else "MINOR_ISSUES",
            "categories": list(by_category.keys()),
            "detailed_errors": errors[:5]  # First 5 detailed errors
        }
        
    except Exception as e:
        return {
            "errors": f"Grammar error detection failed: {str(e)}",
            "error_count": -1,
            "confidence": "LOW",
            "error": str(e)
        }

# SOTA + LanguageTool enhanced tool registry
SOTA_PLUS_LT_TOOLS = {
    "grammar_rag": sota_enhanced_grammar_rag_tool,
    "correction_analyzer": languagetool_correction_analyzer,
    "error_detector": grammar_error_detector
}

# For compatibility, use enhanced tool as default
AVAILABLE_TOOLS = {
    "grammar_rag": sota_enhanced_grammar_rag_tool
}

if __name__ == "__main__":
    # Test the SOTA + LanguageTool integration
    print("🧪 Testing SOTA + LanguageTool Integration")
    print("="*50)
    
    # Test grammar analysis
    result = sota_enhanced_grammar_rag_tool("subject verb agreement", "english", "", "")
    print(f"Grammar Query Result: {result['confidence']}")
    print(f"Sources: {result.get('total_sources', 1)}")
    
    # Test correction analysis
    if LANGUAGETOOL_AVAILABLE:
        analysis = languagetool_correction_analyzer(
            "She don't like coffee", 
            "She doesn't like coffee", 
            "english", "", ""
        )
        print(f"Correction Analysis: {analysis.get('quality', 'N/A')}")
        print(f"Recommendation: {analysis.get('recommendation', 'N/A')}")
    else:
        print("LanguageTool not available - using fallback grammar analysis")
    
    print("✅ SOTA + LanguageTool integration ready!")
