#!/usr/bin/env python3
"""
Ultimate toolkit combining all the most effective tools for maximum accuracy.
Incrementally tested to ensure each tool adds value.
"""

import os
import sys
import json
import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from utils.llm.backends import call_model

# Import existing tools
try:
    from utils.rag.advanced_chunked_rag import AdvancedChunkedRAG
except ImportError:
    AdvancedChunkedRAG = None

try:
    from utils.rag.opensource_grammar_api import OpensourceGrammarAPI
except ImportError:
    OpensourceGrammarAPI = None

try:
    import language_tool_python
    LANGUAGETOOL_AVAILABLE = True
except ImportError:
    LANGUAGETOOL_AVAILABLE = False

def ultimate_chunked_grammar_rag_tool(query: str, language: str = "english") -> Dict[str, Any]:
    """
    Enhanced chunked RAG tool with improved Spanish support and multilingual queries.
    """
    try:
        if not AdvancedChunkedRAG:
            return {"error": "AdvancedChunkedRAG not available", "success": False}
        
        # Map language codes
        lang_map = {"en": "english", "es": "spanish", "de": "german", "fr": "french"}
        rag_language = lang_map.get(language.lower()[:2], language.lower())
        
        # Initialize RAG system
        rag = AdvancedChunkedRAG(language=rag_language)
        
        # Enhanced query processing for Spanish
        if rag_language == "spanish":
            # Add Spanish-specific query enhancements
            spanish_keywords = {
                "agreement": "concordancia",
                "gender": "género", 
                "number": "número",
                "verb": "verbo",
                "tense": "tiempo",
                "subjunctive": "subjuntivo",
                "accent": "acentuación",
                "spelling": "ortografía"
            }
            
            # Enhance query with Spanish terms
            enhanced_query = query
            for en_term, es_term in spanish_keywords.items():
                if en_term in query.lower():
                    enhanced_query += f" {es_term}"
        else:
            enhanced_query = query
        
        # Perform smart query across all chunk types
        result = rag.smart_query(enhanced_query)
        
        if not result or "error" in result:
            return {
                "rules": "No relevant grammar rules found for this query.",
                "success": False,
                "query": enhanced_query,
                "language": rag_language
            }
        
        # Extract rule information from result
        rule_info = result.get("rule", "No rule information available")
        
        return {
            "rules": rule_info,
            "success": True,
            "query": enhanced_query,
            "language": rag_language,
            "confidence": result.get("confidence", "MEDIUM")
        }
        
    except Exception as e:
        return {
            "error": f"Grammar RAG search failed: {str(e)}",
            "success": False,
            "query": query,
            "language": language
        }

def enhanced_meaning_change_tool(original: str, suggested: str, language: str = "english", backend: str = "gpt-4o-mini") -> Dict[str, Any]:
    """
    Enhanced meaning change detector with multilingual support and detailed analysis.
    """
    try:
        # Language-specific prompts for better accuracy
        if language.lower().startswith("es"):
            prompt = f"""Eres un detector de cambios de significado para evaluación de corrección gramatical. Compara las oraciones original y sugerida y califica la severidad del cambio de significado.

NIVELES DE SEVERIDAD:
0 - Sin cambio de significado (solo correcciones gramaticales)
1 - Cambio menor de clarificación o estilo (mismo significado central)
2 - Cambio notable pero aceptable (ligero cambio en énfasis/matiz)
3 - Alteración significativa del significado (información importante cambiada)
4 - Cambio mayor de significado o contradicción (significado completamente diferente)

ORIGINAL:
{original}

SUGERIDA:
{suggested}

Formato de respuesta:
SEVERIDAD: [0-4]
RAZONAMIENTO: [explica el nivel de cambio de significado]"""
        else:
            prompt = f"""You are a meaning-change detector for GEC evaluation. Compare the original and suggested sentences and rate the meaning change severity.

SEVERITY LEVELS:
0 - No meaning change (grammatical corrections only)
1 - Minor clarification or style change (same core meaning)
2 - Noticeable but acceptable change (slight shift in emphasis/nuance)
3 - Significant meaning alteration (important information changed)
4 - Major meaning change or contradiction (completely different meaning)

ORIGINAL:
{original}

SUGGESTED:
{suggested}

Return format:
SEVERITY: [0-4]
REASONING: [explain the meaning change level]"""
        
        # Use smaller model for efficiency
        api_key = os.getenv('OPENAI_API_KEY', '')
        success, response, tokens = call_model(prompt, backend, api_key)
        
        if not success or not response:
            return {"meaning_change_score": 0, "reasoning": "Analysis failed", "success": False}
        
        # Parse response
        severity_match = re.search(r'SEVERIDAD?:\s*([0-4])', response, re.IGNORECASE)
        reasoning_match = re.search(r'RAZONAMIENTO|REASONING:\s*(.+)', response, re.IGNORECASE | re.DOTALL)
        
        severity = int(severity_match.group(1)) if severity_match else 0
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"
        
        return {
            "meaning_change_score": severity,
            "reasoning": reasoning,
            "success": True,
            "original": original,
            "suggested": suggested
        }
        
    except Exception as e:
        return {
            "meaning_change_score": 0,
            "reasoning": f"Error in meaning change analysis: {str(e)}",
            "success": False
        }

def enhanced_quality_reward_tool(original: str, suggested: str, language: str = "english", backend: str = "gpt-4o-mini") -> Dict[str, Any]:
    """
    Enhanced quality/reward assessment with multilingual support and detailed scoring.
    """
    try:
        # Language-specific prompts
        if language.lower().startswith("es"):
            prompt = f"""Eres un evaluador de calidad de texto para corrección gramatical. Compara las oraciones original y sugerida y califica la mejora relativa.

ORIGINAL:
{original}

SUGERIDA:
{suggested}

Califica la mejora relativa en una escala de -3 a +3:
-3: Mucho peor (introduce errores mayores, degrada significativamente la calidad)
-2: Peor (introduce errores notables, degrada la calidad)
-1: Ligeramente peor (degradación menor)
 0: Sin cambio en calidad
+1: Ligeramente mejor (mejora menor)
+2: Mejor (corrige errores, mejora la claridad)
+3: Mucho mejor (mejora significativa, corrección de errores mayores)

Formato de respuesta:
MEJORA: [-3 a +3]
RAZONAMIENTO: [explica la evaluación de mejora relativa]"""
        else:
            prompt = f"""You are a text quality assessor for GEC evaluation. Compare the original and suggested sentences and rate the relative improvement.

ORIGINAL:
{original}

SUGGESTED:
{suggested}

Rate the relative improvement on a scale from -3 to +3:
-3: Much worse (introduces major errors, significantly degrades quality)
-2: Worse (introduces noticeable errors, degrades quality)
-1: Slightly worse (minor degradation)
 0: No change in quality
+1: Slightly better (minor improvement)
+2: Better (fixes errors, improves clarity)
+3: Much better (significant improvement, major error fixes)

Return format:
IMPROVEMENT: [-3 to +3]
REASONING: [explain the relative improvement assessment]"""
        
        api_key = os.getenv('OPENAI_API_KEY', '')
        success, response, tokens = call_model(prompt, backend, api_key)
        
        if not success or not response:
            return {"quality_score": 0, "reasoning": "Analysis failed", "success": False}
        
        # Parse response
        improvement_match = re.search(r'MEJORA|IMPROVEMENT:\s*([-+]?[0-3])', response, re.IGNORECASE)
        reasoning_match = re.search(r'RAZONAMIENTO|REASONING:\s*(.+)', response, re.IGNORECASE | re.DOTALL)
        
        improvement = int(improvement_match.group(1)) if improvement_match else 0
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"
        
        return {
            "quality_score": improvement,
            "reasoning": reasoning,
            "success": True,
            "original": original,
            "suggested": suggested
        }
        
    except Exception as e:
        return {
            "quality_score": 0,
            "reasoning": f"Error in quality analysis: {str(e)}",
            "success": False
        }

def enhanced_nonsense_detector_tool(text: str, language: str = "english", backend: str = "gpt-4o-mini") -> Dict[str, Any]:
    """
    Enhanced nonsense detector with multilingual support and detailed scoring.
    """
    try:
        # Language-specific prompts
        if language.lower().startswith("es"):
            prompt = f"""Eres un detector estricto de sinsentidos para evaluación de corrección gramatical. Determina si el texto sugerido introduce sinsentidos, incoherencia estructural, o hace el texto ininterpretable.

CRITERIOS PARA PUNTUACIÓN:
-1: Reduce sinsentidos (mejora la coherencia)
 0: Neutral (sin cambio en coherencia)
 1: Sinsentido leve (pequeña incoherencia)
 2: Sinsentido medio (incoherencia notable)
 3: Sinsentido mayor/pérdida de información/ruptura sintáctica

Texto a evaluar:
{text}

Formato de respuesta:
PUNTUACIÓN: [-1 a 3]
RAZONAMIENTO: [explica la evaluación de sinsentido]"""
        else:
            prompt = f"""You are a strict nonsense detector for GEC evaluation. Determine if the suggested text introduces nonsense, structural incoherence, or makes the text uninterpretable.

SCORING CRITERIA:
-1: Reduces nonsense (improves coherence)
 0: Neutral (no change in coherence)
 1: Slight nonsense (minor incoherence)
 2: Medium nonsense (noticeable incoherence)
 3: Major nonsense/loss of information/syntax breaking

Text to evaluate:
{text}

Return format:
SCORE: [-1 to 3]
REASONING: [explain the nonsense evaluation]"""
        
        api_key = os.getenv('OPENAI_API_KEY', '')
        success, response, tokens = call_model(prompt, backend, api_key)
        
        if not success or not response:
            return {"nonsense_score": 0, "reasoning": "Analysis failed", "success": False}
        
        # Parse response
        score_match = re.search(r'PUNTUACIÓN|SCORE:\s*([-]?[0-3])', response, re.IGNORECASE)
        reasoning_match = re.search(r'RAZONAMIENTO|REASONING:\s*(.+)', response, re.IGNORECASE | re.DOTALL)
        
        score = int(score_match.group(1)) if score_match else 0
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"
        
        return {
            "nonsense_score": score,
            "reasoning": reasoning,
            "success": True,
            "text": text
        }
        
    except Exception as e:
        return {
            "nonsense_score": 0,
            "reasoning": f"Error in nonsense analysis: {str(e)}",
            "success": False
        }

def enhanced_languagetool_analyzer(original: str, suggested: str, language: str = "english") -> Dict[str, Any]:
    """
    Enhanced LanguageTool analyzer with better error handling and multilingual support.
    """
    if not LANGUAGETOOL_AVAILABLE:
        return {
            "analysis": "LanguageTool not available - install language_tool_python and ensure Java is installed",
            "success": False
        }
    
    try:
        # Map language codes for LanguageTool
        lt_lang_map = {
            "english": "en-US",
            "spanish": "es",
            "german": "de-DE", 
            "french": "fr"
        }
        
        lt_lang = lt_lang_map.get(language.lower(), "en-US")
        
        # Initialize LanguageTool
        tool = language_tool_python.LanguageTool(lt_lang)
        
        # Check both original and suggested
        original_errors = tool.check(original)
        suggested_errors = tool.check(suggested)
        
        # Analyze correction effectiveness
        original_error_count = len(original_errors)
        suggested_error_count = len(suggested_errors)
        
        # Detailed analysis
        analysis_parts = []
        
        if original_error_count > suggested_error_count:
            fixed_count = original_error_count - suggested_error_count
            analysis_parts.append(f"✅ Fixed {fixed_count} error(s)")
        elif suggested_error_count > original_error_count:
            introduced_count = suggested_error_count - original_error_count
            analysis_parts.append(f"❌ Introduced {introduced_count} new error(s)")
        else:
            analysis_parts.append("➡️ Same number of errors")
        
        # Error details
        if original_errors:
            analysis_parts.append(f"Original errors: {[e.message for e in original_errors[:3]]}")
        if suggested_errors:
            analysis_parts.append(f"Remaining errors: {[e.message for e in suggested_errors[:3]]}")
        
        # Quality assessment
        if suggested_error_count == 0 and original_error_count > 0:
            quality = "excellent"
        elif suggested_error_count < original_error_count:
            quality = "good"
        elif suggested_error_count == original_error_count:
            quality = "neutral"
        else:
            quality = "poor"
        
        tool.close()
        
        return {
            "analysis": " | ".join(analysis_parts),
            "quality": quality,
            "original_errors": original_error_count,
            "suggested_errors": suggested_error_count,
            "error_change": original_error_count - suggested_error_count,
            "success": True
        }
        
    except Exception as e:
        return {
            "analysis": f"LanguageTool analysis failed: {str(e)}",
            "quality": "unknown",
            "success": False
        }

def direct_rule_query_tool(rule_name: str, language: str = "english") -> Dict[str, Any]:
    """
    Direct rule querying for comprehensive rule information.
    """
    try:
        if not AdvancedChunkedRAG:
            return {"error": "AdvancedChunkedRAG not available", "success": False}
        
        lang_map = {"en": "english", "es": "spanish", "de": "german", "fr": "french"}
        rag_language = lang_map.get(language.lower()[:2], language.lower())
        
        rag = AdvancedChunkedRAG(language=rag_language)
        
        # Get comprehensive rule information
        rule_info = rag.get_comprehensive_rule_info(rule_name)
        
        if not rule_info:
            return {
                "rule_info": f"No information found for rule: {rule_name}",
                "success": False
            }
        
        return {
            "rule_info": rule_info,
            "success": True,
            "rule_name": rule_name,
            "language": rag_language
        }
        
    except Exception as e:
        return {
            "error": f"Direct rule query failed: {str(e)}",
            "success": False
        }

# Define the ultimate toolkit with all tools
ULTIMATE_TOOLS = {
    "grammar_rag": ultimate_chunked_grammar_rag_tool,
    "meaning_change": enhanced_meaning_change_tool,
    "quality_reward": enhanced_quality_reward_tool,
    "nonsense_detector": enhanced_nonsense_detector_tool,
    "languagetool_analyzer": enhanced_languagetool_analyzer,
    "direct_rule_query": direct_rule_query_tool,
}

# Test function to validate tools
def test_ultimate_tools():
    """Test all tools to ensure they work correctly."""
    print("🧪 Testing Ultimate Tools")
    print("=" * 40)
    
    test_cases = [
        {
            "original": "She don't like coffee.",
            "suggested": "She doesn't like coffee.",
            "language": "english"
        },
        {
            "original": "La casa blanco es grande.",
            "suggested": "La casa blanca es grande.",
            "language": "spanish"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n🔬 Test {i}: {test['language']}")
        print(f"Original: {test['original']}")
        print(f"Suggested: {test['suggested']}")
        
        # Test each tool
        for tool_name, tool_func in ULTIMATE_TOOLS.items():
            try:
                if tool_name == "grammar_rag":
                    result = tool_func("subject verb agreement", test['language'])
                elif tool_name == "direct_rule_query":
                    result = tool_func("Subject-Verb Agreement", test['language'])
                elif tool_name in ["meaning_change", "quality_reward"]:
                    result = tool_func(test['original'], test['suggested'], test['language'])
                elif tool_name == "nonsense_detector":
                    result = tool_func(test['suggested'], test['language'])
                elif tool_name == "languagetool_analyzer":
                    result = tool_func(test['original'], test['suggested'], test['language'])
                else:
                    continue
                
                success = result.get('success', False)
                print(f"  {tool_name}: {'✅' if success else '❌'}")
                
            except Exception as e:
                print(f"  {tool_name}: ❌ Error: {e}")
    
    print("\n✅ Ultimate tools testing complete!")

if __name__ == "__main__":
    test_ultimate_tools()
