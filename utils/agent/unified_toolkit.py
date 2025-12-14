#!/usr/bin/env python3
"""
Unified SOTA Toolkit for Spanish GEC Classification

High-value, debugged tools with detailed accuracy contribution tracking.
Supports enhanced RAG queries: lang+rule, lang+rule+src->tgt patterns.
"""

import os
import sys
import re
import json
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, asdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.rag.massive_multilingual_rag_v4 import MassiveMultilingualRAG
from utils.rag.annotation_guidelines_rag import AnnotationGuidelinesRAG
from utils.agent.fixed_meaning_tool import FixedMeaningTool
from utils.agent.spanish_grammar_checker import SpanishGrammarChecker


@dataclass
class ToolResult:
    """Standardized tool result with accuracy tracking."""
    tool_name: str
    classification: str
    confidence: float
    reasoning: str
    evidence: Dict[str, Any]
    cost: float = 0.0
    processing_time: float = 0.0


@dataclass
class UnifiedAnalysis:
    """Complete analysis from all tools."""
    src: str
    tgt: str
    language: str
    tool_results: List[ToolResult]
    final_classification: str
    final_confidence: float
    final_reasoning: str
    total_cost: float
    debug_info: Dict[str, Any]


class UnifiedSOTAToolkit:
    """Unified toolkit with enhanced RAG and debugging capabilities."""
    
    def __init__(self, language: str = "es", debug: bool = True):
        """Initialize unified toolkit."""
        self.language = language
        self.debug = debug
        
        # Initialize all tools
        os.environ['QUIET_LOGS'] = '1'
        self.multilingual_rag = MassiveMultilingualRAG(verbose=False)
        self.guidelines_rag = AnnotationGuidelinesRAG()
        self.meaning_tool = FixedMeaningTool()
        self.grammar_checker = SpanishGrammarChecker()
        
        # Tool accuracy tracking
        self.tool_stats = {
            'enhanced_rag': {'calls': 0, 'hits': 0, 'accuracy': 0.0},
            'meaning_analysis': {'calls': 0, 'hits': 0, 'accuracy': 0.0},
            'grammar_analysis': {'calls': 0, 'hits': 0, 'accuracy': 0.0},
            'heuristic_patterns': {'calls': 0, 'hits': 0, 'accuracy': 0.0}
        }
    
    def analyze_correction(self, src: str, tgt: str) -> UnifiedAnalysis:
        """Complete analysis using all tools with enhanced RAG."""
        import time
        start_time = time.time()
        
        tool_results = []
        debug_info = {}
        
        # Tool 1: Enhanced RAG Analysis
        rag_result = self._enhanced_rag_analysis(src, tgt)
        tool_results.append(rag_result)
        
        # Tool 2: Meaning Analysis
        meaning_result = self._meaning_analysis(src, tgt)
        tool_results.append(meaning_result)
        
        # Tool 3: Grammar Analysis
        grammar_result = self._grammar_analysis(src, tgt)
        tool_results.append(grammar_result)
        
        # Tool 4: Heuristic Patterns
        heuristic_result = self._heuristic_patterns(src, tgt)
        tool_results.append(heuristic_result)
        
        # Aggregate results
        final_classification, final_confidence, final_reasoning = self._aggregate_results(tool_results)
        
        # Debug information
        if self.debug:
            debug_info = {
                'processing_time': time.time() - start_time,
                'tool_contributions': {tr.tool_name: tr.confidence for tr in tool_results},
                'evidence_summary': {tr.tool_name: tr.evidence for tr in tool_results}
            }
        
        return UnifiedAnalysis(
            src=src, tgt=tgt, language=self.language,
            tool_results=tool_results,
            final_classification=final_classification,
            final_confidence=final_confidence,
            final_reasoning=final_reasoning,
            total_cost=sum(tr.cost for tr in tool_results),
            debug_info=debug_info
        )
    
    def _enhanced_rag_analysis(self, src: str, tgt: str) -> ToolResult:
        """Enhanced RAG with multiple query patterns."""
        self.tool_stats['enhanced_rag']['calls'] += 1
        
        # Extract edit for analysis
        edits = self._extract_edits(src, tgt)
        if not edits:
            return ToolResult(
                tool_name="enhanced_rag",
                classification="TP",
                confidence=0.3,
                reasoning="No clear edits detected",
                evidence={"edits": [], "query_patterns": []}
            )
        
        evidence = {"edits": edits, "query_patterns": []}
        best_classification = "TP"
        best_confidence = 0.0
        best_reasoning = "No RAG match"
        
        for before, after in edits[:2]:  # Analyze top 2 edits
            # Pattern 1: Traditional lang+src->tgt
            rag_results = self.multilingual_rag.search_rules(before, after, self.language)
            if rag_results:
                rule, confidence = rag_results[0]
                if confidence > best_confidence:
                    best_classification = rule.rule_type
                    best_confidence = confidence
                    best_reasoning = f"RAG rule: {rule.description}"
                    evidence["query_patterns"].append(f"lang+edit: {before}→{after}")
            
            # Pattern 2: Enhanced lang+rule query
            rule_queries = [
                f"accent {self.language}",
                f"grammar {self.language}",
                f"synonym {self.language}",
                f"meaning {self.language}"
            ]
            
            for rule_query in rule_queries:
                enhanced_results = self._query_rag_by_rule(rule_query, before, after)
                if enhanced_results and enhanced_results[1] > best_confidence:
                    best_classification = enhanced_results[0]
                    best_confidence = enhanced_results[1]
                    best_reasoning = f"Enhanced RAG: {rule_query} pattern"
                    evidence["query_patterns"].append(f"lang+rule: {rule_query}")
        
        # Check guidelines
        guideline_results = self.guidelines_rag.search_guidelines(src, tgt, self.language)
        if guideline_results:
            guideline, confidence = guideline_results[0]
            if confidence > best_confidence:
                best_classification = guideline.label
                best_confidence = confidence
                best_reasoning = f"Guideline: {guideline.category}"
                evidence["query_patterns"].append("guidelines")
        
        if best_confidence > 0.5:
            self.tool_stats['enhanced_rag']['hits'] += 1
        
        return ToolResult(
            tool_name="enhanced_rag",
            classification=best_classification,
            confidence=best_confidence,
            reasoning=best_reasoning,
            evidence=evidence
        )
    
    def _query_rag_by_rule(self, rule_query: str, before: str, after: str) -> Optional[Tuple[str, float]]:
        """Enhanced RAG query by rule type."""
        # Search for rules matching the pattern
        matching_rules = []
        for rule in self.multilingual_rag.rules_db:
            if rule.language == self.language:
                rule_text = f"{rule.description} {rule.pattern_before} {rule.pattern_after}".lower()
                if any(keyword in rule_text for keyword in rule_query.lower().split()):
                    # Calculate relevance to current edit
                    relevance = self._calculate_rule_relevance(rule, before, after)
                    if relevance > 0.3:
                        matching_rules.append((rule, relevance))
        
        if matching_rules:
            best_rule, best_relevance = max(matching_rules, key=lambda x: x[1])
            return best_rule.rule_type, best_relevance
        
        return None
    
    def _calculate_rule_relevance(self, rule, before: str, after: str) -> float:
        """Calculate rule relevance to current edit."""
        relevance = 0.0
        
        # Exact match bonus
        if rule.pattern_before.lower() == before.lower():
            relevance += 0.5
        if rule.pattern_after.lower() == after.lower():
            relevance += 0.5
        
        # Partial match bonus
        if before.lower() in rule.pattern_before.lower() or rule.pattern_before.lower() in before.lower():
            relevance += 0.2
        if after.lower() in rule.pattern_after.lower() or rule.pattern_after.lower() in after.lower():
            relevance += 0.2
        
        # Pattern similarity
        if len(before) == len(after) and abs(len(before) - len(rule.pattern_before)) <= 2:
            relevance += 0.1
        
        return min(relevance, 1.0)
    
    def _meaning_analysis(self, src: str, tgt: str) -> ToolResult:
        """Meaning change analysis."""
        self.tool_stats['meaning_analysis']['calls'] += 1
        
        meaning_result = self.meaning_tool.analyze_meaning_change(src, tgt, self.language)
        
        classification = "FP1" if not meaning_result.meaning_preserved else "TP"
        confidence = meaning_result.confidence
        
        if not meaning_result.meaning_preserved and confidence > 0.7:
            self.tool_stats['meaning_analysis']['hits'] += 1
        
        return ToolResult(
            tool_name="meaning_analysis",
            classification=classification,
            confidence=confidence,
            reasoning=f"Meaning preserved: {meaning_result.meaning_preserved}",
            evidence=asdict(meaning_result)
        )
    
    def _grammar_analysis(self, src: str, tgt: str) -> ToolResult:
        """Grammar error analysis."""
        self.tool_stats['grammar_analysis']['calls'] += 1
        
        grammar_result = self.grammar_checker.analyze_correction(src, tgt)
        
        classification = "FP2" if grammar_result["introduces_errors"] else "TP"
        confidence = grammar_result["fp2_likelihood"]
        
        if grammar_result["introduces_errors"] and confidence > 0.8:
            self.tool_stats['grammar_analysis']['hits'] += 1
        
        return ToolResult(
            tool_name="grammar_analysis",
            classification=classification,
            confidence=confidence,
            reasoning=f"Grammar errors: {', '.join(grammar_result['descriptions'])}",
            evidence=grammar_result
        )
    
    def _heuristic_patterns(self, src: str, tgt: str) -> ToolResult:
        """Advanced heuristic pattern matching."""
        self.tool_stats['heuristic_patterns']['calls'] += 1
        
        # Extract word differences
        src_words = set(src.lower().split())
        tgt_words = set(tgt.lower().split())
        added_words = tgt_words - src_words
        removed_words = src_words - tgt_words
        
        evidence = {
            "added_words": list(added_words),
            "removed_words": list(removed_words),
            "pattern_matches": []
        }
        
        # High-confidence patterns
        
        # 1. Accent corrections (TP)
        accent_pairs = [
            ('mas', 'más'), ('si', 'sí'), ('tu', 'tú'), ('mi', 'mí'),
            ('el', 'él'), ('se', 'sé'), ('de', 'dé'), ('te', 'té'),
            ('como', 'cómo'), ('que', 'qué'), ('donde', 'dónde'), ('cuando', 'cuándo')
        ]
        
        for before, after in accent_pairs:
            if before in removed_words and after in added_words:
                self.tool_stats['heuristic_patterns']['hits'] += 1
                evidence["pattern_matches"].append(f"accent_correction: {before}→{after}")
                return ToolResult(
                    tool_name="heuristic_patterns",
                    classification="TP",
                    confidence=0.95,
                    reasoning=f"Accent correction: {before} → {after}",
                    evidence=evidence
                )
        
        # 2. Synonym changes (FP3)
        synonym_pairs = [
            ('mando', 'envío'), ('importante', 'clave'), ('gratis', 'gratuito'),
            ('responder', 'contestar'), ('hacer', 'realizar'), ('obtener', 'conseguir'),
            ('enviar', 'mandar'), ('bueno', 'bien'), ('grande', 'gran')
        ]
        
        for word1, word2 in synonym_pairs:
            if (word1 in removed_words and word2 in added_words) or (word2 in removed_words and word1 in added_words):
                self.tool_stats['heuristic_patterns']['hits'] += 1
                evidence["pattern_matches"].append(f"synonym_change: {word1}↔{word2}")
                return ToolResult(
                    tool_name="heuristic_patterns",
                    classification="FP3",
                    confidence=0.9,
                    reasoning=f"Synonym change: {word1} ↔ {word2}",
                    evidence=evidence
                )
        
        # 3. Grammar corrections (TP)
        if len(removed_words) == 1 and len(added_words) == 1:
            removed = list(removed_words)[0]
            added = list(added_words)[0]
            
            grammar_corrections = [
                ('han', 'ha'), ('ha', 'han'),  # verb agreement
                ('esta', 'está'), ('estas', 'estás'),  # accent on verbs
                ('practico', 'práctico'), ('publico', 'público'),  # stress patterns
            ]
            
            for wrong, correct in grammar_corrections:
                if removed == wrong and added == correct:
                    self.tool_stats['heuristic_patterns']['hits'] += 1
                    evidence["pattern_matches"].append(f"grammar_correction: {wrong}→{correct}")
                    return ToolResult(
                        tool_name="heuristic_patterns",
                        classification="TP",
                        confidence=0.9,
                        reasoning=f"Grammar correction: {wrong} → {correct}",
                        evidence=evidence
                    )
        
        return ToolResult(
            tool_name="heuristic_patterns",
            classification="TP",
            confidence=0.3,
            reasoning="No clear heuristic pattern",
            evidence=evidence
        )
    
    def _extract_edits(self, src: str, tgt: str) -> List[Tuple[str, str]]:
        """Extract word-level edits."""
        src_words = src.split()
        tgt_words = tgt.split()
        
        edits = []
        max_len = max(len(src_words), len(tgt_words))
        
        for i in range(max_len):
            src_word = src_words[i] if i < len(src_words) else ""
            tgt_word = tgt_words[i] if i < len(tgt_words) else ""
            
            if src_word != tgt_word and src_word and tgt_word:
                edits.append((src_word, tgt_word))
        
        return edits[:3]  # Return top 3 edits
    
    def _aggregate_results(self, tool_results: List[ToolResult]) -> Tuple[str, float, str]:
        """Aggregate tool results with improved logic."""
        
        # Priority-based aggregation for better accuracy
        
        # 1. Check for ultra-high confidence heuristics (>0.9)
        heuristic_result = next((tr for tr in tool_results 
                               if tr.tool_name == 'heuristic_patterns' and tr.confidence > 0.9), None)
        if heuristic_result:
            return heuristic_result.classification, heuristic_result.confidence, f"High-conf heuristic: {heuristic_result.reasoning}"
        
        # 2. Check for strong grammar errors (FP2)
        grammar_result = next((tr for tr in tool_results 
                             if tr.tool_name == 'grammar_analysis' and tr.confidence > 0.8 and tr.classification == 'FP2'), None)
        if grammar_result:
            return grammar_result.classification, grammar_result.confidence, f"Grammar error: {grammar_result.reasoning}"
        
        # 3. Check for strong meaning changes (FP1)
        meaning_result = next((tr for tr in tool_results 
                             if tr.tool_name == 'meaning_analysis' and tr.confidence > 0.7 and tr.classification == 'FP1'), None)
        if meaning_result:
            return meaning_result.classification, meaning_result.confidence, f"Meaning change: {meaning_result.reasoning}"
        
        # 4. Check for strong RAG matches
        rag_result = next((tr for tr in tool_results 
                         if tr.tool_name == 'enhanced_rag' and tr.confidence > 0.8), None)
        if rag_result:
            return rag_result.classification, rag_result.confidence, f"RAG match: {rag_result.reasoning}"
        
        # 5. Weighted voting for remaining cases
        weights = {
            'heuristic_patterns': 0.4,  # Increased weight for heuristics
            'grammar_analysis': 0.3,
            'enhanced_rag': 0.2,
            'meaning_analysis': 0.1     # Reduced weight to prevent over-conservative TP bias
        }
        
        votes = {}
        total_weight = 0.0
        reasoning_parts = []
        
        for result in tool_results:
            if result.confidence > 0.3:  # Lower threshold for voting
                weight = weights.get(result.tool_name, 0.1) * result.confidence
                votes[result.classification] = votes.get(result.classification, 0) + weight
                total_weight += weight
                reasoning_parts.append(f"{result.tool_name}: {result.reasoning}")
        
        if not votes:
            return "TP", 0.5, "No tool results"
        
        # Find best classification
        best_classification = max(votes.items(), key=lambda x: x[1])[0]
        best_confidence = votes[best_classification] / total_weight if total_weight > 0 else 0.5
        
        # Combine reasoning
        final_reasoning = " | ".join(reasoning_parts[:2])
        
        return best_classification, min(best_confidence, 0.99), final_reasoning
    
    def get_tool_stats(self) -> Dict[str, Any]:
        """Get tool performance statistics."""
        return {
            'tool_stats': self.tool_stats,
            'total_calls': sum(stats['calls'] for stats in self.tool_stats.values()),
            'total_hits': sum(stats['hits'] for stats in self.tool_stats.values())
        }
    
    def debug_example(self, src: str, tgt: str, expected_label: str = None) -> None:
        """Debug a specific example with detailed output."""
        print(f"\n🔍 DEBUGGING EXAMPLE:")
        print(f"  Source: {src}")
        print(f"  Target: {tgt}")
        if expected_label:
            print(f"  Expected: {expected_label}")
        print("=" * 50)
        
        analysis = self.analyze_correction(src, tgt)
        
        print(f"📊 FINAL RESULT: {analysis.final_classification} (confidence: {analysis.final_confidence:.3f})")
        print(f"💭 REASONING: {analysis.final_reasoning}")
        
        print(f"\n🔧 TOOL BREAKDOWN:")
        for result in analysis.tool_results:
            status = "✅" if result.confidence > 0.5 else "⚪"
            print(f"  {status} {result.tool_name}: {result.classification} ({result.confidence:.3f}) - {result.reasoning}")
        
        if expected_label:
            correct = "✅" if analysis.final_classification == expected_label else "❌"
            print(f"\n{correct} ACCURACY: {analysis.final_classification} vs {expected_label}")
        
        print(f"\n⏱️  PERFORMANCE: {analysis.debug_info.get('processing_time', 0):.3f}s, ${analysis.total_cost:.6f}")


def main():
    """Test unified toolkit."""
    toolkit = UnifiedSOTAToolkit(debug=True)
    
    test_cases = [
        ('El equipo ha sido notificado.', 'El equipo han sido notificado.', 'FP2'),
        ('Te mando el archivo.', 'Te envío el archivo.', 'FP3'),
        ('Necesito mas tiempo.', 'Necesito más tiempo.', 'TP'),
        ('No me gusta.', 'Me gusta.', 'FP1'),
        ('La casa es grande.', 'La casa es muy grande.', 'FP3'),
    ]
    
    print("🚀 UNIFIED SOTA TOOLKIT TEST:")
    
    for src, tgt, expected in test_cases:
        toolkit.debug_example(src, tgt, expected)
    
    print(f"\n📈 TOOL STATISTICS:")
    stats = toolkit.get_tool_stats()
    for tool_name, tool_stats in stats['tool_stats'].items():
        hit_rate = tool_stats['hits'] / tool_stats['calls'] if tool_stats['calls'] > 0 else 0
        print(f"  {tool_name}: {tool_stats['hits']}/{tool_stats['calls']} hits ({hit_rate:.1%})")


if __name__ == "__main__":
    main()

Unified SOTA Toolkit for Spanish GEC Classification

High-value, debugged tools with detailed accuracy contribution tracking.
Supports enhanced RAG queries: lang+rule, lang+rule+src->tgt patterns.
"""

import os
import sys
import re
import json
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, asdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.rag.massive_multilingual_rag_v4 import MassiveMultilingualRAG
from utils.rag.annotation_guidelines_rag import AnnotationGuidelinesRAG
from utils.agent.fixed_meaning_tool import FixedMeaningTool
from utils.agent.spanish_grammar_checker import SpanishGrammarChecker


@dataclass
class ToolResult:
    """Standardized tool result with accuracy tracking."""
    tool_name: str
    classification: str
    confidence: float
    reasoning: str
    evidence: Dict[str, Any]
    cost: float = 0.0
    processing_time: float = 0.0


@dataclass
class UnifiedAnalysis:
    """Complete analysis from all tools."""
    src: str
    tgt: str
    language: str
    tool_results: List[ToolResult]
    final_classification: str
    final_confidence: float
    final_reasoning: str
    total_cost: float
    debug_info: Dict[str, Any]


class UnifiedSOTAToolkit:
    """Unified toolkit with enhanced RAG and debugging capabilities."""
    
    def __init__(self, language: str = "es", debug: bool = True):
        """Initialize unified toolkit."""
        self.language = language
        self.debug = debug
        
        # Initialize all tools
        os.environ['QUIET_LOGS'] = '1'
        self.multilingual_rag = MassiveMultilingualRAG(verbose=False)
        self.guidelines_rag = AnnotationGuidelinesRAG()
        self.meaning_tool = FixedMeaningTool()
        self.grammar_checker = SpanishGrammarChecker()
        
        # Tool accuracy tracking
        self.tool_stats = {
            'enhanced_rag': {'calls': 0, 'hits': 0, 'accuracy': 0.0},
            'meaning_analysis': {'calls': 0, 'hits': 0, 'accuracy': 0.0},
            'grammar_analysis': {'calls': 0, 'hits': 0, 'accuracy': 0.0},
            'heuristic_patterns': {'calls': 0, 'hits': 0, 'accuracy': 0.0}
        }
    
    def analyze_correction(self, src: str, tgt: str) -> UnifiedAnalysis:
        """Complete analysis using all tools with enhanced RAG."""
        import time
        start_time = time.time()
        
        tool_results = []
        debug_info = {}
        
        # Tool 1: Enhanced RAG Analysis
        rag_result = self._enhanced_rag_analysis(src, tgt)
        tool_results.append(rag_result)
        
        # Tool 2: Meaning Analysis
        meaning_result = self._meaning_analysis(src, tgt)
        tool_results.append(meaning_result)
        
        # Tool 3: Grammar Analysis
        grammar_result = self._grammar_analysis(src, tgt)
        tool_results.append(grammar_result)
        
        # Tool 4: Heuristic Patterns
        heuristic_result = self._heuristic_patterns(src, tgt)
        tool_results.append(heuristic_result)
        
        # Aggregate results
        final_classification, final_confidence, final_reasoning = self._aggregate_results(tool_results)
        
        # Debug information
        if self.debug:
            debug_info = {
                'processing_time': time.time() - start_time,
                'tool_contributions': {tr.tool_name: tr.confidence for tr in tool_results},
                'evidence_summary': {tr.tool_name: tr.evidence for tr in tool_results}
            }
        
        return UnifiedAnalysis(
            src=src, tgt=tgt, language=self.language,
            tool_results=tool_results,
            final_classification=final_classification,
            final_confidence=final_confidence,
            final_reasoning=final_reasoning,
            total_cost=sum(tr.cost for tr in tool_results),
            debug_info=debug_info
        )
    
    def _enhanced_rag_analysis(self, src: str, tgt: str) -> ToolResult:
        """Enhanced RAG with multiple query patterns."""
        self.tool_stats['enhanced_rag']['calls'] += 1
        
        # Extract edit for analysis
        edits = self._extract_edits(src, tgt)
        if not edits:
            return ToolResult(
                tool_name="enhanced_rag",
                classification="TP",
                confidence=0.3,
                reasoning="No clear edits detected",
                evidence={"edits": [], "query_patterns": []}
            )
        
        evidence = {"edits": edits, "query_patterns": []}
        best_classification = "TP"
        best_confidence = 0.0
        best_reasoning = "No RAG match"
        
        for before, after in edits[:2]:  # Analyze top 2 edits
            # Pattern 1: Traditional lang+src->tgt
            rag_results = self.multilingual_rag.search_rules(before, after, self.language)
            if rag_results:
                rule, confidence = rag_results[0]
                if confidence > best_confidence:
                    best_classification = rule.rule_type
                    best_confidence = confidence
                    best_reasoning = f"RAG rule: {rule.description}"
                    evidence["query_patterns"].append(f"lang+edit: {before}→{after}")
            
            # Pattern 2: Enhanced lang+rule query
            rule_queries = [
                f"accent {self.language}",
                f"grammar {self.language}",
                f"synonym {self.language}",
                f"meaning {self.language}"
            ]
            
            for rule_query in rule_queries:
                enhanced_results = self._query_rag_by_rule(rule_query, before, after)
                if enhanced_results and enhanced_results[1] > best_confidence:
                    best_classification = enhanced_results[0]
                    best_confidence = enhanced_results[1]
                    best_reasoning = f"Enhanced RAG: {rule_query} pattern"
                    evidence["query_patterns"].append(f"lang+rule: {rule_query}")
        
        # Check guidelines
        guideline_results = self.guidelines_rag.search_guidelines(src, tgt, self.language)
        if guideline_results:
            guideline, confidence = guideline_results[0]
            if confidence > best_confidence:
                best_classification = guideline.label
                best_confidence = confidence
                best_reasoning = f"Guideline: {guideline.category}"
                evidence["query_patterns"].append("guidelines")
        
        if best_confidence > 0.5:
            self.tool_stats['enhanced_rag']['hits'] += 1
        
        return ToolResult(
            tool_name="enhanced_rag",
            classification=best_classification,
            confidence=best_confidence,
            reasoning=best_reasoning,
            evidence=evidence
        )
    
    def _query_rag_by_rule(self, rule_query: str, before: str, after: str) -> Optional[Tuple[str, float]]:
        """Enhanced RAG query by rule type."""
        # Search for rules matching the pattern
        matching_rules = []
        for rule in self.multilingual_rag.rules_db:
            if rule.language == self.language:
                rule_text = f"{rule.description} {rule.pattern_before} {rule.pattern_after}".lower()
                if any(keyword in rule_text for keyword in rule_query.lower().split()):
                    # Calculate relevance to current edit
                    relevance = self._calculate_rule_relevance(rule, before, after)
                    if relevance > 0.3:
                        matching_rules.append((rule, relevance))
        
        if matching_rules:
            best_rule, best_relevance = max(matching_rules, key=lambda x: x[1])
            return best_rule.rule_type, best_relevance
        
        return None
    
    def _calculate_rule_relevance(self, rule, before: str, after: str) -> float:
        """Calculate rule relevance to current edit."""
        relevance = 0.0
        
        # Exact match bonus
        if rule.pattern_before.lower() == before.lower():
            relevance += 0.5
        if rule.pattern_after.lower() == after.lower():
            relevance += 0.5
        
        # Partial match bonus
        if before.lower() in rule.pattern_before.lower() or rule.pattern_before.lower() in before.lower():
            relevance += 0.2
        if after.lower() in rule.pattern_after.lower() or rule.pattern_after.lower() in after.lower():
            relevance += 0.2
        
        # Pattern similarity
        if len(before) == len(after) and abs(len(before) - len(rule.pattern_before)) <= 2:
            relevance += 0.1
        
        return min(relevance, 1.0)
    
    def _meaning_analysis(self, src: str, tgt: str) -> ToolResult:
        """Meaning change analysis."""
        self.tool_stats['meaning_analysis']['calls'] += 1
        
        meaning_result = self.meaning_tool.analyze_meaning_change(src, tgt, self.language)
        
        classification = "FP1" if not meaning_result.meaning_preserved else "TP"
        confidence = meaning_result.confidence
        
        if not meaning_result.meaning_preserved and confidence > 0.7:
            self.tool_stats['meaning_analysis']['hits'] += 1
        
        return ToolResult(
            tool_name="meaning_analysis",
            classification=classification,
            confidence=confidence,
            reasoning=f"Meaning preserved: {meaning_result.meaning_preserved}",
            evidence=asdict(meaning_result)
        )
    
    def _grammar_analysis(self, src: str, tgt: str) -> ToolResult:
        """Grammar error analysis."""
        self.tool_stats['grammar_analysis']['calls'] += 1
        
        grammar_result = self.grammar_checker.analyze_correction(src, tgt)
        
        classification = "FP2" if grammar_result["introduces_errors"] else "TP"
        confidence = grammar_result["fp2_likelihood"]
        
        if grammar_result["introduces_errors"] and confidence > 0.8:
            self.tool_stats['grammar_analysis']['hits'] += 1
        
        return ToolResult(
            tool_name="grammar_analysis",
            classification=classification,
            confidence=confidence,
            reasoning=f"Grammar errors: {', '.join(grammar_result['descriptions'])}",
            evidence=grammar_result
        )
    
    def _heuristic_patterns(self, src: str, tgt: str) -> ToolResult:
        """Advanced heuristic pattern matching."""
        self.tool_stats['heuristic_patterns']['calls'] += 1
        
        # Extract word differences
        src_words = set(src.lower().split())
        tgt_words = set(tgt.lower().split())
        added_words = tgt_words - src_words
        removed_words = src_words - tgt_words
        
        evidence = {
            "added_words": list(added_words),
            "removed_words": list(removed_words),
            "pattern_matches": []
        }
        
        # High-confidence patterns
        
        # 1. Accent corrections (TP)
        accent_pairs = [
            ('mas', 'más'), ('si', 'sí'), ('tu', 'tú'), ('mi', 'mí'),
            ('el', 'él'), ('se', 'sé'), ('de', 'dé'), ('te', 'té'),
            ('como', 'cómo'), ('que', 'qué'), ('donde', 'dónde'), ('cuando', 'cuándo')
        ]
        
        for before, after in accent_pairs:
            if before in removed_words and after in added_words:
                self.tool_stats['heuristic_patterns']['hits'] += 1
                evidence["pattern_matches"].append(f"accent_correction: {before}→{after}")
                return ToolResult(
                    tool_name="heuristic_patterns",
                    classification="TP",
                    confidence=0.95,
                    reasoning=f"Accent correction: {before} → {after}",
                    evidence=evidence
                )
        
        # 2. Synonym changes (FP3)
        synonym_pairs = [
            ('mando', 'envío'), ('importante', 'clave'), ('gratis', 'gratuito'),
            ('responder', 'contestar'), ('hacer', 'realizar'), ('obtener', 'conseguir'),
            ('enviar', 'mandar'), ('bueno', 'bien'), ('grande', 'gran')
        ]
        
        for word1, word2 in synonym_pairs:
            if (word1 in removed_words and word2 in added_words) or (word2 in removed_words and word1 in added_words):
                self.tool_stats['heuristic_patterns']['hits'] += 1
                evidence["pattern_matches"].append(f"synonym_change: {word1}↔{word2}")
                return ToolResult(
                    tool_name="heuristic_patterns",
                    classification="FP3",
                    confidence=0.9,
                    reasoning=f"Synonym change: {word1} ↔ {word2}",
                    evidence=evidence
                )
        
        # 3. Grammar corrections (TP)
        if len(removed_words) == 1 and len(added_words) == 1:
            removed = list(removed_words)[0]
            added = list(added_words)[0]
            
            grammar_corrections = [
                ('han', 'ha'), ('ha', 'han'),  # verb agreement
                ('esta', 'está'), ('estas', 'estás'),  # accent on verbs
                ('practico', 'práctico'), ('publico', 'público'),  # stress patterns
            ]
            
            for wrong, correct in grammar_corrections:
                if removed == wrong and added == correct:
                    self.tool_stats['heuristic_patterns']['hits'] += 1
                    evidence["pattern_matches"].append(f"grammar_correction: {wrong}→{correct}")
                    return ToolResult(
                        tool_name="heuristic_patterns",
                        classification="TP",
                        confidence=0.9,
                        reasoning=f"Grammar correction: {wrong} → {correct}",
                        evidence=evidence
                    )
        
        return ToolResult(
            tool_name="heuristic_patterns",
            classification="TP",
            confidence=0.3,
            reasoning="No clear heuristic pattern",
            evidence=evidence
        )
    
    def _extract_edits(self, src: str, tgt: str) -> List[Tuple[str, str]]:
        """Extract word-level edits."""
        src_words = src.split()
        tgt_words = tgt.split()
        
        edits = []
        max_len = max(len(src_words), len(tgt_words))
        
        for i in range(max_len):
            src_word = src_words[i] if i < len(src_words) else ""
            tgt_word = tgt_words[i] if i < len(tgt_words) else ""
            
            if src_word != tgt_word and src_word and tgt_word:
                edits.append((src_word, tgt_word))
        
        return edits[:3]  # Return top 3 edits
    
    def _aggregate_results(self, tool_results: List[ToolResult]) -> Tuple[str, float, str]:
        """Aggregate tool results with improved logic."""
        
        # Priority-based aggregation for better accuracy
        
        # 1. Check for ultra-high confidence heuristics (>0.9)
        heuristic_result = next((tr for tr in tool_results 
                               if tr.tool_name == 'heuristic_patterns' and tr.confidence > 0.9), None)
        if heuristic_result:
            return heuristic_result.classification, heuristic_result.confidence, f"High-conf heuristic: {heuristic_result.reasoning}"
        
        # 2. Check for strong grammar errors (FP2)
        grammar_result = next((tr for tr in tool_results 
                             if tr.tool_name == 'grammar_analysis' and tr.confidence > 0.8 and tr.classification == 'FP2'), None)
        if grammar_result:
            return grammar_result.classification, grammar_result.confidence, f"Grammar error: {grammar_result.reasoning}"
        
        # 3. Check for strong meaning changes (FP1)
        meaning_result = next((tr for tr in tool_results 
                             if tr.tool_name == 'meaning_analysis' and tr.confidence > 0.7 and tr.classification == 'FP1'), None)
        if meaning_result:
            return meaning_result.classification, meaning_result.confidence, f"Meaning change: {meaning_result.reasoning}"
        
        # 4. Check for strong RAG matches
        rag_result = next((tr for tr in tool_results 
                         if tr.tool_name == 'enhanced_rag' and tr.confidence > 0.8), None)
        if rag_result:
            return rag_result.classification, rag_result.confidence, f"RAG match: {rag_result.reasoning}"
        
        # 5. Weighted voting for remaining cases
        weights = {
            'heuristic_patterns': 0.4,  # Increased weight for heuristics
            'grammar_analysis': 0.3,
            'enhanced_rag': 0.2,
            'meaning_analysis': 0.1     # Reduced weight to prevent over-conservative TP bias
        }
        
        votes = {}
        total_weight = 0.0
        reasoning_parts = []
        
        for result in tool_results:
            if result.confidence > 0.3:  # Lower threshold for voting
                weight = weights.get(result.tool_name, 0.1) * result.confidence
                votes[result.classification] = votes.get(result.classification, 0) + weight
                total_weight += weight
                reasoning_parts.append(f"{result.tool_name}: {result.reasoning}")
        
        if not votes:
            return "TP", 0.5, "No tool results"
        
        # Find best classification
        best_classification = max(votes.items(), key=lambda x: x[1])[0]
        best_confidence = votes[best_classification] / total_weight if total_weight > 0 else 0.5
        
        # Combine reasoning
        final_reasoning = " | ".join(reasoning_parts[:2])
        
        return best_classification, min(best_confidence, 0.99), final_reasoning
    
    def get_tool_stats(self) -> Dict[str, Any]:
        """Get tool performance statistics."""
        return {
            'tool_stats': self.tool_stats,
            'total_calls': sum(stats['calls'] for stats in self.tool_stats.values()),
            'total_hits': sum(stats['hits'] for stats in self.tool_stats.values())
        }
    
    def debug_example(self, src: str, tgt: str, expected_label: str = None) -> None:
        """Debug a specific example with detailed output."""
        print(f"\n🔍 DEBUGGING EXAMPLE:")
        print(f"  Source: {src}")
        print(f"  Target: {tgt}")
        if expected_label:
            print(f"  Expected: {expected_label}")
        print("=" * 50)
        
        analysis = self.analyze_correction(src, tgt)
        
        print(f"📊 FINAL RESULT: {analysis.final_classification} (confidence: {analysis.final_confidence:.3f})")
        print(f"💭 REASONING: {analysis.final_reasoning}")
        
        print(f"\n🔧 TOOL BREAKDOWN:")
        for result in analysis.tool_results:
            status = "✅" if result.confidence > 0.5 else "⚪"
            print(f"  {status} {result.tool_name}: {result.classification} ({result.confidence:.3f}) - {result.reasoning}")
        
        if expected_label:
            correct = "✅" if analysis.final_classification == expected_label else "❌"
            print(f"\n{correct} ACCURACY: {analysis.final_classification} vs {expected_label}")
        
        print(f"\n⏱️  PERFORMANCE: {analysis.debug_info.get('processing_time', 0):.3f}s, ${analysis.total_cost:.6f}")


def main():
    """Test unified toolkit."""
    toolkit = UnifiedSOTAToolkit(debug=True)
    
    test_cases = [
        ('El equipo ha sido notificado.', 'El equipo han sido notificado.', 'FP2'),
        ('Te mando el archivo.', 'Te envío el archivo.', 'FP3'),
        ('Necesito mas tiempo.', 'Necesito más tiempo.', 'TP'),
        ('No me gusta.', 'Me gusta.', 'FP1'),
        ('La casa es grande.', 'La casa es muy grande.', 'FP3'),
    ]
    
    print("🚀 UNIFIED SOTA TOOLKIT TEST:")
    
    for src, tgt, expected in test_cases:
        toolkit.debug_example(src, tgt, expected)
    
    print(f"\n📈 TOOL STATISTICS:")
    stats = toolkit.get_tool_stats()
    for tool_name, tool_stats in stats['tool_stats'].items():
        hit_rate = tool_stats['hits'] / tool_stats['calls'] if tool_stats['calls'] > 0 else 0
        print(f"  {tool_name}: {tool_stats['hits']}/{tool_stats['calls']} hits ({hit_rate:.1%})")


if __name__ == "__main__":
    main()















