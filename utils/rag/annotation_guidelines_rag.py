#!/usr/bin/env python3
"""
ANNOTATION GUIDELINES RAG TOOL
Provides context-aware guidance for edge cases using annotation guidelines
"""

import os
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore")

@dataclass(frozen=True)
class GuidelineEntry:
    """Annotation guideline entry."""
    category: str
    label: str  # TP, FP1, FP2, FP3, TN, FN
    description: str
    examples: tuple  # Tuple of (before, after) pairs - hashable
    keywords: tuple  # Tuple of keywords - hashable
    confidence: float
    language: str = "en"


class AnnotationGuidelinesRAG:
    """RAG tool for annotation guidelines and edge case resolution."""
    
    def __init__(self):
        self.guidelines = self._build_guidelines_database()
        self.keyword_index = self._build_keyword_index()
        print(f"📋 Loaded {len(self.guidelines)} guideline entries")
    
    def _build_guidelines_database(self) -> List[GuidelineEntry]:
        """Build comprehensive guidelines database from docs."""
        guidelines = []
        
        # FP1 - CRITICAL ERRORS
        fp1_guidelines = [
            GuidelineEntry(
                category="meaning_change_major",
                label="FP1",
                description="Major meaning-changing suggestions that alter key information",
                examples=(
                    ("scale 2-3/10", "scale 3/10"),
                    ("and change", "'s"),
                    ("crimea", "crime"),
                    ("facepalm", "Facebook"),
                    ("friend", "friends"),
                    ("their", "his or her"),
                ),
                keywords=("meaning", "change", "information", "loss", "alter", "key", "scale", "numbers", "entities"),
                confidence=0.99
            ),
            GuidelineEntry(
                category="nonsensical",
                label="FP1",
                description="Nonsensical suggestions that introduce meaningless content or break structure",
                examples=(
                    ("", "[ [ [ AUTHOR : Please italicize book titles . ] ] ]"),
                    ("", "[ sic ]"),
                    ("Lisa leads", "led"),
                ),
                keywords=("nonsense", "meaningless", "structural", "integrity", "break", "author", "sic", "tense"),
                confidence=0.99
            ),
            GuidelineEntry(
                category="sensitivity",
                label="FP1",
                description="Suggestions triggering sensitivity issues, misgendering, or offensive content",
                examples=(
                    ("crimea", "crime"),
                    ("facepalm", "Facebook"),
                    ("friend", "friends"),
                    ("their", "his or her"),
                    ("", "a disabled"),
                ),
                keywords=("sensitive", "misgender", "identity", "pronoun", "disability", "offensive", "bias"),
                confidence=0.99
            ),
        ]
        
        # FP2 - MEDIUM ERRORS
        fp2_guidelines = [
            GuidelineEntry(
                category="ungrammatical",
                label="FP2",
                description="Ungrammatical suggestions that make sentence incorrect",
                examples=(
                    ("genetic", "genetically"),
                    ("imagine", "Imagine"),
                    ("wat", "at"),
                    ("Father", "The father"),
                    ("wans", ", want"),
                ),
                keywords=("ungrammatical", "incorrect", "grammar", "error", "syntax", "morphology"),
                confidence=0.98
            ),
            GuidelineEntry(
                category="meaning_change_minor",
                label="FP2",
                description="Minor or medium meaning changes that slightly alter interpretation",
                examples=(
                    ("didn't", "would not"),
                    ("Collecting", "Collect"),
                    ("a", "the"),
                    ("Thou", "You"),
                ),
                keywords=("minor", "medium", "meaning", "interpretation", "slightly", "alter", "modal", "article"),
                confidence=0.97
            ),
        ]
        
        # FP3 - STYLISTIC/PREFERENTIAL
        fp3_guidelines = [
            GuidelineEntry(
                category="stylistic",
                label="FP3",
                description="Domain-dependent stylistic suggestions",
                examples=(
                    ("Alice, Bob and Charlie", "Alice, Bob, and Charlie"),  # Oxford comma
                ),
                keywords=("stylistic", "domain", "dependent", "oxford", "comma", "punctuation", "style"),
                confidence=0.95
            ),
            GuidelineEntry(
                category="preferential",
                label="FP3",
                description="Domain-independent preference-based suggestions",
                examples=(
                    ("doesn't", "does not"),
                    ("can't", "cannot"),
                ),
                keywords=("preferential", "preference", "contraction", "expansion", "formal", "informal"),
                confidence=0.95
            ),
            GuidelineEntry(
                category="minor_improvement",
                label="FP3",
                description="Unnecessary clarity improvements to already correct text",
                examples=(
                    ("I finished", "I just finished"),
                    ("completed", "successfully completed"),
                ),
                keywords=("minor", "improvement", "clarity", "unnecessary", "adverb", "redundant"),
                confidence=0.95
            ),
        ]
        
        # TP - TRUE POSITIVES
        tp_guidelines = [
            GuidelineEntry(
                category="minimal_correction",
                label="TP",
                description="Minimal correct suggestions that fix all errors without changing meaning",
                examples=(
                    ("Definately", "Definitely"),
                    ("was", "were"),
                    ("dont", "doesn't"),
                    ("despues", "después"),
                    ("estaba", "estaban"),
                ),
                keywords=("minimal", "correct", "fix", "error", "spelling", "grammar", "agreement", "accent"),
                confidence=0.99
            ),
        ]
        
        # FN - FALSE NEGATIVES
        fn_guidelines = [
            GuidelineEntry(
                category="no_correction",
                label="FN",
                description="No suggestions in incorrect sentence",
                examples=(
                    ("She go to store", "She go to store"),
                    ("They has planned", "They has planned"),
                    ("He eat breakfast", "He eat breakfast"),
                ),
                keywords=("no", "correction", "missed", "error", "agreement", "tense", "uncorrected"),
                confidence=0.99
            ),
            GuidelineEntry(
                category="incomplete_correction",
                label="FN",
                description="Partial correction that misses some errors",
                examples=(
                    ("They was happy", "They were happy"),  # But missed other errors
                    ("students was preparing", "students were preparing"),  # But missed 'submit'
                ),
                keywords=("incomplete", "partial", "missed", "some", "errors", "remaining"),
                confidence=0.98
            ),
        ]
        
        # TN - TRUE NEGATIVES
        tn_guidelines = [
            GuidelineEntry(
                category="correct_pass",
                label="TN",
                description="No changes to already correct sentences",
                examples=(
                    ("The conference starts at 9 AM", "The conference starts at 9 AM"),
                    ("All users must agree", "All users must agree"),
                ),
                keywords=("correct", "pass", "no", "change", "already", "perfect"),
                confidence=0.99
            ),
        ]
        
        # SPANISH-SPECIFIC GUIDELINES
        spanish_guidelines = [
            GuidelineEntry(
                category="spanish_accents",
                label="TP",
                description="Spanish accent corrections",
                examples=(
                    ("mas", "más"),
                    ("como", "cómo"),
                    ("despues", "después"),
                    ("tambien", "también"),
                ),
                keywords=("spanish", "accent", "diacritic", "tilde", "acentuación"),
                confidence=0.99,
                language="es"
            ),
            GuidelineEntry(
                category="spanish_agreement",
                label="TP",
                description="Spanish gender/number agreement",
                examples=(
                    ("casa blanco", "casa blanca"),
                    ("libros roja", "libros rojos"),
                    ("niña alto", "niña alta"),
                ),
                keywords=("spanish", "agreement", "gender", "number", "concordancia", "género"),
                confidence=0.99,
                language="es"
            ),
            GuidelineEntry(
                category="spanish_subjunctive",
                label="TP",
                description="Spanish subjunctive mood corrections",
                examples=(
                    ("espero que vienes", "espero que vengas"),
                    ("quiero que haces", "quiero que hagas"),
                    ("dudo que es", "dudo que sea"),
                ),
                keywords=("spanish", "subjunctive", "mood", "subjuntivo", "modo"),
                confidence=0.98,
                language="es"
            ),
        ]
        
        # Combine all guidelines
        guidelines.extend(fp1_guidelines)
        guidelines.extend(fp2_guidelines)
        guidelines.extend(fp3_guidelines)
        guidelines.extend(tp_guidelines)
        guidelines.extend(fn_guidelines)
        guidelines.extend(tn_guidelines)
        guidelines.extend(spanish_guidelines)
        
        return guidelines
    
    def _build_keyword_index(self) -> Dict[str, List[GuidelineEntry]]:
        """Build keyword index for fast lookup."""
        index = {}
        
        for guideline in self.guidelines:
            for keyword in guideline.keywords:
                if keyword not in index:
                    index[keyword] = []
                index[keyword].append(guideline)
        
        return index
    
    def search_guidelines(self, before: str, after: str, language: str = "en") -> List[Tuple[GuidelineEntry, float]]:
        """Search for relevant guidelines based on edit context."""
        matches = []
        
        # Extract keywords from edit context
        context_keywords = self._extract_keywords(before, after)
        
        # Score guidelines based on keyword matches
        guideline_scores = {}
        
        for keyword in context_keywords:
            if keyword in self.keyword_index:
                for guideline in self.keyword_index[keyword]:
                    # Filter by language
                    if guideline.language != "en" and guideline.language != language:
                        continue
                    
                    if guideline not in guideline_scores:
                        guideline_scores[guideline] = 0
                    guideline_scores[guideline] += 1
        
        # Check for direct example matches
        for guideline in self.guidelines:
            if guideline.language != "en" and guideline.language != language:
                continue
            
            for example_before, example_after in guideline.examples:
                if (self._fuzzy_match(before, example_before) and 
                    self._fuzzy_match(after, example_after)):
                    if guideline not in guideline_scores:
                        guideline_scores[guideline] = 0
                    guideline_scores[guideline] += 10  # High score for direct matches
        
        # Convert to scored list
        for guideline, score in guideline_scores.items():
            confidence = guideline.confidence * min(1.0, score / 5.0)  # Scale by match quality
            matches.append((guideline, confidence))
        
        # Sort by confidence
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:3]  # Top 3 matches
    
    def _extract_keywords(self, before: str, after: str) -> List[str]:
        """Extract relevant keywords from edit context."""
        text = f"{before} {after}".lower()
        
        # Basic keyword extraction
        keywords = []
        
        # Check for specific patterns
        if re.search(r'\d+', text):
            keywords.append("numbers")
        
        if any(word in text for word in ["he", "she", "his", "her", "their", "them"]):
            keywords.append("pronoun")
        
        if any(word in text for word in ["the", "a", "an"]):
            keywords.append("article")
        
        if re.search(r'[áéíóúñü]', text):
            keywords.append("accent")
        
        if any(word in text for word in ["was", "were", "is", "are"]):
            keywords.append("agreement")
        
        if any(word in text for word in ["go", "goes", "went", "going"]):
            keywords.append("tense")
        
        if len(before.split()) != len(after.split()):
            keywords.append("structural")
        
        if before.lower() != after.lower() and len(set(before.lower().split()) & set(after.lower().split())) < len(before.split()):
            keywords.append("meaning")
        
        # Add common words as keywords
        for word in text.split():
            if len(word) > 3:
                keywords.append(word)
        
        return keywords
    
    def _fuzzy_match(self, text1: str, text2: str, threshold: float = 0.8) -> bool:
        """Check if two texts are similar enough."""
        if not text1 or not text2:
            return False
        
        # Simple fuzzy matching
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if len(words1) == 0 or len(words2) == 0:
            return text1.lower() == text2.lower()
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        similarity = intersection / union if union > 0 else 0
        return similarity >= threshold
    
    def get_label_guidance(self, before: str, after: str, language: str = "en") -> Optional[str]:
        """Get the most likely label based on guidelines."""
        matches = self.search_guidelines(before, after, language)
        
        if matches:
            best_match, confidence = matches[0]
            if confidence > 0.5:  # Confidence threshold
                return best_match.label
        
        return None
    
    def get_explanation(self, before: str, after: str, language: str = "en") -> str:
        """Get explanation for the edit based on guidelines."""
        matches = self.search_guidelines(before, after, language)
        
        if matches:
            best_match, confidence = matches[0]
            if confidence > 0.5:
                return f"{best_match.description} (confidence: {confidence:.2f})"
        
        return "No specific guideline match found"
    
    def save_guidelines(self, filepath: str):
        """Save guidelines database to JSON."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        data = {
            "metadata": {
                "total_guidelines": len(self.guidelines),
                "categories": len(set(g.category for g in self.guidelines)),
            },
            "guidelines": [
                {
                    "category": g.category,
                    "label": g.label,
                    "description": g.description,
                    "examples": list(g.examples),
                    "keywords": list(g.keywords),
                    "confidence": g.confidence,
                    "language": g.language,
                }
                for g in self.guidelines
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved {len(self.guidelines)} guidelines to {filepath}")


def main():
    """Test the annotation guidelines RAG."""
    print("🚀 Building Annotation Guidelines RAG")
    print("=" * 50)
    
    # Initialize RAG
    rag = AnnotationGuidelinesRAG()
    
    # Save guidelines
    os.makedirs("_experiments/final_agent/data", exist_ok=True)
    rag.save_guidelines("_experiments/final_agent/data/annotation_guidelines.json")
    
    # Test cases
    test_cases = [
        ("mas", "más", "es"),
        ("he go", "he goes", "en"),
        ("2-3/10", "3/10", "en"),
        ("doesn't", "does not", "en"),
        ("genetic", "genetically", "en"),
        ("", "just", "en"),
    ]
    
    print(f"\n🔍 TESTING GUIDELINES:")
    for before, after, lang in test_cases:
        label = rag.get_label_guidance(before, after, lang)
        explanation = rag.get_explanation(before, after, lang)
        print(f"  '{before}' → '{after}' ({lang}): {label} - {explanation}")
    
    print(f"\n🏆 ANNOTATION GUIDELINES RAG COMPLETE!")


if __name__ == "__main__":
    main()
