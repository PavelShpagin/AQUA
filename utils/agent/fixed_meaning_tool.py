#!/usr/bin/env python3
"""
Fixed Meaning Tool - Corrected entity detection for accurate FP1 classification.
"""

import os
import re
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class MeaningAnalysis:
    """Meaning change analysis result."""
    meaning_preserved: bool
    severity_score: int  # 0-4 scale
    change_type: str  # "none", "clarification", "alteration", "contradiction", "critical"
    critical_changes: List[str]
    confidence: float
    reasoning: str
    fp1_likelihood: float  # 0.0-1.0 probability this is FP1


class FixedMeaningTool:
    """Fixed meaning change detection tool with accurate pattern recognition."""
    
    def __init__(self, api_token: Optional[str] = None):
        self.api_token = api_token or os.getenv('OPENAI_API_KEY')
        
        # Fixed patterns for critical meaning changes
        self.critical_patterns = self._build_fixed_patterns()
    
    def _build_fixed_patterns(self) -> Dict[str, List[str]]:
        """Build FIXED patterns for meaning changes."""
        return {
            "number_changes": [
                r'\b\d+\b',  # Any number
                r'\b\d+-\d+\b',  # Number ranges like "2-3"
            ],
            "negation_changes": [
                r'\bno\b', r'\bni\b', r'\bnunca\b', r'\bnada\b', r'\bnadie\b'
            ],
            "temporal_changes": [
                r'\bayer\b', r'\bhoy\b', r'\bmañana\b', r'\bantes\b', r'\bdespués\b',
                r'\bnunca\b', r'\bsiempre\b', r'\ba veces\b'
            ],
            "modal_changes": [
                r'\btal vez\b', r'\bquizás\b', r'\bpuede\b', r'\bpodría\b',
                r'\bdebe\b', r'\bdebería\b', r'\bdefinitivamente\b', r'\bseguramente\b'
            ],
            "pronoun_changes": [
                r'\bme\b', r'\bte\b', r'\bse\b', r'\bnos\b', r'\bos\b',
                r'\ble\b', r'\bla\b', r'\blo\b', r'\bles\b', r'\blas\b', r'\blos\b'
            ]
        }
    
    def analyze_meaning_change(self, src: str, tgt: str, language: str, complexity: str = "auto") -> MeaningAnalysis:
        """Analyze meaning changes with FIXED logic."""
        
        # Initialize analysis
        critical_changes = []
        severity_score = 0
        fp1_likelihood = 0.0
        
        # 1. NUMBER/QUANTITY CHANGES (High FP1 risk)
        src_numbers = re.findall(r'\d+(?:-\d+)?', src)
        tgt_numbers = re.findall(r'\d+(?:-\d+)?', tgt)
        
        if src_numbers != tgt_numbers:
            critical_changes.append(f"number_change: {src_numbers} → {tgt_numbers}")
            severity_score += 3
            fp1_likelihood += 0.8
        
        # 2. PRONOUN CHANGES (High FP1 risk)
        pronouns = ['me', 'te', 'se', 'nos', 'os', 'le', 'la', 'lo', 'les', 'las', 'los']
        src_pronouns = [w for w in src.lower().split() if w in pronouns]
        tgt_pronouns = [w for w in tgt.lower().split() if w in pronouns]
        
        if src_pronouns != tgt_pronouns:
            critical_changes.append(f"pronoun_change: {src_pronouns} → {tgt_pronouns}")
            severity_score += 2
            fp1_likelihood += 0.6
        
        # 3. NEGATION CHANGES (Critical FP1)
        negations = ['no', 'ni', 'nunca', 'nada', 'nadie', 'ningún', 'ninguna']
        src_neg = sum(1 for w in src.lower().split() if w in negations)
        tgt_neg = sum(1 for w in tgt.lower().split() if w in negations)
        
        if src_neg != tgt_neg:
            critical_changes.append(f"negation_change: {src_neg} → {tgt_neg}")
            severity_score += 4
            fp1_likelihood += 0.9
        
        # 4. TEMPORAL CHANGES (Moderate FP1 risk)
        temporal_words = ['ayer', 'hoy', 'mañana', 'antes', 'después', 'nunca', 'siempre']
        src_temporal = [w for w in src.lower().split() if w in temporal_words]
        tgt_temporal = [w for w in tgt.lower().split() if w in temporal_words]
        
        if src_temporal != tgt_temporal:
            critical_changes.append(f"temporal_change: {src_temporal} → {tgt_temporal}")
            severity_score += 2
            fp1_likelihood += 0.5
        
        # 5. MODAL CHANGES (Moderate FP1 risk)
        modals = ['tal vez', 'quizás', 'puede', 'podría', 'debe', 'debería', 'definitivamente']
        src_modals = [modal for modal in modals if modal in src.lower()]
        tgt_modals = [modal for modal in modals if modal in tgt.lower()]
        
        if src_modals != tgt_modals:
            critical_changes.append(f"modal_change: {src_modals} → {tgt_modals}")
            severity_score += 1
            fp1_likelihood += 0.3
        
        # 6. ACCENT/SPELLING CHANGES (Should NOT be FP1)
        accent_pairs = [('mas', 'más'), ('si', 'sí'), ('tu', 'tú'), ('que', 'qué')]
        is_accent_correction = any(
            without in src.lower() and with_accent in tgt.lower()
            for without, with_accent in accent_pairs
        )
        
        if is_accent_correction:
            # Override - accent corrections are NOT meaning changes
            fp1_likelihood = max(0.0, fp1_likelihood - 0.5)
            severity_score = max(0, severity_score - 2)
            critical_changes.append("accent_correction_not_meaning_change")
        
        # 7. CONTRACTION CHANGES (Should NOT be FP1)
        is_contraction = ('a el' in src.lower() and 'al' in tgt.lower()) or \
                        ('de el' in src.lower() and 'del' in tgt.lower())
        
        if is_contraction:
            # Override - contractions are NOT meaning changes
            fp1_likelihood = max(0.0, fp1_likelihood - 0.5)
            severity_score = max(0, severity_score - 2)
            critical_changes.append("contraction_not_meaning_change")
        
        # Final assessment
        meaning_preserved = fp1_likelihood < 0.3
        
        if fp1_likelihood > 0.7:
            change_type = "critical"
        elif fp1_likelihood > 0.4:
            change_type = "alteration"
        elif fp1_likelihood > 0.2:
            change_type = "clarification"
        else:
            change_type = "none"
        
        confidence = 0.8 if len(critical_changes) > 0 else 0.9
        
        reasoning = f"Detected {len(critical_changes)} critical patterns. FP1 likelihood: {fp1_likelihood:.2f}"
        
        return MeaningAnalysis(
            meaning_preserved=meaning_preserved,
            severity_score=min(4, severity_score),
            change_type=change_type,
            critical_changes=critical_changes,
            confidence=confidence,
            reasoning=reasoning,
            fp1_likelihood=fp1_likelihood
        )


def main():
    """Test the fixed meaning tool."""
    print("🔧 TESTING FIXED MEANING TOOL")
    print("=" * 40)
    
    tool = FixedMeaningTool()
    
    test_cases = [
        # TP cases (should be LOW FP1)
        ("Necesito mas tiempo.", "Necesito más tiempo.", "TP"),
        ("Voy a el médico.", "Voy al médico.", "TP"),
        
        # FP1 cases (should be HIGH FP1)
        ("Tengo 2-3 horas.", "Tengo 3 horas.", "FP1"),
        ("Me dijo que venía.", "Te dijo que venía.", "FP1"),
        ("No me gusta.", "Me gusta.", "FP1"),
        
        # FP3 cases (should be LOW FP1)
        ("Te mando el archivo.", "Te envío el archivo.", "FP3"),
        ("Estoy contento.", "Estoy feliz.", "FP3"),
    ]
    
    correct = 0
    
    for i, (src, tgt, expected) in enumerate(test_cases, 1):
        analysis = tool.analyze_meaning_change(src, tgt, "Spanish")
        
        # Determine if classification matches expectation
        if expected == "FP1":
            correct_prediction = analysis.fp1_likelihood > 0.6
        else:  # TP or FP3
            correct_prediction = analysis.fp1_likelihood < 0.4
        
        if correct_prediction:
            correct += 1
        
        status = "✅" if correct_prediction else "❌"
        print(f"{status} Test {i} ({expected}): FP1 likelihood {analysis.fp1_likelihood:.2f}")
    
    accuracy = correct / len(test_cases)
    print(f"\n🏆 Fixed Meaning Tool Accuracy: {accuracy:.1%} ({correct}/{len(test_cases)})")


if __name__ == "__main__":
    main()




