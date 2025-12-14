#!/usr/bin/env python3
"""
Spanish Grammar Checker for FP2 Detection

Detects grammar errors introduced by corrections (FP2 cases).
Focused on common Spanish grammar patterns that our agent is missing.
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class GrammarError:
    """Detected grammar error."""
    error_type: str
    description: str
    confidence: float
    before: str
    after: str


class SpanishGrammarChecker:
    """Lightweight Spanish grammar checker for FP2 detection."""
    
    def __init__(self):
        """Initialize Spanish grammar patterns."""
        # Subject-verb agreement patterns
        self.verb_patterns = {
            # Singular subjects with plural verbs (ERROR)
            'singular_subject_plural_verb': [
                (r'\b(el|la|este|esta|ese|esa|aquel|aquella)\s+\w+\s+(han|están|son|tienen|van|vienen|funcionan)\b', 
                 'Singular subject with plural verb'),
                (r'\b(él|ella|usted)\s+(han|están|son|tienen|van|vienen|funcionan)\b',
                 'Singular pronoun with plural verb'),
            ],
            # Plural subjects with singular verbs (ERROR)
            'plural_subject_singular_verb': [
                (r'\b(los|las|estos|estas|esos|esas|aquellos|aquellas)\s+\w+\s+(ha|está|es|tiene|va|viene|juega|salió|funciona)\b',
                 'Plural subject with singular verb'),
                (r'\b(ellos|ellas|ustedes)\s+(ha|está|es|tiene|va|viene|juega|salió|funciona)\b',
                 'Plural pronoun with singular verb'),
                (r'\b\w+\s+y\s+\w+\s+(ha|está|es|tiene|va|viene|juega|salió|funciona)\b',
                 'Compound subject with singular verb'),
            ],
            # Person changes (ERROR)
            'person_disagreement': [
                (r'\b(me|te|se|nos|os)\s+(dijiste|enviarás|vamos|van)\b',
                 'Person disagreement in verb'),
                (r'\b(yo|tú|él|ella|nosotros|vosotros|ellos|ellas)\s+\w*(é|ás|á|emos|éis|án)\b',
                 'Wrong person ending'),
            ]
        }
        
        # Gender agreement patterns
        self.gender_patterns = {
            # Masculine noun with feminine adjective (ERROR)
            'masc_noun_fem_adj': [
                (r'\b(el|un|este|ese|aquel)\s+\w+\s+(blanca|negra|roja|azula|alta|baja|guapa|pequeña|grande)\b',
                 'Masculine noun with feminine adjective'),
            ],
            # Feminine noun with masculine adjective (ERROR)
            'fem_noun_masc_adj': [
                (r'\b(la|una|esta|esa|aquella)\s+\w+\s+(blanco|negro|rojo|azul|alto|bajo|guapo|pequeño)\b',
                 'Feminine noun with masculine adjective'),
            ],
            # Quantifier gender disagreement (ERROR)
            'quantifier_gender_error': [
                (r'\b(cuánto|mucho|poco|todo)\s+(personas|mujeres|niñas|casas)\b',
                 'Masculine quantifier with feminine noun'),
                (r'\b(cuánta|mucha|poca|toda)\s+(hombres|niños|coches|problemas)\b',
                 'Feminine quantifier with masculine noun'),
            ]
        }
        
        # Accent/diacritic errors
        self.accent_patterns = {
            # Missing accents on interrogatives/exclamatives
            'missing_interrogative_accent': [
                (r'¿(que|quien|cuando|donde|como|cuanto|cual)\b', 'Missing accent on interrogative'),
                (r'¡(que|quien|cuando|donde|como|cuanto|cual)\b', 'Missing accent on exclamative'),
            ],
            # Missing accents on pronouns
            'missing_pronoun_accent': [
                (r'\ba\s+el\b', 'Missing accent on pronoun "él"'),
                (r'\bse\b(?=\s+[a-z]+)', 'Possible missing accent on "sé"'),
            ],
            # Wrong stress patterns
            'wrong_stress': [
                (r'\bpractica\b', 'Wrong stress on "práctica"'),
                (r'\bpublico\b', 'Wrong stress on "público"'),
                (r'\bgusto\b', 'Wrong stress on "gustó"'),
            ],
            # Missing accents on past tense (more specific)
            'missing_past_accent': [
                (r'\b(gusto|hablo|trabajo|estudio|cambio)\b', 'Missing accent on past tense verb'),
            ]
        }
        
        # Spelling and other errors
        self.spelling_patterns = {
            # Common spelling errors
            'spelling_errors': [
                (r'\bay\b', 'Spelling error: "ay" should be "hay"'),
                (r'\bva\s+haber\b', 'Missing preposition: should be "va a haber"'),
            ]
        }
    
    def check_grammar_errors(self, text: str) -> List[GrammarError]:
        """Check for grammar errors in Spanish text."""
        errors = []
        text_lower = text.lower()
        
        # Check verb agreement
        for error_type, patterns in self.verb_patterns.items():
            for pattern, description in patterns:
                matches = re.finditer(pattern, text_lower)
                for match in matches:
                    errors.append(GrammarError(
                        error_type=error_type,
                        description=description,
                        confidence=0.95,  # Higher confidence for verb patterns
                        before=match.group(),
                        after=""
                    ))
        
        # Check gender agreement
        for error_type, patterns in self.gender_patterns.items():
            for pattern, description in patterns:
                matches = re.finditer(pattern, text_lower)
                for match in matches:
                    errors.append(GrammarError(
                        error_type=error_type,
                        description=description,
                        confidence=0.85,
                        before=match.group(),
                        after=""
                    ))
        
        # Check accent errors
        for error_type, patterns in self.accent_patterns.items():
            for pattern, description in patterns:
                matches = re.finditer(pattern, text_lower)
                for match in matches:
                    errors.append(GrammarError(
                        error_type=error_type,
                        description=description,
                        confidence=0.7,  # Lower confidence for accent patterns
                        before=match.group(),
                        after=""
                    ))
        
        # Check spelling errors
        for error_type, patterns in self.spelling_patterns.items():
            for pattern, description in patterns:
                matches = re.finditer(pattern, text_lower)
                for match in matches:
                    errors.append(GrammarError(
                        error_type=error_type,
                        description=description,
                        confidence=0.85,
                        before=match.group(),
                        after=""
                    ))
        
        return errors
    
    def analyze_correction(self, src: str, tgt: str) -> Dict[str, any]:
        """Analyze if a correction introduces grammar errors (FP2)."""
        src_errors = self.check_grammar_errors(src)
        tgt_errors = self.check_grammar_errors(tgt)
        
        # Check if correction introduced new errors
        new_errors = []
        for tgt_error in tgt_errors:
            # Check if this error wasn't in the source
            found_in_src = any(
                src_error.error_type == tgt_error.error_type and 
                src_error.before == tgt_error.before 
                for src_error in src_errors
            )
            if not found_in_src:
                new_errors.append(tgt_error)
        
        # Calculate FP2 likelihood
        fp2_likelihood = 0.0
        if new_errors:
            fp2_likelihood = max(error.confidence for error in new_errors)
        
        return {
            'introduces_errors': len(new_errors) > 0,
            'new_errors': new_errors,
            'fp2_likelihood': fp2_likelihood,
            'error_types': [error.error_type for error in new_errors],
            'descriptions': [error.description for error in new_errors]
        }


def main():
    """Test the Spanish grammar checker."""
    checker = SpanishGrammarChecker()
    
    # Test FP2 cases (should detect errors)
    test_cases = [
        ('El equipo ha sido notificado.', 'El equipo han sido notificado.'),
        ('Los niños juegan en el patio.', 'Los niños juega en el patio.'),
        ('¿Dónde vives?', '¿Donde vives?'),
        ('Se lo di a él.', 'Se lo di a el.'),
        ('María y Ana salieron temprano.', 'María y Ana salió temprano.'),
        # New test cases
        ('¿Cuántas personas asistirán?', '¿Cuánto personas asistirán?'),
        ('La solución funciona bien.', 'La solución funcionan bien.'),
        ('No hay nadie en casa.', 'No ay nadie en casa.'),
        ('Me gustó la película.', 'Me gusto la película.'),
        ('El lunes va a haber una reunión.', 'El lunes va haber una reunión.'),
    ]
    
    print("Testing Spanish Grammar Checker:")
    for src, tgt in test_cases:
        result = checker.analyze_correction(src, tgt)
        print(f"\n{src} → {tgt}")
        print(f"  Introduces errors: {result['introduces_errors']}")
        print(f"  FP2 likelihood: {result['fp2_likelihood']:.2f}")
        if result['new_errors']:
            for error in result['new_errors']:
                print(f"    - {error.description}: {error.before}")


if __name__ == "__main__":
    main()

Spanish Grammar Checker for FP2 Detection

Detects grammar errors introduced by corrections (FP2 cases).
Focused on common Spanish grammar patterns that our agent is missing.
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class GrammarError:
    """Detected grammar error."""
    error_type: str
    description: str
    confidence: float
    before: str
    after: str


class SpanishGrammarChecker:
    """Lightweight Spanish grammar checker for FP2 detection."""
    
    def __init__(self):
        """Initialize Spanish grammar patterns."""
        # Subject-verb agreement patterns
        self.verb_patterns = {
            # Singular subjects with plural verbs (ERROR)
            'singular_subject_plural_verb': [
                (r'\b(el|la|este|esta|ese|esa|aquel|aquella)\s+\w+\s+(han|están|son|tienen|van|vienen|funcionan)\b', 
                 'Singular subject with plural verb'),
                (r'\b(él|ella|usted)\s+(han|están|son|tienen|van|vienen|funcionan)\b',
                 'Singular pronoun with plural verb'),
            ],
            # Plural subjects with singular verbs (ERROR)
            'plural_subject_singular_verb': [
                (r'\b(los|las|estos|estas|esos|esas|aquellos|aquellas)\s+\w+\s+(ha|está|es|tiene|va|viene|juega|salió|funciona)\b',
                 'Plural subject with singular verb'),
                (r'\b(ellos|ellas|ustedes)\s+(ha|está|es|tiene|va|viene|juega|salió|funciona)\b',
                 'Plural pronoun with singular verb'),
                (r'\b\w+\s+y\s+\w+\s+(ha|está|es|tiene|va|viene|juega|salió|funciona)\b',
                 'Compound subject with singular verb'),
            ],
            # Person changes (ERROR)
            'person_disagreement': [
                (r'\b(me|te|se|nos|os)\s+(dijiste|enviarás|vamos|van)\b',
                 'Person disagreement in verb'),
                (r'\b(yo|tú|él|ella|nosotros|vosotros|ellos|ellas)\s+\w*(é|ás|á|emos|éis|án)\b',
                 'Wrong person ending'),
            ]
        }
        
        # Gender agreement patterns
        self.gender_patterns = {
            # Masculine noun with feminine adjective (ERROR)
            'masc_noun_fem_adj': [
                (r'\b(el|un|este|ese|aquel)\s+\w+\s+(blanca|negra|roja|azula|alta|baja|guapa|pequeña|grande)\b',
                 'Masculine noun with feminine adjective'),
            ],
            # Feminine noun with masculine adjective (ERROR)
            'fem_noun_masc_adj': [
                (r'\b(la|una|esta|esa|aquella)\s+\w+\s+(blanco|negro|rojo|azul|alto|bajo|guapo|pequeño)\b',
                 'Feminine noun with masculine adjective'),
            ],
            # Quantifier gender disagreement (ERROR)
            'quantifier_gender_error': [
                (r'\b(cuánto|mucho|poco|todo)\s+(personas|mujeres|niñas|casas)\b',
                 'Masculine quantifier with feminine noun'),
                (r'\b(cuánta|mucha|poca|toda)\s+(hombres|niños|coches|problemas)\b',
                 'Feminine quantifier with masculine noun'),
            ]
        }
        
        # Accent/diacritic errors
        self.accent_patterns = {
            # Missing accents on interrogatives/exclamatives
            'missing_interrogative_accent': [
                (r'¿(que|quien|cuando|donde|como|cuanto|cual)\b', 'Missing accent on interrogative'),
                (r'¡(que|quien|cuando|donde|como|cuanto|cual)\b', 'Missing accent on exclamative'),
            ],
            # Missing accents on pronouns
            'missing_pronoun_accent': [
                (r'\ba\s+el\b', 'Missing accent on pronoun "él"'),
                (r'\bse\b(?=\s+[a-z]+)', 'Possible missing accent on "sé"'),
            ],
            # Wrong stress patterns
            'wrong_stress': [
                (r'\bpractica\b', 'Wrong stress on "práctica"'),
                (r'\bpublico\b', 'Wrong stress on "público"'),
                (r'\bgusto\b', 'Wrong stress on "gustó"'),
            ],
            # Missing accents on past tense (more specific)
            'missing_past_accent': [
                (r'\b(gusto|hablo|trabajo|estudio|cambio)\b', 'Missing accent on past tense verb'),
            ]
        }
        
        # Spelling and other errors
        self.spelling_patterns = {
            # Common spelling errors
            'spelling_errors': [
                (r'\bay\b', 'Spelling error: "ay" should be "hay"'),
                (r'\bva\s+haber\b', 'Missing preposition: should be "va a haber"'),
            ]
        }
    
    def check_grammar_errors(self, text: str) -> List[GrammarError]:
        """Check for grammar errors in Spanish text."""
        errors = []
        text_lower = text.lower()
        
        # Check verb agreement
        for error_type, patterns in self.verb_patterns.items():
            for pattern, description in patterns:
                matches = re.finditer(pattern, text_lower)
                for match in matches:
                    errors.append(GrammarError(
                        error_type=error_type,
                        description=description,
                        confidence=0.95,  # Higher confidence for verb patterns
                        before=match.group(),
                        after=""
                    ))
        
        # Check gender agreement
        for error_type, patterns in self.gender_patterns.items():
            for pattern, description in patterns:
                matches = re.finditer(pattern, text_lower)
                for match in matches:
                    errors.append(GrammarError(
                        error_type=error_type,
                        description=description,
                        confidence=0.85,
                        before=match.group(),
                        after=""
                    ))
        
        # Check accent errors
        for error_type, patterns in self.accent_patterns.items():
            for pattern, description in patterns:
                matches = re.finditer(pattern, text_lower)
                for match in matches:
                    errors.append(GrammarError(
                        error_type=error_type,
                        description=description,
                        confidence=0.7,  # Lower confidence for accent patterns
                        before=match.group(),
                        after=""
                    ))
        
        # Check spelling errors
        for error_type, patterns in self.spelling_patterns.items():
            for pattern, description in patterns:
                matches = re.finditer(pattern, text_lower)
                for match in matches:
                    errors.append(GrammarError(
                        error_type=error_type,
                        description=description,
                        confidence=0.85,
                        before=match.group(),
                        after=""
                    ))
        
        return errors
    
    def analyze_correction(self, src: str, tgt: str) -> Dict[str, any]:
        """Analyze if a correction introduces grammar errors (FP2)."""
        src_errors = self.check_grammar_errors(src)
        tgt_errors = self.check_grammar_errors(tgt)
        
        # Check if correction introduced new errors
        new_errors = []
        for tgt_error in tgt_errors:
            # Check if this error wasn't in the source
            found_in_src = any(
                src_error.error_type == tgt_error.error_type and 
                src_error.before == tgt_error.before 
                for src_error in src_errors
            )
            if not found_in_src:
                new_errors.append(tgt_error)
        
        # Calculate FP2 likelihood
        fp2_likelihood = 0.0
        if new_errors:
            fp2_likelihood = max(error.confidence for error in new_errors)
        
        return {
            'introduces_errors': len(new_errors) > 0,
            'new_errors': new_errors,
            'fp2_likelihood': fp2_likelihood,
            'error_types': [error.error_type for error in new_errors],
            'descriptions': [error.description for error in new_errors]
        }


def main():
    """Test the Spanish grammar checker."""
    checker = SpanishGrammarChecker()
    
    # Test FP2 cases (should detect errors)
    test_cases = [
        ('El equipo ha sido notificado.', 'El equipo han sido notificado.'),
        ('Los niños juegan en el patio.', 'Los niños juega en el patio.'),
        ('¿Dónde vives?', '¿Donde vives?'),
        ('Se lo di a él.', 'Se lo di a el.'),
        ('María y Ana salieron temprano.', 'María y Ana salió temprano.'),
        # New test cases
        ('¿Cuántas personas asistirán?', '¿Cuánto personas asistirán?'),
        ('La solución funciona bien.', 'La solución funcionan bien.'),
        ('No hay nadie en casa.', 'No ay nadie en casa.'),
        ('Me gustó la película.', 'Me gusto la película.'),
        ('El lunes va a haber una reunión.', 'El lunes va haber una reunión.'),
    ]
    
    print("Testing Spanish Grammar Checker:")
    for src, tgt in test_cases:
        result = checker.analyze_correction(src, tgt)
        print(f"\n{src} → {tgt}")
        print(f"  Introduces errors: {result['introduces_errors']}")
        print(f"  FP2 likelihood: {result['fp2_likelihood']:.2f}")
        if result['new_errors']:
            for error in result['new_errors']:
                print(f"    - {error.description}: {error.before}")


if __name__ == "__main__":
    main()















