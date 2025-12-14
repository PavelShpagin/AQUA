#!/usr/bin/env python3
"""
Advanced chunked grammar rulebooks with intelligent text splitting.
Each chunk represents a detailed rule with examples, counter-examples, and context.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re
from sentence_transformers import SentenceTransformer

@dataclass
class RuleChunk:
    """Represents a chunk of a grammar rule with detailed information."""
    chunk_id: str
    rule_name: str
    rule_category: str
    chunk_type: str  # "definition", "examples", "counter_examples", "usage", "context"
    content: str
    language: str
    source: str
    difficulty: str
    related_chunks: List[str]
    keywords: List[str]
    
    def to_document(self) -> str:
        """Convert chunk to document format for embedding."""
        return f"{self.rule_name} ({self.chunk_type}): {self.content}"
    
    def to_metadata(self) -> Dict[str, Any]:
        """Convert chunk to metadata for vector database."""
        return {
            "chunk_id": self.chunk_id,
            "rule_name": self.rule_name,
            "rule_category": self.rule_category,
            "chunk_type": self.chunk_type,
            "language": self.language,
            "source": self.source,
            "difficulty": self.difficulty,
            "keywords": ", ".join(self.keywords)  # Convert list to string
        }

class ChunkedRulebookProcessor:
    """Advanced processor for creating chunked grammar rulebooks."""
    
    def __init__(self, language: str = "english"):
        self.language = language
        self.embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
    def create_comprehensive_english_chunks(self) -> List[RuleChunk]:
        """Create comprehensive English grammar chunks with detailed examples."""
        chunks = []
        
        # Subject-Verb Agreement - Multiple chunks
        chunks.extend([
            RuleChunk(
                chunk_id="en_sva_001_def",
                rule_name="Subject-Verb Agreement",
                rule_category="grammar",
                chunk_type="definition",
                content="The subject and verb must agree in number (singular or plural) and person (first, second, third). This is a fundamental rule of English grammar that affects sentence correctness.",
                language="english",
                source="Oxford English Grammar",
                difficulty="beginner",
                related_chunks=["en_sva_001_examples", "en_sva_001_counter"],
                keywords=["subject", "verb", "agreement", "number", "person", "singular", "plural"]
            ),
            RuleChunk(
                chunk_id="en_sva_001_examples",
                rule_name="Subject-Verb Agreement",
                rule_category="grammar", 
                chunk_type="examples",
                content="Correct examples: 'She walks to school' (singular subject + singular verb), 'They walk to school' (plural subject + plural verb), 'The team is ready' (collective noun treated as singular), 'The students are studying' (plural subject + plural verb), 'Everyone has arrived' (indefinite pronoun + singular verb).",
                language="english",
                source="Oxford English Grammar",
                difficulty="beginner",
                related_chunks=["en_sva_001_def", "en_sva_001_counter"],
                keywords=["examples", "singular", "plural", "collective", "indefinite", "pronoun"]
            ),
            RuleChunk(
                chunk_id="en_sva_001_counter",
                rule_name="Subject-Verb Agreement",
                rule_category="grammar",
                chunk_type="counter_examples", 
                content="Common errors: 'She walk to school' (singular subject + plural verb), 'They walks to school' (plural subject + singular verb), 'The team are ready' (collective noun incorrectly treated as plural), 'The students is studying' (plural subject + singular verb).",
                language="english",
                source="Oxford English Grammar",
                difficulty="beginner",
                related_chunks=["en_sva_001_def", "en_sva_001_examples"],
                keywords=["errors", "common", "mistakes", "incorrect"]
            ),
            RuleChunk(
                chunk_id="en_sva_001_usage",
                rule_name="Subject-Verb Agreement",
                rule_category="grammar",
                chunk_type="usage",
                content="Usage notes: With compound subjects joined by 'and', use plural verbs ('John and Mary are here'). With subjects joined by 'or' or 'nor', the verb agrees with the nearest subject ('Neither John nor his friends are coming'). Collective nouns can be singular or plural depending on context.",
                language="english",
                source="Oxford English Grammar", 
                difficulty="intermediate",
                related_chunks=["en_sva_001_def", "en_sva_001_examples"],
                keywords=["compound", "subjects", "collective", "nouns", "context"]
            )
        ])
        
        # Pronoun Case - Multiple chunks
        chunks.extend([
            RuleChunk(
                chunk_id="en_pron_001_def",
                rule_name="Pronoun Case",
                rule_category="grammar",
                chunk_type="definition",
                content="Pronouns must be in the correct case: subjective (I, he, she, we, they), objective (me, him, her, us, them), or possessive (my, his, her, our, their). The case depends on the pronoun's function in the sentence.",
                language="english",
                source="Oxford English Grammar",
                difficulty="intermediate",
                related_chunks=["en_pron_001_examples", "en_pron_001_counter"],
                keywords=["pronoun", "case", "subjective", "objective", "possessive"]
            ),
            RuleChunk(
                chunk_id="en_pron_001_examples",
                rule_name="Pronoun Case",
                rule_category="grammar",
                chunk_type="examples",
                content="Correct examples: 'Between you and me' (objective case after preposition), 'She and I went shopping' (subjective case as compound subject), 'The teacher gave him and her the books' (objective case as indirect objects), 'It's mine' (possessive case).",
                language="english",
                source="Oxford English Grammar",
                difficulty="intermediate", 
                related_chunks=["en_pron_001_def", "en_pron_001_counter"],
                keywords=["preposition", "compound", "subject", "indirect", "object"]
            ),
            RuleChunk(
                chunk_id="en_pron_001_counter",
                rule_name="Pronoun Case",
                rule_category="grammar",
                chunk_type="counter_examples",
                content="Common errors: 'Between you and I' (incorrect subjective case after preposition), 'Me and her went shopping' (incorrect objective case as subject), 'The teacher gave he and she the books' (incorrect subjective case as objects).",
                language="english",
                source="Oxford English Grammar",
                difficulty="intermediate",
                related_chunks=["en_pron_001_def", "en_pron_001_examples"],
                keywords=["errors", "incorrect", "preposition", "subject", "object"]
            )
        ])
        
        # Past Participle Forms - Multiple chunks
        chunks.extend([
            RuleChunk(
                chunk_id="en_part_001_def",
                rule_name="Past Participle Forms",
                rule_category="grammar",
                chunk_type="definition",
                content="Past participles are used with auxiliary verbs (have, has, had) to form perfect tenses, and with 'be' verbs for passive voice. Regular verbs add -ed, but irregular verbs have unique forms that must be memorized.",
                language="english",
                source="Oxford English Grammar",
                difficulty="intermediate",
                related_chunks=["en_part_001_examples", "en_part_001_counter"],
                keywords=["past", "participle", "auxiliary", "perfect", "passive", "irregular"]
            ),
            RuleChunk(
                chunk_id="en_part_001_examples", 
                rule_name="Past Participle Forms",
                rule_category="grammar",
                chunk_type="examples",
                content="Correct examples: 'I have gone to the store' (gone, not went), 'She has written a letter' (written, not wrote), 'They have eaten dinner' (eaten, not ate), 'The book was written by Shakespeare' (passive voice with past participle).",
                language="english",
                source="Oxford English Grammar",
                difficulty="intermediate",
                related_chunks=["en_part_001_def", "en_part_001_counter"],
                keywords=["gone", "written", "eaten", "passive", "voice"]
            ),
            RuleChunk(
                chunk_id="en_part_001_counter",
                rule_name="Past Participle Forms", 
                rule_category="grammar",
                chunk_type="counter_examples",
                content="Common errors: 'I have went to the store' (incorrect past tense instead of past participle), 'She has wrote a letter' (incorrect past tense), 'They have ate dinner' (incorrect past tense). Remember: past tense ≠ past participle.",
                language="english",
                source="Oxford English Grammar",
                difficulty="intermediate",
                related_chunks=["en_part_001_def", "en_part_001_examples"],
                keywords=["went", "wrote", "ate", "past", "tense", "errors"]
            )
        ])
        
        return chunks
    
    def create_comprehensive_spanish_chunks(self) -> List[RuleChunk]:
        """Create comprehensive Spanish grammar chunks with detailed examples."""
        chunks = []
        
        # Concordancia de género y número - Multiple chunks
        chunks.extend([
            RuleChunk(
                chunk_id="es_conc_001_def",
                rule_name="Concordancia de género y número",
                rule_category="gramática",
                chunk_type="definition",
                content="Los artículos, adjetivos y sustantivos deben concordar en género (masculino/femenino) y número (singular/plural). Esta concordancia es obligatoria y afecta la corrección gramatical de la oración.",
                language="spanish",
                source="Real Academia Española",
                difficulty="básico",
                related_chunks=["es_conc_001_examples", "es_conc_001_counter"],
                keywords=["concordancia", "género", "número", "artículos", "adjetivos", "sustantivos"]
            ),
            RuleChunk(
                chunk_id="es_conc_001_examples",
                rule_name="Concordancia de género y número",
                rule_category="gramática",
                chunk_type="examples", 
                content="Ejemplos correctos: 'La casa blanca' (femenino singular), 'Los coches rojos' (masculino plural), 'Las niñas pequeñas' (femenino plural), 'El problema principal' (masculino singular - nota: problema es masculino), 'Una solución práctica' (femenino singular).",
                language="spanish",
                source="Real Academia Española",
                difficulty="básico",
                related_chunks=["es_conc_001_def", "es_conc_001_counter"],
                keywords=["ejemplos", "femenino", "masculino", "singular", "plural", "problema"]
            ),
            RuleChunk(
                chunk_id="es_conc_001_counter",
                rule_name="Concordancia de género y número",
                rule_category="gramática",
                chunk_type="counter_examples",
                content="Errores comunes: 'La casa blanco' (error de género), 'Los coche rojo' (error de número), 'La problema principal' (error: problema es masculino), 'Una análisis detallado' (error: análisis es masculino). Atención especial a sustantivos como 'problema', 'análisis', 'sistema'.",
                language="spanish",
                source="Real Academia Española", 
                difficulty="básico",
                related_chunks=["es_conc_001_def", "es_conc_001_examples"],
                keywords=["errores", "género", "número", "problema", "análisis", "sistema"]
            )
        ])
        
        # Acentuación - Multiple chunks
        chunks.extend([
            RuleChunk(
                chunk_id="es_acent_001_def",
                rule_name="Reglas de acentuación",
                rule_category="ortografía",
                chunk_type="definition",
                content="Las reglas de acentuación determinan cuándo una palabra lleva tilde. Se basan en la posición de la sílaba tónica: agudas (última sílaba), llanas (penúltima), esdrújulas (antepenúltima) y sobresdrújulas.",
                language="spanish",
                source="Real Academia Española",
                difficulty="intermedio",
                related_chunks=["es_acent_001_examples", "es_acent_001_counter"],
                keywords=["acentuación", "tilde", "agudas", "llanas", "esdrújulas", "sílaba", "tónica"]
            ),
            RuleChunk(
                chunk_id="es_acent_001_examples",
                rule_name="Reglas de acentuación",
                rule_category="ortografía",
                chunk_type="examples",
                content="Ejemplos correctos: 'análisis' (esdrújula, siempre lleva tilde), 'López' (aguda terminada en consonante que no es n/s), 'árboles' (llana terminada en consonante que no es n/s), 'participación' (aguda terminada en n), 'más' (monosílabo diacrítico vs. mas = pero).",
                language="spanish",
                source="Real Academia Española",
                difficulty="intermedio",
                related_chunks=["es_acent_001_def", "es_acent_001_counter"],
                keywords=["análisis", "López", "árboles", "participación", "más", "diacrítico"]
            ),
            RuleChunk(
                chunk_id="es_acent_001_counter",
                rule_name="Reglas de acentuación",
                rule_category="ortografía", 
                chunk_type="counter_examples",
                content="Errores frecuentes: 'analisis' (falta tilde en esdrújula), 'Lopez' (falta tilde en aguda), 'arboles' (falta tilde en llana), 'participacion' (falta tilde en aguda), 'mas' sin tilde cuando significa 'además' (debe ser 'más').",
                language="spanish",
                source="Real Academia Española",
                difficulty="intermedio",
                related_chunks=["es_acent_001_def", "es_acent_001_examples"],
                keywords=["errores", "tilde", "esdrújula", "aguda", "llana"]
            )
        ])
        
        return chunks
    
    def chunk_external_rulebook(self, text: str, language: str, source: str) -> List[RuleChunk]:
        """Intelligently chunk external grammar rulebooks."""
        chunks = []
        
        # Split by major sections (rules)
        rule_sections = re.split(r'\n\s*(?:Rule|Regla|Règle)\s*\d+', text)
        
        for i, section in enumerate(rule_sections[1:], 1):  # Skip first empty split
            if len(section.strip()) < 50:  # Skip very short sections
                continue
                
            # Extract rule name (first line or heading)
            lines = section.strip().split('\n')
            rule_name = lines[0].strip()[:100]  # First 100 chars as rule name
            
            # Split section into logical chunks
            subsections = self._split_section_intelligently(section)
            
            for j, (chunk_type, content) in enumerate(subsections):
                if len(content.strip()) < 20:  # Skip very short content
                    continue
                    
                chunk = RuleChunk(
                    chunk_id=f"{language[:2]}_ext_{i:03d}_{j:02d}",
                    rule_name=rule_name,
                    rule_category="external",
                    chunk_type=chunk_type,
                    content=content.strip(),
                    language=language,
                    source=source,
                    difficulty="unknown",
                    related_chunks=[],
                    keywords=self._extract_keywords(content, language)
                )
                chunks.append(chunk)
        
        return chunks
    
    def _split_section_intelligently(self, section: str) -> List[Tuple[str, str]]:
        """Split a rule section into logical chunks."""
        subsections = []
        
        # Look for common patterns
        patterns = {
            "examples": r"(?:Examples?|Ejemplos?|Exemples?):\s*(.*?)(?=\n\s*(?:[A-Z]|$))",
            "counter_examples": r"(?:Incorrect|Wrong|Errores?|Incorrecto):\s*(.*?)(?=\n\s*(?:[A-Z]|$))", 
            "usage": r"(?:Usage|Note|Nota|Uso):\s*(.*?)(?=\n\s*(?:[A-Z]|$))",
            "definition": r"^(.*?)(?=(?:Examples?|Ejemplos?|Incorrect|Usage|Note))"
        }
        
        for chunk_type, pattern in patterns.items():
            matches = re.findall(pattern, section, re.MULTILINE | re.DOTALL | re.IGNORECASE)
            for match in matches:
                if len(match.strip()) > 20:
                    subsections.append((chunk_type, match.strip()))
        
        # If no patterns found, split by paragraphs
        if not subsections:
            paragraphs = [p.strip() for p in section.split('\n\n') if len(p.strip()) > 20]
            for i, para in enumerate(paragraphs):
                chunk_type = "definition" if i == 0 else "content"
                subsections.append((chunk_type, para))
        
        return subsections
    
    def _extract_keywords(self, text: str, language: str) -> List[str]:
        """Extract keywords from text content."""
        # Simple keyword extraction - can be enhanced with NLP
        words = re.findall(r'\b\w{3,}\b', text.lower())
        
        # Language-specific stop words (simplified)
        stop_words = {
            "english": {"the", "and", "are", "for", "with", "this", "that", "have", "been"},
            "spanish": {"los", "las", "una", "para", "con", "que", "por", "del", "son"},
            "french": {"les", "des", "une", "pour", "avec", "que", "par", "dans"}
        }
        
        lang_stops = stop_words.get(language, set())
        keywords = [w for w in set(words) if w not in lang_stops and len(w) > 3]
        
        return keywords[:10]  # Return top 10 keywords

if __name__ == "__main__":
    processor = ChunkedRulebookProcessor("english")
    
    # Test English chunks
    en_chunks = processor.create_comprehensive_english_chunks()
    print(f"Created {len(en_chunks)} English chunks")
    
    # Test Spanish chunks  
    es_chunks = processor.create_comprehensive_spanish_chunks()
    print(f"Created {len(es_chunks)} Spanish chunks")
    
    # Show sample chunk
    if en_chunks:
        sample = en_chunks[0]
        print(f"\nSample chunk: {sample.chunk_id}")
        print(f"Type: {sample.chunk_type}")
        print(f"Content: {sample.content[:100]}...")
