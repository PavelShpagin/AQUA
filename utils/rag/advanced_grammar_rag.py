#!/usr/bin/env python3
"""
Advanced Grammar RAG System with Multiple Backends and Comprehensive Rule Sources.

Supports:
- ChromaDB (current)
- FAISS (fast similarity search)
- Pinecone (cloud vector database)
- Multiple embedding models
- Comprehensive grammar rulebooks for Spanish and other languages
"""

import os
import json
import yaml
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import logging

# Vector database backends
try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    import faiss
    import numpy as np
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    import pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

# Embedding models
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


@dataclass
class GrammarRule:
    """Structured grammar rule with comprehensive metadata."""
    rule_id: str
    rule_name: str
    description: str
    examples: List[str]
    counter_examples: List[str]
    keywords: List[str]
    category: str  # grammar, syntax, morphology, orthography, punctuation
    subcategory: str  # specific area like "verb_conjugation", "article_usage"
    severity: str  # critical, high, medium, low
    language: str
    source: str  # reference source (e.g., "RAE", "Nueva Gramática")
    difficulty_level: str  # beginner, intermediate, advanced
    related_rules: List[str]  # IDs of related rules
    tags: List[str]  # additional searchable tags


class EmbeddingProvider:
    """Abstract base for embedding providers."""
    
    def embed_text(self, text: str) -> List[float]:
        raise NotImplementedError
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError


class SentenceTransformerEmbedding(EmbeddingProvider):
    """SentenceTransformer embedding provider."""
    
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers not available")
        self.model = SentenceTransformer(model_name)
    
    def embed_text(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()


class OpenAIEmbedding(EmbeddingProvider):
    """OpenAI embedding provider."""
    
    def __init__(self, model: str = "text-embedding-3-small", api_key: Optional[str] = None):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai not available")
        self.client = openai.OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
        self.model = model
    
    def embed_text(self, text: str) -> List[float]:
        response = self.client.embeddings.create(input=text, model=self.model)
        return response.data[0].embedding
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(input=texts, model=self.model)
        return [item.embedding for item in response.data]


class VectorDatabase:
    """Abstract base for vector databases."""
    
    def add_rules(self, rules: List[GrammarRule], embeddings: List[List[float]]):
        raise NotImplementedError
    
    def search(self, query_embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        raise NotImplementedError
    
    def delete_all(self):
        raise NotImplementedError


class ChromaDBDatabase(VectorDatabase):
    """ChromaDB implementation."""
    
    def __init__(self, collection_name: str, persist_directory: str = "data/rag/chroma_advanced"):
        if not CHROMADB_AVAILABLE:
            raise ImportError("chromadb not available")
        
        self.client = chromadb.PersistentClient(
            path=persist_directory, 
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection_name = collection_name
        
        # Create collection with cosine similarity
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
    
    def add_rules(self, rules: List[GrammarRule], embeddings: List[List[float]]):
        documents = []
        metadatas = []
        ids = []
        
        for rule, embedding in zip(rules, embeddings):
            # Create comprehensive searchable document
            doc_text = f"{rule.rule_name}: {rule.description}"
            if rule.examples:
                doc_text += f" Examples: {'; '.join(rule.examples)}"
            if rule.keywords:
                doc_text += f" Keywords: {', '.join(rule.keywords)}"
            
            documents.append(doc_text)
            metadatas.append({
                'rule_id': rule.rule_id,
                'rule_name': rule.rule_name,
                'category': rule.category,
                'subcategory': rule.subcategory,
                'severity': rule.severity,
                'language': rule.language,
                'source': rule.source,
                'difficulty_level': rule.difficulty_level,
                'tags': ','.join(rule.tags)
            })
            ids.append(rule.rule_id)
        
        # Add with pre-computed embeddings
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings
        )
    
    def search(self, query_embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                'rule_id': results['metadatas'][0][i]['rule_id'],
                'rule_name': results['metadatas'][0][i]['rule_name'],
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else 0
            })
        
        return formatted_results
    
    def delete_all(self):
        try:
            self.collection.delete(where={})
        except:
            pass


class FAISSDatabase(VectorDatabase):
    """FAISS implementation for fast similarity search."""
    
    def __init__(self, dimension: int = 384):
        if not FAISS_AVAILABLE:
            raise ImportError("faiss not available")
        
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.rules = []  # Store rule metadata
    
    def add_rules(self, rules: List[GrammarRule], embeddings: List[List[float]]):
        # Normalize embeddings for cosine similarity
        embeddings_array = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings_array)
        
        self.index.add(embeddings_array)
        self.rules.extend(rules)
    
    def search(self, query_embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        # Normalize query embedding
        query_array = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_array)
        
        distances, indices = self.index.search(query_array, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.rules):
                rule = self.rules[idx]
                results.append({
                    'rule_id': rule.rule_id,
                    'rule_name': rule.rule_name,
                    'document': f"{rule.rule_name}: {rule.description}",
                    'metadata': {
                        'category': rule.category,
                        'severity': rule.severity,
                        'language': rule.language
                    },
                    'distance': float(distances[0][i])
                })
        
        return results
    
    def delete_all(self):
        self.index.reset()
        self.rules.clear()


class AdvancedGrammarRAG:
    """Advanced Grammar RAG system with multiple backends and comprehensive rules."""
    
    def __init__(self, 
                 backend: str = "chromadb",
                 embedding_provider: str = "openai",
                 language: str = "spanish",
                 **kwargs):
        
        self.language = language
        self.logger = logging.getLogger(__name__)
        
        # Initialize embedding provider
        if embedding_provider == "sentence_transformer":
            model_name = kwargs.get('embedding_model', 'paraphrase-multilingual-MiniLM-L12-v2')
            self.embedder = SentenceTransformerEmbedding(model_name)
        elif embedding_provider == "openai":
            self.embedder = OpenAIEmbedding(**kwargs)
        else:
            raise ValueError(f"Unknown embedding provider: {embedding_provider}")
        
        # Initialize vector database
        if backend == "chromadb":
            collection_name = f"advanced_grammar_{language}"
            self.db = ChromaDBDatabase(collection_name, **kwargs)
        elif backend == "faiss":
            dimension = kwargs.get('dimension', 384)
            self.db = FAISSDatabase(dimension)
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    def load_comprehensive_spanish_rules(self) -> List[GrammarRule]:
        """Load comprehensive Spanish grammar rules from multiple sources."""
        
        rules = []
        
        # Real Academia Española (RAE) inspired rules
        rae_rules = [
            GrammarRule(
                rule_id="es_rae_001",
                rule_name="Concordancia de género y número",
                description="Los artículos, adjetivos y participios deben concordar en género y número con el sustantivo al que acompañan o se refieren.",
                examples=[
                    "la casa blanca (fem. sing.)",
                    "los libros interesantes (masc. pl.)",
                    "una mujer trabajadora (fem. sing.)",
                    "unos niños pequeños (masc. pl.)"
                ],
                counter_examples=[
                    "la casa blanco (incorrecto)",
                    "los libro interesante (incorrecto)",
                    "una mujer trabajador (incorrecto)"
                ],
                keywords=["concordancia", "género", "número", "artículo", "adjetivo", "sustantivo"],
                category="grammar",
                subcategory="agreement",
                severity="critical",
                language="spanish",
                source="RAE - Nueva Gramática de la Lengua Española",
                difficulty_level="beginner",
                related_rules=["es_rae_002", "es_rae_003"],
                tags=["concordancia", "morfología", "determinantes"]
            ),
            
            GrammarRule(
                rule_id="es_rae_002",
                rule_name="Uso de ser y estar",
                description="'Ser' se usa para características permanentes, identidad y definiciones. 'Estar' se usa para estados temporales, ubicación y condiciones.",
                examples=[
                    "María es médica (profesión permanente)",
                    "El libro es interesante (característica inherente)",
                    "María está enferma (estado temporal)",
                    "El libro está en la mesa (ubicación)"
                ],
                counter_examples=[
                    "María está médica (incorrecto para profesión)",
                    "El libro está interesante (incorrecto para característica)"
                ],
                keywords=["ser", "estar", "copulativo", "atributo", "predicado"],
                category="grammar",
                subcategory="verb_usage",
                severity="high",
                language="spanish",
                source="RAE - Nueva Gramática de la Lengua Española",
                difficulty_level="intermediate",
                related_rules=["es_rae_001", "es_rae_010"],
                tags=["verbos", "copulativos", "predicado"]
            ),
            
            GrammarRule(
                rule_id="es_rae_003",
                rule_name="Acentuación de palabras agudas",
                description="Las palabras agudas (acentuadas en la última sílaba) llevan tilde cuando terminan en vocal, -n o -s.",
                examples=[
                    "café (termina en vocal)",
                    "ratón (termina en -n)",
                    "compás (termina en -s)",
                    "sofá, bebé, colibrí"
                ],
                counter_examples=[
                    "reloj (no lleva tilde, termina en consonante que no es -n o -s)",
                    "papel (no lleva tilde, termina en -l)"
                ],
                keywords=["acentuación", "tilde", "aguda", "oxítona", "acento"],
                category="orthography",
                subcategory="accentuation",
                severity="high",
                language="spanish",
                source="RAE - Ortografía de la Lengua Española",
                difficulty_level="intermediate",
                related_rules=["es_rae_004", "es_rae_005"],
                tags=["ortografía", "acentos", "prosodia"]
            ),
            
            GrammarRule(
                rule_id="es_rae_004",
                rule_name="Acentuación de palabras graves o llanas",
                description="Las palabras graves (acentuadas en la penúltima sílaba) llevan tilde cuando NO terminan en vocal, -n o -s.",
                examples=[
                    "árbol (termina en -l)",
                    "cárcel (termina en -l)",
                    "fácil (termina en -l)",
                    "mártir (termina en -r)"
                ],
                counter_examples=[
                    "casa (no lleva tilde, termina en vocal)",
                    "examen (no lleva tilde, termina en -n)"
                ],
                keywords=["acentuación", "tilde", "grave", "llana", "paroxítona"],
                category="orthography",
                subcategory="accentuation",
                severity="high",
                language="spanish",
                source="RAE - Ortografía de la Lengua Española",
                difficulty_level="intermediate",
                related_rules=["es_rae_003", "es_rae_005"],
                tags=["ortografía", "acentos", "prosodia"]
            ),
            
            GrammarRule(
                rule_id="es_rae_005",
                rule_name="Acentuación de palabras esdrújulas",
                description="Las palabras esdrújulas (acentuadas en la antepenúltima sílaba) SIEMPRE llevan tilde.",
                examples=[
                    "médico", "rápido", "último", "música",
                    "teléfono", "matemáticas", "gramática"
                ],
                counter_examples=[
                    "medico (sin tilde es incorrecto)",
                    "rapido (sin tilde es incorrecto)"
                ],
                keywords=["acentuación", "tilde", "esdrújula", "proparoxítona"],
                category="orthography",
                subcategory="accentuation",
                severity="critical",
                language="spanish",
                source="RAE - Ortografía de la Lengua Española",
                difficulty_level="beginner",
                related_rules=["es_rae_003", "es_rae_004"],
                tags=["ortografía", "acentos", "prosodia"]
            ),
            
            GrammarRule(
                rule_id="es_rae_006",
                rule_name="Uso del subjuntivo en oraciones subordinadas",
                description="El subjuntivo se usa en oraciones subordinadas que expresan duda, deseo, emoción, mandato o irrealidad.",
                examples=[
                    "Espero que vengas (deseo)",
                    "Dudo que sea verdad (duda)",
                    "Me alegra que estés bien (emoción)",
                    "Te pido que me ayudes (mandato)"
                ],
                counter_examples=[
                    "Espero que vienes (incorrecto, debe ser subjuntivo)",
                    "Dudo que es verdad (incorrecto, debe ser subjuntivo)"
                ],
                keywords=["subjuntivo", "subordinada", "modo", "irrealis", "duda", "deseo"],
                category="grammar",
                subcategory="verb_mood",
                severity="high",
                language="spanish",
                source="RAE - Nueva Gramática de la Lengua Española",
                difficulty_level="advanced",
                related_rules=["es_rae_007", "es_rae_008"],
                tags=["verbos", "modos", "sintaxis"]
            ),
            
            GrammarRule(
                rule_id="es_rae_007",
                rule_name="Concordancia temporal en subjuntivo",
                description="La concordancia temporal determina qué tiempo del subjuntivo usar según el tiempo del verbo principal.",
                examples=[
                    "Espero que venga (presente + presente subj.)",
                    "Esperaba que viniera (imperfecto + imperfecto subj.)",
                    "He esperado que haya venido (perfecto + perfecto subj.)"
                ],
                counter_examples=[
                    "Esperaba que venga (incorrecto, falta concordancia temporal)",
                    "Espero que viniera (incorrecto, falta concordancia temporal)"
                ],
                keywords=["concordancia", "temporal", "subjuntivo", "consecutio", "temporum"],
                category="grammar",
                subcategory="tense_agreement",
                severity="high",
                language="spanish",
                source="RAE - Nueva Gramática de la Lengua Española",
                difficulty_level="advanced",
                related_rules=["es_rae_006", "es_rae_009"],
                tags=["verbos", "tiempos", "concordancia"]
            ),
            
            GrammarRule(
                rule_id="es_rae_008",
                rule_name="Uso de artículos definidos",
                description="Los artículos definidos (el, la, los, las) se usan para referirse a entidades específicas o conocidas por el hablante y oyente.",
                examples=[
                    "El libro que me prestaste (específico)",
                    "La mesa de la cocina (conocida)",
                    "Los estudiantes de esta clase (definidos)"
                ],
                counter_examples=[
                    "Dame libro (falta artículo)",
                    "El agua está fría (correcto: el agua, no la agua)"
                ],
                keywords=["artículo", "definido", "determinante", "especificidad"],
                category="grammar",
                subcategory="determiners",
                severity="medium",
                language="spanish",
                source="RAE - Nueva Gramática de la Lengua Española",
                difficulty_level="beginner",
                related_rules=["es_rae_001", "es_rae_009"],
                tags=["determinantes", "artículos", "definitud"]
            ),
            
            GrammarRule(
                rule_id="es_rae_009",
                rule_name="Leísmo, laísmo y loísmo",
                description="Uso correcto de los pronombres átonos: 'le/les' para complemento indirecto, 'lo/los' para CD masculino, 'la/las' para CD femenino.",
                examples=[
                    "Le dije la verdad (CI, correcto)",
                    "Lo vi ayer (CD masculino, correcto)",
                    "La llamé por teléfono (CD femenino, correcto)"
                ],
                counter_examples=[
                    "Le vi ayer (leísmo, incorrecto para CD)",
                    "La dije la verdad (laísmo, incorrecto para CI)",
                    "Los dije que vinieran (loísmo, incorrecto para CI)"
                ],
                keywords=["pronombres", "átonos", "leísmo", "laísmo", "loísmo", "complemento"],
                category="grammar",
                subcategory="pronouns",
                severity="high",
                language="spanish",
                source="RAE - Nueva Gramática de la Lengua Española",
                difficulty_level="advanced",
                related_rules=["es_rae_008", "es_rae_010"],
                tags=["pronombres", "complementos", "dialectos"]
            ),
            
            GrammarRule(
                rule_id="es_rae_010",
                rule_name="Perífrasis verbales de obligación",
                description="Las perífrasis 'tener que + infinitivo', 'haber de + infinitivo' y 'deber + infinitivo' expresan obligación con diferentes matices.",
                examples=[
                    "Tengo que estudiar (obligación fuerte)",
                    "He de terminar esto (obligación formal)",
                    "Debo ayudar a mi familia (obligación moral)"
                ],
                counter_examples=[
                    "Tengo de estudiar (incorrecto)",
                    "He que terminar (incorrecto)"
                ],
                keywords=["perífrasis", "verbal", "obligación", "modalidad", "deber"],
                category="grammar",
                subcategory="verbal_periphrasis",
                severity="medium",
                language="spanish",
                source="RAE - Nueva Gramática de la Lengua Española",
                difficulty_level="intermediate",
                related_rules=["es_rae_002", "es_rae_006"],
                tags=["perífrasis", "modalidad", "verbos"]
            )
        ]
        
        rules.extend(rae_rules)
        
        # Add punctuation rules
        punctuation_rules = [
            GrammarRule(
                rule_id="es_punct_001",
                rule_name="Uso de la coma en enumeraciones",
                description="En las enumeraciones, se separan los elementos con comas, excepto el último que va precedido de 'y', 'e', 'o', 'u'.",
                examples=[
                    "Compré manzanas, peras, naranjas y plátanos",
                    "Estudia matemáticas, física, química e inglés"
                ],
                counter_examples=[
                    "Compré manzanas, peras, naranjas, y plátanos (coma innecesaria antes de 'y')",
                    "Estudia matemáticas física química inglés (faltan comas)"
                ],
                keywords=["coma", "enumeración", "serie", "conjunción"],
                category="punctuation",
                subcategory="comma_usage",
                severity="medium",
                language="spanish",
                source="RAE - Ortografía de la Lengua Española",
                difficulty_level="beginner",
                related_rules=["es_punct_002"],
                tags=["puntuación", "comas", "listas"]
            ),
            
            GrammarRule(
                rule_id="es_punct_002",
                rule_name="Signos de interrogación y exclamación",
                description="En español se usan signos de apertura y cierre para interrogaciones (¿?) y exclamaciones (¡!).",
                examples=[
                    "¿Cómo estás?",
                    "¡Qué sorpresa!",
                    "¿Vienes o no vienes?"
                ],
                counter_examples=[
                    "Como estás? (falta signo de apertura)",
                    "Que sorpresa! (falta signo de apertura)",
                    "¿Vienes o no vienes. (falta signo de cierre)"
                ],
                keywords=["interrogación", "exclamación", "signos", "apertura", "cierre"],
                category="punctuation",
                subcategory="question_exclamation",
                severity="critical",
                language="spanish",
                source="RAE - Ortografía de la Lengua Española",
                difficulty_level="beginner",
                related_rules=["es_punct_001"],
                tags=["puntuación", "interrogación", "exclamación"]
            )
        ]
        
        rules.extend(punctuation_rules)
        
        return rules
    
    def load_rules_from_file(self, file_path: str) -> List[GrammarRule]:
        """Load grammar rules from JSON or YAML file."""
        
        path = Path(file_path)
        if not path.exists():
            self.logger.warning(f"Rules file not found: {file_path}")
            return []
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                if path.suffix.lower() == '.json':
                    data = json.load(f)
                elif path.suffix.lower() in ['.yml', '.yaml']:
                    data = yaml.safe_load(f)
                else:
                    self.logger.error(f"Unsupported file format: {path.suffix}")
                    return []
            
            rules = []
            for rule_data in data.get('rules', []):
                rule = GrammarRule(
                    rule_id=rule_data.get('rule_id', ''),
                    rule_name=rule_data.get('rule_name', ''),
                    description=rule_data.get('description', ''),
                    examples=rule_data.get('examples', []),
                    counter_examples=rule_data.get('counter_examples', []),
                    keywords=rule_data.get('keywords', []),
                    category=rule_data.get('category', 'general'),
                    subcategory=rule_data.get('subcategory', ''),
                    severity=rule_data.get('severity', 'medium'),
                    language=rule_data.get('language', self.language),
                    source=rule_data.get('source', 'Unknown'),
                    difficulty_level=rule_data.get('difficulty_level', 'intermediate'),
                    related_rules=rule_data.get('related_rules', []),
                    tags=rule_data.get('tags', [])
                )
                rules.append(rule)
            
            return rules
            
        except Exception as e:
            self.logger.error(f"Error loading rules from {file_path}: {e}")
            return []
    
    def build_database(self, rules: Optional[List[GrammarRule]] = None):
        """Build the vector database with grammar rules."""
        
        if rules is None:
            if self.language.lower() == 'spanish':
                rules = self.load_comprehensive_spanish_rules()
            else:
                # Try to load from file
                rules_file = f"data/rag/{self.language.lower()}/comprehensive_rules.json"
                rules = self.load_rules_from_file(rules_file)
                
                if not rules:
                    self.logger.warning(f"No comprehensive rules found for {self.language}")
                    return False
        
        if not rules:
            self.logger.error("No rules to add to database")
            return False
        
        # Clear existing data
        self.db.delete_all()
        
        # Create embeddings for all rules
        self.logger.info(f"Creating embeddings for {len(rules)} rules...")
        
        # Prepare texts for embedding
        texts = []
        for rule in rules:
            # Create comprehensive text for embedding
            text = f"{rule.rule_name}. {rule.description}"
            if rule.examples:
                text += f" Examples: {' '.join(rule.examples)}"
            if rule.keywords:
                text += f" Keywords: {' '.join(rule.keywords)}"
            texts.append(text)
        
        # Create embeddings in batches
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = self.embedder.embed_batch(batch_texts)
            all_embeddings.extend(batch_embeddings)
        
        # Add to database
        self.logger.info("Adding rules to vector database...")
        self.db.add_rules(rules, all_embeddings)
        
        self.logger.info(f"Successfully built database with {len(rules)} rules")
        return True
    
    def search_rules(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for grammar rules using semantic similarity."""
        
        # Create query embedding
        query_embedding = self.embedder.embed_text(query)
        
        # Search database
        results = self.db.search(query_embedding, k)
        
        return results
    
    def get_rule_by_category(self, category: str, k: int = 10) -> List[Dict[str, Any]]:
        """Get rules by category (grammar, punctuation, orthography, etc.)."""
        
        query = f"{category} rules in {self.language}"
        return self.search_rules(query, k)
    
    def export_rules_to_file(self, output_path: str, rules: List[GrammarRule]):
        """Export rules to JSON file for future use."""
        
        rules_data = {
            'language': self.language,
            'total_rules': len(rules),
            'rules': []
        }
        
        for rule in rules:
            rule_dict = {
                'rule_id': rule.rule_id,
                'rule_name': rule.rule_name,
                'description': rule.description,
                'examples': rule.examples,
                'counter_examples': rule.counter_examples,
                'keywords': rule.keywords,
                'category': rule.category,
                'subcategory': rule.subcategory,
                'severity': rule.severity,
                'language': rule.language,
                'source': rule.source,
                'difficulty_level': rule.difficulty_level,
                'related_rules': rule.related_rules,
                'tags': rule.tags
            }
            rules_data['rules'].append(rule_dict)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(rules_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Exported {len(rules)} rules to {output_path}")


def main():
    """Example usage and testing."""
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test with Spanish rules
    print("=== Testing Advanced Grammar RAG for Spanish ===")
    
    # Initialize with different backends
    backends_to_test = []
    
    if CHROMADB_AVAILABLE:
        backends_to_test.append(("chromadb", "ChromaDB"))
    
    if FAISS_AVAILABLE and SENTENCE_TRANSFORMERS_AVAILABLE:
        backends_to_test.append(("faiss", "FAISS"))
    
    for backend, backend_name in backends_to_test:
        print(f"\n--- Testing {backend_name} Backend ---")
        
        try:
            # Initialize RAG system
            rag = AdvancedGrammarRAG(
                backend=backend,
                embedding_provider="openai",
                language="spanish"
            )
            
            # Build database
            success = rag.build_database()
            if not success:
                print(f"Failed to build database for {backend_name}")
                continue
            
            # Test queries
            test_queries = [
                "concordancia de género y número",
                "uso de ser y estar",
                "acentuación de palabras",
                "subjuntivo en oraciones subordinadas",
                "signos de interrogación",
                "comma usage in lists",
                "verb agreement"
            ]
            
            for query in test_queries:
                print(f"\nQuery: '{query}'")
                results = rag.search_rules(query, k=3)
                
                for i, result in enumerate(results, 1):
                    print(f"  {i}. {result['rule_name']} (distance: {result.get('distance', 0):.3f})")
            
            # Export rules for future use
            if backend == "chromadb":  # Export once
                spanish_rules = rag.load_comprehensive_spanish_rules()
                output_path = "data/rag/spanish/comprehensive_rules.json"
                rag.export_rules_to_file(output_path, spanish_rules)
            
        except Exception as e:
            print(f"Error testing {backend_name}: {e}")
    
    print("\n=== Advanced Grammar RAG Testing Complete ===")


if __name__ == "__main__":
    main()
