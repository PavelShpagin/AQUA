#!/usr/bin/env python3
"""
Grammar RAG system for language-specific grammar rules.
Uses ChromaDB for vector storage and retrieval.
"""

import os
import json
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# Initialize ChromaDB client
CHROMA_PATH = "data/rag/chroma_db"
client = chromadb.PersistentClient(path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False))

# Use sentence transformer for embeddings (no API needed)
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)


def get_or_create_collection(language: str):
    """Get or create a collection for a specific language."""
    collection_name = f"grammar_rules_{language.lower()}"
    try:
        return client.get_collection(name=collection_name, embedding_function=embedding_func)
    except:
        return client.create_collection(name=collection_name, embedding_function=embedding_func)


def load_grammar_rules(language: str) -> bool:
    """Load grammar rules from JSON files into ChromaDB."""
    rules_file = f"data/rag/{language.lower()}/grammar_rules.json"
    
    if not os.path.exists(rules_file):
        # Create default rules if file doesn't exist
        create_default_rules(language)
    
    try:
        with open(rules_file, 'r', encoding='utf-8') as f:
            rules_data = json.load(f)
        
        collection = get_or_create_collection(language)
        
        # Clear existing data
        try:
            collection.delete(where={})
        except:
            pass
        
        # Add rules to collection
        documents = []
        metadatas = []
        ids = []
        
        for i, rule in enumerate(rules_data.get('rules', [])):
            # Create searchable document combining all rule information
            doc_text = f"{rule['rule_name']}: {rule['description']}"
            if rule.get('examples'):
                doc_text += f" Examples: {', '.join(rule['examples'])}"
            if rule.get('keywords'):
                doc_text += f" Keywords: {', '.join(rule['keywords'])}"
            
            documents.append(doc_text)
            metadatas.append({
                'rule_name': rule['rule_name'],
                'category': rule.get('category', 'general'),
                'severity': rule.get('severity', 'medium'),
                'language': language
            })
            ids.append(f"{language}_{i}")
        
        if documents:
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
        
        return True
        
    except Exception as e:
        print(f"Error loading grammar rules for {language}: {e}")
        return False


def search_grammar_rules(query: str, language: str, n_results: int = 3) -> List[Dict[str, Any]]:
    """Search for relevant grammar rules."""
    try:
        collection = get_or_create_collection(language)
        
        # Perform similarity search
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        if not results['documents'][0]:
            # Load rules if collection is empty
            load_grammar_rules(language)
            results = collection.query(
                query_texts=[query],
                n_results=n_results
            )
        
        # Format results
        rules = []
        for i, doc in enumerate(results['documents'][0]):
            metadata = results['metadatas'][0][i]
            
            # Parse the document to extract components
            parts = doc.split(': ', 1)
            rule_name = parts[0] if parts else metadata.get('rule_name', 'Unknown')
            description = parts[1] if len(parts) > 1 else doc
            
            # Extract examples if present
            examples = []
            if 'Examples:' in description:
                desc_part, examples_part = description.split(' Examples:', 1)
                description = desc_part
                if ' Keywords:' in examples_part:
                    examples_part, _ = examples_part.split(' Keywords:', 1)
                examples = [ex.strip() for ex in examples_part.split(',')]
            
            rules.append({
                'rule_name': rule_name,
                'description': description,
                'examples': examples,
                'category': metadata.get('category', 'general'),
                'severity': metadata.get('severity', 'medium'),
                'relevance_score': results['distances'][0][i] if 'distances' in results else 0
            })
        
        return rules
        
    except Exception as e:
        print(f"Error searching grammar rules: {e}")
        return []


def create_default_rules(language: str):
    """Create default grammar rules for a language."""
    os.makedirs(f"data/rag/{language.lower()}", exist_ok=True)
    
    default_rules = {
        'english': {
            'rules': [
                {
                    'rule_name': 'Subject-Verb Agreement',
                    'description': 'The subject and verb must agree in number (singular or plural)',
                    'examples': ['She walks (not walk)', 'They walk (not walks)'],
                    'keywords': ['agreement', 'subject', 'verb', 'singular', 'plural'],
                    'category': 'grammar',
                    'severity': 'high'
                },
                {
                    'rule_name': 'Article Usage',
                    'description': 'Use "a" before consonant sounds, "an" before vowel sounds, "the" for specific items',
                    'examples': ['a book', 'an apple', 'the specific book'],
                    'keywords': ['article', 'a', 'an', 'the', 'determiner'],
                    'category': 'grammar',
                    'severity': 'medium'
                },
                {
                    'rule_name': 'Comma Splice',
                    'description': 'Two independent clauses cannot be joined with just a comma',
                    'examples': ['Wrong: I went, she stayed. Right: I went, but she stayed.'],
                    'keywords': ['comma', 'splice', 'punctuation', 'independent clause'],
                    'category': 'punctuation',
                    'severity': 'medium'
                },
                {
                    'rule_name': 'Its vs It\'s',
                    'description': '"Its" is possessive, "it\'s" is a contraction of "it is"',
                    'examples': ['Its color is blue', 'It\'s raining'],
                    'keywords': ['its', 'it\'s', 'possessive', 'contraction'],
                    'category': 'spelling',
                    'severity': 'high'
                },
                {
                    'rule_name': 'Oxford Comma',
                    'description': 'Optional comma before "and" in a list; depends on style guide',
                    'examples': ['apples, oranges, and bananas'],
                    'keywords': ['oxford', 'serial', 'comma', 'list'],
                    'category': 'style',
                    'severity': 'low'
                }
            ]
        },
        'spanish': {
            'rules': [
                {
                    'rule_name': 'Género y Concordancia',
                    'description': 'Los artículos y adjetivos deben concordar en género y número con el sustantivo',
                    'examples': ['la casa blanca', 'los coches rojos'],
                    'keywords': ['género', 'concordancia', 'artículo', 'adjetivo'],
                    'category': 'grammar',
                    'severity': 'high'
                },
                {
                    'rule_name': 'Acentuación',
                    'description': 'Las palabras agudas, graves y esdrújulas siguen reglas específicas de acentuación',
                    'examples': ['café (aguda)', 'árbol (grave)', 'médico (esdrújula)'],
                    'keywords': ['acento', 'tilde', 'aguda', 'grave', 'esdrújula'],
                    'category': 'orthography',
                    'severity': 'high'
                },
                {
                    'rule_name': 'Ser vs Estar',
                    'description': 'Ser para características permanentes, estar para estados temporales',
                    'examples': ['Ella es alta', 'Ella está cansada'],
                    'keywords': ['ser', 'estar', 'verbo', 'estado'],
                    'category': 'grammar',
                    'severity': 'high'
                }
            ]
        },
        'german': {
            'rules': [
                {
                    'rule_name': 'Kasus (Cases)',
                    'description': 'German has four cases: Nominativ, Akkusativ, Dativ, Genitiv',
                    'examples': ['der Mann (Nom)', 'den Mann (Akk)', 'dem Mann (Dat)', 'des Mannes (Gen)'],
                    'keywords': ['Kasus', 'Fall', 'Nominativ', 'Akkusativ', 'Dativ', 'Genitiv'],
                    'category': 'grammar',
                    'severity': 'high'
                },
                {
                    'rule_name': 'Verbposition',
                    'description': 'Verb position changes based on sentence type',
                    'examples': ['Ich gehe heute. (V2)', 'Gehe ich heute? (V1)'],
                    'keywords': ['Verb', 'Position', 'V2', 'Wortstellung'],
                    'category': 'syntax',
                    'severity': 'high'
                }
            ]
        },
        'ukrainian': {
            'rules': [
                {
                    'rule_name': 'Відмінки (Cases)',
                    'description': 'Ukrainian has seven cases that affect noun endings',
                    'examples': ['стіл (Nom)', 'стола (Gen)', 'столу (Dat)'],
                    'keywords': ['відмінок', 'називний', 'родовий', 'давальний'],
                    'category': 'grammar',
                    'severity': 'high'
                },
                {
                    'rule_name': 'Апостроф',
                    'description': 'Apostrophe is used after consonants before я, ю, є, ї',
                    'examples': ["м'ясо", "в'їзд", "об'єкт"],
                    'keywords': ['апостроф', 'правопис'],
                    'category': 'orthography',
                    'severity': 'medium'
                }
            ]
        }
    }
    
    # Get rules for the specified language or use English as default
    rules_data = default_rules.get(language.lower(), default_rules['english'])
    
    # Save to file
    rules_file = f"data/rag/{language.lower()}/grammar_rules.json"
    with open(rules_file, 'w', encoding='utf-8') as f:
        json.dump(rules_data, f, ensure_ascii=False, indent=2)
    
    print(f"Created default grammar rules for {language} at {rules_file}")


# Initialize rules for common languages on module load
def initialize_rag():
    """Initialize RAG databases for all languages."""
    languages = ['english', 'spanish', 'german', 'ukrainian']
    for lang in languages:
        try:
            load_grammar_rules(lang)
        except Exception as e:
            print(f"Warning: Could not initialize {lang} RAG: {e}")