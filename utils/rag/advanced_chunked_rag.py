#!/usr/bin/env python3
"""
Advanced chunked RAG system with intelligent rule retrieval and direct querying.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings
import openai
import numpy as np

from utils.rag.chunked_rulebooks import ChunkedRulebookProcessor, RuleChunk

class AdvancedChunkedRAG:
    """Advanced RAG system using chunked grammar rulebooks."""
    
    def __init__(self, language: str = "english", persist_directory: str = "data/rag/chunked_advanced"):
        self.language = language
        self.persist_directory = persist_directory
        self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.embedding_model = "text-embedding-3-small"
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create collections for different chunk types
        self.collections = {}
        chunk_types = ["definition", "examples", "counter_examples", "usage", "content"]
        
        for chunk_type in chunk_types:
            collection_name = f"{language}_{chunk_type}_chunks"
            try:
                self.collections[chunk_type] = self.client.get_collection(name=collection_name)
            except:
                self.collections[chunk_type] = self.client.create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
    
    def build_chunked_database(self):
        """Build the chunked grammar database."""
        logging.info(f"Building chunked database for {self.language}")
        
        processor = ChunkedRulebookProcessor(self.language)
        
        # Get chunks based on language
        if self.language == "english":
            chunks = processor.create_comprehensive_english_chunks()
        elif self.language == "spanish":
            chunks = processor.create_comprehensive_spanish_chunks()
        else:
            logging.warning(f"No chunked rules available for {self.language}")
            return
        
        # Group chunks by type
        chunks_by_type = {}
        for chunk in chunks:
            chunk_type = chunk.chunk_type
            if chunk_type not in chunks_by_type:
                chunks_by_type[chunk_type] = []
            chunks_by_type[chunk_type].append(chunk)
        
        # Add chunks to appropriate collections
        for chunk_type, type_chunks in chunks_by_type.items():
            if chunk_type in self.collections:
                self._add_chunks_to_collection(type_chunks, self.collections[chunk_type])
                logging.info(f"Added {len(type_chunks)} {chunk_type} chunks")
    
    def _add_chunks_to_collection(self, chunks: List[RuleChunk], collection):
        """Add chunks to a ChromaDB collection."""
        if not chunks:
            return
        
        # Create embeddings
        documents = [chunk.to_document() for chunk in chunks]
        response = self.openai_client.embeddings.create(input=documents, model=self.embedding_model)
        embeddings = [item.embedding for item in response.data]
        
        # Prepare data for ChromaDB
        ids = [chunk.chunk_id for chunk in chunks]
        metadatas = [chunk.to_metadata() for chunk in chunks]
        
        # Add to collection
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
    
    def search_by_query(self, query: str, chunk_types: List[str] = None, k: int = 3) -> List[Dict[str, Any]]:
        """Search for relevant chunks by query."""
        if chunk_types is None:
            chunk_types = ["definition", "examples", "counter_examples", "usage"]
        
        all_results = []
        
        # Search in each specified chunk type
        for chunk_type in chunk_types:
            if chunk_type in self.collections:
                # Create query embedding
                response = self.openai_client.embeddings.create(input=[query], model=self.embedding_model)
                query_embedding = response.data[0].embedding
                
                # Search in collection
                results = self.collections[chunk_type].query(
                    query_embeddings=[query_embedding],
                    n_results=k
                )
                
                # Format results
                for i in range(len(results['ids'][0])):
                    result = {
                        'chunk_id': results['ids'][0][i],
                        'document': results['documents'][0][i],
                        'distance': results['distances'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'chunk_type': chunk_type,
                        'relevance_score': 1 - results['distances'][0][i]
                    }
                    all_results.append(result)
        
        # Sort by relevance and return top results
        all_results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return all_results[:k*2]  # Return more results for better coverage
    
    def search_by_rule_name(self, rule_name: str, chunk_types: List[str] = None) -> List[Dict[str, Any]]:
        """Search for chunks by specific rule name."""
        if chunk_types is None:
            chunk_types = ["definition", "examples", "counter_examples", "usage"]
        
        all_results = []
        
        for chunk_type in chunk_types:
            if chunk_type in self.collections:
                # Search by exact metadata match
                results = self.collections[chunk_type].get(
                    where={"rule_name": {"$eq": rule_name}},
                    include=["documents", "metadatas"]
                )
                
                # Format results
                for i in range(len(results['ids'])):
                    result = {
                        'chunk_id': results['ids'][i],
                        'document': results['documents'][i],
                        'distance': 0.0,  # Exact match
                        'metadata': results['metadatas'][i],
                        'chunk_type': chunk_type,
                        'relevance_score': 1.0
                    }
                    all_results.append(result)
        
        return all_results
    
    def search_by_category(self, category: str, chunk_types: List[str] = None) -> List[Dict[str, Any]]:
        """Search for chunks by grammar category."""
        if chunk_types is None:
            chunk_types = ["definition", "examples", "counter_examples", "usage"]
        
        all_results = []
        
        for chunk_type in chunk_types:
            if chunk_type in self.collections:
                # Search by category filter
                results = self.collections[chunk_type].get(
                    where={"rule_category": {"$eq": category}},
                    include=["documents", "metadatas"]
                )
                
                # Format results
                for i in range(len(results['ids'])):
                    result = {
                        'chunk_id': results['ids'][i],
                        'document': results['documents'][i],
                        'distance': 0.0,
                        'metadata': results['metadatas'][i],
                        'chunk_type': chunk_type,
                        'relevance_score': 1.0
                    }
                    all_results.append(result)
        
        return all_results
    
    def get_comprehensive_rule_info(self, rule_name: str) -> Dict[str, Any]:
        """Get comprehensive information about a specific rule."""
        # Get all chunks for this rule
        chunks = self.search_by_rule_name(rule_name)
        
        if not chunks:
            return {"error": f"No information found for rule: {rule_name}"}
        
        # Organize by chunk type
        organized = {}
        for chunk in chunks:
            chunk_type = chunk['chunk_type']
            if chunk_type not in organized:
                organized[chunk_type] = []
            organized[chunk_type].append(chunk)
        
        # Build comprehensive response
        response = {
            "rule_name": rule_name,
            "language": self.language,
            "total_chunks": len(chunks)
        }
        
        # Add each type of information
        for chunk_type in ["definition", "examples", "counter_examples", "usage"]:
            if chunk_type in organized:
                response[chunk_type] = [chunk['document'] for chunk in organized[chunk_type]]
        
        return response
    
    def smart_query(self, query: str, query_type: str = "auto") -> Dict[str, Any]:
        """Smart query that determines the best search strategy."""
        query_lower = query.lower()
        
        # Determine query type if auto
        if query_type == "auto":
            if any(word in query_lower for word in ["rule", "regla", "règle"]):
                query_type = "rule_name"
            elif any(word in query_lower for word in ["grammar", "gramática", "grammaire", "syntax"]):
                query_type = "category"
            elif any(word in query_lower for word in ["example", "ejemplo", "exemple"]):
                query_type = "examples_focused"
            else:
                query_type = "general"
        
        # Execute appropriate search
        if query_type == "rule_name":
            # Extract potential rule name
            rule_name = query.replace("rule", "").replace("regla", "").strip()
            results = self.search_by_rule_name(rule_name)
        elif query_type == "category":
            # Search by category
            results = self.search_by_category("grammar")
        elif query_type == "examples_focused":
            # Focus on examples and counter-examples
            results = self.search_by_query(query, chunk_types=["examples", "counter_examples"])
        else:
            # General search across all types
            results = self.search_by_query(query)
        
        if not results:
            return {
                "rule": f"No relevant grammar information found for: '{query}'",
                "source": "Chunked Grammar Database",
                "confidence": "LOW",
                "query_matched": query,
                "error": "No matching chunks found"
            }
        
        # Get best result
        best_result = results[0]
        
        # Build comprehensive response
        rule_name = best_result['metadata']['rule_name']
        comprehensive_info = self.get_comprehensive_rule_info(rule_name)
        
        # Format response
        rule_text = f"**{rule_name}**"
        
        if "definition" in comprehensive_info:
            rule_text += f"\n\n**Definition**: {comprehensive_info['definition'][0]}"
        
        if "examples" in comprehensive_info:
            examples_text = "; ".join(comprehensive_info['examples'])
            rule_text += f"\n\n**Examples**: {examples_text}"
        
        if "counter_examples" in comprehensive_info:
            counter_text = "; ".join(comprehensive_info['counter_examples'])
            rule_text += f"\n\n**Common Errors**: {counter_text}"
        
        if "usage" in comprehensive_info:
            usage_text = "; ".join(comprehensive_info['usage'])
            rule_text += f"\n\n**Usage Notes**: {usage_text}"
        
        return {
            "rule": rule_text,
            "source": "Chunked Grammar Database",
            "confidence": "HIGH" if best_result['relevance_score'] > 0.8 else "MEDIUM",
            "query_matched": query,
            "relevance_score": best_result['relevance_score'],
            "rule_name": rule_name,
            "chunk_type": best_result['chunk_type'],
            "total_chunks": comprehensive_info.get('total_chunks', 1)
        }

def build_all_chunked_databases():
    """Build chunked databases for all supported languages."""
    languages = ["english", "spanish"]
    
    for language in languages:
        print(f"\n🔧 Building chunked database for {language.upper()}")
        rag = AdvancedChunkedRAG(language=language)
        rag.build_chunked_database()
        print(f"✅ Completed {language} chunked database")

if __name__ == "__main__":
    build_all_chunked_databases()
