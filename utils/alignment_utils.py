#!/usr/bin/env python3
"""
utils/alignment_utils.py - Optimized alignment utility for GEC methods
======================================================================

Provides clean, optimized alignment generation for methods that need it:
- sentence/baseline
- feedback/baseline  
- edit/baseline

Features:
- Thread-safe model caching
- Robust error handling with fallbacks
- Supports all optimization levels (sequential, parallel, batch)
- Clean API for different alignment types
"""

import os
import threading
from typing import Optional, Tuple

# Global model cache with thread safety
_ALIGNMENT_MODELS = {}
_ALIGNMENT_LOCK = threading.Lock()

def get_alignment_models(language: str = 'es') -> Tuple[Optional[object], Optional[object]]:
    """
    Get cached spaCy and ERRANT models for alignment generation.
    Thread-safe singleton pattern.
    
    Args:
        language: Language code (es, en, de, ua)
        
    Returns:
        Tuple of (nlp, annotator) or (None, None) if loading fails
    """
    if language in _ALIGNMENT_MODELS:
        return _ALIGNMENT_MODELS[language]
    
    with _ALIGNMENT_LOCK:
        # Double-check pattern
        if language in _ALIGNMENT_MODELS:
            return _ALIGNMENT_MODELS[language]
        
        try:
            from utils.processing.model_setup import setup_language_models
            nlp, annotator = setup_language_models(language)
            _ALIGNMENT_MODELS[language] = (nlp, annotator)
            return nlp, annotator
        except Exception as e:
            print(f"Warning: Failed to load alignment models for {language}: {e}")
            _ALIGNMENT_MODELS[language] = (None, None)
            return None, None

def create_clean_alignment(src: str, tgt: str, language: str = 'es') -> str:
    """
    Create clean alignment with robust fallbacks.
    
    Args:
        src: Source text
        tgt: Target text
        language: Language code
        
    Returns:
        Clean alignment string
    """
    # Quick check for identical texts
    if src.strip() == tgt.strip():
        return "No changes detected"
    
    # Try ERRANT alignment first
    try:
        nlp, annotator = get_alignment_models(language)
        if nlp and annotator:
            from utils.errant_align import create_alignment
            alignment = create_alignment(src, tgt, language=language, merge=True)
            if alignment and alignment.strip() and alignment != src:
                return alignment
    except Exception:
        # Suppress ERRANT errors and use fallback
        pass
    
    # Clean fallback alignment
    src_clean = src.strip()[:100]
    tgt_clean = tgt.strip()[:100]
    return f"Original: {src_clean} → Suggested: {tgt_clean}"

def create_simple_alignment(src: str, tgt: str) -> str:
    """
    Create simple alignment without ERRANT (for methods that don't need complex alignment).
    
    Args:
        src: Source text
        tgt: Target text
        
    Returns:
        Simple alignment string
    """
    if src.strip() == tgt.strip():
        return "No changes detected"
    
    src_clean = src.strip()[:100]
    tgt_clean = tgt.strip()[:100]
    return f"Original: {src_clean} → Suggested: {tgt_clean}"

def create_diff_alignment(src: str, tgt: str) -> str:
    """
    Create diff-style alignment for edit-based methods.
    
    Args:
        src: Source text
        tgt: Target text
        
    Returns:
        Diff-style alignment
    """
    if src.strip() == tgt.strip():
        return "No edits"
    
    # Simple word-level diff
    src_words = src.split()
    tgt_words = tgt.split()
    
    if len(src_words) != len(tgt_words):
        return f"Length change: {len(src_words)} → {len(tgt_words)} words"
    
    changes = []
    for i, (s_word, t_word) in enumerate(zip(src_words, tgt_words)):
        if s_word != t_word:
            changes.append(f"{s_word}→{t_word}")
    
    if changes:
        return f"Changes: {', '.join(changes[:5])}" + ("..." if len(changes) > 5 else "")
    else:
        return "Minor changes detected"

# Method-specific alignment functions
def get_sentence_alignment(src: str, tgt: str, language: str = 'es') -> str:
    """Alignment for sentence-based methods."""
    return create_clean_alignment(src, tgt, language)

def get_feedback_alignment(src: str, tgt: str, language: str = 'es') -> str:
    """Alignment for feedback-based methods."""
    return create_clean_alignment(src, tgt, language)

def get_edit_alignment(src: str, tgt: str) -> str:
    """Alignment for edit-based methods."""
    return create_diff_alignment(src, tgt)

def get_legacy_alignment(src: str, tgt: str) -> str:
    """Simple alignment for legacy methods (no ERRANT)."""
    return create_simple_alignment(src, tgt)

# Preload models for common languages
def preload_alignment_models(languages: list = ['es', 'en']):
    """Preload alignment models for specified languages."""
    for lang in languages:
        get_alignment_models(lang)

# Export main functions
__all__ = [
    'get_sentence_alignment',
    'get_feedback_alignment', 
    'get_edit_alignment',
    'get_legacy_alignment',
    'preload_alignment_models'
]







