"""
Language Model Setup for GEC Processing

Simplified setup for spaCy models used in GEC data processing.
"""

import sys
import os

# Add third-party to path for ERRANT
third_party_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'third-party')
if os.path.exists(third_party_path) and third_party_path not in sys.path:
    sys.path.insert(0, third_party_path)

# Global cache for loaded models to avoid repeated loading
_model_cache = {}

# Process-level cache to avoid repeated model loading within the same process
_process_model_cache = {}
_load_locks = {}


def setup_language_models(language: str):
    """
    Setup language models for GEC processing with aggressive caching.
    
    Args:
        language: Language code (en, de, ua, ru, es)
        
    Returns:
        Tuple of (nlp, annotator) - for compatibility with ERRANT-based processing
    """
    # Fast path: process cache
    if language in _process_model_cache:
        return _process_model_cache[language]
    # Fast path: global cache
    if language in _model_cache:
        result = _model_cache[language]
        _process_model_cache[language] = result
        return result
    
    try:
        # Ensure single initialization across threads
        import threading  # type: ignore
        lock = _load_locks.setdefault(language, threading.Lock())
        with lock:
            # Re-check caches inside lock
            if language in _process_model_cache:
                return _process_model_cache[language]
            if language in _model_cache:
                result = _model_cache[language]
                _process_model_cache[language] = result
                return result

        # Add third-party/errant to path for ERRANT if not already done
        import sys
        import os
        # Add the errant package directory directly
        # Get the project root (gec_judge directory)
        current_dir = os.path.dirname(__file__)  # utils/processing
        project_root = os.path.dirname(os.path.dirname(current_dir))  # gec_judge
        errant_path = os.path.join(project_root, 'third-party', 'errant')
        if os.path.exists(errant_path) and errant_path not in sys.path:
            sys.path.insert(0, errant_path)
        
        # Lazy import to avoid hard dependency during module import
        import spacy  # type: ignore
        import errant  # type: ignore
        # Red Sparta: prefer sparta_lib downloader if available
        try:
            from sparta_lib.utils import spacy_download as _sparta_spacy_download  # type: ignore
        except Exception:
            _sparta_spacy_download = None
        
        # Try to load spaCy models (extend coverage; fall back to blank tokenizer if missing)
        model_names = {
            'en': 'en_core_web_sm',
            'de': 'de_core_news_sm',
            'es': 'es_core_news_sm',
            'fr': 'fr_core_news_sm',
            'it': 'it_core_news_sm',
            'pt': 'pt_core_news_sm',
            'nl': 'nl_core_news_sm',
            'sv': 'sv_core_news_sm',
            'da': 'da_core_news_sm',
            'no': 'nb_core_news_sm',  # spaCy uses nb_* for Norwegian Bokmål
            'nb': 'nb_core_news_sm',
            'fi': 'fi_core_news_sm',
            'pl': 'pl_core_news_sm',
            'ru': 'ru_core_news_sm',
            'ro': 'ro_core_news_sm',
            'uk': 'uk_core_news_sm',
            'ua': 'uk_core_news_sm',
            'lt': 'lt_core_news_sm',
            'lv': 'lv_core_news_sm',
            'et': 'et_core_news_sm',
            'cs': 'cs_core_news_sm',
            'sk': 'sk_core_news_sm',
            'sl': 'sl_core_news_sm',
            'hr': 'hr_core_news_sm',
            'bg': 'bg_core_news_sm',
            'el': 'el_core_news_sm'
        }
        
        model_name = model_names.get(language)
        
        nlp = None
        if model_name:
            print(f"Loading spaCy model: {model_name}")
            try:
                nlp = spacy.load(model_name)
            except OSError:
                # Red Sparta downloader if present (no direct internet)
                if _sparta_spacy_download is not None:
                    try:
                        _sparta_spacy_download(model_name)  # type: ignore
                        nlp = spacy.load(model_name)
                    except Exception:
                        nlp = None
                # Try local wheel fallback if available
                import glob, subprocess, sys as _sys, os as _os
                base = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(__file__))), 'third-party', 'spacy-models')
                wheels = glob.glob(_os.path.join(base, f"{model_name}-*.whl"))
                if nlp is None and wheels:
                    wheel = wheels[0]
                    print(f"Installing local wheel for {model_name}: {wheel}")
                    # Install without dependency resolution to avoid internet requirement on Sparta
                    subprocess.run([_sys.executable, '-m', 'pip', 'install', '--no-deps', '--no-input', '--quiet', wheel], check=False)
                    nlp = spacy.load(model_name)
        # If we still have no nlp (no model or not installed), try tokenizer-only for the requested language
        if nlp is None:
            print(f"Using tokenizer-only spaCy for '{language}' via spacy.blank('{language}')")
            try:
                nlp = spacy.blank(language)
            except Exception:
                # Fallback to multilingual or English tokenizer with explicit log
                try:
                    print(f"WARNING: spacy.blank('{language}') not available. Falling back to spacy.blank('xx') tokenizer-only.")
                    nlp = spacy.blank('xx')
                except Exception:
                    print(f"WARNING: spacy.blank('xx') not available. Falling back to spacy.blank('en') tokenizer-only.")
                    nlp = spacy.blank('en')
        
        # For alignment we do not need POS/lemma components. Use tokenizer-only for speed.
        try:
            if nlp.pipe_names:
                nlp.disable_pipes(*list(nlp.pipe_names))
                print("Using tokenizer-only spaCy pipeline for alignment")
        except Exception:
            pass

        # For Ukrainian, use English ERRANT with Ukrainian spaCy
        if language in ['ua', 'uk']:
            print(f"Using English ERRANT with Ukrainian spaCy for {language}")
            annotator = errant.load('en')  # Load English ERRANT
            annotator.nlp = nlp  # But use Ukrainian spaCy model
            print(f"Successfully loaded models for {language}")
            result = (nlp, annotator)
            _model_cache[language] = result
            _process_model_cache[language] = result
            return result
        
        # For other languages, try native ERRANT support
        try:
            # ERRANT 3.0 only supports 'en' natively
            # For other languages, we'll use English ERRANT with language-specific spaCy
            if language in ['de', 'es', 'ru', 'fr']:
                print(f"Using English ERRANT with {language} spaCy")
                annotator = errant.load('en')
                annotator.nlp = nlp
            else:
                # Default to English
                annotator = errant.load('en', nlp)
            
            print(f"Successfully loaded models for {language}")
            result = (nlp, annotator)
            _model_cache[language] = result
            _process_model_cache[language] = result
            return result
        except Exception as e:
            print(f"Warning: ERRANT setup failed for {language}: {e}")
            print(f"Using spaCy tokenization only for {language}")
            result = (nlp, None)
            _model_cache[language] = result
            return result
        
    except (OSError, ImportError, KeyError) as e:
        print(f"Could not load models for {language}: {e}")
        print(f"Falling back to simplified mode for {language}")
        result = (None, None)
        _model_cache[language] = result
        return result