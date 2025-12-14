"""
Language Model Setup for GEC Processing

Simplified setup for spaCy models used in GEC data processing.
"""

# Global cache for loaded models to avoid repeated loading
_model_cache = {}
_loading_lock = None

def _get_lock():
    """Get or create threading lock for model loading."""
    global _loading_lock
    if _loading_lock is None:
        import threading
        _loading_lock = threading.Lock()
    return _loading_lock


def setup_language_models(language: str):
    """
    Setup language models for GEC processing with thread-safe caching.
    
    Args:
        language: Language code (en, de, ua, ru, es)
        
    Returns:
        Tuple of (nlp, annotator) - for compatibility with ERRANT-based processing
    """
    # Check cache first (double-checked locking pattern)
    if language in _model_cache:
        return _model_cache[language]
    
    # Use lock to prevent multiple threads from loading the same model
    with _get_lock():
        # Check cache again inside lock
        if language in _model_cache:
            return _model_cache[language]
        
        try:
            # Lazy import to avoid hard dependency during module import
            import spacy  # type: ignore
            import errant  # type: ignore
            
            # Try to load spaCy models
            model_names = {
                'en': 'en_core_web_sm',
                'de': 'de_core_news_sm', 
                'ua': 'uk_core_news_sm',
                'uk': 'uk_core_news_sm',
                'es': 'es_core_news_sm',
                'fr': 'fr_core_news_sm'
            }
            
            model_name = model_names.get(language, 'en_core_web_sm')
            
            print(f"Loading spaCy model: {model_name}")
            nlp = spacy.load(model_name)
            
            # For Ukrainian, use English ERRANT with Ukrainian spaCy
            # This is the approach used by MultiGEC 2025
            if language in ['ua', 'uk']:
                print(f"Using English ERRANT with Ukrainian spaCy for {language}")
                annotator = errant.load('en')  # Load English ERRANT
                annotator.nlp = nlp  # But use Ukrainian spaCy model
                print(f"Successfully loaded models for {language}")
                result = (nlp, annotator)
                _model_cache[language] = result
                return result
            else:
                # For other languages, try to load matching ERRANT
                try:
                    annotator = errant.load(language, nlp)
                except:
                    # If language-specific ERRANT fails, fall back to English
                    annotator = errant.load('en')
                    annotator.nlp = nlp
                else:
                    # Default to English
                    annotator = errant.load('en', nlp)
                
                print(f"Successfully loaded models for {language}")
                result = (nlp, annotator)
                _model_cache[language] = result
                return result
                
        except Exception as e:
            print(f"Error loading models for {language}: {e}")
            # Cache the failure to avoid repeated attempts
            result = (None, None)
            _model_cache[language] = result
            return result







