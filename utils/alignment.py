"""
Text Alignment Utility using ERRANT only.

We rely on the ERRANT annotator constructed in model setup; no Levenshtein fallback.
Returns compact auxiliary markup: '<c>old->new</c> ...'.
"""


def get_alignment_for_language(src_text: str, tgt_text: str, language: str, nlp=None, annotator=None, merge: bool = True) -> str:
    """
    Get aligned text for GEC training data processing.
    
    Uses ERRANT Annotator when provided. Returns None if alignment fails.
    
    Returns:
        Aligned text string with corrections marked as <c>old->new</c>, or None on failure
    """
    if annotator is None:
        # No annotator available - return None to indicate failure
        return None

    try:
        # Tokenize the texts first if nlp is provided
        if nlp is not None:
            src_doc = nlp(src_text)
            tgt_doc = nlp(tgt_text)
        else:
            src_doc = src_text
            tgt_doc = tgt_text
            
        # Prefer annotate(); fall back to parse() if needed
        if hasattr(annotator, 'annotate'):
            doc_edits = annotator.annotate(src_doc, tgt_doc)
        else:
            doc_edits = annotator.parse(src_doc, tgt_doc)
        if not doc_edits:
            return tgt_text
        parts = []
        for edit in doc_edits:
            old = ''
            new = ''
            for attr in ('o_str', 'orig_str', 'o_text'):
                if hasattr(edit, attr):
                    old = getattr(edit, attr) or ''
                    break
            for attr in ('c_str', 'cor_str', 'c_text'):
                if hasattr(edit, attr):
                    new = getattr(edit, attr) or ''
                    break
            if not old and hasattr(edit, 'o_toks'):
                try:
                    old = ' '.join(t.text if hasattr(t, 'text') else str(t) for t in edit.o_toks)
                except Exception:
                    old = ''
            if not new and hasattr(edit, 'c_toks'):
                try:
                    new = ' '.join(t.text if hasattr(t, 'text') else str(t) for t in edit.c_toks)
                except Exception:
                    new = ''
            parts.append(f"<c>{old}->{new}</c>")
        return ' '.join(parts)
    except Exception:
        # No fallback; return None on failure
        return None