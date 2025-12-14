#!/usr/bin/env python3
"""
Lightweight safety/consistency checks for GEC evaluation spans.

Goals:
- Detect numeric/unit/date/entity changes that risk FP1/FP2
- Detect structural punctuation integrity issues (quotes/brackets)

No external heavy deps; spaCy NER optional (best-effort when available).
"""
from __future__ import annotations

import re
from typing import Dict, Any, Tuple
import os
import math


_NUM_RE = re.compile(r"[0-9]+([.,][0-9]+)?")
_UNIT_TOKENS = {
    'mg','g','kg','ml','l','km','m','cm','mm','°c','°f','%','usd','eur','$','€','¥'
}
_DATE_HINTS = re.compile(r"\b(\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?|\b\d{4}\b|enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\b", re.I)


def _tokenize_simple(text: str) -> Tuple[set, set, set]:
    t = (text or '').strip()
    lower = t.lower()
    nums = set(_NUM_RE.findall(lower)) or (set([m.group(0) for m in _NUM_RE.finditer(lower)]))
    # capture tokens for unit detection
    toks = re.findall(r"[\w°%€$¥]+", lower)
    units = set([x for x in toks if x in _UNIT_TOKENS])
    dates = set()
    if _DATE_HINTS.search(lower):
        dates.add('date_hint')
    return set([m.group(0) for m in _NUM_RE.finditer(lower)]), units, dates


def analyze_span_risk(src: str, tgt: str, x: str, y: str, lang_code: str) -> Dict[str, Any]:
    """Return risk signals for a single edit span x=>y in the context of src/tgt.

    Signals:
    - number_changed: numeric token set differs
    - unit_changed: unit token set differs
    - date_changed: date pattern difference (heuristic)
    - entity_changed: NER set differs (best-effort with spaCy)
    """
    x_nums, x_units, x_dates = _tokenize_simple(x)
    y_nums, y_units, y_dates = _tokenize_simple(y)
    number_changed = (x_nums != y_nums)
    unit_changed = (x_units != y_units)
    date_changed = (bool(x_dates) != bool(y_dates))

    entity_changed = False
    try:
        import spacy
        nlp = None
        if lang_code == 'es':
            try:
                nlp = spacy.load('es_core_news_sm')
            except Exception:
                nlp = spacy.blank('es')
        elif lang_code == 'en':
            try:
                nlp = spacy.load('en_core_web_sm')
            except Exception:
                nlp = spacy.blank('en')
        else:
            nlp = spacy.blank(lang_code)
        if nlp is not None and 'ner' in nlp.pipe_names:
            ex = set([ent.text.lower() for ent in nlp(x).ents])
            ey = set([ent.text.lower() for ent in nlp(y).ents])
            entity_changed = (ex != ey)
    except Exception:
        entity_changed = False

    # Optional semantic change using Sentence-Transformers (lazy, gated)
    semantic_bucket = 'none'
    semantic_score = 0.0
    if os.getenv('RISK_SEMANTIC', '1') in {'1','true','on','yes'}:
        try:
            # lazy import and singleton model cache
            from sentence_transformers import SentenceTransformer
            model_name = os.getenv('RISK_SEMANTIC_MODEL', 'paraphrase-multilingual-MiniLM-L12-v2')
            global _SEM_MODEL
            if '_SEM_MODEL' not in globals() or getattr(globals().get('_SEM_MODEL'), '_name', '') != model_name:
                _SEM_MODEL = SentenceTransformer(model_name)
                _SEM_MODEL._name = model_name
            # short-circuit trivial cases
            if (x or '').strip().lower() == (y or '').strip().lower():
                semantic_score = 0.0
            else:
                emb = _SEM_MODEL.encode([x, y], normalize_embeddings=True)
                # cosine distance = 1 - cosine similarity
                sim = float(emb[0].dot(emb[1]))
                semantic_score = max(0.0, min(1.0, 1.0 - sim))
            # Map distance to bucket conservatively
            if semantic_score < 0.10:
                semantic_bucket = 'none'
            elif semantic_score < 0.20:
                semantic_bucket = 'weak'
            elif semantic_score < 0.35:
                semantic_bucket = 'moderate'
            elif semantic_score < 0.50:
                semantic_bucket = 'strong'
            else:
                semantic_bucket = 'extreme'
        except Exception:
            semantic_bucket = 'none'
            semantic_score = 0.0

    # Information loss heuristic: significant content deletion without replacement
    # Heuristic: large relative length drop AND content token Jaccard drop
    def _content_tokens(t: str):
        # Unicode-friendly word tokens; then keep tokens that contain letters and length>=3
        toks = [w.lower() for w in re.findall(r"[\w]+", t, flags=re.UNICODE)]
        kept = []
        for w in toks:
            if len(w) < 3:
                continue
            if any(ch.isalpha() for ch in w):
                kept.append(w)
        return set(kept)

    info_loss = False
    try:
        x_tokens = _content_tokens(x)
        y_tokens = _content_tokens(y)
        # relative length drop on content tokens
        xl = max(1, len(x_tokens))
        yl = max(1, len(y_tokens))
        length_drop = (xl - yl) / float(xl)
        # jaccard similarity
        inter = len(x_tokens & y_tokens)
        union = len(x_tokens | y_tokens) or 1
        jacc = inter / union
        # info loss if large drop and low overlap, or entities removed
        removed_entities = False
        try:
            import spacy
            nlp = None
            if lang_code == 'es':
                try:
                    nlp = spacy.load('es_core_news_sm')
                except Exception:
                    nlp = spacy.blank('es')
            elif lang_code == 'en':
                try:
                    nlp = spacy.load('en_core_web_sm')
                except Exception:
                    nlp = spacy.blank('en')
            else:
                nlp = spacy.blank(lang_code)
            if nlp is not None and 'ner' in nlp.pipe_names:
                ex = set([ent.text.lower() for ent in nlp(x).ents])
                ey = set([ent.text.lower() for ent in nlp(y).ents])
                removed_entities = bool(ex - ey)
        except Exception:
            removed_entities = False

        info_loss = (length_drop >= 0.5 and jacc <= 0.4) or removed_entities
    except Exception:
        info_loss = False

    return {
        'number_changed': number_changed,
        'unit_changed': unit_changed,
        'date_changed': date_changed,
        'entity_changed': entity_changed,
        'semantic_bucket': semantic_bucket,
        'semantic_distance': semantic_score,
        'info_loss': info_loss,
    }


_OPEN_CLOSED = {
    '"': '"',
    "'": "'",
    '«': '»',
    '“': '”',
    '‘': '’',
    '(': ')',
    '[': ']',
    '{': '}',
}


def _balance_score(text: str) -> int:
    stack = []
    mismatches = 0
    for ch in text:
        if ch in _OPEN_CLOSED:
            # Treat symmetrical quotes as both open/close by context length
            if _OPEN_CLOSED[ch] == ch:
                if stack and stack[-1] == ch:
                    stack.pop()
                else:
                    stack.append(ch)
            else:
                stack.append(ch)
        elif ch in _OPEN_CLOSED.values():
            # find matching last open
            if stack and ch == _OPEN_CLOSED.get(stack[-1], ''):
                stack.pop()
            else:
                mismatches += 1
    return len(stack) + mismatches


def structural_regression(src: str, tgt: str) -> bool:
    """Return True if bracket/quote balance worsened from src to tgt."""
    return _balance_score(tgt) > _balance_score(src)


