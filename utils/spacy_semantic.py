from typing import List, Tuple


def analyze_edits_with_spacy(src: str, spans: List[str], lang_code: str) -> List[str]:
    """Produce concise spaCy-based cues for alignment spans.

    - For each {x=>y} in spans (from alignment), compute minimal cues in src:
      - agreement-likely: subj-verb, det-noun, adj-noun number/person mismatch
      - lexical vs inflectional: lemma(x) == lemma(y) (best-effort)
      - dep/pos context (very short)

    Returns up to 3 cue lines to limit prompt bloat.
    """
    lines: List[str] = []
    try:
        import spacy
        # Prepare nlp
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
            try:
                nlp = spacy.blank(lang_code)
            except Exception:
                nlp = spacy.blank('xx')

        doc = nlp(src)

        def morph_val(tok, key: str) -> str:
            vals = tok.morph.get(key)
            return vals[0] if vals else ''

        def agree_num(a, b) -> bool:
            va, vb = morph_val(a, 'Number'), morph_val(b, 'Number')
            return bool(va and vb and va == vb)

        def agree_pers(a, b) -> bool:
            va, vb = morph_val(a, 'Person'), morph_val(b, 'Person')
            return bool(va and vb and va == vb)

        for span in spans:
            if len(lines) >= 3:
                break
            try:
                body = span[1:-1]
                x, y = body.split('=>', 1)
                x = x.strip(); y = y.strip()

                # Find token for x (best-effort by surface)
                idx = next((i for i, t in enumerate(doc) if t.text == x), -1)
                if idx == -1:
                    idx = next((i for i, t in enumerate(doc) if t.text.lower() == x.lower()), -1)

                label = 'lexical-likely'
                ctx = ''
                if idx != -1:
                    t = doc[idx]
                    ctx = t.pos_ or ''
                    if t.pos_ == 'VERB':
                        subj = next((ch for ch in t.children if 'subj' in ch.dep_), None)
                        if subj and (not agree_num(t, subj) or not agree_pers(t, subj)):
                            label = 'agreement-likely'
                    elif t.pos_ in ('NOUN','PROPN'):
                        mismatch = False
                        num_t = morph_val(t, 'Number')
                        if num_t:
                            for ch in t.children:
                                if ch.pos_ in ('DET','ADJ'):
                                    if morph_val(ch, 'Number') and morph_val(ch, 'Number') != num_t:
                                        mismatch = True; break
                            if not mismatch and idx > 0 and doc[idx-1].pos_ == 'DET':
                                if morph_val(doc[idx-1], 'Number') and morph_val(doc[idx-1], 'Number') != num_t:
                                    mismatch = True
                        if mismatch:
                            label = 'agreement-likely'
                    elif t.pos_ == 'ADJ' and t.head is not None and t.head.pos_ in ('NOUN','PROPN'):
                        if not agree_num(t, t.head):
                            label = 'agreement-likely'

                # Lemma check (approximate inflectional change vs lexical):
                try:
                    y_doc = nlp(y)
                    if idx != -1 and len(y_doc):
                        x_lemma = doc[idx].lemma_ if doc[idx].lemma_ else doc[idx].text
                        y_lemma = y_doc[0].lemma_ if y_doc[0].lemma_ else y_doc[0].text
                        if x_lemma == y_lemma:
                            # keep lexical-likely if context suggests; otherwise mark inflectional
                            if label != 'agreement-likely':
                                label = 'inflectional-likely'
                except Exception:
                    pass

                context_short = f" ({ctx})" if ctx else ''
                lines.append(f"- {span}: {label}{context_short}")
            except Exception:
                continue
        return lines
    except Exception:
        return lines









