from typing import Dict, List, Tuple
import re

def _cosine(u, v) -> float:
    import math
    nu = math.sqrt(sum(x*x for x in u)) or 1e-9
    nv = math.sqrt(sum(x*x for x in v)) or 1e-9
    return sum(x*y for x, y in zip(u, v)) / (nu * nv)

def _mean_pool(last_hidden_state, attention_mask):
    import numpy as np
    masked = last_hidden_state * attention_mask[:, :, None]
    denom = attention_mask.sum(axis=1)[:, None].clip(min=1)
    return masked.sum(axis=1) / denom

def _bucketize(score: float) -> str:
    # score is cosine similarity in [−1,1]; convert to distance = 1 - sim in [0,2]
    dist = 1.0 - score
    # 5 buckets: none, slight, moderate, strong, extreme
    if dist < 0.10:
        return 'none'
    if dist < 0.25:
        return 'slight'
    if dist < 0.45:
        return 'moderate'
    if dist < 0.70:
        return 'strong'
    return 'extreme'

def compute_pair_similarity(texts: List[str], model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2') -> List[List[float]]:
    """Return cosine similarity matrix for texts (len N) using best-available backend.

    Tries sentence-transformers; falls back to a simple hashed bag-of-words vector to remain robust.
    """
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        m = SentenceTransformer(model_name)
        embs = m.encode(texts, normalize_embeddings=True)
        sims = (embs @ embs.T).tolist()
        return sims
    except Exception:
        # Fallback: hashed TF with char ngrams
        import numpy as np
        from collections import Counter
        vecs = []
        for t in texts:
            t = (t or '').lower()
            grams = [t[i:i+3] for i in range(max(0, len(t)-2))]
            c = Counter(grams)
            keys = sorted(c.keys())
            vec = [c[k] for k in keys]
            # simple L2 norm
            import math
            n = math.sqrt(sum(x*x for x in vec)) or 1e-9
            vecs.append([x/n for x in vec])
        # pad to same length
        maxlen = max((len(v) for v in vecs), default=0)
        padded = [v + [0.0]*(maxlen-len(v)) for v in vecs]
        embs = np.array(padded, dtype=float)
        sims = (embs @ embs.T).tolist()
        return sims

def meaning_change_buckets(edits: List[Tuple[str, str]], model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2') -> Dict[str, str]:
    """For each (x,y) span pair compute similarity bucket and return mapping "{x=>y}": bucket.
    """
    # Prepare unique strings to embed once
    uniq: Dict[str, int] = {}
    seq: List[str] = []
    def idx(s: str) -> int:
        if s not in uniq:
            uniq[s] = len(seq)
            seq.append(s)
        return uniq[s]

    pairs: List[Tuple[int,int,str]] = []
    for x, y in edits:
        ix = idx(x.strip())
        iy = idx(y.strip())
        pairs.append((ix, iy, f"{{{x.strip()}=>{y.strip()}}}"))

    sims = compute_pair_similarity(seq, model_name)
    out: Dict[str, str] = {}
    for ix, iy, key in pairs:
        sim = sims[ix][iy]
        out[key] = _bucketize(sim)
    return out



def _neg_markers_for_lang(lang_code: str) -> List[str]:
    lc = (lang_code or '').lower()
    if lc.startswith('es'):
        return [
            'no','nunca','jamás','ni','tampoco','sin','nada','nadie','ningún','ninguna','ninguno','ningunas','ningunos','jamás',
            "n't"  # in case of mixed text
        ]
    # default/en
    return ['no','not',"n't",'never','neither','nor','without','none','nobody','nothing']


def negation_flip(x: str, y: str, lang_code: str) -> bool:
    """Detects introduction/removal of negation markers between x and y.

    Conservative: returns True only if negation presence differs and tokens are not purely punctuation changes.
    """
    try:
        markers = _neg_markers_for_lang(lang_code)
        def tokens(t: str) -> List[str]:
            return re.findall(r"[\w']+", (t or '').lower(), flags=re.UNICODE)
        tx = tokens(x)
        ty = tokens(y)
        has_neg_x = any(m in tx for m in markers)
        has_neg_y = any(m in ty for m in markers)
        if has_neg_x == has_neg_y:
            return False
        # require some lexical difference beyond punctuation
        letters_x = any(ch.isalpha() for ch in x)
        letters_y = any(ch.isalpha() for ch in y)
        if not (letters_x and letters_y):
            return False
        return True
    except Exception:
        return False


