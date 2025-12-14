#!/usr/bin/env python3
"""
Rulebooks → RAG index builder (research-grade, reproducible)

Features:
- Loads public rulebooks listed in docs/rag.md (JSON mapping)
- Parses Spanish PDF rulebook first (others supported generically)
- Heuristically extracts rule-like segments (filters non-rule text)
- Builds vector index (sentence-transformers) with FAISS if available,
  otherwise falls back to cosine search over NumPy for reproducibility
- Provides debug query CLI for:
  1) rule text
  2) src->tgt pair
  3) combined: rule + src->tgt

Output artifacts (per language, e.g., es):
- data/rulebooks_rag/es/meta.jsonl   (one JSON per rule: {key, title, text, source})
- data/rulebooks_rag/es/embeddings.npy
- data/rulebooks_rag/es/faiss.index  (when FAISS is available)

Usage:
  Build Spanish index:
    PYTHONPATH=. python processing/rules_to_rag.py build --lang es

  Debug queries (top-k results):
    PYTHONPATH=. python processing/rules_to_rag.py debug --lang es \
      --query "acento diacrítico" --k 5
    PYTHONPATH=. python processing/rules_to_rag.py debug --lang es \
      --src "mas" --tgt "más" --k 5
    PYTHONPATH=. python processing/rules_to_rag.py debug --lang es \
      --rule "concordancia de género" --src "libros roja" --tgt "libros rojos"

Notes:
- Spanish PDF parsing requires pdfminer.six (recommended). If unavailable, the
  script will warn and skip PDF extraction for that language.
"""

import os
import sys
import json
import re
import argparse
import pathlib
from typing import List, Dict, Tuple, Optional

import numpy as np

from tqdm import tqdm


# -------------------------------
# Utilities
# -------------------------------

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
RULEBOOKS_DIR = PROJECT_ROOT / "data" / "rulebooks"
RAG_OUT_DIR = PROJECT_ROOT / "data" / "rulebooks_rag"
DOCS_RAG = PROJECT_ROOT / "docs" / "rag.md"


def load_rag_sources() -> Dict[str, Dict[str, str]]:
    """Load docs/rag.md which contains a JSON mapping of languages to sources."""
    with open(DOCS_RAG, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"docs/rag.md is not valid JSON: {e}")


def find_rulebook_for_lang(lang: str) -> Optional[pathlib.Path]:
    """Locate a downloaded rulebook file for a given language in data/rulebooks."""
    if not RULEBOOKS_DIR.exists():
        return None
    candidates = list(RULEBOOKS_DIR.glob(f"{lang}_*"))
    return candidates[0] if candidates else None


def try_extract_text_from_pdf(pdf_path: pathlib.Path) -> str:
    """Extract text from PDF using pdfminer.six if available; else return empty string."""
    try:
        from pdfminer.high_level import extract_text  # type: ignore
    except Exception:
        print("[warn] pdfminer.six not available; cannot parse PDF:", pdf_path)
        return ""
    try:
        text = extract_text(str(pdf_path))
        return text or ""
    except Exception as e:
        print(f"[warn] PDF extraction failed for {pdf_path}: {e}")
        return ""


def try_extract_text_from_html(html_path: pathlib.Path) -> str:
    """Extract text from HTML using a minimal fallback (regex-based) to avoid new deps.
    For higher quality, install beautifulsoup4 and lxml.
    """
    try:
        raw = html_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        raw = html_path.read_text(errors="ignore")
    # Very naive strip of tags; good enough to prototype without adding deps
    text = re.sub(r"<script[\s\S]*?</script>", " ", raw, flags=re.IGNORECASE)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# -------------------------------
# Spanish-specific cleaning/segmentation
# -------------------------------

SPANISH_GRAMMAR_KEYWORDS = [
    # Orthography & accents
    "acento", "tilde", "diacrític", "ortograf", "mayúscul", "minúscul",
    # Agreement & morphology
    "concordancia", "género", "número", "verbo", "conjugación", "tiempo", "modo",
    # Syntax & clause
    "subjuntivo", "indicativo", "condicional", "oración", "sintaxis",
    # Determiners & contractions
    "artículo", "articulos", "artículo", "contracción", "preposición",
]


def clean_spanish_text(text: str) -> str:
    """Remove obvious non-rule lines (page numbers, headers) and normalize spaces."""
    lines = [ln.strip() for ln in text.splitlines()]
    cleaned: List[str] = []
    for ln in lines:
        if not ln:
            cleaned.append("")
            continue
        # Drop page numbers or isolated numbers
        if re.fullmatch(r"\d+", ln):
            continue
        # Drop typical header/footer fragments
        if len(ln) < 5 and not re.search(r"[a-zA-ZáéíóúñÁÉÍÓÚÑ]", ln):
            continue
        cleaned.append(ln)
    cleaned_text = "\n".join(cleaned)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    # Normalize bullets/section markers
    cleaned_text = re.sub(r"^[•·]\s*", "- ", cleaned_text, flags=re.MULTILINE)
    # Collapse excessive spaces
    cleaned_text = re.sub(r"[ \t]{2,}", " ", cleaned_text)
    return cleaned_text


def segment_into_rule_like_chunks_es(text: str) -> List[Tuple[str, str]]:
    """Heuristically segment Spanish text into rule-like chunks.

    Returns list of (rule_title, rule_body).
    """
    # Split paragraphs
    # First identify headings (ALLCAPS lines or numbered headings)
    lines = text.splitlines()
    blocks: List[str] = []
    buf: List[str] = []
    def is_heading(ln: str) -> bool:
        ln_s = ln.strip()
        if not ln_s:
            return False
        # ALLCAPS or numbered heading like 3.1.2 Concordancia
        if re.match(r"^[A-ZÁÉÍÓÚÑ0-9][A-ZÁÉÍÓÚÑ0-9 .:-]{3,}$", ln_s):
            return True
        if re.match(r"^\d+(?:\.\d+)*\s+[A-ZÁÉÍÓÚÑa-záéíóúñ].+", ln_s):
            return True
        return False

    for ln in lines:
        if is_heading(ln):
            if buf:
                blocks.append(" ".join(buf).strip())
                buf = []
            buf.append(ln.strip())
        else:
            buf.append(ln.strip())
    if buf:
        blocks.append(" ".join(buf).strip())

    paragraphs = [b for b in blocks if b]
    chunks: List[Tuple[str, str]] = []

    # Target chunk sizes
    min_len = 300
    max_len = 1200

    for p in paragraphs:
        p_norm = re.sub(r"\s+", " ", p).strip()

        # Split into sentence-like units
        sentences = re.split(r"(?<=[\.!?])\s+(?=[A-ZÁÉÍÓÚÑ])", p_norm)
        if len(sentences) == 1:
            # Try further split on semicolons if needed
            sentences = re.split(r";\s+", p_norm)

        # Walk a sliding window accumulating into mid-sized rule chunks
        buf: List[str] = []
        for s in sentences:
            if not s or s.isspace():
                continue
            buf.append(s)
            cur = " ".join(buf)
            cur_l = cur.lower()
            # If we reached a good size and content is grammar-related, emit
            if len(cur) >= min_len and any(kw in cur_l for kw in SPANISH_GRAMMAR_KEYWORDS):
                # Title: try to extract a short heading or first sentence
                heading_match = re.match(r"^([A-ZÁÉÍÓÚÑ0-9][A-ZÁÉÍÓÚÑ0-9 .:-]{3,}):?\s+", cur)
                if heading_match:
                    title = heading_match.group(1).strip()
                else:
                    first_sentence = re.split(r"(?<=[\.!?])\s+", cur)[0]
                    title = (first_sentence[:140].rsplit(" ", 1)[0] if len(first_sentence) > 140 else first_sentence).strip()

                # Bound maximum size by cutting at sentence boundary near max_len
                body = cur
                if len(body) > max_len:
                    cutoff_idx = body.rfind(". ", 0, max_len)
                    if cutoff_idx > 0:
                        body = body[:cutoff_idx+1]

                # Filter trivial titles
                if len(title) >= 20 and len(body) >= min_len:
                    chunks.append((title, body))
                buf = []

        # Flush remainder if meaningful
        if buf:
            cur = " ".join(buf)
            cur_l = cur.lower()
            if len(cur) >= min_len and any(kw in cur_l for kw in SPANISH_GRAMMAR_KEYWORDS):
                first_sentence = re.split(r"(?<=[\.!?])\s+", cur)[0]
                title = (first_sentence[:140].rsplit(" ", 1)[0] if len(first_sentence) > 140 else first_sentence).strip()
                if len(title) >= 20:
                    chunks.append((title, cur))

    # Deduplicate by title
    seen = set()
    deduped: List[Tuple[str, str]] = []
    for t, b in chunks:
        if t in seen:
            continue
        seen.add(t)
        deduped.append((t, b))
    return deduped


# -------------------------------
# Embedding + FAISS / Cosine backend
# -------------------------------

class VectorIndexer:
    """Embedding indexer with graceful fallbacks.

    Priority:
    1) sentence-transformers + FAISS (fast, high quality)
    2) TF-IDF char n-grams + NumPy cosine (no external deps)
    """
    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        self.mode = 'st_faiss'  # or 'tfidf'
        self.model = None
        self.dim = 0
        self.faiss = None
        self.index = None
        self.embeddings = None  # np.ndarray or sparse matrix
        self.vectorizer = None  # for TF-IDF fallback

        # Try sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self.model = SentenceTransformer(model_name)
            self.dim = getattr(self.model, 'get_sentence_embedding_dimension', lambda: 384)()
            try:
                import faiss  # type: ignore
                self.faiss = faiss
            except Exception:
                # Fall back to TF-IDF if FAISS unavailable
                self.mode = 'tfidf'
        except Exception:
            self.mode = 'tfidf'

    def encode(self, texts: List[str]):
        if self.mode == 'st_faiss' and self.model is not None:
            return np.asarray(self.model.encode(texts, normalize_embeddings=True))
        # TF-IDF fallback: if vectorizer exists, transform; else initialize dummy until build()
        if self.vectorizer is None:
            # Will be fitted in build(); here return placeholder
            return None
        return self.vectorizer.transform(texts)

    def build(self, embeddings):
        if self.mode == 'st_faiss' and embeddings is not None and self.faiss is not None:
            self.embeddings = embeddings.astype('float32')
            index = self.faiss.IndexFlatIP(self.embeddings.shape[1])
            index.add(self.embeddings)
            self.index = index
            return
        # TF-IDF fallback: fit vectorizer on corpus texts and store sparse matrix
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3,5), min_df=1, norm='l2')
        # embeddings arg in this mode is the raw texts list
        corpus_texts: List[str] = embeddings  # type: ignore
        self.embeddings = self.vectorizer.fit_transform(corpus_texts)

    def search(self, query_vec, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        if self.mode == 'st_faiss' and self.index is not None and query_vec is not None:
            q = query_vec.astype('float32')
            D, I = self.index.search(q, top_k)
            return I, D
        # TF-IDF fallback: compute cosine by dot product (L2-normalized)
        if self.vectorizer is None or self.embeddings is None:
            raise RuntimeError("TF-IDF index not initialized")
        # query_vec is sparse row
        if query_vec is None:
            raise RuntimeError("Query embedding missing")
        # Compute scores = q * E.T
        scores = query_vec @ self.embeddings.T  # shape (1, N)
        scores = np.asarray(scores.todense()).ravel()
        top_idx = np.argsort(scores)[::-1][:top_k]
        top_scores = scores[top_idx]
        return top_idx.reshape(1, -1), top_scores.reshape(1, -1)

    def save(self, out_dir: pathlib.Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        if self.mode == 'st_faiss':
            if self.embeddings is not None:
                np.save(out_dir / "embeddings.npy", self.embeddings)
            if self.faiss is not None and self.index is not None:
                try:
                    self.faiss.write_index(self.index, str(out_dir / "faiss.index"))
                except Exception as e:
                    print(f"[warn] Failed to save FAISS index: {e}")
        else:
            # Save vectorizer and skip FAISS
            try:
                import joblib  # type: ignore
            except Exception:
                joblib = None
            if joblib is not None and self.vectorizer is not None:
                joblib.dump(self.vectorizer, out_dir / "tfidf.joblib")

    def load(self, out_dir: pathlib.Path):
        # Try ST+FAISS first
        E_path = out_dir / "embeddings.npy"
        if E_path.exists():
            try:
                self.embeddings = np.load(E_path)
                import faiss  # type: ignore
                self.index = faiss.read_index(str(out_dir / "faiss.index"))
                self.faiss = faiss
                self.mode = 'st_faiss'
                return
            except Exception as e:
                print(f"[warn] Failed to load FAISS index: {e}")
                self.index = None
        # Fallback: TF-IDF
        try:
            import joblib  # type: ignore
            vec_path = out_dir / "tfidf.joblib"
            if vec_path.exists():
                self.vectorizer = joblib.load(vec_path)
                # embeddings will be produced on the fly; keep None
                self.mode = 'tfidf'
                return
        except Exception:
            pass
        # If nothing available
        raise SystemExit("No index artifacts found. Build the index first.")


# -------------------------------
# Build pipeline
# -------------------------------

def build_language(lang: str) -> None:
    src = load_rag_sources()
    if lang not in src:
        raise SystemExit(f"Language '{lang}' not found in docs/rag.md")

    rulebook_path = find_rulebook_for_lang(lang)
    if not rulebook_path:
        raise SystemExit(f"No rulebook found for '{lang}' in {RULEBOOKS_DIR}")

    print(f"[info] Building RAG for {lang} from {rulebook_path.name}")

    # Extract text based on file type
    text = ""
    if rulebook_path.suffix.lower() == ".pdf":
        text = try_extract_text_from_pdf(rulebook_path)
    elif rulebook_path.suffix.lower() in {".html", ".htm"}:
        text = try_extract_text_from_html(rulebook_path)
    else:
        try:
            text = rulebook_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            text = rulebook_path.read_text(errors="ignore")

    if not text:
        raise SystemExit(f"Failed to extract text from {rulebook_path}")

    # Language-specific cleaning and segmentation
    if lang == "es":
        cleaned = clean_spanish_text(text)
        chunks = segment_into_rule_like_chunks_es(cleaned)
    else:
        # Generic: split into paragraphs and keep longer ones
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        chunks = []
        for p in paragraphs:
            p_norm = re.sub(r"\s+", " ", p)
            if len(p_norm) < 80:
                continue
            title = p_norm[:120].rsplit(" ", 1)[0] if len(p_norm) > 120 else p_norm
            chunks.append((title, p_norm))

    if not chunks:
        raise SystemExit("No rule-like chunks extracted; tune segmentation heuristics.")

    # Build metadata and embeddings
    out_dir = RAG_OUT_DIR / lang
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = out_dir / "meta.jsonl"
    with open(meta_path, "w", encoding="utf-8") as f:
        for idx, (title, body) in enumerate(chunks):
            key = f"{lang}::rule_{idx:05d}"
            rec = {
                "key": key,
                "lang": lang,
                "title": title,
                "text": body,
                "source": str(rulebook_path.name),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Encode
    indexer = VectorIndexer()
    corpus_texts = [f"{title}\n\n{body}" for (title, body) in chunks]
    print(f"[info] Encoding/Indexing {len(corpus_texts)} segments…")
    if indexer.mode == 'st_faiss':
        embeddings = indexer.encode(corpus_texts)
        indexer.build(embeddings)
    else:
        # TF-IDF: build directly from raw texts
        indexer.build(corpus_texts)
    indexer.save(out_dir)

    # Simple quality stats
    avg_len = int(np.mean([len(t) for t in corpus_texts]))
    print(f"[ok] Built RAG for {lang}: {len(corpus_texts)} entries, avg chars {avg_len}")


def load_index(lang: str) -> Tuple[VectorIndexer, List[Dict[str, str]]]:
    out_dir = RAG_OUT_DIR / lang
    indexer = VectorIndexer()
    indexer.load(out_dir)
    meta: List[Dict[str, str]] = []
    meta_path = out_dir / "meta.jsonl"
    if not meta_path.exists():
        raise SystemExit(f"Index not found for '{lang}'. Build it first.")
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            meta.append(json.loads(line))
    return indexer, meta


def search(lang: str, rule: Optional[str], src: Optional[str], tgt: Optional[str], k: int) -> List[Dict[str, str]]:
    indexer, meta = load_index(lang)
    queries: List[str] = []

    if rule and src and tgt:
        queries = [f"{rule}\nsrc: {src}\ntgt: {tgt}"]
    elif src and tgt:
        queries = [f"src: {src}\ntgt: {tgt}"]
    elif rule:
        queries = [rule]
    else:
        raise SystemExit("Provide --rule and/or --src + --tgt")

    q_emb = indexer.encode(queries)
    # For TF-IDF, encode returns sparse; for ST, ndarray
    I, D = indexer.search(q_emb, top_k=k)

    results: List[Dict[str, str]] = []
    for rank, (idx, score) in enumerate(zip(I[0].tolist(), D[0].tolist()), start=1):
        rec = meta[idx]
        snippet = rec["text"][:220].replace("\n", " ") + ("…" if len(rec["text"]) > 220 else "")
        results.append({
            "rank": rank,
            "score": float(score),
            "key": rec["key"],
            "title": rec["title"],
            "snippet": snippet,
            "source": rec["source"],
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="Build and debug rulebook indices (RAG or LLM-assisted)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="Build FAISS/embeddings index for a language")
    p_build.add_argument("--lang", required=True, help="Language code, e.g., es")

    p_debug = sub.add_parser("debug", help="Run debug queries against an index")
    p_debug.add_argument("--lang", required=True)
    p_debug.add_argument("--rule", default=None)
    p_debug.add_argument("--src", default=None)
    p_debug.add_argument("--tgt", default=None)
    p_debug.add_argument("--k", type=int, default=5)

    p_llm = sub.add_parser("llm_build", help="LLM-assisted extraction of concise rules to data/rag/{lang}_rules_llm.json")
    p_llm.add_argument("--lang", required=True)
    p_llm.add_argument("--backend", default=os.getenv('RULEBOOK_BACKEND', 'gpt-4.1'))

    args = parser.parse_args()

    if args.cmd == "build":
        build_language(args.lang)
        return

    if args.cmd == "debug":
        res = search(args.lang, args.rule, args.src, args.tgt, args.k)
        print(json.dumps({"query": {"rule": args.rule, "src": args.src, "tgt": args.tgt},
                          "results": res}, ensure_ascii=False, indent=2))
        return

    if args.cmd == "llm_build":
        # Extract high-precision, compact rules using an LLM and save to JSON
        path = find_rulebook_for_lang(args.lang)
        if not path:
            raise SystemExit(f"No rulebook found for '{args.lang}' in {RULEBOOKS_DIR}")
        text = ""
        if path.suffix.lower() == ".pdf":
            text = try_extract_text_from_pdf(path)
        elif path.suffix.lower() in {".html", ".htm"}:
            text = try_extract_text_from_html(path)
        else:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                text = path.read_text(errors="ignore")
        if not text:
            raise SystemExit(f"Failed to extract text from {path}")

        from utils.judge import call_model_with_pricing
        prompt = (
            "Extract 120-200 concise, objective grammar/orthography/punctuation rules suitable for GEC judging.\n"
            "Each rule: id (lang-###), category (agreement/tense/orthography/punctuation/etc.), short description, 1-2 minimal examples.\n"
            "Return strict JSON: {\"rules\":[{\"id\":\"es-001\",\"category\":\"agreement\",\"description\":\"...\",\"examples\":[\"...\",\"...\"]}]}\n\n"
            "SOURCE TEXT (truncated):\n" + text[:150000]
        )
        ok, content, _toks, _pricing = call_model_with_pricing(prompt, args.backend, api_token=os.getenv('API_TOKEN',''), moderation=False)
        if not ok:
            raise SystemExit("LLM extraction failed")
        try:
            start = content.find('{'); end = content.rfind('}')
            data = json.loads(content[start:end+1]) if start!=-1 and end!=-1 else json.loads(content)
            if not isinstance(data.get('rules', []), list):
                raise ValueError('No rules list')
        except Exception as e:
            raise SystemExit(f"Bad JSON from LLM: {e}")

        out_dir = PROJECT_ROOT / 'data' / 'rag'
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{args.lang}_rules_llm.json"
        out_file.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        print(f"[ok] Saved LLM-extracted rules to {out_file}")
        return


if __name__ == "__main__":
    main()


