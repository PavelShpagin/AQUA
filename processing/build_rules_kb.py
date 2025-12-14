#!/usr/bin/env python3
"""
Build a compact rule Knowledge Base (KB) per language using an LLM (default: o3/gpt-4o-mini).

Pipeline (research-grade, reproducible, laconic):
1) Load rulebook for --lang from data/rulebooks/{lang}_*.pdf (or .html/.txt)
2) Extract headings and short content cues (LLM-friendly context)
3) Phase 1 LLM: generate ≥100 rule keys (titles) as pure JSON
4) Phase 2 LLM (batched): expand keys into {key,title,brief,category,examples}
5) Save to data/rules_kb/{lang}.json and run quick query tests

Usage:
  Build KB:   PYTHONPATH=. python processing/build_rules_kb.py build --lang es --backend o3
  Query test: PYTHONPATH=. python processing/build_rules_kb.py query --lang es --q "concordancia de género"
"""

import os
import sys
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.judge import call_model_with_pricing


RULEBOOKS_DIR = ROOT / "data" / "rulebooks"
OUT_DIR = ROOT / "data" / "rules_kb"


# Add simple PDF page splitter when pdfminer is available
def _split_pdf_to_pages(path: Path) -> List[str]:
    try:
        from pdfminer.high_level import extract_text
    except Exception:
        return []
    # Fallback: pdfminer doesn't provide per-page extraction directly without layout; naive chunk
    text = extract_text(str(path)) or ""
    # Heuristic: split by form feed or big gaps
    parts = re.split(r"\f|\n\s*\n\s*\n", text)
    return [p for p in parts if p and p.strip()]


def _load_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pdfminer.high_level import extract_text  # pdfminer.six
        return extract_text(str(path)) or ""
    elif path.suffix.lower() in {".html", ".htm"}:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        raw = re.sub(r"<script[\s\S]*?</script>", " ", raw, flags=re.I)
        raw = re.sub(r"<style[\s\S]*?</style>", " ", raw, flags=re.I)
        text = re.sub(r"<[^>]+>", " ", raw)
        return re.sub(r"\s+", " ", text).strip()
    else:
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return path.read_text(errors="ignore")


def _find_rulebook(lang: str) -> Path:
    cands = list(RULEBOOKS_DIR.glob(f"{lang}_*"))
    if not cands:
        raise SystemExit(f"No rulebook found in {RULEBOOKS_DIR} for lang={lang}")
    # Prefer PDF
    pdfs = [p for p in cands if p.suffix.lower() == ".pdf"]
    rb = pdfs[0] if pdfs else cands[0]
    print(f"[info] Rulebook selected: {rb}")
    return rb


def _extract_headings_context(text: str, max_items: int = 220) -> List[str]:
    """Extract pseudo-headings and short lines; limit count to keep context compact."""
    lines = [ln.strip() for ln in text.splitlines()]
    heads: List[str] = []
    def is_heading(ln: str) -> bool:
        ln_s = ln.strip()
        if not ln_s:
            return False
        if re.match(r"^[A-ZÁÉÍÓÚÑ0-9][A-ZÁÉÍÓÚÑ0-9 .,:;\-]{3,}$", ln_s):
            return True
        if re.match(r"^\d+(?:\.\d+)*\s+.+", ln_s):
            return True
        return False
    for ln in lines:
        if is_heading(ln):
            heads.append(ln.strip())
        elif 40 <= len(ln) <= 160 and any(k in ln.lower() for k in ["concord", "género", "número", "subjunt", "acento", "tilde", "contracci", "prepos", "artícul", "verbo", "tiempo", "modo"]):
            heads.append(ln.strip())
        if len(heads) >= max_items:
            break
    print(f"[info] Extracted heading cues: {len(heads)}")
    return heads


def _call_llm(prompt: str, backend: str) -> Tuple[bool, str]:
    # Minimal same-backend retry for transient proxy/network hiccups (no fallbacks)
    last_err = ""
    for attempt in range(3):
        ok, content, tokens, pricing = call_model_with_pricing(
            prompt,
            backend,
            api_token=os.getenv('OPENAI_API_KEY') or os.getenv('API_TOKEN'),
            temperature_override=0.0,
        )
        if ok and content:
            return True, content
        last_err = content or ""
    raise SystemExit(f"LLM call failed for backend={backend}{(': ' + last_err) if last_err else ''}")


def _phase1_keys(lang: str, heads: List[str], backend: str, target_n: int = 120) -> List[str]:
    ctx = "\n".join(f"- {h}" for h in heads[:220])
    prompt = f"""
You are building a compact rule Knowledge Base for {lang} grammar.
Task: From the provided headings/snippets, output a JSON object {{"keys": [k1, k2, ...]}} with at least {target_n} unique, prescriptive rule keys (short, snake_case, ≤60 chars). No prose.
Rules should cover orthography (accents), agreement (sujeto–verbo, sustantivo–adjetivo), contractions, articles, prepositions, verb moods/tenses, punctuation, and common FP patterns.

HEADINGS:
{ctx}

Output ONLY valid JSON with a list named "keys".
"""
    print(f"[phase1] Requesting ≥{target_n} keys via {backend}…")
    ok, content = _call_llm(prompt, backend)
    if not ok:
        raise SystemExit("Phase1 key generation failed")
    try:
        start = content.find('{'); end = content.rfind('}') + 1
        data = json.loads(content[start:end])
        keys = [str(k).strip() for k in data.get('keys', [])]
        # Dedup + filter
        out = []
        seen = set()
        for k in keys:
            if len(k) >= 4 and len(k) <= 60 and k not in seen:
                seen.add(k)
                out.append(k)
        out = out[:max(target_n, 100)]
        print(f"[phase1] Got keys: {len(out)}")
        return out
    except Exception as e:
        raise SystemExit(f"Phase1 JSON parse error: {e}")


def _phase2_expand(lang: str, heads: List[str], keys: List[str], backend: str, batch: int = 40) -> List[Dict[str, Any]]:
    ctx = "\n".join(f"- {h}" for h in heads[:120])
    results: List[Dict[str, Any]] = []
    for i in range(0, len(keys), batch):
        sub = keys[i:i+batch]
        prompt = f"""
You are generating KB entries for {lang} grammar rules. Use the cues to stay faithful and concise.
Input:
HEADINGS:\n{ctx}
KEYS:\n{json.dumps(sub, ensure_ascii=False)}

For each key, output an object with:
- key (same as input)
- title (≤80 chars, human-readable)
- brief (≤160 chars, prescriptive, actionable)
- category (one of: orthography, agreement, contractions, articles, prepositions, verbs, punctuation, style)
- examples: array of 1–2 objects {{"before":"...","after":"..."}}

Respond ONLY JSON array for these keys.
"""
        print(f"[phase2] Expanding keys {i+1}..{i+len(sub)} via {backend}…")
        ok, content = _call_llm(prompt, backend)
        if not ok:
            continue
        try:
            block = content[content.find('['): content.rfind(']')+1]
            arr = json.loads(block)
            for item in arr:
                if isinstance(item, dict) and item.get('key') in sub:
                    results.append(item)
        except Exception:
            continue
    print(f"[phase2] Expanded entries: {len(results)}")
    return results


def build_kb(lang: str, backend: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rulebook = _find_rulebook(lang)
    if rulebook.suffix.lower() != ".pdf":
        raise SystemExit("Only PDF rulebooks are supported in strict chunking mode.")
    print("[info] Strict chunking mode (≈10 pages per chunk)…")
    pages = _split_pdf_to_pages(rulebook)
    if not pages:
        raise SystemExit("PDF could not be split into pages.")
    from tqdm import tqdm
    chunk_size = 10
    all_keys: List[str] = []
    chunk_heads: List[List[str]] = []
    for i in tqdm(range(0, len(pages), chunk_size), desc="chunks(≈10pp)"):
        chunk_text = "\n".join(pages[i:i+chunk_size])
        h = _extract_headings_context(chunk_text)
        chunk_heads.append(h)
        ks = _phase1_keys(lang, h, backend, target_n=40)
        all_keys.extend(ks)
    # Deduplicate keys
    seen = set()
    keys = []
    for k in all_keys:
        if k not in seen:
            seen.add(k)
            keys.append(k)
    print(f"[chunked] Aggregated keys (dedup): {len(keys)}")
    # Distribute keys across chunks and expand
    uniq_entries: Dict[str, Any] = {}
    if not chunk_heads:
        raise SystemExit("No chunk heads extracted.")
    per_chunk = max(1, (len(keys) + len(chunk_heads) - 1) // len(chunk_heads))
    for idx, h in enumerate(tqdm(chunk_heads, desc="expand(≈50pp)")):
        start = idx * per_chunk
        end = min(len(keys), (idx + 1) * per_chunk)
        if start >= end:
            break
        sub_keys = keys[start:end]
        sub = _phase2_expand(lang, h, sub_keys, backend, batch=40)
        for e in sub:
            k = e.get('key')
            if k and k not in uniq_entries:
                uniq_entries[k] = e
    entries = list(uniq_entries.values())
    print(f"[final] Entries total before dedup: {len(entries)}")
    # Dedup by key (final safety)
    uniq = {}
    for e in entries:
        k = e.get('key')
        if k and k not in uniq:
            uniq[k] = e
    out = {
        'lang': lang,
        'source': rulebook.name,
        'num_rules': len(uniq),
        'rules': list(uniq.values())
    }
    out_path = OUT_DIR / f"{lang}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"KB saved: {out_path} with {out['num_rules']} rules")
    return out_path


def query_kb(lang: str, q: str, top: int = 5) -> None:
    kb_path = OUT_DIR / f"{lang}.json"
    if not kb_path.exists():
        print(f"KB not found: {kb_path}")
        return
    data = json.loads(kb_path.read_text())
    rules = data.get('rules', [])
    # Simple scoring: count keyword overlap
    ql = q.lower()
    scored = []
    for r in rules:
        text = f"{r.get('key','')} {r.get('title','')} {r.get('brief','')}".lower()
        score = sum(1 for tok in ql.split() if tok in text)
        scored.append((score, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    for s, r in scored[:top]:
        ex = r.get('examples') or []
        exs = "; ".join([f"{e.get('before','')} => {e.get('after','')}" for e in ex][:2])
        print(f"[{s}] {r.get('key','')} | {r.get('title','')} | {r.get('brief','')} | ex: {exs}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)

    p_build = sub.add_parser('build')
    p_build.add_argument('--lang', required=True)
    p_build.add_argument('--backend', default='o3', help='o3 or gpt-4o-mini etc.')

    p_query = sub.add_parser('query')
    p_query.add_argument('--lang', required=True)
    p_query.add_argument('--q', required=True)

    args = ap.parse_args()
    if args.cmd == 'build':
        build_kb(args.lang, args.backend)
    elif args.cmd == 'query':
        query_kb(args.lang, args.q)


if __name__ == '__main__':
    main()



