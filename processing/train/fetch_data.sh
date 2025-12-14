#!/usr/bin/env bash
set -euo pipefail

# Unified data fetcher for training corpora
# - cLang-8: expected at third-party/clang8/output_data
# - One Billion Word (Troy-1BW base)
# - Blog Authorship Corpus (Troy-Blogs base) — best-effort, may require manual
# - UberText is pulled via Hugging Face loader at runtime

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RAW_DIR="$ROOT_DIR/data/raw"
TP_DIR="$ROOT_DIR/third-party"

mkdir -p "$RAW_DIR/troy-1bw" "$RAW_DIR/troy-blogs" "$RAW_DIR/ubertext"

echo "== Fetch: One Billion Word (Troy-1BW base) =="
if [ ! -d "$RAW_DIR/troy-1bw/1-billion-word-language-modeling-benchmark-r13output" ]; then
  cd "$RAW_DIR/troy-1bw"
  if [ ! -f 1-billion-word-language-modeling-benchmark.tar.gz ]; then
    curl -L -o 1-billion-word-language-modeling-benchmark.tar.gz \
      "http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz" || true
  fi
  if [ -f 1-billion-word-language-modeling-benchmark.tar.gz ]; then
    tar -xzf 1-billion-word-language-modeling-benchmark.tar.gz || true
  fi
  cd - >/dev/null
else
  echo "Already present."
fi

echo "\n== Fetch: Blog Authorship Corpus (Troy-Blogs base) =="
cd "$RAW_DIR/troy-blogs"
if [ ! -d blogs ]; then
  if [ ! -f blogs.tar.gz ]; then
    echo "Attempting download from UCI..."
    curl -L -o blogs.tar.gz \
      "https://archive.ics.uci.edu/ml/machine-learning-databases/00304/blogs.tar.gz" || true
  fi
  if [ -f blogs.tar.gz ] && [ $(stat -f%z blogs.tar.gz || echo 0) -gt 1000000 ]; then
    tar -xzf blogs.tar.gz || true
  else
    echo "WARNING: blogs.tar.gz not available or too small. Please download manually from UCI."
  fi
else
  echo "Already present."
fi
cd - >/dev/null

echo "\n== Check: cLang-8 =="
if [ -f "$TP_DIR/clang8/output_data/clang8_source_target_en.tsv" ]; then
  echo "cLang-8 present."
else
  echo "cLang-8 missing. Please prepare third-party/clang8/output_data."
fi

echo "\n== Note: UberText =="
echo "UberText is loaded from Hugging Face at runtime: lang-uk/UberText-GEC"
echo "Done."


