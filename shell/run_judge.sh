#!/usr/bin/env bash
# Orchestrator for GEC Judges - calls ensembles which call judges
#
# Usage example:
#   bash shell/run_judge.sh \
#     --judge feedback \
#     --method modular \
#     --backends "gpt-4o-mini gpt-4o" \
#     --lang en \
#     --ensemble weighted \
#     --n_judges 2
#
set -euo pipefail

# Defaults
CONFIG=""
JUDGE=""
METHOD=""
BACKENDS_STR=""
LANG="en"
ENSEMBLE="weighted"
N_JUDGES=1
WORKERS=200
INPUT=""
PREF=""
TGT_COL=""
MODERATION="off"
GOLD=""
DEBUG="off"
OPTIMIZATION="off"
BATCH="off"
SAMPLES=""
FILTER=""
SHARD_SIZE=""
POST_PASS="off"
ROUTER=""
ESCALATE_LABELS=""
ESCALATE_FLAGS=""

# Initialize environment
export PYTHONPATH="${PYTHONPATH:-.}:."
export QUIET_LOGS="${QUIET_LOGS:-0}"

# Prefer project venv Python on PATH automatically
ROOT_DIR="$(cd "$(dirname "$0")"/.. && pwd)"
if [[ -d "$ROOT_DIR/venv/bin" ]]; then
  export PATH="$ROOT_DIR/venv/bin:$PATH"
fi

# Performance tuning environment variables (optimized based on bottleneck analysis)
export MAX_WORKERS="${MAX_WORKERS:-200}"
export ASYNC_WORKERS="${ASYNC_WORKERS:-200}" 
export CONN_TIMEOUT="${CONN_TIMEOUT:-1.5}"
export READ_TIMEOUT="${READ_TIMEOUT:-8.0}"
export LLM_BATCH_SIZE="${LLM_BATCH_SIZE:-20}"
# Let Python auto-detect cores unless user explicitly sets ALIGN_NPROC
export ALIGN_NPROC="${ALIGN_NPROC:-}"
# High-performance defaults; user can still override via env
export ALIGN_DOC_BATCH="${ALIGN_DOC_BATCH:-2048}"
export ALIGN_CHUNK="${ALIGN_CHUNK:-16384}"
unset SAFE_MAX_WORKERS || true

if [[ "${QUIET_LOGS:-0}" != "1" ]]; then
  echo "INFO: Initializing optimized processing infrastructure" >&2
fi

# Load ENV file if exists - FIXED: ensure proper export to subprocesses
if [[ -f .env ]]; then
  # Export all variables from .env to ensure they're inherited by Python subprocesses
  while IFS='=' read -r key value; do
    # Skip comments and empty lines
    [[ $key =~ ^[[:space:]]*# ]] && continue
    [[ -z "$key" ]] && continue
    # Remove quotes if present and export
    value=$(echo "$value" | sed 's/^"\(.*\)"$/\1/' | sed "s/^'\(.*\)'$/\1/")
    export "$key=$value"
  done < .env
  
  # Normalize lowercase to uppercase for keys the code expects
  if [[ -n "${openai_api_key:-}" && -z "${OPENAI_API_KEY:-}" ]]; then export OPENAI_API_KEY="${openai_api_key}"; fi
  if [[ -n "${openai_api_token:-}" && -z "${OPENAI_API_KEY:-}" ]]; then export OPENAI_API_KEY="${openai_api_token}"; fi
  if [[ -n "${api_token:-}" && -z "${API_TOKEN:-}" ]]; then export API_TOKEN="${api_token}"; fi
  # Support multiple OpenAI keys via OPENAI_API_KEYS (comma/space separated)
  if [[ -n "${OPENAI_KEYS:-}" && -z "${OPENAI_API_KEYS:-}" ]]; then export OPENAI_API_KEYS="${OPENAI_KEYS}"; fi
fi

# Helper to uppercase variable names from YAML using Python
load_config() {
  local cfg_path="$1"
  if [[ -f "$cfg_path" ]]; then
    echo "Loading config from $cfg_path"
    # Export CFG_* variables to current shell
    eval "$(python3 - "$cfg_path" <<'PY'
import yaml, sys, shlex
cfg = yaml.safe_load(open(sys.argv[1])) or {}
for k, v in cfg.items():
    if isinstance(v, list):
        v = ' '.join(map(str, v))
    print(f'CFG_{k.upper()}={shlex.quote(str(v))}')
PY
)"
  else
    echo "Config file $cfg_path not found"; exit 1
  fi
}

# Argument parsing
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2;;
    --judge) JUDGE="$2"; shift 2;;
    --method) METHOD="$2"; shift 2;;
    --backends) BACKENDS_STR="$2"; shift 2;;
    --lang) LANG="$2"; shift 2;;
    --ensemble) ENSEMBLE="$2"; shift 2;;
    --n_judges) N_JUDGES="$2"; shift 2;;
    --workers) WORKERS="$2"; shift 2;;
    --input) INPUT="$2"; shift 2;;
    --pref) PREF="$2"; shift 2;;
    --tgt) TGT_COL="$2"; shift 2;;
    --moderation) MODERATION="on"; shift 1;;
    --gold) GOLD="$2"; shift 2;;
    --debug) DEBUG="on"; shift 1;;
    --optimization) OPTIMIZATION="${2:-speed}"; shift 2;;
    --filter) FILTER="$2"; shift 2;;
    --samples) SAMPLES="$2"; shift 2;;
    --shard_size) SHARD_SIZE="$2"; shift 2;;
    --post_pass) POST_PASS="$2"; shift 2;;
    --router) ROUTER="$2"; shift 2;;
    --escalate_labels) ESCALATE_LABELS="$2"; shift 2;;
    --escalate_flags) ESCALATE_FLAGS="$2"; shift 2;;
    *) echo "Unknown argument: $1"; exit 1;;
  esac
done

# Load YAML config if provided *after* CLI args so CLI overrides
if [[ -n "$CONFIG" ]]; then
  load_config "$CONFIG"
  # YAML vars are now in CFG_* - use config values when CLI args are empty
  [[ -z "$JUDGE" ]] && JUDGE="${CFG_JUDGE:-}"
  [[ -z "$METHOD" ]] && METHOD="${CFG_METHOD:-}"
  [[ -z "$BACKENDS_STR" ]] && BACKENDS_STR="${CFG_BACKENDS:-}"
  [[ "$LANG" == "en" && -n "${CFG_LANG:-}" ]] && LANG="${CFG_LANG}"
  [[ "$ENSEMBLE" == "weighted" && -n "${CFG_ENSEMBLE:-}" ]] && ENSEMBLE="${CFG_ENSEMBLE}"
  [[ "$N_JUDGES" == "1" && -n "${CFG_N_JUDGES:-}" ]] && N_JUDGES="${CFG_N_JUDGES}"
  [[ -z "$INPUT" ]] && INPUT="${CFG_INPUT:-}"
  [[ -z "$PREF" ]] && PREF="${CFG_PREF:-}"
  [[ -z "$TGT_COL" ]] && TGT_COL="${CFG_TGT:-}"
  [[ "$WORKERS" == "200" && -n "${CFG_WORKERS:-}" ]] && WORKERS="${CFG_WORKERS}"
  [[ -z "$GOLD" ]] && GOLD="${CFG_GOLD:-}"
  [[ -z "$FILTER" && -n "${CFG_FILTER:-}" ]] && FILTER="${CFG_FILTER}"
  [[ -z "$SAMPLES" && -n "${CFG_SAMPLES:-}" ]] && SAMPLES="${CFG_SAMPLES}"
  [[ -z "$ROUTER" && -n "${CFG_ROUTER:-}" ]] && ROUTER="${CFG_ROUTER}"
  [[ -z "$ESCALATE_LABELS" && -n "${CFG_ESCALATE_LABELS:-}" ]] && ESCALATE_LABELS="${CFG_ESCALATE_LABELS}"
  [[ -z "$ESCALATE_FLAGS" && -n "${CFG_ESCALATE_FLAGS:-}" ]] && ESCALATE_FLAGS="${CFG_ESCALATE_FLAGS}"
  # Map moderation from config (accept on/true/1/yes)
  if [[ "$MODERATION" == "off" && -n "${CFG_MODERATION:-}" ]]; then
    val="${CFG_MODERATION}"
    shopt -s nocasematch
    if [[ "$val" == "on" || "$val" == "true" || "$val" == "yes" || "$val" == "1" ]]; then
      MODERATION="on"
    fi
    shopt -u nocasematch
  fi
  # Map debug from config (accept on/true/1/yes)
  if [[ "$DEBUG" == "off" && -n "${CFG_DEBUG:-}" ]]; then
    val="${CFG_DEBUG}"
    shopt -s nocasematch
    if [[ "$val" == "on" || "$val" == "true" || "$val" == "yes" || "$val" == "1" ]]; then
      DEBUG="on"
    fi
    shopt -u nocasematch
  fi
  # Map optimization from config (accept on/true/1/yes)
  if [[ "$OPTIMIZATION" == "off" && -n "${CFG_OPTIMIZATION:-}" ]]; then
    val="${CFG_OPTIMIZATION}"
    shopt -s nocasematch
    if [[ "$val" == "on" || "$val" == "true" || "$val" == "yes" || "$val" == "1" ]]; then
      OPTIMIZATION="on"
    fi
    shopt -u nocasematch
  fi
  # Map batch from config (accept on/true/1/yes)
  if [[ "$BATCH" == "off" && -n "${CFG_BATCH:-}" ]]; then
    val="${CFG_BATCH}"
    shopt -s nocasematch
    if [[ "$val" == "on" || "$val" == "true" || "$val" == "yes" || "$val" == "1" ]]; then
      BATCH="on"
    fi
    shopt -u nocasematch
  fi
  # Remove optional safety caps to keep config clean unless explicitly set via env
  unset SAFE_MAX_WORKERS || true
  unset HTTP_MAX_OUTSTANDING || true
fi

# Sanity checks
if [[ -z "$JUDGE" || -z "$METHOD" || -z "$BACKENDS_STR" ]]; then
  echo "Parameters --judge, --method, and --backends are required (or via --config)."; exit 1
fi

IFS=' ' read -ra BACKENDS <<< "$BACKENDS_STR"
BACKENDS_LEN=${#BACKENDS[@]}
if [[ $BACKENDS_LEN -eq 0 ]]; then
  echo "No backends supplied"; exit 1
fi

# Require explicit input unless GOLD is provided; if GOLD present, use it as input
if [[ -z "$INPUT" ]]; then
  if [[ -n "$GOLD" && -f "$GOLD" ]]; then
    echo "No input specified. Using GOLD as input: $GOLD"
    INPUT="$GOLD"
  else
    echo "No input specified. Provide --input or set input in config.yaml"; exit 1
  fi
fi

JOINED_BACKENDS=$(echo "$BACKENDS_STR" | tr ' ' '_')
# By default: data/results/{judge}_{method}_{backends}_{lang}
# If --pref is provided, use it as a PREFIX for the folder name
if [[ -n "$PREF" ]]; then
  OUTPUT_DIR="data/results/${PREF}_${JUDGE}_${METHOD}_${JOINED_BACKENDS}_${LANG}"
else
  OUTPUT_DIR="data/results/${JUDGE}_${METHOD}_${JOINED_BACKENDS}_${LANG}"
fi
mkdir -p "$OUTPUT_DIR"

# If SAMPLES requested, create a sampled input so ERRANT runs only on that subset
ORIG_INPUT="$INPUT"
if [[ -n "$SAMPLES" && "$SAMPLES" -gt 0 ]]; then
  SAMPLE_INPUT="$OUTPUT_DIR/input_head_${SAMPLES}.csv"
  python3 - << PY
import pandas as pd
pd.read_csv(r"$ORIG_INPUT").head(int($SAMPLES)).to_csv(r"$SAMPLE_INPUT", index=False)
print(f"Sampled input saved to: $SAMPLE_INPUT")
PY
  INPUT="$SAMPLE_INPUT"
fi

# Preprocess: batch ERRANT alignment for non-legacy methods (skip if already aligned exists)
PREPROCESSED_INPUT="$INPUT"
if [[ "$METHOD" != "legacy" ]]; then
  # If feedback judge and ERRANT disabled, skip pre-alignment entirely
  shopt -s nocasematch
  if [[ "$JUDGE" == "feedback" && "${FEEDBACK_ERRANT:-on}" =~ ^(off|false|0|no)$ ]]; then
    echo "Skipping ERRANT pre-alignment for feedback (ERRANT=off)"
    PREPROCESSED_INPUT="$INPUT"
  else
    # Use a per-input prealigned path to avoid cross-run mixing
    PRE_BASENAME="$(basename "$INPUT")"
    PRE_STEM="${PRE_BASENAME%.*}"
    PREPROCESSED_INPUT="$OUTPUT_DIR/preprocessed_with_aligned_${PRE_STEM}.csv"
    # Always run ERRANT pre-alignment to ensure fresh alignments (no caching)
    RJ_INPUT="$INPUT" RJ_OUT="$PREPROCESSED_INPUT" RJ_LANG="$LANG" RJ_FILTER="$FILTER" RJ_FILTER_ALIGNED="${CFG_FILTER_ALIGNED:-}" python3 - << 'PY'
import os, sys, time
import pandas as pd
sys.path.insert(0, '.')
from utils.errant_align import _align_worker  # reuse long-lived workers
from utils.progress import SingleLineProgress
import langid
from concurrent.futures import ProcessPoolExecutor

inp = os.environ.get('RJ_INPUT')
outp = os.environ.get('RJ_OUT')
lang = os.environ.get('RJ_LANG','auto')
filter_cols = [c for c in os.environ.get('RJ_FILTER','').split() if c]
filter_aligned_cols = [c for c in os.environ.get('RJ_FILTER_ALIGNED','').split() if c]
quiet = os.environ.get('QUIET_LOGS') == '1'

first = True
total = 0
start = time.time()

USE_PRECOMP_ALIGNED = bool(filter_aligned_cols)

# Auto-detect language if requested and not relying solely on precomputed alignments
if (not lang or lang.lower() == 'auto') and not USE_PRECOMP_ALIGNED:
    sample_texts = []
    for chunk in pd.read_csv(inp, encoding='utf-8', quotechar='"', escapechar=None, chunksize=20000):
        sample_texts.extend(chunk.get('src', pd.Series(dtype=str)).astype(str).head(200).tolist())
        if len(sample_texts) >= 1000:
            break
    if sample_texts:
        # Vote by langid over sample
        counts = {}
        for t in sample_texts:
            code, _ = langid.classify(t[:200])
            counts[code] = counts.get(code, 0) + 1
        lang = max(counts.items(), key=lambda kv: kv[1])[0]
    else:
        lang = 'en'
    if not quiet:
        print(f"Auto-detected language: {lang}")

# Count total rows for progress (cheap pass)
try:
    total_rows = 0
    for c in pd.read_csv(inp, encoding='utf-8', quotechar='"', escapechar=None, chunksize=50000):
        total_rows += len(c)
except Exception:
    total_rows = 0

prog = SingleLineProgress(total_rows or 1, desc="ERRANT pre-align", update_every=1, enabled=(not quiet))

# Long-lived worker pool to avoid model reloads between CSV chunks
import multiprocessing as _mp
try:
    if 'fork' in _mp.get_all_start_methods():
        _mp.set_start_method('fork', force=True)
except Exception:
    pass

try:
    nproc = int(os.environ.get('ALIGN_NPROC', '0'))
except Exception:
    nproc = 0
if nproc <= 0:
    try:
        nproc = max(1, min((_mp.cpu_count() or 1), 16))
    except Exception:
        nproc = 1

with ProcessPoolExecutor(max_workers=nproc) as ex:
    for chunk in pd.read_csv(inp, encoding='utf-8', quotechar='"', escapechar=None, chunksize=50000):
        if 'tgt' not in chunk.columns and filter_cols:
            for c in filter_cols:
                if c in chunk.columns:
                    chunk['tgt'] = chunk[c]
                    break
        src_list = chunk['src'].astype(str).tolist()
        tgt_list = chunk['tgt'].astype(str).tolist() if 'tgt' in chunk.columns else [''] * len(chunk)

        if filter_aligned_cols:
            # Do not compute ERRANT; rely on precomputed columns. Ensure columns are strings.
            for col in filter_aligned_cols:
                if col in chunk.columns:
                    chunk[col] = chunk[col].astype(str)
            # Determine primary aligned from available explicit names
            primary_key = None
            for key in ['aligned', 'aligned_sentence'] + filter_aligned_cols:
                if key in chunk.columns:
                    primary_key = key; break
            if primary_key is None:
                chunk['aligned'] = chunk['tgt'].astype(str)
            else:
                chunk['aligned'] = chunk[primary_key].astype(str)
        else:
            # Compute ERRANT alignments (main + optional per-filter) using exact names
            futures = {}
            pairs = list(zip(src_list, tgt_list))
            futures['__main__'] = ex.submit(_align_worker, (lang, pairs))
            for col in filter_aligned_cols:
                if col in chunk.columns:
                    col_list = chunk[col].astype(str).tolist()
                    col_pairs = list(zip(src_list, col_list))
                    futures[col] = ex.submit(_align_worker, (lang, col_pairs))
            main_aligned = futures.pop('__main__').result()
            chunk['aligned'] = main_aligned
            for col, f in futures.items():
                try:
                    res = f.result()
                except Exception:
                    res = ["Error"] * len(chunk)
                # Write back into the exact requested column name
                chunk[col] = res
        chunk.to_csv(outp, index=False, mode='w' if first else 'a', header=first)
        first = False
        total += len(chunk)
        prog.update(total)

elapsed = time.time() - start
prog.finish()
if not quiet:
    print(f"Pre-alignment wrote {total} rows to {outp} in {elapsed:.1f}s")
PY
  fi
  shopt -u nocasematch
fi

# Optional: exit after pre-alignment for benchmarking alignment speed only
if [[ "${EXIT_AFTER_PREALIGN:-0}" == "1" ]]; then
  echo "EXIT_AFTER_PREALIGN=1: Exiting after pre-alignment benchmark."
  exit 0
fi

# Call ensemble directly (which will call judges)
case "$ENSEMBLE" in
  weighted) ENS_SCRIPT="ensembles/weighted.py";;
  consistency) ENS_SCRIPT="ensembles/consistency.py";;
  iter_critic) ENS_SCRIPT="ensembles/iter_critic.py";;
  inner_debate) ENS_SCRIPT="ensembles/inner_debate.py";;
  iter_escalation) ENS_SCRIPT="ensembles/iter_escalation.py";;
  escalation) ENS_SCRIPT="ensembles/escalation.py";;
  moe) ENS_SCRIPT="ensembles/moe.py";;
  *) echo "Unknown ensemble $ENSEMBLE"; exit 1;;
esac

# Name outputs by input file to allow multiple runs into same method folder
INPUT_BASENAME="$(basename "$ORIG_INPUT")"
INPUT_STEM="${INPUT_BASENAME%.*}"
FINAL_PRED="$OUTPUT_DIR/${INPUT_STEM}_labeled.csv"

echo "Running $ENSEMBLE ensemble with $N_JUDGES judges over backends: ${BACKENDS[*]}"
unset LLM_DEBUG || true

# Debug: Check environment on Red Sparta
if [[ -n "${SPARTA_ENV:-}" || -n "${LLM_PROXY_PROD_HOST:-}" ]]; then
  # Keep debug off by default; user can enable explicitly by setting LLM_DEBUG=1
  unset LLM_DEBUG || true
fi

# Export OPTIMIZATION to child processes so Python can read it
export OPTIMIZATION="$OPTIMIZATION"
# Prefer ultra-batch path automatically for single-judge optimized runs (except feedback judge for stability)
# if [[ "$OPTIMIZATION" == "on" && "$N_JUDGES" == "1" && "$JUDGE" != "feedback" ]]; then
#   export ULTIMATE_BATCH=1
# else
#   unset ULTIMATE_BATCH || true
# fi
# Cap only baseline for stability; allow optimized to use requested workers
EFFECTIVE_WORKERS="$WORKERS"
if [[ "$OPTIMIZATION" != "on" && "$EFFECTIVE_WORKERS" -gt 50 ]]; then EFFECTIVE_WORKERS=50; fi
# Enable batch-pricing when optimization or batch are on
if [[ "$OPTIMIZATION" == "on" || "$BATCH" == "on" ]]; then export BATCH_PRICING=1; else unset BATCH_PRICING || true; fi
# Ensure calling service header is set for batch uploads (Sparta/gateway compatibility)
if [[ "$BATCH" == "on" ]]; then
  export LLM_PROXY_CALLING_SERVICE="gec_judge"

  # Fail early if no usable key detected for batch endpoints
  if [[ -z "${OPENAI_API_KEY:-}" && -z "${API_TOKEN:-}" ]]; then
    echo "Missing OPENAI_API_KEY or API_TOKEN in .env (needed for Batch)." >&2
    exit 1
  fi

  # For local/non-Sparta runs, force OpenAI direct Batch endpoints for speed/reliability
  if [[ -z "${SPARTA_ENV:-}" && -z "${LLM_PROXY_PROD_HOST:-}" ]]; then
    # Local/default: Pure Batch (sync). Prefer Transparent API unless explicitly forced to OpenAI Direct
    # Respect pre-set FORCE_OPENAI_DIRECT_BATCH; default to 'off' for better reliability
    if [[ -z "${FORCE_OPENAI_DIRECT_BATCH:-}" ]]; then export FORCE_OPENAI_DIRECT_BATCH="off"; fi
    export BATCH_ASYNC="off"
    # Enforce pure batch: disable hybrid by default
    export BATCH_HYBRID="${BATCH_HYBRID:-off}"
    export HYBRID_THRESHOLD_PCT="${HYBRID_THRESHOLD_PCT:-0.9}"
    # Safer defaults for completion (smaller shards, longer wait)
    export BATCH_SHARD_SIZE="${BATCH_SHARD_SIZE:-20}"
    export BATCH_MAX_FILE_MB="${BATCH_MAX_FILE_MB:-10}"
    export BATCH_MAX_WAIT_SECS="${BATCH_MAX_WAIT_SECS:-5400}"
  else
    # Red Sparta (CLAPI): Pure Batch (sync); allow GW fallback for status only
    export BATCH_ASYNC="off"
    export BATCH_HYBRID="${BATCH_HYBRID:-off}"
    export HYBRID_THRESHOLD_PCT="${HYBRID_THRESHOLD_PCT:-0.9}"
    # Hardcoded safe defaults for big jobs behind proxies
    export BATCH_SHARD_SIZE="${BATCH_SHARD_SIZE:-50}"
    export BATCH_MAX_FILE_MB="${BATCH_MAX_FILE_MB:-10}"
    export DISABLE_GATEWAY_FALLBACK="off"
    export BATCH_MAX_WAIT_SECS="${BATCH_MAX_WAIT_SECS:-1800}"
  fi
fi

# POST_PASS is already captured from args; default remains 'off'

# Basic optimization for reliability: use in-process judge path (cheap, avoids subprocess flakiness)
# Prefer in-process by default, but allow caller to override via USE_IN_PROCESS_JUDGE
if [[ -z "${USE_IN_PROCESS_JUDGE:-}" ]]; then
  if [[ "$JUDGE" == "edit" && "$METHOD" == "agent" ]]; then
      export USE_IN_PROCESS_JUDGE=1
  else
      export USE_IN_PROCESS_JUDGE=1
  fi
fi

if [[ -n "$SHARD_SIZE" && "$SHARD_SIZE" -gt 0 ]]; then
  echo "Auto-sharding enabled: shard_size=$SHARD_SIZE rows"
  SHARDS_DIR="$OUTPUT_DIR/shards"
  mkdir -p "$SHARDS_DIR"
  NUM_SHARDS=$(RJ_INPUT="$PREPROCESSED_INPUT" RJ_SHARDS_DIR="$SHARDS_DIR" RJ_SHARD_SIZE="$SHARD_SIZE" python3 - << 'PY'
import os, pandas as pd
inp=os.environ['RJ_INPUT']
outd=os.environ['RJ_SHARDS_DIR']
sz=int(os.environ['RJ_SHARD_SIZE'])
i=0
for chunk in pd.read_csv(inp, chunksize=sz):
    path=os.path.join(outd, f'shard_{i:05d}.csv')
    chunk.to_csv(path, index=False)
    i+=1
print(i)
PY
)
  echo "Created $NUM_SHARDS shard(s) in $SHARDS_DIR"
  rm -f "$FINAL_PRED"
  first=1
  for shard in "$SHARDS_DIR"/shard_*.csv; do
    OUT_SHARD="$OUTPUT_DIR/${INPUT_STEM}_labeled_$(basename "$shard" .csv).csv"
    RJ_INPUT="$INPUT" RJ_OUT="$PREPROCESSED_INPUT" RJ_LANG="$LANG" python3 "$ENS_SCRIPT" \
      --judge "$JUDGE" \
      --method "$METHOD" \
      --backends ${BACKENDS[*]} \
      --lang "$LANG" \
      --n_judges "$N_JUDGES" \
      --input "$shard" \
      --output "$OUT_SHARD" \
      --workers "$EFFECTIVE_WORKERS" \
      $([ -n "$FILTER" ] && echo --filter "$FILTER" || echo "") \
      $([ -n "$SAMPLES" ] && echo --samples "$SAMPLES" || echo "") \
      $([ -n "$TGT_COL" ] && echo --tgt "$TGT_COL" || echo "") \
      $([ "$BATCH" == "on" ] && echo "--batch on" || echo "") \
      $([ "$OPTIMIZATION" == "on" ] && echo "--optimization on" || echo "") \
      $([ "$MODERATION" == "on" ] && echo "--moderation" || echo "") \
      $([ -n "${ROUTER:-}" ] && echo --router "$ROUTER" || echo "") \
      $([ -n "${ESCALATE_LABELS:-}" ] && echo --escalate_labels "$ESCALATE_LABELS" || echo "") \
      $([ -n "${ESCALATE_FLAGS:-}" ] && echo --escalate_flags "$ESCALATE_FLAGS" || echo "")
    if [[ $first -eq 1 ]]; then
      cp "$OUT_SHARD" "$FINAL_PRED"
      first=0
    else
      tail -n +2 "$OUT_SHARD" >> "$FINAL_PRED"
    fi
  done
else
  if [[ "$ENSEMBLE" == "escalation" || "$ENSEMBLE" == "iter_escalation" || "$ENSEMBLE" == "moe" ]]; then
    # iter_escalation does not accept --moderation flag
    MOD_ARG=""
    if [[ "$ENSEMBLE" == "escalation" ]]; then
      MOD_ARG=$([ "$MODERATION" == "on" ] && echo "--moderation on" || echo "--moderation off")
    fi
    RJ_INPUT="$INPUT" RJ_OUT="$PREPROCESSED_INPUT" RJ_LANG="$LANG" python3 "$ENS_SCRIPT" \
      --judge "$JUDGE" \
      --method "$METHOD" \
      --backends ${BACKENDS[*]} \
      --lang "$LANG" \
      --input "$PREPROCESSED_INPUT" \
      --output "$FINAL_PRED" \
      --workers "$EFFECTIVE_WORKERS" \
      $([ "$OPTIMIZATION" == "on" ] && echo "--optimization on" || echo "") \
      $MOD_ARG \
      $([ -n "${ROUTER:-}" ] && echo --router "$ROUTER" || echo "") \
      $([ -n "${ESCALATE_LABELS:-}" ] && echo --escalate_labels "$ESCALATE_LABELS" || echo "") \
      $([ -n "${ESCALATE_FLAGS:-}" ] && echo --escalate_flags "$ESCALATE_FLAGS" || echo "")
  else
    RJ_INPUT="$INPUT" RJ_OUT="$PREPROCESSED_INPUT" RJ_LANG="$LANG" python3 "$ENS_SCRIPT" \
      --judge "$JUDGE" \
      --method "$METHOD" \
      --backends ${BACKENDS[*]} \
      --lang "$LANG" \
      --n_judges "$N_JUDGES" \
      --input "$PREPROCESSED_INPUT" \
      --output "$FINAL_PRED" \
      --workers "$EFFECTIVE_WORKERS" \
      $([ -n "$FILTER" ] && echo --filter "$FILTER" || echo "") \
      $([ -n "$SAMPLES" ] && echo --samples "$SAMPLES" || echo "") \
      $([ -n "$TGT_COL" ] && echo --tgt "$TGT_COL" || echo "") \
      $([ "$BATCH" == "on" ] && echo "--batch on" || echo "") \
      $([ "$OPTIMIZATION" == "on" ] && echo "--optimization on" || echo "") \
      $([ "$MODERATION" == "on" ] && echo "--moderation" || echo "")
  fi

  # Post-pass: pick error rows and re-run only those, then merge back
  if [[ "$POST_PASS" == "on" ]]; then
    ERR_IN="$OUTPUT_DIR/errors_input_${INPUT_STEM}.csv"
    ERR_OUT="$OUTPUT_DIR/errors_labeled_${INPUT_STEM}.csv"
    PREPROCESSED_INPUT="$PREPROCESSED_INPUT" FINAL_PRED="$FINAL_PRED" ERR_IN="$ERR_IN" python3 - <<PY
import pandas as pd, os
pre = os.environ.get('PREPROCESSED_INPUT')
lab = os.environ.get('FINAL_PRED')
err_in = os.environ.get('ERR_IN')
# Find error rows by idx
df = pd.read_csv(lab)
if 'tp_fp_label' in df.columns:
    err_idx = df.loc[df['tp_fp_label'] == 'Error', 'idx'].astype(str)
elif 'pred_specialized' in df.columns:
    err_idx = df.loc[df['pred_specialized'] == 'Error', 'idx'].astype(str)
else:
    err_idx = pd.Series(dtype=str)
if len(err_idx) == 0:
    print('POST_PASS: no errors to rejudge')
    raise SystemExit(0)
pre_df = pd.read_csv(pre)
pre_df['idx'] = pre_df['idx'].astype(str)
sub = pre_df[pre_df['idx'].isin(err_idx)].copy()
sub.to_csv(err_in, index=False)
print(f'POST_PASS: prepared {len(sub)} error rows for rejudging -> {err_in}')
PY
    if [[ -f "$ERR_IN" ]]; then
      RJ_INPUT="$INPUT" RJ_OUT="$PREPROCESSED_INPUT" RJ_LANG="$LANG" python3 "$ENS_SCRIPT" \
        --judge "$JUDGE" \
        --method "$METHOD" \
        --backends ${BACKENDS[*]} \
        --lang "$LANG" \
        --n_judges "$N_JUDGES" \
        --input "$ERR_IN" \
        --output "$ERR_OUT" \
        --workers "$EFFECTIVE_WORKERS" \
        $([ -n "$FILTER" ] && echo --filter "$FILTER" || echo "") \
        $([ -n "$TGT_COL" ] && echo --tgt "$TGT_COL" || echo "") \
        $([ "$BATCH" == "on" ] && echo "--batch on" || echo "") \
        $([ "$OPTIMIZATION" == "on" ] && echo "--optimization on" || echo "") \
        $([ "$MODERATION" == "on" ] && echo "--moderation" || echo "")
      # Merge back
      FINAL_PRED="$FINAL_PRED" ERR_OUT="$ERR_OUT" python3 - <<PY
import pandas as pd, os
lab = os.environ.get('FINAL_PRED')
err = os.environ.get('ERR_OUT')
base = pd.read_csv(lab)
upd = pd.read_csv(err)
base['idx'] = base['idx'].astype(str)
upd['idx'] = upd['idx'].astype(str)
cols = [c for c in upd.columns if c != 'idx']
base = base.set_index('idx')
upd = upd.set_index('idx')
base.update(upd[cols])
base = base.reset_index()
base.to_csv(lab, index=False)
print('POST_PASS: merged back rejudged rows; total now:', len(base))
PY
    fi
  fi
fi

# Always produce a filtered CSV with only original columns + *_label/*_reasoning/*_aligned
FILTERED_CSV="${FINAL_PRED%.csv}_filtered.csv"
python3 - "$FINAL_PRED" "$ORIG_INPUT" "$FILTER" "${CFG_FILTER_ALIGNED:-}" <<'PY'
import sys
import os
import pandas as pd
inp = sys.argv[1]
orig = sys.argv[2]
cfg_filter = [c for c in (sys.argv[3] or '').split() if c]
cfg_filter_aligned = [c for c in (sys.argv[4] or '').split() if c]
outp = os.path.splitext(inp)[0] + '_filtered.csv'
first = True
orig_cols = []
try:
    orig_cols = list(pd.read_csv(orig, nrows=0).columns)
except Exception:
    orig_cols = []
for chunk in pd.read_csv(inp, chunksize=100000):
    # Do not drop any rows in the filtered CSV, even if there are errors
    # Keep only original input columns plus filter annotation columns
    # Keep original columns including 'src'; drop internal-only columns
    deny = {'tgt','aligned','aligned_sentence','alert'}  # always drop
    keep_orig = [c for c in orig_cols if c in chunk.columns and c not in deny]

    # Build extras restricted to config filter prefixes when provided
    def is_allowed_extra(col: str) -> bool:
        # Always allow top-level tp_fp_label and reasoning if present
        if col in {'tp_fp_label','reasoning'}:
            return True
        # Allow per-filter columns if a filter list is set
        if cfg_filter:
            for base in cfg_filter:
                if col.startswith(base + '_') and (col.endswith('_label') or col.endswith('_reasoning') or col.endswith('_aligned')):
                    return True
            return False
        # Otherwise, allow any *_label/*_reasoning/*_aligned
        return col.endswith('_label') or col.endswith('_reasoning') or col.endswith('_aligned')

    extras = [c for c in chunk.columns if is_allowed_extra(c)]

    # Also include precomputed alignment columns from filter_aligned verbatim if present
    for aline in cfg_filter_aligned:
        if aline in chunk.columns and aline not in extras:
            extras.append(aline)

    # Deterministic order: original columns first, then extras sorted
    extras_sorted = sorted([c for c in extras if c not in keep_orig])
    cols_out = (keep_orig + extras_sorted) if keep_orig else extras_sorted
    if not cols_out:
        cols_out = extras_sorted
    chunk.to_csv(outp, index=False, mode='w' if first else 'a', header=first, columns=cols_out)
    first = False
print(f"Filtered predictions saved: {outp}")
PY

# Benchmark against gold if using default datasets (heuristic)
GOLD_FILE="$GOLD"

# Check if gold should be skipped

if [[ -n "$GOLD" && "$GOLD" != "SKIP" && -f "$GOLD" ]]; then
  echo "Benchmarking against gold..."
  python3 -m gold_eval.run_gold \
      "$GOLD" \
      "$FINAL_PRED" \
      --llm_backend "$JOINED_BACKENDS" \
      --judge "$JUDGE" \
      --method "$METHOD" \
      --lang "$LANG" \
      --pref "$PREF"
else
  echo "Benchmark skipped (no gold provided)"
fi

echo "All done. Results saved to $OUTPUT_DIR"