#!/bin/bash

# Download gold_tp_fp3_fp2_fp1_en.csv from Grammarly Data Catalog into data/eval/
# Usage: run with no args

set -e

# Resolve directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Load .env if present (for DATA_CATALOG_APPLICATION_TOKEN or DATACAT_TOKEN, optional DATACAT_BASE_URL)
ENV_FILE="$PROJECT_ROOT/.env"
if [[ -f "$ENV_FILE" ]]; then
    echo "Loading environment variables from .env"
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

# Config (can be overridden via env)
BASE_URL="${DATACAT_BASE_URL:-https://datacat.nlp-processing.grammarlyaws.com}"
ARTIFACT="${DATACAT_ARTIFACT:-gold_tp_fp3_fp2_fp1_en}"
VERSION="${DATACAT_VERSION:-1}"

# Default destination: EFS path, with fallback to local data/eval if not writeable
DEFAULT_DEST="/home/ray/efs/team/pavel.shpagin/multigec/feedbacks/gold_tp_fp3_fp2_fp1_en.csv"
OUTPUT_FILE="${DATACAT_DEST:-$DEFAULT_DEST}"

# Ensure destination directory exists or fallback
DEST_PARENT="$(dirname "$OUTPUT_FILE")"
if ! mkdir -p "$DEST_PARENT" 2>/dev/null; then
    echo "Destination $DEST_PARENT not writeable; falling back to local data/eval"
    mkdir -p "data/eval"
    OUTPUT_FILE="data/eval/gold_tp_fp3_fp2_fp1_en.csv"
fi

if [[ -f "$OUTPUT_FILE" ]]; then
    echo "Dataset already exists: $OUTPUT_FILE"
    echo "Nothing to do."
    exit 0
fi

echo "Downloading Data Catalog artifact: ${ARTIFACT}:${VERSION} -> $OUTPUT_FILE"

# Prefer explicit download URL if provided (from Data Catalog → Snippets)
if [[ -n "${DATACAT_DOWNLOAD_URL:-}" ]]; then
    echo "Using DATACAT_DOWNLOAD_URL"
    curl -fL --retry 3 \
        -H "Authorization: Bearer ${DATA_CATALOG_APPLICATION_TOKEN:-${DATA_CATALOG_ACCESS_TOKEN:-${DATACAT_TOKEN:-}}}" \
        -H "Accept: text/csv,application/octet-stream" \
        -o "$OUTPUT_FILE.tmp" "$DATACAT_DOWNLOAD_URL" || true
    if [[ -s "$OUTPUT_FILE.tmp" ]] && ! head -c 100 "$OUTPUT_FILE.tmp" | grep -qi "<!doctype html>\|<html"; then
        mv -f "$OUTPUT_FILE.tmp" "$OUTPUT_FILE"
        echo "Downloaded via DATACAT_DOWNLOAD_URL"
        exit 0
    fi
    rm -f "$OUTPUT_FILE.tmp" 2>/dev/null || true
    echo "Fallback: DATACAT_DOWNLOAD_URL failed; trying standard endpoints"
fi

# Prefer Python client if available
if python - <<'PY' 2>/dev/null
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec("datacat") else 1)
PY
then
    echo "Using Python datacat client"
    export OUTPUT_FILE
    python - <<PY
import os
from pathlib import Path
from datacat.client import DataCatalog

base_url = os.environ.get('DATACAT_BASE_URL', 'https://datacat.nlp-processing.grammarlyaws.com')
artifact_id = os.environ.get('DATACAT_ARTIFACT', 'gold_tp_fp3_fp2_fp1_en')
version = os.environ.get('DATACAT_VERSION', '1')
token = os.environ.get('DATA_CATALOG_APPLICATION_TOKEN') or os.environ.get('DATA_CATALOG_ACCESS_TOKEN') or os.environ.get('DATACAT_TOKEN')
dest = os.environ.get('OUTPUT_FILE')

if not token:
    raise SystemExit('DATA_CATALOG_APPLICATION_TOKEN is not set')

dc = DataCatalog(catalog_url=base_url, application_token=token)

def get_artifact(dc, artifact_id, version):
    if str(version).lower() in ('latest', ''):
        return dc.search_by_artifact_id(artifact_id, latest_only=True)[0]
    try:
        return dc.search_by_artifact_version(artifact_id, str(version))
    except Exception:
        return dc.search_by_artifact_id(artifact_id, latest_only=True)[0]

artifact = get_artifact(dc, artifact_id, version)
files = dc.use_artifact(artifact)

# Prefer exact CSV filename; otherwise take first file
target_name = 'gold_tp_fp3_fp2_fp1_en.csv'
selected_name = target_name if target_name in files else next(iter(files.keys()))

content = files[selected_name]()
data = content.read()

out_path = Path(dest)
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, 'wb') as f:
    f.write(data)

print(f'Downloaded via Python client to {out_path}')
PY
    exit 0
fi

# Require token for API access (prefer application token)
APP_TOKEN="${DATA_CATALOG_APPLICATION_TOKEN:-}"
ACCESS_TOKEN="${DATA_CATALOG_ACCESS_TOKEN:-}"
LEGACY_TOKEN="${DATACAT_TOKEN:-}"
TOKEN_TO_USE="${APP_TOKEN:-${ACCESS_TOKEN:-${LEGACY_TOKEN:-}}}"

if [[ -z "$TOKEN_TO_USE" ]]; then
    echo "Error: DATA_CATALOG_APPLICATION_TOKEN is not set."
    echo "Obtain a token from Data Catalog → Authentication tokens, then either:"
    echo "  - put it into .env as DATA_CATALOG_APPLICATION_TOKEN=..."
    echo "  - or export DATA_CATALOG_APPLICATION_TOKEN in your shell"
    exit 1
fi

TMP_FILE="$OUTPUT_FILE.tmp"
rm -f "$TMP_FILE" 2>/dev/null || true

# Try a small set of common API download endpoints used by Data Catalog
declare -a CANDIDATE_URLS=(
  "$BASE_URL/api/v1/artifacts/$ARTIFACT/versions/$VERSION/download?filename=gold_tp_fp3_fp2_fp1_en.csv"
  "$BASE_URL/api/v1/artifacts/$ARTIFACT/versions/$VERSION/download?path=gold_tp_fp3_fp2_fp1_en.csv"
  "$BASE_URL/api/v1/artifacts/$ARTIFACT/versions/$VERSION/files/gold_tp_fp3_fp2_fp1_en.csv?download=1"
  "$BASE_URL/api/artifacts/$ARTIFACT/versions/$VERSION/files/gold_tp_fp3_fp2_fp1_en.csv?download=1"
  "$BASE_URL/artifacts/$ARTIFACT/versions/$VERSION/files/gold_tp_fp3_fp2_fp1_en.csv?download=1"
)

for url in "${CANDIDATE_URLS[@]}"; do
    echo "Trying: $url"
    if curl -fL --retry 3 \
        -H "Authorization: Bearer $TOKEN_TO_USE" \
        -H "X-API-Token: $TOKEN_TO_USE" \
        -H "Accept: text/csv,application/octet-stream" \
        -o "$TMP_FILE" "$url" 2>/dev/null; then
        if [[ -s "$TMP_FILE" ]] && ! head -c 100 "$TMP_FILE" | grep -qi "<!doctype html>\|<html"; then
            mv -f "$TMP_FILE" "$OUTPUT_FILE"
            echo "Downloaded: $OUTPUT_FILE"
            exit 0
        fi
    fi
    rm -f "$TMP_FILE" 2>/dev/null || true
done

echo "Could not download via known endpoints."
echo "Hints:"
echo "  1) From the artifact page, open Snippets and copy a direct file download URL; set it as DATACAT_DOWNLOAD_URL and re-run."
echo "  2) Ensure your DATACAT_TOKEN is valid (not expired) and has access."
exit 2


