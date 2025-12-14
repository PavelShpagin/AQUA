#!/bin/bash

set -e

mkdir -p data/candidates

MODE="gold"
NO_MERGE=0

# Parse args
while [[ $# -gt 0 ]]; do
  case $1 in
    --mode)
      MODE="$2"; shift 2 ;;
    --no-merge)
      NO_MERGE=1; shift 1 ;;
    --help)
      echo "Usage: $0 [--mode gold|silver] [--no-merge]"; exit 0 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

MERGE_FLAG=""
if [ "$NO_MERGE" -eq 1 ]; then MERGE_FLAG="--no-merge"; fi

if [ "$MODE" = "silver" ]; then
  ./shell/proc_all.sh --wi 0 --fce 0 --ua-gec 0 --merlin 0 --omni 500 $MERGE_FLAG
else
  ./shell/proc_all.sh --wi 300 --fce 200 --ua-gec 500 --merlin 500 --omni 0 $MERGE_FLAG
fi

python test/check_alignment.py --lang en
./shell/edit_judge.sh --lang en
python test/check_alignment.py --lang ua
./shell/edit_judge.sh --lang ua
python test/check_alignment.py --lang de
./shell/edit_judge.sh --lang de
python test/avg_edits.py

# Count :::ERROR tags in processed results
echo "Analyzing :::ERROR tags in processed results..."
python test/error_tags.py --lang all