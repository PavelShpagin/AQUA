#!/bin/bash

# Process all three languages with specified dataset distributions
# Usage: ./shell/proc_all.sh --wi 300 --fce 200 --ua-gec 300 --merlin 300 --omni 300

set -e

# Default values
WI=0
FCE=0
UA_GEC=0
MERLIN=0
OMNI=0
NO_MERGE=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --wi)
            WI="$2"
            shift 2
            ;;
        --fce)
            FCE="$2"
            shift 2
            ;;
        --ua-gec)
            UA_GEC="$2"
            shift 2
            ;;
        --merlin)
            MERLIN="$2"
            shift 2
            ;;
        --omni)
            OMNI="$2"
            shift 2
            ;;
        --no-merge)
            NO_MERGE=1
            shift 1
            ;;
        --help)
            echo "Usage: $0 [--wi NUM] [--fce NUM] [--ua-gec NUM] [--merlin NUM] [--omni NUM]"
            echo ""
            echo "Options:"
            echo "  --wi NUM       Number of W&I+LOCNESS samples for English"
            echo "  --fce NUM      Number of FCE samples for English"
            echo "  --ua-gec NUM   Number of UA-GEC samples for Ukrainian"
            echo "  --merlin NUM   Number of FalkoMerlin samples for German"
            echo "  --omni NUM     Number of OmniGEC samples for all languages"
            echo "  --no-merge     Disable merging of adjacent edits during alignment"
            echo "  --help         Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --wi 300 --ua-gec 300 --merlin 300"
            echo "  $0 --omni 300"
            echo "  $0 --wi 300 --fce 200 --ua-gec 500 --merlin 500 --no-merge"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Processing All Languages with Distribution:"
echo "  English: W&I=$WI, FCE=$FCE, OmniGEC=$OMNI"
echo "  Ukrainian: UA-GEC=$UA_GEC, OmniGEC=$OMNI"
echo "  German: FalkoMerlin=$MERLIN, OmniGEC=$OMNI"
if [ "$NO_MERGE" -eq 1 ]; then echo "  Merge edits: DISABLED"; else echo "  Merge edits: ENABLED"; fi
echo "=========================================="

# Process English
echo ""
echo "Processing English..."
MERGE_FLAG=""
MAX_FLAG="--max"
if [ "$NO_MERGE" -eq 1 ]; then MERGE_FLAG="--no-merge"; fi
python processing/en.py --wi "$WI" --fce "$FCE" --omni "$OMNI" $MERGE_FLAG "$MAX_FLAG"

# Process Ukrainian
echo ""
echo "Processing Ukrainian..."
python processing/ua.py --ua-gec "$UA_GEC" --omni "$OMNI" $MERGE_FLAG "$MAX_FLAG"

# Process German
echo ""
echo "Processing German..."
python processing/de.py --merlin "$MERLIN" --omni "$OMNI" $MERGE_FLAG "$MAX_FLAG"

echo ""
echo "=========================================="
echo "FINAL DISTRIBUTION SUMMARY"
echo "=========================================="

# Count distributions from each output file
echo ""
echo "ENGLISH (data/processed/en-judge.csv):"
python -c "
import csv
datasets = {}
total = 0
with open('data/processed/en-judge.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        total += 1
        dataset = row['idx'].split('_')[0]
        datasets[dataset] = datasets.get(dataset, 0) + 1

print(f'  Total: {total} samples')
for dataset, count in sorted(datasets.items()):
    print(f'  {dataset}: {count} samples ({count/total*100:.1f}%)')
"

echo ""
echo "UKRAINIAN (data/processed/ua-judge.csv):"
python -c "
import csv
datasets = {}
total = 0
with open('data/processed/ua-judge.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        total += 1
        dataset = row['idx'].split('_')[0]
        datasets[dataset] = datasets.get(dataset, 0) + 1

print(f'  Total: {total} samples')
for dataset, count in sorted(datasets.items()):
    print(f'  {dataset}: {count} samples ({count/total*100:.1f}%)')
"

echo ""
echo "GERMAN (data/processed/de-judge.csv):"
python -c "
import csv
datasets = {}
total = 0
with open('data/processed/de-judge.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        total += 1
        dataset = row['idx'].split('_')[0]
        datasets[dataset] = datasets.get(dataset, 0) + 1

print(f'  Total: {total} samples')
for dataset, count in sorted(datasets.items()):
    print(f'  {dataset}: {count} samples ({count/total*100:.1f}%)')
"

echo ""
echo "=========================================="
echo "Processing completed successfully!"
echo "=========================================="