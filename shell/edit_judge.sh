#!/bin/bash

# Edit-Level GEC Judge Runner Script
# Usage: ./shell/edit_judge.sh --lang en --backend gpt-4o-mini --sample 100

set -e  # Exit on any error

# Default values
LANG=""
BACKEND="gpt-4o"
SAMPLE=""
DEMO=false
OUTPUT=""
N_PARALLEL_JOBS=50

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --lang)
            LANG="$2"
            shift 2
            ;;
        --backend)
            BACKEND="$2"
            shift 2
            ;;
        --sample)
            SAMPLE="$2"
            shift 2
            ;;
        --demo)
            DEMO=true
            shift
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --n_parallel_jobs)
            N_PARALLEL_JOBS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 --lang LANG [--backend BACKEND] [--sample SAMPLE] [--demo] [--output OUTPUT] [--n_parallel_jobs N]"
            echo ""
            echo "Arguments:"
            echo "  --lang              Language code (required): en, de, ua, es"
            echo "  --backend           LLM backend (default: gpt-4o-mini): gpt-4o, gpt-4o-mini"
            echo "  --sample            Number of samples to process (default: all)"
            echo "  --demo              Run in demo mode without API calls"
            echo "  --output            Custom output file path"
            echo "  --n_parallel_jobs   Number of parallel workers (default: 50)"
            echo "  -h, --help          Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --lang en --sample 100"
            echo "  $0 --lang de --backend gpt-4o --sample 50"
            echo "  $0 --lang ua --demo --sample 10"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$LANG" ]]; then
    echo "Error: --lang is required"
    echo "Use --help for usage information."
    exit 1
fi

# Map language to data file
case $LANG in
    en)
        DATA_FILE="data/processed/en-judge.csv"
        ;;
    de)
        DATA_FILE="data/processed/de-judge.csv"
        ;;
    ua)
        DATA_FILE="data/processed/ua-judge.csv"
        ;;
    es)
        echo "Warning: Spanish data file not found, using English data for demo"
        DATA_FILE="data/processed/en-judge.csv"
        ;;
    *)
        echo "Error: Unsupported language '$LANG'. Supported: en, de, ua, es"
        exit 1
        ;;
esac

# Check if data file exists
if [[ ! -f "$DATA_FILE" ]]; then
    echo "Error: Data file not found: $DATA_FILE"
    exit 1
fi

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Load environment variables from .env if it exists
ENV_FILE="$PROJECT_ROOT/.env"
if [[ -f "$ENV_FILE" ]]; then
    echo "Loading environment variables from .env"
    set -a  # Export all variables
    source "$ENV_FILE"
    set +a  # Stop exporting
else
    echo "Warning: .env file not found at $ENV_FILE"
    echo "You may need to set API keys manually:"
    echo "  export OPENAI_API_KEY=your_openai_key"
    echo "  export LLM_PROXY_API_TOKEN=your_llm_proxy_token"
fi

# Build command arguments
CMD_ARGS=()
CMD_ARGS+=("--data" "$DATA_FILE")
CMD_ARGS+=("--lang" "$LANG")
CMD_ARGS+=("--backend" "$BACKEND")

if [[ -n "$SAMPLE" ]]; then
    CMD_ARGS+=("--sample" "$SAMPLE")
fi

if [[ "$DEMO" == true ]]; then
    CMD_ARGS+=("--demo")
fi

if [[ -n "$OUTPUT" ]]; then
    CMD_ARGS+=("--output" "$OUTPUT")
fi

CMD_ARGS+=("--n_parallel_jobs" "$N_PARALLEL_JOBS")

# Set API key based on backend and demo mode
if [[ "$DEMO" != true ]]; then
    case $BACKEND in
        gpt-4o|gpt-4o-mini)
            if [[ -z "$OPENAI_API_KEY" ]]; then
                echo "Error: OPENAI_API_KEY environment variable is required for backend '$BACKEND'"
                echo "Please set it in .env file or export it directly:"
                echo "  export OPENAI_API_KEY=your_openai_key"
                exit 1
            fi
            CMD_ARGS+=("--api_key" "$OPENAI_API_KEY")
            ;;
        *)
            # For future backends, might need LLM_PROXY_API_TOKEN
            if [[ -n "$LLM_PROXY_API_TOKEN" ]]; then
                CMD_ARGS+=("--api_key" "$LLM_PROXY_API_TOKEN")
            fi
            ;;
    esac
fi

# Print configuration
echo "=================================="
echo "Edit-Level GEC Judge Configuration"
echo "=================================="
echo "Language: $LANG"
echo "Data file: $DATA_FILE"
echo "Backend: $BACKEND"
echo "Sample size: ${SAMPLE:-all}"
echo "Demo mode: $DEMO"
echo "Output: ${OUTPUT:-auto-generated}"
echo "Parallel workers: $N_PARALLEL_JOBS"
echo "=================================="
echo ""

# Change to project root directory
cd "$PROJECT_ROOT"

# Run the edit-level judge
echo "Running edit-level judge..."
python gold_eval/edit_level_judge.py "${CMD_ARGS[@]}"

exit_code=$?

if [[ $exit_code -eq 0 ]]; then
    echo ""
    echo "✓ Edit-level judge completed successfully!"
else
    echo ""
    echo "✗ Edit-level judge failed with exit code $exit_code"
    exit $exit_code
fi
