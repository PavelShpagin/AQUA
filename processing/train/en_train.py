#!/usr/bin/env python3
"""
English GEC training entrypoint (production wrapper).

Thin wrapper around `processing/process_bea_training.py` to keep a stable
CLI for reproducible runs.

Usage (recommended):
  python processing/en_train.py --output data/training/bea \
    --clang8 3000 --troy-1bw 2000 --wi-locness 500 --fce 0 --lang8 0 --nucle 0

Notes:
- Random seed is fixed to 42 inside the underlying processor.
- Data detokenization is preserved; alignment uses a simplified ERRANT proxy.
"""

import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from processing.process_bea_training import main as _run_bea


def main() -> int:
    return _run_bea()


if __name__ == "__main__":
    raise SystemExit(main())


