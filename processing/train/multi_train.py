#!/usr/bin/env python3
"""
Multilingual GEC training entrypoint (production wrapper).

Thin wrapper around `processing/process_multigec2025_training.py` to keep a
stable CLI for reproducible runs.

Example:
  python processing/multi_train.py --output data/training/multigec2025 \
    --languages en de ua ru --omnigec-samples 500 --clang8-samples 3000 \
    --ubertext-samples 1500
"""

import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from processing.process_multigec2025_training import main as _run_multi


def main() -> int:
    return _run_multi()


if __name__ == "__main__":
    raise SystemExit(main())


