#!/usr/bin/env python3
"""
utils/optimization/runner.py
============================

Central optimized runner with FIXED race conditions.
Guarantees deterministic result ordering identical to non-optimized version.

Fixes:
- Removed complex sharding that caused ordering issues
- Uses simple ThreadPoolExecutor with futures mapping
- Preserves exact input order via index-based result array
- No fallback position tracking that could cause misalignment
"""

from __future__ import annotations

import time
from typing import Callable, Any, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


# Removed _process_slice - no longer needed with simplified approach


def run_optimized_process_rows(
    df,
    process_fn: Callable[[Any], Dict[str, Any]],
    *,
    desc: str = "Optimized",
    target_shards: int | None = None,
    workers_per_shard: int = 50,
) -> List[Dict[str, Any]]:
    """Run optimized process with FIXED race conditions.
    
    This version eliminates the complex sharding system that caused ordering issues.
    Uses the same simple ThreadPoolExecutor pattern as non-optimized but with more workers.
    Guarantees identical result ordering to non-optimized version.
    
    Args:
        df: pandas DataFrame of rows
        process_fn: function(row) -> result dict
        desc: progress bar label
        target_shards: ignored (for compatibility)
        workers_per_shard: max workers to use
    Returns:
        List of result dicts in EXACT input order
    """
    import os
    import sys
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    total = len(df)
    if total == 0:
        return []
    
    # Use more aggressive worker count for optimization
    max_workers = min(200, workers_per_shard, total)
    
    # Pre-allocate results array to guarantee ordering
    results: List[Dict[str, Any]] = [None] * total  # type: ignore
    
    # Simple, deterministic ThreadPoolExecutor (same as non-optimized)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create futures with explicit index mapping
        futures = {executor.submit(process_fn, row): i for i, (_, row) in enumerate(df.iterrows())}
        
        # Progress bar
        pbar = tqdm(total=total, desc=desc, leave=True, mininterval=0.1, file=sys.stdout, disable=(os.getenv('QUIET_LOGS') == '1'))
        
        try:
            # Collect results in completion order but store in input order
            for future in as_completed(futures):
                result = future.result()
                original_index = futures[future]
                results[original_index] = result
                pbar.update(1)
        finally:
            pbar.close()
    
    # Fill any None results with error dicts (should not happen)
    for i, result in enumerate(results):
        if result is None:
            results[i] = {
                'idx': i, 
                'tp_fp_label': 'API_FAILED', 
                'reasoning': 'Processing failed', 
                'success': False
            }
    
    return results


