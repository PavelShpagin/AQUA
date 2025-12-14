#!/usr/bin/env python3
"""
GEC Judge Optimization Integration
================================

Seamless integration system that patches existing infrastructure for optimal performance.
Combines ensemble patching and environment configuration for transparent upgrades.
"""

import os
import sys
from typing import Dict, List, Any, Callable
import pandas as pd
from functools import wraps


# Global optimization context
_optimization_context = {
    'judge': 'sentence',
    'method': 'legacy', 
    'backend': 'gemini-2.0-flash-lite',
    'lang': 'es',
    'enabled': True
}


def configure_environment():
    """Configure environment for optimal performance."""
    
    # Set processing mode for bulk operations
    if os.getenv('PROCESSING_MODE') != 'bulk':
        os.environ['PROCESSING_MODE'] = 'bulk'
    
    # Configure performance parameters if not already set
    performance_vars = {
        'MAX_WORKERS': '200',
        'ASYNC_WORKERS': '200', 
        'CONN_TIMEOUT': '1.5',
        'READ_TIMEOUT': '8.0',
        'LLM_BATCH_SIZE': '20'
    }
    
    for var, default in performance_vars.items():
        if var not in os.environ:
            os.environ[var] = default
    
    # Enable speed optimizations
    if os.getenv('ENABLE_SPEED_OPTIMIZATIONS') != 'true':
        os.environ['ENABLE_SPEED_OPTIMIZATIONS'] = 'true'
    
    # Environment configured quietly


def set_optimization_context(judge: str, method: str, backend: str, lang: str):
    """Set context for optimization processing."""
    global _optimization_context
    _optimization_context.update({
        'judge': judge,
        'method': method, 
        'backend': backend,
        'lang': lang
    })


def patch_ensemble_system():
    """Patch ensemble system for transparent performance upgrades."""
    # IMPORTANT: Do not override process_rows_parallel.
    # We keep ensemble row orchestration intact to preserve full reasoning.
    # Optimizations are applied at the judge-call level only.
    try:
        from utils import ensemble  # noqa: F401
        # Explicitly signal no-op success for clarity
        # (We intentionally do not wrap ensemble.process_rows_parallel)
        return True
    except ImportError:
        print("WARN: Ensemble module not available for patching")
        return False
    except Exception as e:
        print(f"WARN: Failed to patch ensemble system: {e}")
        return False


def patch_judge_system():
    """No-op patch to preserve judge-specific high-quality prompts (no simplified fallbacks)."""
    return True


def verify_optimization_status():
    """Verify optimization system is properly configured."""
    
    status = {
        'environment_configured': os.getenv('PROCESSING_MODE') == 'bulk',
        'performance_vars_set': bool(os.getenv('MAX_WORKERS')),
        'ensemble_patched': False,
        'optimization_available': False
    }
    
    # Check ensemble patching
    try:
        from utils import ensemble  # noqa: F401
        # We intentionally do not wrap ensemble functions anymore.
        # Consider it patched for environment/config purposes only.
        status['ensemble_patched'] = True
    except ImportError:
        pass
    
    # Check optimization availability  
    try:
        from utils.optimization.core import process_batch_optimized
        status['optimization_available'] = True
    except ImportError:
        pass
    
    return status


def apply_optimizations(judge: str = 'sentence', method: str = 'legacy', 
                       backend: str = 'gemini-2.0-flash-lite', lang: str = 'es'):
    """Apply all optimization patches and configurations."""
    
    # Configure environment quietly
    configure_environment()
    
    # Set optimization context
    set_optimization_context(judge, method, backend, lang)
    
    # Patch systems
    patches_applied = 0
    
    if patch_ensemble_system():
        patches_applied += 1
    
    if patch_judge_system():
        patches_applied += 1
    
    # Verify status and report only if needed
    status = verify_optimization_status()
    
    if all([status['environment_configured'], status['ensemble_patched'], 
           status['optimization_available']]):
        return True
    else:
        if os.getenv('QUIET_LOGS') != '1':
            print("WARN: Optimization system partially active")
        return False


def get_optimization_status() -> Dict[str, Any]:
    """Get detailed optimization system status."""
    
    status = verify_optimization_status()
    
    status.update({
        'context': _optimization_context.copy(),
        'environment': {
            'processing_mode': os.getenv('PROCESSING_MODE'),
            'max_workers': os.getenv('MAX_WORKERS'),
            'conn_timeout': os.getenv('CONN_TIMEOUT'),
            'read_timeout': os.getenv('READ_TIMEOUT')
        }
    })
    
    return status
