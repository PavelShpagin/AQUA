#!/usr/bin/env python3
"""
Centralized settings for LLM parameters.

Provides a single accessor for temperature so it can be controlled globally
via environment variable without touching multiple call sites.

Env overrides:
- LLM_TEMPERATURE (preferred)
- TEMPERATURE (fallback)
"""

import os


def get_llm_temperature(default: float = 1.0) -> float:
    return default


def get_final_judge_temperature(default: float = 0.0) -> float:
    """Return temperature specifically for final judges (inner_debate, iter_critic).

    Currently returns the provided default (0.0) to keep final judges deterministic.
    Hook for future env-based control, e.g., FINAL_JUDGE_TEMPERATURE.
    """
    return default


