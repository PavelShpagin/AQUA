# Aligned Sentences Update for Feedback Judge

## Summary
Updated the feedback judge components to use the SOTA (State-of-the-Art) TPFP prompt from legacy with proper aligned sentence support using ERRANT alignment. Removed all fallback mechanisms - the system now returns "Error" if alignment cannot be generated.

## Changes Made

### 1. Updated Prompt (judges/feedback/prompts.py)
- Replaced `TPFP_PROMPT_BASELINE` with the SOTA version from `legacy/gold_eval/feedback_tpfp_prompt.py`
- Made it language-agnostic (supports multiple languages via {0} placeholder)
- Added aligned changes field ({3} placeholder) to show edits in format: `{original=>corrected}`
- Includes comprehensive examples with aligned changes

### 2. Baseline Implementation (judges/feedback/baseline.py)
- Added imports for ERRANT alignment utilities
- Preloads spaCy models once at startup for detected languages
- Generates aligned sentences automatically if not provided in input
- Returns "Error" label if alignment generation fails
- Uses merged alignment format for clarity (e.g., "I {loves=>love} pizza")
- Updated `build_prompt` to accept aligned parameter

### 3. Modular Implementation (judges/feedback/modular.py)
- Similar changes as baseline.py
- Added support for aligned sentences in both final_judge and modular modes
- Returns "Error" label if alignment generation fails
- Preloads spaCy models for performance
- Includes aligned_sentence in output results

### 4. Alignment Utilities (utils/errant_align.py & utils/alignment.py)
- **Removed all diff-based fallback mechanisms**
- Functions now return `None` when alignment cannot be generated
- No more fallback to simple diff alignment or returning source/target unchanged
- Strict requirement for ERRANT models to be available

## Technical Details

### Alignment Generation
- Uses ERRANT library with language-specific spaCy models
- Supports English, German, and Ukrainian
- **NO FALLBACKS**: Returns None if ERRANT models unavailable or alignment fails
- Preserves spaces and punctuation accurately
- Merges consecutive edits for readability

### Error Handling
- If alignment generation fails, the system returns:
  - `tp_fp_label`: "Error"
  - `reasoning`: "Error: Failed to generate alignment for text comparison"
  - `aligned_sentence`: empty string
- No silent failures or fallback values

### Model Preloading
- Detects languages from dataset sample (first 100 rows)
- Loads appropriate spaCy models: en_core_web_sm, de_core_news_sm, uk_core_news_sm
- Reuses loaded models across all rows for efficiency
- Returns error if models unavailable

## Usage
The aligned changes are automatically generated and included in the prompt, providing better context for the LLM to evaluate grammatical corrections. The format shows exactly what changed:
- Insertion: `{=>word}`
- Deletion: `{word=>}`
- Substitution: `{old=>new}`

## Backward Compatibility
- Still accepts aligned_sentence from input CSV if available
- Falls back to 'alert' column for legacy format
- Returns "Error" if alignment cannot be generated (no silent fallbacks)
