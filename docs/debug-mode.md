# Debug Mode for GEC Judges

## Overview

The debug mode provides a quick way to test judge methods with sample data from both English and Ukrainian datasets. When enabled, it runs the judge on one random sample per label category (TP, FP1, FP2, FP3, etc.) and displays detailed information about:

- The full prompt sent to the model
- The raw model output
- The extracted reasoning
- The predicted label vs expected label
- Token usage and cost breakdown
- Execution time

## Usage

### Via Command Line

```bash
# Using the --debug flag
bash shell/run_judge.sh --judge feedback --method baseline --backends gas_gemini20_flash_lite --debug

# With multiple backends
bash shell/run_judge.sh --judge sentence --method baseline --backends "gpt-4o-mini gas_gemini20_flash_lite" --debug
```

### Via Config File

Set `debug: on` in `config.yaml`:

```yaml
judge: feedback
method: baseline
backends:
  - gas_gemini20_flash_lite
lang: en
debug: on  # Enable debug mode
```

Then run:
```bash
bash shell/run_judge.sh --config config.yaml
```

### Direct Python Execution

Each judge method also supports debug mode directly:

```bash
# Feedback judge
python -m judges.feedback.baseline --llm_backend gas_gemini20_flash_lite --debug on

# Sentence judge  
python -m judges.sentence.baseline --llm_backend gpt-4o-mini --debug on
```

## Sample Output

When debug mode is enabled, you'll see output like:

```
================================================================================
DEBUG OUTPUT - FEEDBACK JUDGE (baseline)
================================================================================

📊 SAMPLE INFO:
  Language: en
  Expected Label: TP
  Sample Index: 50

📝 INPUT:
  Source: Approve birthday email and automation.
  Target: Approve the birthday email and automation.

🤖 MODEL: gas_gemini20_flash_lite
⏱️  Execution Time: 3.40 seconds

📤 PROMPT (first 500 chars):
----------------------------------------
You are an **Error Severity Classifier** for grammatical error correction evaluation...

📥 RAW OUTPUT:
----------------------------------------
{
  "classification": "TP",
  "reason": "The edit correctly adds the definite article 'the'...",
  "tags": ["grammar", "article usage", "clarity"],
  "type_of_writing": "Professional"
}

🎯 PREDICTED LABEL: TP
✅ EXPECTED LABEL: TP
✓ CORRECT

📎 ADDITIONAL INFO:
  token_usage: {'total_tokens': 1388, 'input_tokens': 1302, 'output_tokens': 86}
  cost_breakdown: {'total_cost_usd': 0.00012345}
  model: gas_gemini20_flash_lite

================================================================================
```

## How It Works

1. **Sample Selection**: The debug mode loads evaluation data files and randomly selects one sample per label category:
   - For feedback judge: TP, FP1, FP2, FP3
   - For sentence/edit judges: TP, FP1, FP2, FP3, TN, FN
   - For TNFN judge: TP, FP, TN, FN

2. **Language Testing**: By default, tests both English and Ukrainian samples to ensure multi-language support

3. **Model Execution**: Runs the actual judge logic with the selected samples

4. **Result Analysis**: Compares predicted labels with expected labels and displays accuracy

5. **Logging**: Saves detailed logs to `debug_logs/` directory in JSON format for later analysis

## Supported Judges

Debug mode is currently implemented for:

- **feedback/baseline**: Single model classification
- **feedback/legacy**: Legacy implementation
- **feedback/agent**: Agent-based classification
- **sentence/baseline**: Two-model approach with voting
- **sentence/legacy**: Legacy sentence-level judge
- **edit/baseline**: Edit-level classification
- **tnfn/baseline**: True Negative/False Negative classification

## Implementation Details

### Shared Utilities

The debug functionality is implemented in `utils/debug.py` with these key functions:

- `load_debug_samples()`: Loads test samples from evaluation datasets
- `log_debug_output()`: Displays formatted debug information
- `save_debug_log()`: Saves debug results to JSON files
- `run_debug_test()`: Orchestrates the debug testing process

### Judge Integration

Each judge method can add debug support by:

1. Adding `--debug` argument to the argument parser
2. Checking if debug mode is enabled
3. Creating a debug wrapper function that calls the judge logic
4. Using `run_debug_test()` to execute the test
5. Saving results with `save_debug_log()`

Example implementation in a judge:

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', default='off', choices=['on', 'off'])
    args = parser.parse_args()
    
    if args.debug == 'on':
        from utils.debug import run_debug_test, save_debug_log
        
        def debug_wrapper(src, tgt, llm_backend, lang, **kwargs):
            # Your judge logic here
            prompt = build_prompt(src, tgt)
            result = call_model(prompt, llm_backend)
            return {
                'prompt': prompt,
                'output': result,
                'label': parse_label(result)
            }
        
        logs = run_debug_test(debug_wrapper, 'judge_type', 'method', 
                              [args.llm_backend], 'all')
        save_debug_log(logs, 'judge_type', 'method')
        return
```

## Benefits

1. **Quick Validation**: Test if a judge is working correctly without processing entire datasets
2. **Cost Effective**: Uses only a few samples, minimizing API costs during development
3. **Debugging Aid**: Shows full prompts and outputs for troubleshooting
4. **Multi-language Testing**: Automatically tests both English and Ukrainian samples
5. **Reproducible**: Uses deterministic sample selection for consistent testing

## Troubleshooting

If debug mode shows errors:

1. **Check API Keys**: Ensure your `.env` file has the correct API tokens
2. **Verify Data Files**: Check that evaluation datasets exist in `data/eval/`
3. **Model Availability**: Confirm the specified backend is properly configured
4. **Language Detection**: Verify langid is installed (`pip install langid`)

## Future Enhancements

Potential improvements to debug mode:

1. Add support for custom sample selection
2. Include performance benchmarking across models
3. Generate comparison reports across different judge methods
4. Add interactive mode for manual sample testing
5. Support for testing with specific error types
