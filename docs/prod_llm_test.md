# Production LLM Endpoint Testing

## Overview
The `test/prod_llm.py` script tests both production and QA LLM endpoints to verify connectivity and functionality.

## Prerequisites

1. Set up environment variables in a `.env` file in the project root:
```bash
API_TOKEN=your_api_token_here
OPENAI_API_KEY=your_openai_api_key_here  # Optional for OpenAI Direct tests
```

2. If running on Red Sparta (production network), ensure proxy variables are set:
```bash
INTERNAL_HTTP_PROXY=http://proxy.example.com:8080
INTERNAL_HTTPS_PROXY=http://proxy.example.com:8080
```

## Running the Tests

```bash
cd scripts/multilingual/gec_judge
python test/prod_llm.py
```

## Endpoints Tested

### LLM Proxy
- **Production**: `http://clapi.prod-cheetah.grammarlyaws.com/api/v0/llm-proxy`
- **QA**: `http://clapi.qa-text-processing.grammarlyaws.com/api/v0/llm-proxy`

Models tested:
- `openai_direct_chat_gpt4o_mini`
- `openai_direct_gpt41_nano`
- `gas_gemini20_flash_lite`

### Transparent API
- **Production**: `http://clapi.prod-cheetah.grammarlyaws.com/transparent/openai/v1`
- **QA**: `http://clapi.qa-text-processing.grammarlyaws.com/transparent/openai/v1`

Models tested:
- `gpt-4o-mini`
- `gpt-4.1-nano`

### OpenAI Direct API
- **Endpoint**: `https://api.openai.com/v1/chat/completions`

Models tested:
- `gpt-5-nano`
- `gpt-5-mini`

## Test Output

The script provides:
1. Individual test results with response times and colored status indicators
2. Error messages for failed tests
3. Connectivity check for all endpoints
4. Summary statistics including success rate and average response time
5. List of working endpoints
6. Automatic troubleshooting tips based on error patterns

## Troubleshooting

### No API Token
- Ensure `API_TOKEN` is set in your environment or `.env` file
- The script will continue with limited testing if no token is found

### Connection Errors on Red Sparta (Production)
- Production endpoints return HTML authentication pages when accessed from outside the production network
- Verify that `INTERNAL_HTTP_PROXY` is correctly set when running on production network
- May require VPN access or running directly on Red Sparta infrastructure

### Invalid Token Errors
- QA LLM Proxy requires a valid `API_TOKEN` with proper permissions
- Contact your team for valid API credentials

### Model-Specific Issues
- GPT-5 models don't support the `temperature` parameter (the script handles this automatically)
- Some models may require specific permissions or may not be available in certain environments
- Transparent API requires `X-LLM-Proxy-Calling-Service` header (automatically included)

## Current Status (Based on Test Results)

### Working Endpoints
- **QA Transparent API**: Fully functional with models like gpt-4o-mini and gpt-4.1-nano
- **OpenAI Direct API**: Working with gpt-5-nano and gpt-5-mini models

### Endpoints Requiring Additional Setup
- **Production Endpoints**: Require VPN or running from production network
- **QA LLM Proxy**: Requires valid API_TOKEN with proper permissions

## Exit Codes
- `0`: All tests passed
- `1`: One or more tests failed

## Features

### Connectivity Check
The script performs a quick connectivity check on all endpoints before running the actual tests, showing:
- Whether the endpoint is reachable
- HTTP status code returned
- Connection timeouts or failures

### Color-Coded Output
The script uses ANSI color codes for better readability:
- Green: Successful tests and working endpoints
- Red: Failed tests and errors
- Yellow: Warnings and tips
- Blue: Response content

### Smart Error Handling
- Detects HTML responses (authentication pages)
- Identifies API error messages
- Provides context-specific troubleshooting tips
- Handles model-specific requirements (e.g., GPT-5 temperature parameter)
