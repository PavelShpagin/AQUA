#!/usr/bin/env python3
"""
Agent-as-a-judge implementation per docs/general.md.

ReAct-style agent with tools for GEC evaluation:
- Language rulebooks with grammar rules (RAG)
- Guidelines and knowledge base
- Web search for domain knowledge
- Meaning-change tool
- Nonsense detector
- Reward/quality tool
"""

import os
import re
from typing import Tuple, List, Dict, Any
from utils.llm.backends import call_model
try:
    from utils.agent.enhanced_tools import AVAILABLE_TOOLS
except ImportError:
    from utils.agent.tools import AVAILABLE_TOOLS


def classify_agent(s: str, t: str, *, language_label: str, prompts, backend: str = "gpt-4o-mini", api_token: str = None, aligned: str = "") -> Tuple[str, str]:
    """
    Agent-as-a-judge with tool access for edge cases and complex evaluations.
    
    Uses ReAct pattern: Reasoning -> Acting (tool use) -> Observing -> repeat until conclusion.
    Uses judge-specific agent prompts from the prompts module.
    """
    # Prefer explicit token, otherwise read from environment. Do not hard-fail
    # if API_TOKEN is missing when OPENAI_API_KEY is available (direct path).
    if not api_token:
        # Try OPENAI_API_KEY first as it's more reliable
        api_token = os.getenv('OPENAI_API_KEY', '') or os.getenv('API_TOKEN', '')
    if not api_token:
        return 'Error', 'LLM credentials missing (set API_TOKEN or OPENAI_API_KEY)'
    
    # Use judge-specific agent prompt from prompts.py
    # Build agent system prompt; include alignment when template supports it
    try:
        system_prompt = prompts.AGENT_PROMPT.format(
            language=language_label,
            original=s,
            suggested=t,
            aligned=aligned or ""
        )
    except Exception:
        system_prompt = prompts.AGENT_PROMPT.format(
            language=language_label,
            original=s,
            suggested=t
        )

    conversation_history = [system_prompt]
    max_iterations = 5
    
    for iteration in range(max_iterations):
        # Get agent response
        full_prompt = "\n\n".join(conversation_history)
        
        ok, response, _tokens = call_model(full_prompt, backend, api_token)
        
        if not ok:
            return 'Error', f'Agent call failed at iteration {iteration}: {response}'
        
        conversation_history.append(response)
        
        # Check for final answer
        final_match = re.search(r'Final Answer:\s*(TP|FP[123])\s*[-–]\s*(.+)', response, re.IGNORECASE | re.DOTALL)
        if final_match:
            classification = final_match.group(1).upper()
            reasoning = final_match.group(2).strip()
            
            # Include full conversation in reasoning for transparency
            full_reasoning = f"Agent reasoning: {reasoning}\n\nFull analysis:\n" + "\n".join(conversation_history[-3:])
            return classification, full_reasoning
        
        # Parse and execute tool calls
        action_match = re.search(r'Action:\s*(\w+)\s*\(([^)]*)\)', response)
        if action_match:
            tool_name = action_match.group(1)
            args_str = action_match.group(2)
            
            if tool_name in AVAILABLE_TOOLS:
                try:
                    # Parse arguments (simplified - assumes string args)
                    args = [arg.strip().strip('"\'') for arg in args_str.split(',') if arg.strip()]
                    
                    # Execute tool with appropriate arguments
                    if tool_name == "grammar_rag":
                        # New signature: grammar_rag(query, language, backend, api_token)
                        query = args[0] if args else "general grammar rules"
                        lang = args[1] if len(args) > 1 else language_label
                        result = AVAILABLE_TOOLS[tool_name](query, lang, backend, api_token)
                    else:
                        result = {"error": f"Unknown tool: {tool_name}"}
                    
                    # Format observation for better agent understanding
                    if isinstance(result, dict):
                        obs_parts = []
                        if 'result' in result:
                            obs_parts.append(f"Result: {result['result']}")
                        if 'severity' in result:
                            obs_parts.append(f"Severity: {result['severity']}")
                        if 'score' in result:
                            obs_parts.append(f"Score: {result['score']}")
                        if 'confidence' in result:
                            obs_parts.append(f"Confidence: {result['confidence']}")
                        if 'reasoning' in result:
                            obs_parts.append(f"Reasoning: {result['reasoning'][:200]}")
                        observation = f"Observation: {' | '.join(obs_parts)}"
                    else:
                        observation = f"Observation: {result}"
                    conversation_history.append(observation)
                    
                except Exception as e:
                    error_obs = f"Observation: Tool error - {str(e)}"
                    conversation_history.append(error_obs)
            else:
                error_obs = f"Observation: Unknown tool '{tool_name}'. Available tools: {list(AVAILABLE_TOOLS.keys())}"
                conversation_history.append(error_obs)
        
        # If no action found, encourage the agent to conclude
        elif iteration >= 2:
            prompt_conclusion = "Please provide your Final Answer now in the format: Final Answer: [TP|FP1|FP2|FP3] - [reasoning]"
            conversation_history.append(prompt_conclusion)
    
    # Fallback: Use tool outputs to make best guess
    tool_scores = {'nonsense': None, 'meaning': None, 'quality': None}
    
    # Extract scores from conversation history
    for entry in conversation_history:
        if 'Observation:' in entry:
            if 'Severity:' in entry:
                try:
                    tool_scores['meaning'] = int(re.search(r'Severity:\s*(\d+)', entry).group(1))
                except:
                    pass
            if 'Score:' in entry and 'nonsense' in conversation_history[conversation_history.index(entry)-1].lower():
                try:
                    tool_scores['nonsense'] = int(re.search(r'Score:\s*([-]?\d+)', entry).group(1))
                except:
                    pass
            if 'Score:' in entry and 'quality' in conversation_history[conversation_history.index(entry)-1].lower():
                try:
                    tool_scores['quality'] = int(re.search(r'Score:\s*([-]?\d+)', entry).group(1))
                except:
                    pass
    
    # Apply same logic as modular judge
    if tool_scores['nonsense'] is not None and tool_scores['nonsense'] >= 2:
        return 'FP1', f'Agent timeout - but detected major nonsense (score: {tool_scores["nonsense"]})'
    elif tool_scores['meaning'] is not None and tool_scores['meaning'] >= 2:
        return 'FP1', f'Agent timeout - but detected meaning change (severity: {tool_scores["meaning"]})'
    elif tool_scores['quality'] is not None and tool_scores['quality'] < 0:
        return 'FP2', f'Agent timeout - but detected quality degradation (score: {tool_scores["quality"]})'
    elif tool_scores['quality'] is not None and 0 <= tool_scores['quality'] < 1:
        return 'FP3', f'Agent timeout - but minimal improvement (quality: {tool_scores["quality"]})'
    elif tool_scores['quality'] is not None and tool_scores['quality'] >= 1:
        return 'TP', f'Agent timeout - but quality improvement detected (score: {tool_scores["quality"]})'
    else:
        return 'FP3', f'Agent timeout after {max_iterations} iterations - insufficient tool data for classification'


