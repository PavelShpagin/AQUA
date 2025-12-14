# Inner Debate Ensemble Method

## Overview

The **Inner Debate Ensemble** is a novel ensemble method that improves GEC (Grammatical Error Correction) judgment accuracy by creating structured debates between competing viewpoints before making final decisions.

## Algorithm

### 1. Dominance Priority System

The ensemble uses a strict dominance hierarchy to identify the most important competing opinions:

```
FP1 > FP2 > FP3 > FN > TP/TN
```

Where:
- **FP1**: Critical False Positive (highest priority - most important to identify)
- **FP2**: Medium False Positive  
- **FP3**: Minor False Positive
- **FN**: False Negative
- **TP/TN**: True Positive/True Negative (lowest priority, with tie-breaking)

**TP/TN Tie-Breaking**: When `src == tgt`, TN dominates; otherwise TP dominates.

### 2. Debate Creation Process

1. **Collect Initial Judgments**: Run n_judges (minimum 2) in parallel using different backends
2. **Check for Consensus**: If all judges agree on the same label, return that unanimous decision immediately (no debate needed)
3. **Find Top 2 Dominant Classes**: Identify the two most dominant label classes using the priority system
4. **Select Balanced Debaters**: Take the last `min(m, k)` judges from each dominant class (where m and k are the vote counts)
5. **Create Alternating Arguments**: Format reasoning from both sides in alternating order
6. **Final Arbitration**: Use a final judge with custom debate prompts to make the decision

### 3. Example Scenarios

#### Scenario A: Unanimous Consensus
**Input Judges**: `[TP(R1), TP(R2), TP(R3), TP(R4)]`
**Result**: Return `TP` immediately (no debate needed)

#### Scenario B: Debate Required
**Input Judges**: `[TP(R1), TP(R2), FP3(R3), FP1(R4)]`

**Step 1**: Check consensus → Multiple unique labels, proceed to debate
**Step 2**: Identify dominant classes → `TP` (2 votes) and `FP1` (1 vote, but higher priority)
**Step 3**: Select debaters → `min(2, 1) = 1` → Use `TP(R2)` and `FP1(R4)`
**Step 4**: Create debate:
```
FP1 Argument:
R4

TP Argument:
R2
```
**Step 5**: Final judge sees the debate and makes informed decision

## Implementation Details

### Files Modified/Created

- **`ensembles/inner_debate.py`**: Main ensemble implementation
- **`ensembles/prompts.py`**: Added debate-specific prompts for each judge type
- **`test_inner_debate_logic.py`**: Comprehensive unit tests for core logic

### Key Functions

- `get_dominance_priority()`: Calculates dominance scores for labels
- `get_two_most_dominant_classes()`: Finds competing viewpoints
- `create_debate_opinions()`: Formats alternating arguments
- `call_debate_judge_for_row()`: Custom final judge with debate prompts

### Debate Prompts

Each judge type (feedback, sentence, edit) has specialized debate prompts that:
- Present the structured debate clearly
- Ask for final decision after weighing arguments
- Request explanation of which side was more convincing

## Usage

```bash
python -m ensembles.inner_debate \
    --judge feedback \
    --method baseline \
    --backends gpt-4o-mini gpt-3.5-turbo \
    --lang en \
    --n_judges 4 \
    --input data.csv \
    --output results.csv
```

## Expected Benefits

1. **Better Conflict Resolution**: Structured debates help resolve disagreements between high-quality judges
2. **Transparency**: Final decisions include reasoning from both sides
3. **Robust Decision Making**: Uses dominance hierarchy to focus on most important disagreements
4. **Balanced Arguments**: min(m, k) logic ensures fair representation of both sides
5. **Efficiency**: Unanimous decisions skip the debate process entirely, saving computational cost and reducing latency

## Testing

The implementation includes comprehensive unit tests that verify:
- ✅ Dominance priority calculation
- ✅ Two most dominant class identification  
- ✅ Alternating debate creation
- ✅ Edge cases (single class, various scenarios)
- ✅ TP/TN tie-breaking logic
- ✅ Unanimous consensus handling (skips debate)

Run tests with:
```bash
python test_inner_debate_logic.py
```

## Integration

The inner_debate ensemble follows the same interface as other ensembles (consistency, weighted) and integrates seamlessly with the existing infrastructure:
- Uses shared utilities from `utils.ensemble`
- Compatible with all judge types (feedback, sentence, edit, tnfn)
- Supports all backend configurations
- Includes proper error handling and pricing tracking

This method represents a significant advancement in ensemble techniques for GEC evaluation, combining the benefits of multiple judges with structured argumentation for more reliable decisions.
