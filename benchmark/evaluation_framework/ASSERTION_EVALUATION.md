# Assertion-Based Evaluation

The evaluation framework supports two types of evaluation:

1. **Score-based evaluation** - Uses LLM judge to score agent outputs against rubrics
2. **Assertion-based evaluation** - Validates agent outputs against specific assertions

## Assertion Types

### 1. contains-all

Checks if the agent response contains specified value(s).

**Single value:**
```yaml
- vars:
    question: "Load the file 'data.tiff'. Respond with <1> if successful, or <0> if failed."
  assert:
    - type: contains-all
      value: "<1>"
```

**Multiple values (ALL must be present):**
```yaml
- vars:
    question: "List all loaded layers."
  assert:
    - type: contains-all
      value: ["layer1", "layer2", "layer3"]
```

### 2. not-contains

Checks if the agent response does NOT contain specified value(s).

**Single value:**
```yaml
- vars:
    question: "Load the file 'data.tiff'. Respond with <1> if successful, or <0> if failed."
  assert:
    - type: not-contains
      value: "<0>"
```

**Multiple values (NONE should be present):**
```yaml
- vars:
    question: "Run the task without errors."
  assert:
    - type: not-contains
      value: ["ERROR", "FAILED", "EXCEPTION"]
```

### 3. llm-rubric

Uses LLM to evaluate the response against a rubric.

```yaml
- vars:
    question: "Create a volume rendering of the dataset."
  assert:
    - type: llm-rubric
      value: |
        The response should:
        1. Confirm that volume rendering was created
        2. Mention appropriate color mapping
        3. Indicate successful visualization
```

## Binary Response Pattern

A common pattern for simple pass/fail tasks:

```yaml
- vars:
    question: |
      Load the file "dataset.tiff".
      Respond with <1> if successful, or <0> if failed. Only respond with <1> or <0>.
  assert:
    - type: contains-all
      value: "<1>"
    - type: not-contains
      value: "<0>"
```

When the framework detects this pattern (both `contains-all: <1>` and `not-contains: <0>`), it treats it as a single assertion:
- Response contains `<1>` → **PASS** (score = 1)
- Response contains `<0>` → **FAIL** (score = 0)
- Response is anything else → **INVALID** (score = 0)

## Evaluation Result

Assertion-based evaluation returns:

```json
{
  "status": "completed",
  "case_name": "test_case",
  "model": "rule-based",
  "agent_response": "<agent's full response>",
  "assertion_results": [
    {
      "assertion_index": 0,
      "type": "contains-all",
      "value": "<1>",
      "passed": true,
      "score": 1,
      "details": "Checking if response contains '<1>': ✓"
    }
  ],
  "scores": {
    "total_score": 1,
    "total_passed": 1,
    "total_assertions": 1,
    "pass_rate": 1.0,
    "average_score": 1.0
  },
  "score": 1,
  "timestamp": "2025-01-15T10:30:00"
}
```

## Usage with Evaluation Framework

The framework automatically detects assertion-based test cases:

```bash
python -m benchmark.evaluation_framework.run_evaluation \
    --agent napari_mcp \
    --config config.json \
    --yaml benchmark/eval_cases/napari/0_actions/eval_basic_napari_functions.yaml \
    --cases benchmark/eval_cases/napari/0_actions \
    --data-dir /path/to/napari/data
```

No special configuration is needed - the framework automatically uses `AssertionEvaluator` when it detects assertion-based YAML test cases.

## Mixing Assertion Types

You can combine different assertion types:

```yaml
- vars:
    question: "Create a visualization and save it."
  assert:
    - type: contains-all
      value: ["visualization created", "saved"]
    - type: not-contains
      value: ["error", "failed"]
    - type: llm-rubric
      value: "The response should describe the visualization in detail."
```

All assertions must pass for the test case to be considered successful (score = 1).

## Comparison with Score-Based Evaluation

| Feature | Score-Based | Assertion-Based |
|---------|-------------|-----------------|
| Evaluation Method | LLM judge with rubric | Rule-based + optional LLM |
| Result | Numeric score (0-100) | Binary pass/fail (0 or 1) |
| Use Case | Complex visual quality assessment | Simple function testing |
| Example | "Create a volume rendering with proper color mapping" | "Load file. Respond <1> if success, <0> if fail" |
| Typical Benchmarks | main, chatvis_bench, sci_volume_data | bioimage_data (napari actions) |

## Implementation Details

The `AssertionEvaluator` class (in `benchmark/evaluation_helpers/assertion_evaluator.py`) handles assertion-based evaluation:

1. Loads agent response from test results
2. Evaluates each assertion against the response
3. Returns pass/fail result with detailed assertion results
4. Saves evaluation result to `evaluation_results/{eval_mode}/evaluation_result_*.json`

Both MCP and PVPython eval modes are supported.
