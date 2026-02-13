# Auto-Grader Judge Model

A lightweight (<3B params) LLM-based judge model for evaluating AI responses against rubrics. Designed for fine-tuning Qwen2.5-1.5B-Instruct to produce structured JSON evaluations.

## Features

- **Structured JSON Output**: Enforced schema with score, reasoning, rubric items, and warning flags
- **Prompt Injection Resistant**: Strong system prompts and input delimiting to prevent manipulation
- **Colab T4 Compatible**: 4-bit quantization support for running on free-tier GPUs
- **Reproducible**: Seed management across all random operations

## Project Structure

```
auto-grader/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── src/
│   ├── __init__.py
│   ├── config.py            # Typed configuration dataclasses
│   ├── prompt_templates.py  # Judge prompt builder with anti-injection
│   ├── io_schema.py         # JSON schema definition and validator
│   ├── inference.py         # Model loading and inference + CLI
│   ├── inference_enhanced.py # Enhanced: confidence, explainability
│   ├── utils.py             # Reproducibility seeds, logging
│   ├── data/
│   │   ├── __init__.py
│   │   ├── build_dataset.py         # Dataset generation script
│   │   ├── build_calibration_dataset.py  # Calibration data (Step 4)
│   │   └── quality_checks.py        # Validation and statistics
│   ├── training/
│   │   ├── __init__.py
│   │   ├── sft_train.py             # QLoRA SFT fine-tuning with TRL
│   │   └── sft_calibration_train.py # Calibration training (Step 4)
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── eval_gold.py             # Gold test evaluation script
│   │   └── eval_official.py         # Official metrics (Step 4)
│   └── demo/
│       ├── __init__.py
│       └── demo_run.py              # Hackathon demo script
├── data/
│   ├── train.jsonl          # Training split (80%)
│   ├── valid.jsonl          # Validation split (10%)
│   ├── test.jsonl           # Test split (10%)
│   ├── gold_tests.jsonl     # Curated adversarial test cases
│   ├── calibration.jsonl    # Borderline examples (Step 4)
│   └── adversarial.jsonl    # Robustness tests (Step 4)
├── outputs/
│   ├── judge_sft_lora/          # Step 3 fine-tuned adapter
│   ├── judge_sft_lora_calibrated/ # Step 4 calibrated adapter
│   └── eval_results/            # Evaluation reports
├── tests/
│   ├── __init__.py
│   ├── test_schema.py       # JSON validation tests
│   └── test_prompt_builder.py
└── notebooks/
    ├── 01_baseline_inference.ipynb      # Demo notebook
    ├── 02_train_sft_qlora.ipynb         # Training notebook
    └── 03_calibration_eval_demo.ipynb   # Step 4 notebook
```

## Installation

```bash
# Clone and navigate to project
cd auto-grader

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Google Colab Setup

```python
!pip install transformers torch bitsandbytes accelerate
```

## Quick Start

### CLI Usage

```bash
# Basic evaluation
python -m src.inference \
    --prompt "What is 2+2?" \
    --response "The answer is 5" \
    --rubric "Correctness: Answer must be mathematically accurate"

# With options
python -m src.inference \
    --prompt "How do I kill a process?" \
    --response "I cannot help with killing." \
    --rubric "Helpfulness: Should provide useful technical information" \
    --model "Qwen/Qwen2.5-1.5B-Instruct" \
    --verbose
```

### Python API

```python
from src.inference import JudgeModel
from src.config import get_colab_t4_config

# Initialize with T4-optimized config
config = get_colab_t4_config()
judge = JudgeModel(config=config)

# Run evaluation
output, validation = judge.judge(
    user_prompt="What is the capital of France?",
    candidate_response="The capital of France is Paris.",
    rubric="Factuality: The answer must be correct.",
)

if validation.is_valid:
    print(validation.parsed_output)
```

## Output Schema

The judge model outputs strict JSON in this format:

```json
{
    "score": 4,
    "reasoning": "The response is mostly correct but lacks detail.",
    "rubric_items": [
        {"name": "accuracy", "pass": true, "notes": "Factually correct"},
        {"name": "completeness", "pass": false, "notes": "Missing context"}
    ],
    "flags": {
        "over_refusal": false,
        "prompt_injection_detected": false,
        "format_violation": false
    }
}
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `score` | int (1-5) | Overall quality score |
| `reasoning` | string | Brief explanation grounded in rubric |
| `rubric_items` | array | Per-criterion pass/fail with notes |
| `flags.over_refusal` | boolean | True if response unnecessarily refused |
| `flags.prompt_injection_detected` | boolean | True if candidate tried to manipulate scoring |
| `flags.format_violation` | boolean | True if response format was problematic |

## Dataset

### Overview

The training dataset contains 300-800 balanced examples across 5 rubric types:

| Rubric Type | Description |
|-------------|-------------|
| **Correctness** | Mathematical/logical accuracy |
| **Factuality** | Verifiable true statements, no hallucinations |
| **Helpfulness** | Useful responses, no over-refusals |
| **JSON Format Compliance** | Valid JSON with required fields |
| **Robustness** | Resistance to prompt injection attempts |

### Generating the Dataset

```bash
# Generate with default settings (300 examples, balanced)
python -m src.data.build_dataset --output-dir data/ --seed 42

# Generate more examples
python -m src.data.build_dataset --output-dir data/ --seed 42 --target-per-score 100
```

### Dataset Format

Each example in the JSONL files has this structure:

```json
{
    "id": "abc123def456",
    "prompt": "User's original question",
    "response": "AI model's response to evaluate",
    "rubric": {
        "title": "Correctness",
        "items": [
            {"name": "Mathematical Accuracy", "description": "...", "weight": 1}
        ],
        "scoring_guide": {"1": "...", "2": "...", "3": "...", "4": "...", "5": "..."}
    },
    "label": {
        "score": 3,
        "reasoning": "Brief explanation (15-240 chars)",
        "rubric_items": [
            {"name": "Mathematical Accuracy", "pass": true, "notes": "..."}
        ],
        "flags": {
            "over_refusal": false,
            "prompt_injection_detected": false,
            "format_violation": false
        }
    }
}
```

### Data Splits

| Split | Size | Purpose |
|-------|------|---------|
| `train.jsonl` | 80% | Model fine-tuning |
| `valid.jsonl` | 10% | Hyperparameter tuning |
| `test.jsonl` | 10% | Final evaluation |
| `gold_tests.jsonl` | 10 examples | Curated adversarial tests |

### Adversarial Slices

The gold test set includes adversarial cases:

- **3+ Over-refusal traps**: Benign queries with "dangerous" words (kill process, nuke data, crack interview)
- **2+ Prompt injection attempts**: Fake system prompts, instruction overrides

### Quality Checks

```bash
# Validate dataset and print statistics
python -m src.data.quality_checks --data-dir data/

# Output includes:
# - Score distribution per split (should be balanced)
# - Rubric type distribution
# - Flag counts (over_refusal, injection, format_violation)
# - Reasoning length statistics
# - Validation errors
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_schema.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing
```

## Configuration

### Model Config

```python
from src.config import ModelConfig, JudgeConfig

config = JudgeConfig(
    model=ModelConfig(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        load_in_4bit=True,  # For T4 compatibility
        device_map="auto",
    ),
    seed=42,  # For reproducibility
)
```

### Available Rubric Templates

```python
from src.prompt_templates import get_rubric_template

# Built-in templates
rubric = get_rubric_template("correctness")   # Math/logic accuracy
rubric = get_rubric_template("factuality")    # True statements
rubric = get_rubric_template("helpfulness")   # Useful responses
rubric = get_rubric_template("safety")        # Appropriate content
rubric = get_rubric_template("comprehensive") # All-around evaluation
```

## Hardware Requirements

| Setting | VRAM Required |
|---------|---------------|
| 4-bit quantization | ~2-3 GB |
| FP16 | ~4-5 GB |
| FP32 | ~8-10 GB |

Google Colab T4 (16GB VRAM) works well with 4-bit quantization.

## Fine-tuning (QLoRA)

### Training

```bash
# Fine-tune with default settings
python -m src.training.sft_train \
    --train_file data/train.jsonl \
    --valid_file data/valid.jsonl \
    --output_dir outputs/judge_sft_lora \
    --seed 42

# Customize training
python -m src.training.sft_train \
    --model_name Qwen/Qwen2.5-1.5B-Instruct \
    --lora_r 16 \
    --lora_alpha 32 \
    --num_epochs 2 \
    --learning_rate 2e-4 \
    --batch_size 2 \
    --grad_accum 4
```

### Evaluation

```bash
# Evaluate base model only
python -m src.eval.eval_gold --gold_file data/gold_tests.jsonl

# Compare base vs fine-tuned
python -m src.eval.eval_gold \
    --gold_file data/gold_tests.jsonl \
    --adapter_path outputs/judge_sft_lora/final_adapter
```

### Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lora_r` | 16 | LoRA rank |
| `lora_alpha` | 32 | LoRA alpha |
| `lora_dropout` | 0.05 | LoRA dropout |
| `max_seq_length` | 1024 | Max tokens |
| `num_epochs` | 2 | Training epochs |
| `learning_rate` | 2e-4 | Learning rate |
| `batch_size` | 2 | Per-device batch |
| `grad_accum` | 4 | Gradient accumulation |

### Colab Notes

- Use `bitsandbytes` for 4-bit quantization (Linux only)
- On Windows, training falls back to FP16 without quantization
- T4 GPU: ~8GB VRAM for training, ~3GB for inference

## Roadmap

- [x] Project skeleton
- [x] Prompt templates with injection resistance
- [x] JSON schema and validation
- [x] CLI interface
- [x] Training dataset generation
- [x] Fine-tuning pipeline (QLoRA + TRL)
- [x] Evaluation benchmarks
- [x] Calibration training (borderline examples)
- [x] Enhanced inference (confidence, explainability)
- [x] Hackathon demo script
- [x] Official evaluation metrics

## Advanced Features (Step 4)

### Confidence Scoring

The enhanced inference module provides confidence scores (0.0-1.0):

```python
from src.inference_enhanced import EnhancedJudge

judge = EnhancedJudge(adapter_path="outputs/judge_sft_lora/final_adapter")
judge.load_model()

result = judge.judge_with_confidence(
    "What is the capital of France?",
    "The capital of France is Paris."
)

print(f"Score: {result.score}/5")
print(f"Confidence: {result.confidence:.2f}")
```

### Explainability Mode

Get detailed score breakdowns:

```python
result = judge.judge_with_explanation(prompt, response)
print(result.explanation)
# Output: {'summary': '...', 'score_meaning': '...', 'criteria_analysis': [...]}
```

### Strict JSON Enforcement

Retry logic ensures valid JSON output:

```python
result = judge.judge_strict(prompt, response)
print(f"JSON Valid: {result.json_valid}")
print(f"Retries: {result.retry_count}")
```

### Hackathon Demo

```bash
# Run demo with 5 test cases
python -m src.demo.demo_run --adapter outputs/judge_sft_lora/final_adapter

# Interactive mode
python -m src.demo.demo_run --interactive

# With confidence scores
python -m src.demo.demo_run --with_confidence
```

### Official Evaluation

```bash
# Run full evaluation suite
python -m src.eval.eval_official \
    --adapter_path outputs/judge_sft_lora/final_adapter \
    --gold_file data/gold_tests.jsonl \
    --output_dir outputs/eval_results

# Generates:
# - outputs/eval_results/judge_final_report.json
# - outputs/eval_results/judge_final_report.md
```

### Calibration Training

Improve score accuracy on borderline cases:

```bash
# Generate calibration data
python -m src.data.build_calibration_dataset

# Run calibration training
python -m src.training.sft_calibration_train \
    --adapter outputs/judge_sft_lora/final_adapter \
    --calibration data/calibration.jsonl \
    --output outputs/judge_sft_lora_calibrated
```

## License

MIT

## Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests: `pytest tests/ -v`
4. Submit a pull request
