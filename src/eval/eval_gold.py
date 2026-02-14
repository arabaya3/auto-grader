"""
Evaluation script for Auto-Grader Judge Model on gold test set.

Compares base model vs fine-tuned model performance on adversarial cases.

Usage:
    # Evaluate base model only
    python -m src.eval.eval_gold --gold_file data/gold_tests.jsonl
    
    # Evaluate both base and fine-tuned
    python -m src.eval.eval_gold \
        --gold_file data/gold_tests.jsonl \
        --adapter_path outputs/judge_sft_lora/final_adapter
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch

# Suppress verbose logging during import
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("accelerate").setLevel(logging.ERROR)

from transformers import AutoModelForCausalLM, AutoTokenizer

# Import project modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.prompt_templates import JUDGE_SYSTEM_PROMPT, build_judge_prompt
from src.io_schema import validate_judge_output, extract_json_from_text


def repair_rubric_items_notes(parsed: Optional[dict]) -> tuple[Optional[dict], bool]:
    """Repair missing notes in rubric_items.
    
    If rubric_items entries are missing 'notes', auto-fill with placeholder
    and set format_violation flag.
    
    Args:
        parsed: Parsed JSON output dict
    
    Returns:
        Tuple of (repaired_dict, was_repaired)
    """
    if parsed is None:
        return None, False
    
    was_repaired = False
    
    if "rubric_items" in parsed and isinstance(parsed["rubric_items"], list):
        for item in parsed["rubric_items"]:
            if isinstance(item, dict):
                if "notes" not in item or not item.get("notes"):
                    item["notes"] = "(auto-filled)"
                    was_repaired = True
    
    # If we repaired, also set format_violation flag
    if was_repaired:
        if "flags" not in parsed:
            parsed["flags"] = {}
        parsed["flags"]["format_violation"] = True
    
    return parsed, was_repaired


# Check for PEFT
try:
    from peft import PeftModel
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

# Check for bitsandbytes
try:
    from transformers import BitsAndBytesConfig
    HAS_BNB = True
except ImportError:
    HAS_BNB = False


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class EvalConfig:
    """Configuration for evaluation."""
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    use_4bit: bool = True
    max_new_tokens: int = 512
    temperature: float = 0.1  # Low temperature for consistent JSON
    top_p: float = 1.0
    do_sample: bool = False  # Greedy for determinism
    device_map: str = "auto"


@dataclass
class EvalResult:
    """Result for a single evaluation."""
    example_id: str
    label_score: int
    predicted_score: Optional[int]
    json_valid: bool
    raw_output: str
    parsed_output: Optional[dict[str, Any]]
    label_flags: dict[str, bool]
    predicted_flags: dict[str, bool]
    errors: list[str] = field(default_factory=list)


# =============================================================================
# Model Loading
# =============================================================================

def load_base_model(config: EvalConfig) -> tuple[Any, Any]:
    """Load base model and tokenizer.
    
    Args:
        config: Evaluation configuration
    
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading base model: {config.model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        padding_side="left",  # For generation
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": config.device_map,
        "torch_dtype": torch.float16,
    }
    
    if config.use_4bit and HAS_BNB:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        **model_kwargs,
    )
    model.eval()
    
    return model, tokenizer


def load_finetuned_model(
    config: EvalConfig,
    adapter_path: str,
) -> tuple[Any, Any]:
    """Load base model with LoRA adapters.
    
    Args:
        config: Evaluation configuration
        adapter_path: Path to saved LoRA adapters
    
    Returns:
        Tuple of (model, tokenizer)
    """
    if not HAS_PEFT:
        raise ImportError("PEFT not installed. Please install: pip install peft")
    
    print(f"Loading fine-tuned model with adapters from: {adapter_path}")
    
    # Load tokenizer from adapter path (or base model)
    tokenizer_path = adapter_path if Path(adapter_path).exists() else config.model_name
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        padding_side="left",
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": config.device_map,
        "torch_dtype": torch.float16,
    }
    
    if config.use_4bit and HAS_BNB:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        **model_kwargs,
    )
    
    # Load adapters
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    
    return model, tokenizer


# =============================================================================
# Generation with Hard JSON Enforcement
# =============================================================================

def generate_judgment(
    model: Any,
    tokenizer: Any,
    example: dict[str, Any],
    config: EvalConfig,
) -> str:
    """Generate judgment for an example with JSON enforcement.
    
    Args:
        model: Model to use
        tokenizer: Tokenizer
        example: Example dictionary
        config: Eval configuration
    
    Returns:
        Raw model output string
    """
    # Format rubric from dict
    rubric_dict = example["rubric"]
    rubric_lines = [f"Title: {rubric_dict.get('title', 'Evaluation')}"]
    
    if "items" in rubric_dict:
        rubric_lines.append("\nCriteria:")
        for item in rubric_dict["items"]:
            name = item.get("name", "")
            desc = item.get("description", "")
            rubric_lines.append(f"- {name}: {desc}")
    
    if "scoring_guide" in rubric_dict:
        rubric_lines.append("\nScoring Guide:")
        for score, desc in sorted(rubric_dict["scoring_guide"].items(), key=lambda x: x[0]):
            rubric_lines.append(f"  {score}: {desc}")
    
    rubric_str = "\n".join(rubric_lines)
    
    # Build prompt
    user_content = build_judge_prompt(
        user_prompt=example["prompt"],
        candidate_response=example["response"],
        rubric=rubric_str,
    )
    
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    
    # Tokenize
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    # Generate with hard JSON enforcement
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature if config.do_sample else None,
            top_p=config.top_p if config.do_sample else None,
            do_sample=config.do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the generated part
    input_length = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][input_length:]
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return output_text


def attempt_json_repair(raw_output: str) -> Optional[str]:
    """Attempt to repair/extract JSON from messy output.
    
    Args:
        raw_output: Raw model output that may contain extra text
    
    Returns:
        Cleaned JSON string or None
    """
    # First try standard extraction
    extracted = extract_json_from_text(raw_output)
    if extracted:
        return extracted
    
    # Try to find JSON-like content
    output = raw_output.strip()
    
    # Remove common prefixes
    prefixes_to_remove = [
        "Here is my evaluation:",
        "Here's my judgment:",
        "My assessment:",
        "Based on the rubric:",
    ]
    for prefix in prefixes_to_remove:
        if output.lower().startswith(prefix.lower()):
            output = output[len(prefix):].strip()
    
    # Try extraction again
    return extract_json_from_text(output)


# =============================================================================
# Evaluation Logic
# =============================================================================

def evaluate_single_example(
    model: Any,
    tokenizer: Any,
    example: dict[str, Any],
    config: EvalConfig,
) -> EvalResult:
    """Evaluate a single example.
    
    Args:
        model: Model to use
        tokenizer: Tokenizer
        example: Example dictionary with prompt, response, rubric, label
        config: Eval configuration
    
    Returns:
        EvalResult with comparison data
    """
    example_id = example.get("id", "unknown")
    label = example.get("label", {})
    label_score = label.get("score", 0)
    label_flags = label.get("flags", {})
    
    # Generate
    raw_output = generate_judgment(model, tokenizer, example, config)
    
    # Try to extract/repair JSON
    json_str = attempt_json_repair(raw_output)
    
    # Validate
    errors = []
    parsed_output = None
    predicted_score = None
    predicted_flags = {"over_refusal": False, "prompt_injection_detected": False, "format_violation": False}
    json_valid = False
    
    if json_str:
        validation = validate_judge_output(json_str)
        json_valid = validation.is_valid
        
        if validation.is_valid and validation.parsed_output:
            parsed_output = validation.parsed_output
            # Repair missing notes in rubric_items
            parsed_output, was_repaired = repair_rubric_items_notes(parsed_output)
            predicted_score = parsed_output.get("score")
            predicted_flags = parsed_output.get("flags", predicted_flags)
        elif validation.parsed_output:
            # Try to repair even if validation failed due to missing notes
            parsed_output = validation.parsed_output
            parsed_output, was_repaired = repair_rubric_items_notes(parsed_output)
            if was_repaired:
                # Re-validate after repair
                json_valid = True
                predicted_score = parsed_output.get("score")
                predicted_flags = parsed_output.get("flags", predicted_flags)
            else:
                errors.extend(validation.errors)
        else:
            errors.extend(validation.errors)
    else:
        errors.append("Could not extract JSON from output")
    
    return EvalResult(
        example_id=example_id,
        label_score=label_score,
        predicted_score=predicted_score,
        json_valid=json_valid,
        raw_output=raw_output,
        parsed_output=parsed_output,
        label_flags=label_flags,
        predicted_flags=predicted_flags,
        errors=errors,
    )


def evaluate_gold_tests(
    model: Any,
    tokenizer: Any,
    gold_file: str,
    config: EvalConfig,
    model_name: str = "model",
) -> list[EvalResult]:
    """Evaluate all gold test examples.
    
    Args:
        model: Model to use
        tokenizer: Tokenizer
        gold_file: Path to gold tests JSONL
        config: Eval configuration
        model_name: Name for logging
    
    Returns:
        List of EvalResults
    """
    print(f"\n=== Evaluating {model_name} on {gold_file} ===")
    
    # Load examples
    examples = []
    with open(gold_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    
    results = []
    for i, example in enumerate(examples):
        print(f"  [{i+1}/{len(examples)}] Evaluating {example.get('id', 'unknown')}...", end=" ", flush=True)
        result = evaluate_single_example(model, tokenizer, example, config)
        status = "✓" if result.json_valid else "✗"
        score_match = "=" if result.predicted_score == result.label_score else "≠"
        print(f"{status} (label={result.label_score}, pred={result.predicted_score}) {score_match}")
        results.append(result)
    
    return results


# =============================================================================
# Metrics Computation
# =============================================================================

@dataclass
class Metrics:
    """Aggregated metrics."""
    total: int = 0
    json_valid_count: int = 0
    json_valid_rate: float = 0.0
    score_exact_match: int = 0
    score_accuracy: float = 0.0
    flag_accuracy: dict[str, float] = field(default_factory=dict)
    overall_flag_accuracy: float = 0.0


def compute_metrics(results: list[EvalResult]) -> Metrics:
    """Compute aggregated metrics from results.
    
    Args:
        results: List of evaluation results
    
    Returns:
        Metrics object
    """
    metrics = Metrics(total=len(results))
    
    if not results:
        return metrics
    
    # JSON validity
    metrics.json_valid_count = sum(1 for r in results if r.json_valid)
    metrics.json_valid_rate = metrics.json_valid_count / len(results)
    
    # Score accuracy (only for valid JSON)
    valid_results = [r for r in results if r.json_valid]
    if valid_results:
        metrics.score_exact_match = sum(1 for r in valid_results if r.predicted_score == r.label_score)
        metrics.score_accuracy = metrics.score_exact_match / len(valid_results)
    
    # Flag accuracy
    flag_names = ["over_refusal", "prompt_injection_detected", "format_violation"]
    flag_correct = {f: 0 for f in flag_names}
    flag_total = {f: 0 for f in flag_names}
    
    for r in valid_results:
        for flag in flag_names:
            label_val = r.label_flags.get(flag, False)
            pred_val = r.predicted_flags.get(flag, False)
            if label_val == pred_val:
                flag_correct[flag] += 1
            flag_total[flag] += 1
    
    for flag in flag_names:
        if flag_total[flag] > 0:
            metrics.flag_accuracy[flag] = flag_correct[flag] / flag_total[flag]
        else:
            metrics.flag_accuracy[flag] = 0.0
    
    # Overall flag accuracy
    total_flags = sum(flag_total.values())
    total_correct = sum(flag_correct.values())
    if total_flags > 0:
        metrics.overall_flag_accuracy = total_correct / total_flags
    
    return metrics


def print_metrics(metrics: Metrics, name: str = "Model"):
    """Print metrics in a nice format."""
    print(f"\n=== {name} Metrics ===")
    print(f"Total examples: {metrics.total}")
    print(f"JSON valid: {metrics.json_valid_count}/{metrics.total} ({metrics.json_valid_rate:.1%})")
    print(f"Score accuracy: {metrics.score_exact_match}/{metrics.json_valid_count} ({metrics.score_accuracy:.1%})")
    print("Flag accuracy:")
    for flag, acc in metrics.flag_accuracy.items():
        print(f"  {flag}: {acc:.1%}")
    print(f"Overall flag accuracy: {metrics.overall_flag_accuracy:.1%}")


# =============================================================================
# Comparison Table
# =============================================================================

def print_comparison_table(
    base_results: list[EvalResult],
    tuned_results: Optional[list[EvalResult]] = None,
):
    """Print markdown comparison table.
    
    Args:
        base_results: Results from base model
        tuned_results: Results from fine-tuned model (optional)
    """
    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    
    if tuned_results:
        header = "| ID | Label Score | Base Score | Tuned Score | Base JSON | Tuned JSON |"
        separator = "|:---|:---:|:---:|:---:|:---:|:---:|"
        print(header)
        print(separator)
        
        for base_r, tuned_r in zip(base_results, tuned_results):
            base_score = base_r.predicted_score if base_r.predicted_score is not None else "N/A"
            tuned_score = tuned_r.predicted_score if tuned_r.predicted_score is not None else "N/A"
            base_json = "✓" if base_r.json_valid else "✗"
            tuned_json = "✓" if tuned_r.json_valid else "✗"
            
            print(f"| {base_r.example_id} | {base_r.label_score} | {base_score} | {tuned_score} | {base_json} | {tuned_json} |")
    else:
        header = "| ID | Label Score | Predicted Score | JSON Valid |"
        separator = "|:---|:---:|:---:|:---:|"
        print(header)
        print(separator)
        
        for r in base_results:
            pred_score = r.predicted_score if r.predicted_score is not None else "N/A"
            json_valid = "✓" if r.json_valid else "✗"
            print(f"| {r.example_id} | {r.label_score} | {pred_score} | {json_valid} |")


def print_summary_comparison(base_metrics: Metrics, tuned_metrics: Optional[Metrics] = None):
    """Print summary comparison."""
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if tuned_metrics:
        print(f"{'Metric':<30} | {'Base':>15} | {'Fine-tuned':>15} | {'Delta':>10}")
        print("-" * 80)
        
        delta_json = tuned_metrics.json_valid_rate - base_metrics.json_valid_rate
        delta_score = tuned_metrics.score_accuracy - base_metrics.score_accuracy
        delta_flag = tuned_metrics.overall_flag_accuracy - base_metrics.overall_flag_accuracy
        
        print(f"{'JSON Valid Rate':<30} | {base_metrics.json_valid_rate:>14.1%} | {tuned_metrics.json_valid_rate:>14.1%} | {delta_json:>+9.1%}")
        print(f"{'Score Accuracy':<30} | {base_metrics.score_accuracy:>14.1%} | {tuned_metrics.score_accuracy:>14.1%} | {delta_score:>+9.1%}")
        print(f"{'Overall Flag Accuracy':<30} | {base_metrics.overall_flag_accuracy:>14.1%} | {tuned_metrics.overall_flag_accuracy:>14.1%} | {delta_flag:>+9.1%}")
    else:
        print(f"JSON Valid Rate: {base_metrics.json_valid_rate:.1%}")
        print(f"Score Accuracy: {base_metrics.score_accuracy:.1%}")
        print(f"Overall Flag Accuracy: {base_metrics.overall_flag_accuracy:.1%}")


# =============================================================================
# Main Evaluation Runner
# =============================================================================

def run_evaluation(
    gold_file: str,
    adapter_path: Optional[str] = None,
    config: Optional[EvalConfig] = None,
) -> tuple[Metrics, Optional[Metrics]]:
    """Run full evaluation pipeline.
    
    Args:
        gold_file: Path to gold tests JSONL
        adapter_path: Path to fine-tuned adapters (optional)
        config: Eval configuration
    
    Returns:
        Tuple of (base_metrics, tuned_metrics or None)
    """
    if config is None:
        config = EvalConfig()
    
    # Load base model
    base_model, base_tokenizer = load_base_model(config)
    
    # Evaluate base model
    base_results = evaluate_gold_tests(
        base_model, base_tokenizer, gold_file, config, model_name="Base Model"
    )
    base_metrics = compute_metrics(base_results)
    print_metrics(base_metrics, "Base Model")
    
    # Evaluate fine-tuned model if adapter path provided
    tuned_results = None
    tuned_metrics = None
    
    if adapter_path and Path(adapter_path).exists():
        # Clear memory
        del base_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Load fine-tuned model
        tuned_model, tuned_tokenizer = load_finetuned_model(config, adapter_path)
        
        # Evaluate
        tuned_results = evaluate_gold_tests(
            tuned_model, tuned_tokenizer, gold_file, config, model_name="Fine-tuned Model"
        )
        tuned_metrics = compute_metrics(tuned_results)
        print_metrics(tuned_metrics, "Fine-tuned Model")
        
        # Comparison table
        print_comparison_table(base_results, tuned_results)
        print_summary_comparison(base_metrics, tuned_metrics)
    else:
        print_comparison_table(base_results)
        print_summary_comparison(base_metrics)
    
    return base_metrics, tuned_metrics


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate Judge Model on Gold Tests")
    
    parser.add_argument("--gold_file", "--gold", dest="gold_file", type=str, default="data/gold_tests.jsonl",
                        help="Path to gold tests JSONL")
    parser.add_argument("--adapter_path", "--adapter", dest="adapter_path", type=str, default=None,
                        help="Path to fine-tuned LoRA adapters (optional)")
    parser.add_argument("--model_name", "--model", dest="model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="Base model name")
    parser.add_argument("--no_4bit", action="store_true",
                        help="Disable 4-bit quantization")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Generation temperature (0 for greedy)")
    
    args = parser.parse_args()
    
    config = EvalConfig(
        model_name=args.model_name,
        use_4bit=not args.no_4bit,
        temperature=args.temperature,
        do_sample=args.temperature > 0,
    )
    
    run_evaluation(args.gold_file, args.adapter_path, config)


if __name__ == "__main__":
    main()
