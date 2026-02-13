"""
Official Evaluation Script for Auto-Grader Judge Model.

Computes comprehensive metrics:
- Score Accuracy (exact match)
- Score MAE (mean absolute error)
- Pearson/Spearman correlation
- JSON validity rate
- Flag accuracy (exact match per flag type)
- Robustness suite (adversarial examples)

Outputs:
- judge_final_report.json (structured metrics)
- judge_final_report.md (markdown summary)

Usage:
    python -m src.eval.eval_official \
        --adapter_path outputs/judge_sft_lora_calibrated/final_adapter \
        --gold_file data/gold_tests.jsonl \
        --adversarial_file data/adversarial.jsonl
"""

import argparse
import json
import logging
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# Try to import scipy for correlation
try:
    from scipy.stats import pearsonr, spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Note: scipy not installed, correlation metrics will be skipped")

# Try bitsandbytes
try:
    from transformers import BitsAndBytesConfig
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

# Try PEFT
try:
    from peft import PeftModel
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

# Suppress verbose logging
logging.getLogger("transformers").setLevel(logging.WARNING)

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.prompt_templates import JUDGE_SYSTEM_PROMPT


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    adapter_path: Optional[str] = None
    use_4bit: bool = True
    max_new_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.95


@dataclass
class EvalMetrics:
    """Container for all evaluation metrics."""
    # Core metrics
    total_examples: int = 0
    json_valid_count: int = 0
    json_validity_rate: float = 0.0
    
    # Score metrics
    score_exact_match: int = 0
    score_accuracy: float = 0.0
    score_mae: float = 0.0
    score_within_1: int = 0
    score_within_1_rate: float = 0.0
    
    # Correlation (optional)
    pearson_r: Optional[float] = None
    pearson_p: Optional[float] = None
    spearman_r: Optional[float] = None
    spearman_p: Optional[float] = None
    
    # Flag metrics
    flag_accuracy: Dict[str, float] = None
    flag_precision: Dict[str, float] = None
    flag_recall: Dict[str, float] = None
    flag_f1: Dict[str, float] = None
    
    # Per-score breakdown
    per_score_accuracy: Dict[int, float] = None
    per_score_count: Dict[int, int] = None
    
    # Robustness metrics (adversarial)
    adversarial_results: Dict[str, Dict] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_examples": self.total_examples,
            "json_validity_rate": self.json_validity_rate,
            "score_accuracy": self.score_accuracy,
            "score_mae": self.score_mae,
            "score_within_1_rate": self.score_within_1_rate,
            "pearson_r": self.pearson_r,
            "spearman_r": self.spearman_r,
            "flag_accuracy": self.flag_accuracy,
            "flag_f1": self.flag_f1,
            "per_score_accuracy": self.per_score_accuracy,
            "adversarial_results": self.adversarial_results,
        }


def load_model(config: EvalConfig) -> Tuple:
    """Load model and tokenizer."""
    print(f"Loading model: {config.base_model}")
    
    # Quantization
    quantization_config = None
    if config.use_4bit and HAS_BNB:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model,
        trust_remote_code=True,
        padding_side="left",
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load adapter if provided
    if config.adapter_path and HAS_PEFT:
        if Path(config.adapter_path).exists():
            print(f"Loading adapter: {config.adapter_path}")
            model = PeftModel.from_pretrained(model, config.adapter_path)
        else:
            print(f"WARNING: Adapter not found: {config.adapter_path}")
    
    model.eval()
    return model, tokenizer


def build_prompt(example: dict) -> str:
    """Build evaluation prompt from example."""
    prompt = example["prompt"]
    response = example["response"]
    rubric = example["rubric"]
    
    # Build rubric text
    rubric_items = "\n".join([
        f"- {item['name']}: {item['description']} (weight: {item['weight']})"
        for item in rubric["items"]
    ])
    
    scoring_guide = "\n".join([
        f"  {score}: {desc}"
        for score, desc in rubric["scoring_guide"].items()
    ])
    
    return f"""Evaluate this response:

**User Prompt:** {prompt}

**Assistant Response:** {response}

**Rubric: {rubric['title']}**
{rubric_items}

**Scoring Guide:**
{scoring_guide}

Provide your evaluation as JSON with: reasoning, score (1-5), and flags."""


def generate_judgment(
    model, tokenizer, prompt: str, config: EvalConfig
) -> Tuple[str, Optional[dict]]:
    """Generate judgment and parse JSON."""
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    
    # Parse JSON
    parsed = None
    try:
        # Try direct parse
        parsed = json.loads(response)
    except json.JSONDecodeError:
        # Try to extract JSON from text
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
            except:
                pass
    
    return response, parsed


def compute_flag_metrics(
    predictions: List[dict],
    labels: List[dict],
) -> Tuple[Dict, Dict, Dict, Dict]:
    """Compute flag precision, recall, F1."""
    flag_types = ["factual_error", "hallucination", "refusal_appropriate", 
                  "safety_concern", "incomplete_response", "off_topic"]
    
    accuracy = {}
    precision = {}
    recall = {}
    f1 = {}
    
    for flag in flag_types:
        tp = fp = fn = tn = 0
        
        for pred, label in zip(predictions, labels):
            pred_flag = pred.get("flags", {}).get(flag, False)
            true_flag = label.get("flags", {}).get(flag, False)
            
            if pred_flag and true_flag:
                tp += 1
            elif pred_flag and not true_flag:
                fp += 1
            elif not pred_flag and true_flag:
                fn += 1
            else:
                tn += 1
        
        # Compute metrics
        total = tp + fp + fn + tn
        accuracy[flag] = (tp + tn) / total if total > 0 else 0
        precision[flag] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[flag] = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1[flag] = 2 * precision[flag] * recall[flag] / (precision[flag] + recall[flag]) \
            if (precision[flag] + recall[flag]) > 0 else 0
    
    return accuracy, precision, recall, f1


def evaluate_gold_set(
    model, tokenizer, gold_file: str, config: EvalConfig
) -> Tuple[EvalMetrics, List[dict]]:
    """Evaluate on gold test set."""
    print(f"\nEvaluating on: {gold_file}")
    
    # Load gold data
    examples = []
    with open(gold_file) as f:
        for line in f:
            examples.append(json.loads(line))
    
    print(f"Gold examples: {len(examples)}")
    
    metrics = EvalMetrics()
    metrics.total_examples = len(examples)
    
    predictions = []
    labels = []
    true_scores = []
    pred_scores = []
    per_score_correct = defaultdict(int)
    per_score_total = defaultdict(int)
    
    results = []
    
    for i, example in enumerate(examples):
        prompt = build_prompt(example)
        raw_response, parsed = generate_judgment(model, tokenizer, prompt, config)
        
        label = example["label"]
        true_score = label["score"]
        true_scores.append(true_score)
        per_score_total[true_score] += 1
        
        result = {
            "id": i,
            "prompt_preview": example["prompt"][:100],
            "true_score": true_score,
            "raw_response": raw_response,
        }
        
        if parsed:
            metrics.json_valid_count += 1
            pred_score = parsed.get("score")
            
            result["parsed"] = parsed
            result["pred_score"] = pred_score
            
            if pred_score is not None:
                pred_scores.append(pred_score)
                predictions.append(parsed)
                labels.append(label)
                
                # Score accuracy
                if pred_score == true_score:
                    metrics.score_exact_match += 1
                    per_score_correct[true_score] += 1
                
                if abs(pred_score - true_score) <= 1:
                    metrics.score_within_1 += 1
            else:
                pred_scores.append(0)  # Default for correlation
        else:
            result["parsed"] = None
            result["pred_score"] = None
            pred_scores.append(0)
        
        results.append(result)
        
        if (i + 1) % 10 == 0:
            print(f"Progress: {i + 1}/{len(examples)}")
    
    # Compute metrics
    metrics.json_validity_rate = metrics.json_valid_count / metrics.total_examples * 100
    
    if pred_scores:
        valid_pairs = [(t, p) for t, p in zip(true_scores, pred_scores) if p > 0]
        if valid_pairs:
            true_valid, pred_valid = zip(*valid_pairs)
            metrics.score_mae = sum(abs(t - p) for t, p in valid_pairs) / len(valid_pairs)
            metrics.score_accuracy = metrics.score_exact_match / len(valid_pairs) * 100
            metrics.score_within_1_rate = metrics.score_within_1 / len(valid_pairs) * 100
            
            # Correlation
            if HAS_SCIPY and len(valid_pairs) > 2:
                r, p = pearsonr(true_valid, pred_valid)
                metrics.pearson_r = r
                metrics.pearson_p = p
                
                r, p = spearmanr(true_valid, pred_valid)
                metrics.spearman_r = r
                metrics.spearman_p = p
    
    # Per-score accuracy
    metrics.per_score_accuracy = {}
    metrics.per_score_count = dict(per_score_total)
    for score in range(1, 6):
        if per_score_total[score] > 0:
            metrics.per_score_accuracy[score] = per_score_correct[score] / per_score_total[score] * 100
        else:
            metrics.per_score_accuracy[score] = 0.0
    
    # Flag metrics
    if predictions and labels:
        acc, prec, rec, f1 = compute_flag_metrics(predictions, labels)
        metrics.flag_accuracy = acc
        metrics.flag_precision = prec
        metrics.flag_recall = rec
        metrics.flag_f1 = f1
    
    return metrics, results


def evaluate_adversarial(
    model, tokenizer, adversarial_file: str, config: EvalConfig
) -> Dict[str, Dict]:
    """Evaluate on adversarial examples."""
    print(f"\nEvaluating adversarial: {adversarial_file}")
    
    if not Path(adversarial_file).exists():
        print("Adversarial file not found, skipping...")
        return {}
    
    # Load adversarial data
    examples = []
    with open(adversarial_file) as f:
        for line in f:
            examples.append(json.loads(line))
    
    print(f"Adversarial examples: {len(examples)}")
    
    results = {}
    
    for example in examples:
        test_name = example.get("test_name", "unknown")
        expected = example.get("expected", {})
        
        prompt = build_prompt(example)
        raw_response, parsed = generate_judgment(model, tokenizer, prompt, config)
        
        result = {
            "json_valid": parsed is not None,
            "expected": expected,
            "raw_response": raw_response[:500],
        }
        
        if parsed:
            result["parsed"] = parsed
            
            # Check expectations
            checks = {}
            
            if "score_range" in expected:
                score = parsed.get("score", 0)
                min_s, max_s = expected["score_range"]
                checks["score_in_range"] = min_s <= score <= max_s
            
            if "required_flags" in expected:
                for flag in expected["required_flags"]:
                    checks[f"flag_{flag}"] = parsed.get("flags", {}).get(flag, False)
            
            if "forbidden_content" in expected:
                response_lower = raw_response.lower()
                for forbidden in expected["forbidden_content"]:
                    checks[f"no_{forbidden}"] = forbidden.lower() not in response_lower
            
            result["checks"] = checks
            result["passed"] = all(checks.values()) if checks else True
        else:
            result["passed"] = False
            result["checks"] = {"json_valid": False}
        
        results[test_name] = result
        print(f"  {test_name}: {'PASS' if result['passed'] else 'FAIL'}")
    
    return results


def generate_report(
    metrics: EvalMetrics,
    results: List[dict],
    output_dir: str,
    config: EvalConfig,
) -> Tuple[str, str]:
    """Generate JSON and Markdown reports."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().isoformat()
    
    # JSON report
    json_report = {
        "timestamp": timestamp,
        "config": {
            "base_model": config.base_model,
            "adapter_path": config.adapter_path,
        },
        "metrics": metrics.to_dict(),
        "detailed_results": results[:20],  # First 20 for review
    }
    
    json_path = output_path / "judge_final_report.json"
    with open(json_path, "w") as f:
        json.dump(json_report, f, indent=2)
    
    # Markdown report
    md_lines = [
        "# Auto-Grader Judge Model - Official Evaluation Report",
        f"\n**Generated:** {timestamp}",
        f"\n**Model:** `{config.base_model}`",
        f"\n**Adapter:** `{config.adapter_path or 'None (baseline)'}`",
        "\n---\n",
        "## Summary Metrics",
        "\n| Metric | Value |",
        "|--------|-------|",
        f"| Total Examples | {metrics.total_examples} |",
        f"| JSON Validity Rate | {metrics.json_validity_rate:.1f}% |",
        f"| Score Accuracy | {metrics.score_accuracy:.1f}% |",
        f"| Score MAE | {metrics.score_mae:.2f} |",
        f"| Score Within ±1 | {metrics.score_within_1_rate:.1f}% |",
    ]
    
    if metrics.pearson_r is not None:
        md_lines.append(f"| Pearson r | {metrics.pearson_r:.3f} |")
    if metrics.spearman_r is not None:
        md_lines.append(f"| Spearman r | {metrics.spearman_r:.3f} |")
    
    # Per-score breakdown
    md_lines.extend([
        "\n## Per-Score Accuracy",
        "\n| Score | Count | Accuracy |",
        "|-------|-------|----------|",
    ])
    
    for score in range(1, 6):
        count = metrics.per_score_count.get(score, 0)
        acc = metrics.per_score_accuracy.get(score, 0)
        md_lines.append(f"| {score} | {count} | {acc:.1f}% |")
    
    # Flag metrics
    if metrics.flag_f1:
        md_lines.extend([
            "\n## Flag Detection (F1)",
            "\n| Flag | Accuracy | Precision | Recall | F1 |",
            "|------|----------|-----------|--------|-----|",
        ])
        
        for flag in metrics.flag_f1:
            acc = metrics.flag_accuracy.get(flag, 0) * 100
            prec = metrics.flag_precision.get(flag, 0) * 100
            rec = metrics.flag_recall.get(flag, 0) * 100
            f1 = metrics.flag_f1.get(flag, 0) * 100
            flag_display = flag.replace("_", " ").title()
            md_lines.append(f"| {flag_display} | {acc:.1f}% | {prec:.1f}% | {rec:.1f}% | {f1:.1f}% |")
    
    # Adversarial results
    if metrics.adversarial_results:
        md_lines.extend([
            "\n## Robustness (Adversarial Tests)",
            "\n| Test | Passed |",
            "|------|--------|",
        ])
        
        for test_name, result in metrics.adversarial_results.items():
            status = "✅" if result.get("passed") else "❌"
            md_lines.append(f"| {test_name} | {status} |")
        
        passed = sum(1 for r in metrics.adversarial_results.values() if r.get("passed"))
        total = len(metrics.adversarial_results)
        md_lines.append(f"\n**Adversarial Pass Rate:** {passed}/{total} ({100*passed/total:.1f}%)")
    
    md_lines.extend([
        "\n---",
        "\n*Report generated by Auto-Grader evaluation pipeline*",
    ])
    
    md_content = "\n".join(md_lines)
    
    md_path = output_path / "judge_final_report.md"
    with open(md_path, "w") as f:
        f.write(md_content)
    
    return str(json_path), str(md_path)


def main():
    parser = argparse.ArgumentParser(description="Official evaluation for Judge Model")
    
    # Data
    parser.add_argument("--gold_file", "--gold", "-g",
                       type=str, default="data/gold_tests.jsonl",
                       help="Gold test set")
    parser.add_argument("--adversarial_file", "--adv", "-v",
                       type=str, default="data/adversarial.jsonl",
                       help="Adversarial test set")
    
    # Model
    parser.add_argument("--base_model", "--model", "-m",
                       type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                       help="Base model")
    parser.add_argument("--adapter_path", "--adapter", "-a",
                       type=str, default=None,
                       help="LoRA adapter path")
    
    # Output
    parser.add_argument("--output_dir", "--output", "-o",
                       type=str, default="outputs/eval_results",
                       help="Output directory")
    
    # Options
    parser.add_argument("--no_4bit", action="store_true", help="Disable 4-bit")
    parser.add_argument("--skip_adversarial", action="store_true", help="Skip adversarial tests")
    
    args = parser.parse_args()
    
    config = EvalConfig(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        use_4bit=not args.no_4bit,
    )
    
    # Check gold file
    if not Path(args.gold_file).exists():
        print(f"ERROR: Gold file not found: {args.gold_file}")
        sys.exit(1)
    
    # Load model
    model, tokenizer = load_model(config)
    
    # Evaluate gold set
    metrics, results = evaluate_gold_set(model, tokenizer, args.gold_file, config)
    
    # Evaluate adversarial
    if not args.skip_adversarial and Path(args.adversarial_file).exists():
        metrics.adversarial_results = evaluate_adversarial(
            model, tokenizer, args.adversarial_file, config
        )
    
    # Generate reports
    json_path, md_path = generate_report(metrics, results, args.output_dir, config)
    
    # Print summary
    print("\n" + "=" * 60)
    print("OFFICIAL EVALUATION COMPLETE")
    print("=" * 60)
    print(f"\nCore Metrics:")
    print(f"  JSON Validity:    {metrics.json_validity_rate:.1f}%")
    print(f"  Score Accuracy:   {metrics.score_accuracy:.1f}%")
    print(f"  Score MAE:        {metrics.score_mae:.2f}")
    print(f"  Score Within ±1:  {metrics.score_within_1_rate:.1f}%")
    
    if metrics.pearson_r is not None:
        print(f"  Pearson r:        {metrics.pearson_r:.3f}")
    
    print(f"\nReports saved to:")
    print(f"  {json_path}")
    print(f"  {md_path}")


if __name__ == "__main__":
    main()
