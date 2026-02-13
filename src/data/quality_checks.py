"""
Quality checks and validation for Auto-Grader training dataset.

Validates:
- Score distribution per split
- Rubric items match rubric definition
- Score in [1..5]
- Reasoning length 15-240 characters
- Flag presence and consistency

Usage:
    python -m src.data.quality_checks --data-dir data/
"""

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ValidationResult:
    """Result of a validation check."""
    passed: bool
    message: str
    details: list[str] | None = None


@dataclass
class DatasetStats:
    """Statistics for a dataset split."""
    total: int
    score_distribution: dict[int, int]
    rubric_distribution: dict[str, int]
    flag_counts: dict[str, int]
    reasoning_length_min: int
    reasoning_length_max: int
    reasoning_length_avg: float
    errors: list[str]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load examples from JSONL file."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON on line {line_num}: {e}")
    return examples


def validate_example(example: dict[str, Any], idx: int) -> list[str]:
    """Validate a single example and return list of errors."""
    errors = []
    
    # Check required fields
    required = ["id", "prompt", "response", "rubric", "label"]
    for field in required:
        if field not in example:
            errors.append(f"Example {idx}: Missing required field '{field}'")
    
    if errors:
        return errors  # Can't continue validation without required fields
    
    # Validate label
    label = example.get("label", {})
    
    # Score validation
    score = label.get("score")
    if not isinstance(score, int) or score < 1 or score > 5:
        errors.append(f"Example {idx} ({example['id']}): Invalid score {score}, must be 1-5")
    
    # Reasoning validation
    reasoning = label.get("reasoning", "")
    if not isinstance(reasoning, str):
        errors.append(f"Example {idx} ({example['id']}): Reasoning must be a string")
    elif len(reasoning) < 15:
        errors.append(f"Example {idx} ({example['id']}): Reasoning too short ({len(reasoning)} chars, min 15)")
    elif len(reasoning) > 240:
        errors.append(f"Example {idx} ({example['id']}): Reasoning too long ({len(reasoning)} chars, max 240)")
    
    # Rubric items validation
    rubric = example.get("rubric", {})
    rubric_items = rubric.get("items", [])
    label_items = label.get("rubric_items", [])
    
    rubric_item_names = {item.get("name") for item in rubric_items}
    label_item_names = {item.get("name") for item in label_items}
    
    # Check that label items match rubric items
    missing_in_label = rubric_item_names - label_item_names
    extra_in_label = label_item_names - rubric_item_names
    
    if missing_in_label:
        errors.append(f"Example {idx} ({example['id']}): Missing rubric items in label: {missing_in_label}")
    if extra_in_label:
        errors.append(f"Example {idx} ({example['id']}): Extra items in label not in rubric: {extra_in_label}")
    
    # Validate flags
    flags = label.get("flags", {})
    expected_flags = ["over_refusal", "prompt_injection_detected", "format_violation"]
    for flag in expected_flags:
        if flag in flags and not isinstance(flags[flag], bool):
            errors.append(f"Example {idx} ({example['id']}): Flag '{flag}' must be boolean")
    
    return errors


def compute_stats(examples: list[dict[str, Any]]) -> DatasetStats:
    """Compute statistics for a dataset split."""
    scores = []
    rubrics = []
    flags: dict[str, int] = {"over_refusal": 0, "prompt_injection_detected": 0, "format_violation": 0}
    reasoning_lengths = []
    errors = []
    
    for idx, ex in enumerate(examples):
        # Validate
        ex_errors = validate_example(ex, idx)
        errors.extend(ex_errors)
        
        # Collect stats
        label = ex.get("label", {})
        score = label.get("score")
        if isinstance(score, int):
            scores.append(score)
        
        rubric = ex.get("rubric", {})
        rubric_title = rubric.get("title", "Unknown")
        rubrics.append(rubric_title)
        
        ex_flags = label.get("flags", {})
        for flag_name in flags:
            if ex_flags.get(flag_name, False):
                flags[flag_name] += 1
        
        reasoning = label.get("reasoning", "")
        if isinstance(reasoning, str):
            reasoning_lengths.append(len(reasoning))
    
    return DatasetStats(
        total=len(examples),
        score_distribution=dict(Counter(scores)),
        rubric_distribution=dict(Counter(rubrics)),
        flag_counts=flags,
        reasoning_length_min=min(reasoning_lengths) if reasoning_lengths else 0,
        reasoning_length_max=max(reasoning_lengths) if reasoning_lengths else 0,
        reasoning_length_avg=sum(reasoning_lengths) / len(reasoning_lengths) if reasoning_lengths else 0,
        errors=errors,
    )


def check_balance(stats: DatasetStats, tolerance: float = 0.3) -> ValidationResult:
    """Check if scores are reasonably balanced."""
    if not stats.score_distribution:
        return ValidationResult(False, "No scores found")
    
    counts = list(stats.score_distribution.values())
    avg = sum(counts) / len(counts)
    
    imbalanced = []
    for score, count in stats.score_distribution.items():
        deviation = abs(count - avg) / avg if avg > 0 else 0
        if deviation > tolerance:
            imbalanced.append(f"Score {score}: {count} ({deviation:.1%} deviation)")
    
    if imbalanced:
        return ValidationResult(
            False,
            f"Score distribution imbalanced (>{tolerance:.0%} tolerance)",
            imbalanced
        )
    
    return ValidationResult(True, "Score distribution is balanced")


def print_stats(name: str, stats: DatasetStats) -> None:
    """Print statistics for a split."""
    print(f"\n{'='*60}")
    print(f" {name.upper()} SPLIT")
    print(f"{'='*60}")
    print(f"Total examples: {stats.total}")
    
    print(f"\nScore Distribution:")
    for score in sorted(stats.score_distribution.keys()):
        count = stats.score_distribution[score]
        pct = count / stats.total * 100 if stats.total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  Score {score}: {count:4d} ({pct:5.1f}%) {bar}")
    
    print(f"\nRubric Distribution:")
    for rubric, count in sorted(stats.rubric_distribution.items(), key=lambda x: -x[1]):
        pct = count / stats.total * 100 if stats.total > 0 else 0
        print(f"  {rubric:25s}: {count:4d} ({pct:5.1f}%)")
    
    print(f"\nFlags:")
    for flag, count in stats.flag_counts.items():
        print(f"  {flag:30s}: {count:4d}")
    
    print(f"\nReasoning Length:")
    print(f"  Min: {stats.reasoning_length_min} chars")
    print(f"  Max: {stats.reasoning_length_max} chars")
    print(f"  Avg: {stats.reasoning_length_avg:.1f} chars")
    
    if stats.errors:
        print(f"\n⚠️  Validation Errors ({len(stats.errors)}):")
        for error in stats.errors[:10]:  # Show first 10
            print(f"  - {error}")
        if len(stats.errors) > 10:
            print(f"  ... and {len(stats.errors) - 10} more")


def validate_gold_tests(path: Path) -> list[str]:
    """Validate gold test set has required cases."""
    errors = []
    
    if not path.exists():
        errors.append(f"Gold test file not found: {path}")
        return errors
    
    examples = load_jsonl(path)
    
    if len(examples) != 10:
        errors.append(f"Gold tests should have exactly 10 examples, found {len(examples)}")
    
    # Check for required adversarial cases
    over_refusal_count = 0
    injection_count = 0
    
    for ex in examples:
        flags = ex.get("label", {}).get("flags", {})
        if flags.get("over_refusal", False):
            over_refusal_count += 1
        if flags.get("prompt_injection_detected", False):
            injection_count += 1
    
    if over_refusal_count < 3:
        errors.append(f"Need at least 3 over-refusal traps, found {over_refusal_count}")
    
    if injection_count < 2:
        errors.append(f"Need at least 2 jailbreak/injection attempts, found {injection_count}")
    
    return errors


def main():
    parser = argparse.ArgumentParser(description="Validate Auto-Grader dataset")
    parser.add_argument("--data-dir", "--data_dir", dest="data_dir", type=str, default="data", help="Directory containing JSONL files")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    all_passed = True
    total_errors = []
    
    # Check each split
    splits = ["train", "valid", "test"]
    for split in splits:
        path = data_dir / f"{split}.jsonl"
        if not path.exists():
            print(f"Warning: {path} not found")
            all_passed = False
            continue
        
        examples = load_jsonl(path)
        stats = compute_stats(examples)
        print_stats(split, stats)
        
        # Check balance
        balance_result = check_balance(stats)
        if not balance_result.passed:
            print(f"\n⚠️  {balance_result.message}")
            if balance_result.details:
                for detail in balance_result.details:
                    print(f"    - {detail}")
        else:
            print(f"\n✅ {balance_result.message}")
        
        total_errors.extend(stats.errors)
        if stats.errors:
            all_passed = False
    
    # Check gold tests
    gold_path = data_dir / "gold_tests.jsonl"
    print(f"\n{'='*60}")
    print(" GOLD TESTS VALIDATION")
    print(f"{'='*60}")
    
    gold_errors = validate_gold_tests(gold_path)
    if gold_errors:
        print("⚠️  Gold test validation errors:")
        for error in gold_errors:
            print(f"  - {error}")
        all_passed = False
    else:
        print("✅ Gold tests valid (10 examples, 3+ over-refusal, 2+ injection)")
    
    # Summary
    print(f"\n{'='*60}")
    print(" SUMMARY")
    print(f"{'='*60}")
    
    if all_passed and not total_errors:
        print("✅ All quality checks passed!")
        return 0
    else:
        print(f"⚠️  Found {len(total_errors)} validation errors")
        print("Please fix the errors and re-run quality checks.")
        return 1


if __name__ == "__main__":
    exit(main())
