"""
Rebalance training dataset for better flag and extreme score performance.

This script addresses the following issues:
1. Flag examples are underrepresented (~5% positive cases)
2. Gold tests only use extreme scores (1 and 5)
3. Robustness rubric examples may be underrepresented

Usage:
    python -m src.data.rebalance_dataset --input data/train.jsonl --output data/train_rebalanced.jsonl
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load examples from JSONL file."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def write_jsonl(examples: list[dict[str, Any]], path: Path) -> None:
    """Write examples to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Wrote {len(examples)} examples to {path}")


def analyze_dataset(examples: list[dict[str, Any]], name: str = "Dataset") -> dict[str, Any]:
    """Analyze dataset distribution and print statistics."""
    print(f"\n{'=' * 60}")
    print(f"{name} Analysis ({len(examples)} examples)")
    print("=" * 60)
    
    # Score distribution
    scores = [ex["label"]["score"] for ex in examples]
    score_counts = Counter(scores)
    print("\nScore Distribution:")
    for score in sorted(score_counts.keys()):
        count = score_counts[score]
        pct = count / len(examples) * 100
        bar = "█" * int(pct / 2)
        print(f"  Score {score}: {count:3d} ({pct:5.1f}%) {bar}")
    
    # Flag distribution
    flag_counts = {
        "over_refusal": {"true": 0, "false": 0},
        "prompt_injection_detected": {"true": 0, "false": 0},
        "format_violation": {"true": 0, "false": 0},
    }
    for ex in examples:
        flags = ex["label"].get("flags", {})
        for flag_name in flag_counts:
            if flags.get(flag_name, False):
                flag_counts[flag_name]["true"] += 1
            else:
                flag_counts[flag_name]["false"] += 1
    
    print("\nFlag Distribution:")
    for flag_name, counts in flag_counts.items():
        total = counts["true"] + counts["false"]
        pct = counts["true"] / total * 100 if total > 0 else 0
        print(f"  {flag_name:30}: {counts['true']:3d} true ({pct:5.1f}%)")
    
    # Rubric distribution
    rubric_counts = Counter(ex["rubric"]["title"] for ex in examples)
    print("\nRubric Distribution:")
    for rubric, count in rubric_counts.most_common():
        pct = count / len(examples) * 100
        print(f"  {rubric:25}: {count:3d} ({pct:5.1f}%)")
    
    return {
        "scores": dict(score_counts),
        "flags": flag_counts,
        "rubrics": dict(rubric_counts),
    }


def get_flag_examples(examples: list[dict[str, Any]], flag_name: str) -> list[dict[str, Any]]:
    """Get all examples where a specific flag is True."""
    return [
        ex for ex in examples 
        if ex["label"].get("flags", {}).get(flag_name, False)
    ]


def get_extreme_score_examples(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Get examples with score 1 or 5."""
    return [ex for ex in examples if ex["label"]["score"] in (1, 5)]


def get_rubric_examples(examples: list[dict[str, Any]], rubric_title: str) -> list[dict[str, Any]]:
    """Get examples with a specific rubric type."""
    return [ex for ex in examples if ex["rubric"]["title"] == rubric_title]


def create_variant_id(original_id: str, variant_num: int) -> str:
    """Create a unique ID for an oversampled variant."""
    # Remove any existing variant suffix
    base_id = original_id.split("_dup")[0].split("_var")[0]
    return f"{base_id}_var{variant_num}"


def get_score_examples(examples: list[dict[str, Any]], score: int) -> list[dict[str, Any]]:
    """Get examples with a specific score."""
    return [ex for ex in examples if ex["label"]["score"] == score]


def rebalance_dataset(
    examples: list[dict[str, Any]],
    flag_oversample_factor: int = 3,
    extreme_score_boost: float = 1.5,
    robustness_boost: float = 1.5,
    balance_all_scores: bool = True,
    target_score_pct: float = 0.15,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Rebalance dataset by oversampling underrepresented patterns.
    
    Args:
        examples: Original training examples
        flag_oversample_factor: How many times to duplicate flag=True examples
        extreme_score_boost: Multiplier for score 1 and 5 examples
        robustness_boost: Multiplier for Robustness rubric examples
        balance_all_scores: Whether to ensure all scores have minimum representation
        target_score_pct: Target minimum percentage for each score (default: 15%)
        seed: Random seed
        
    Returns:
        Rebalanced list of examples
    """
    random.seed(seed)
    
    rebalanced = list(examples)  # Start with all original examples
    variant_counter = {}  # Track variants per original ID
    
    def add_variants(source_examples: list[dict[str, Any]], num_copies: int, reason: str):
        """Add oversampled variants of examples."""
        nonlocal rebalanced
        added = 0
        for ex in source_examples:
            original_id = ex["id"]
            for _ in range(num_copies):
                variant_counter[original_id] = variant_counter.get(original_id, 0) + 1
                new_ex = ex.copy()
                new_ex["id"] = create_variant_id(original_id, variant_counter[original_id])
                rebalanced.append(new_ex)
                added += 1
        print(f"  Added {added} examples ({reason})")
    
    def add_sampled_variants(source_examples: list[dict[str, Any]], num_to_add: int, reason: str):
        """Add a specific number of randomly sampled variants."""
        nonlocal rebalanced
        if num_to_add <= 0 or not source_examples:
            return
        sampled = random.choices(source_examples, k=num_to_add)
        for ex in sampled:
            original_id = ex["id"]
            variant_counter[original_id] = variant_counter.get(original_id, 0) + 1
            new_ex = ex.copy()
            new_ex["id"] = create_variant_id(original_id, variant_counter[original_id])
            rebalanced.append(new_ex)
        print(f"  Added {num_to_add} examples ({reason})")
    
    print("\nRebalancing dataset...")
    
    # 1. Oversample flag=True examples
    for flag_name in ["over_refusal", "prompt_injection_detected", "format_violation"]:
        flag_examples = get_flag_examples(examples, flag_name)
        if flag_examples:
            add_variants(flag_examples, flag_oversample_factor - 1, f"{flag_name}=True × {flag_oversample_factor}")
    
    # 2. Boost extreme score examples (scores 1 and 5)
    extreme_examples = get_extreme_score_examples(examples)
    extra_copies = int(len(extreme_examples) * (extreme_score_boost - 1))
    if extra_copies > 0:
        sampled = random.choices(extreme_examples, k=extra_copies)
        for ex in sampled:
            original_id = ex["id"]
            variant_counter[original_id] = variant_counter.get(original_id, 0) + 1
            new_ex = ex.copy()
            new_ex["id"] = create_variant_id(original_id, variant_counter[original_id])
            rebalanced.append(new_ex)
        print(f"  Added {extra_copies} examples (extreme scores 1 & 5 × {extreme_score_boost})")
    
    # 3. Boost Robustness rubric examples (important for prompt injection tests)
    robustness_examples = get_rubric_examples(examples, "Robustness")
    extra_copies = int(len(robustness_examples) * (robustness_boost - 1))
    if extra_copies > 0:
        sampled = random.choices(robustness_examples, k=extra_copies)
        for ex in sampled:
            original_id = ex["id"]
            variant_counter[original_id] = variant_counter.get(original_id, 0) + 1
            new_ex = ex.copy()
            new_ex["id"] = create_variant_id(original_id, variant_counter[original_id])
            rebalanced.append(new_ex)
        print(f"  Added {extra_copies} examples (Robustness rubric × {robustness_boost})")
    
    # 4. Balance all scores to ensure minimum representation for middle scores
    if balance_all_scores:
        # Calculate current distribution
        from collections import Counter
        current_scores = Counter(ex["label"]["score"] for ex in rebalanced)
        total = len(rebalanced)
        target_count = int(total * target_score_pct)
        
        print(f"  Balancing scores to minimum {target_score_pct*100:.0f}% ({target_count} examples)...")
        
        for score in [2, 3, 4]:  # Middle scores that may be underrepresented
            current_count = current_scores.get(score, 0)
            if current_count < target_count:
                needed = target_count - current_count
                score_examples = get_score_examples(examples, score)
                if score_examples:
                    add_sampled_variants(score_examples, needed, f"score={score} balance")
    
    # Shuffle the final dataset
    random.shuffle(rebalanced)
    
    return rebalanced


def main():
    parser = argparse.ArgumentParser(description="Rebalance training dataset")
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="data/train.jsonl",
        help="Input training JSONL file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/train_rebalanced.jsonl",
        help="Output rebalanced JSONL file"
    )
    parser.add_argument(
        "--flag-factor",
        type=int,
        default=4,
        help="Oversample factor for flag=True examples (default: 4)"
    )
    parser.add_argument(
        "--extreme-boost",
        type=float,
        default=1.5,
        help="Boost factor for extreme scores 1 and 5 (default: 1.5)"
    )
    parser.add_argument(
        "--robustness-boost",
        type=float,
        default=2.0,
        help="Boost factor for Robustness rubric examples (default: 2.0)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace original train.jsonl instead of creating new file"
    )
    parser.add_argument(
        "--balance-scores",
        action="store_true",
        default=True,
        help="Balance all scores to ensure min 15%% representation (default: True)"
    )
    parser.add_argument(
        "--no-balance-scores",
        dest="balance_scores",
        action="store_false",
        help="Disable score balancing"
    )
    parser.add_argument(
        "--target-score-pct",
        type=float,
        default=0.15,
        help="Target minimum percentage for each score (default: 0.15)"
    )
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output) if not args.replace else input_path
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return
    
    # Load original dataset
    print(f"Loading dataset from: {input_path}")
    examples = load_jsonl(input_path)
    
    # Analyze original
    analyze_dataset(examples, "Original Dataset")
    
    # Rebalance
    rebalanced = rebalance_dataset(
        examples,
        flag_oversample_factor=args.flag_factor,
        extreme_score_boost=args.extreme_boost,
        robustness_boost=args.robustness_boost,
        balance_all_scores=args.balance_scores,
        target_score_pct=args.target_score_pct,
        seed=args.seed,
    )
    
    # Analyze rebalanced
    analyze_dataset(rebalanced, "Rebalanced Dataset")
    
    # Write output
    write_jsonl(rebalanced, output_path)
    
    print(f"\nDone! Increased from {len(examples)} to {len(rebalanced)} examples")
    print(f"  (+{len(rebalanced) - len(examples)} examples, {(len(rebalanced)/len(examples) - 1)*100:.1f}% increase)")
    
    if args.replace:
        print(f"\nOriginal file replaced: {output_path}")
    else:
        print(f"\nTo use rebalanced data, update notebook to use: {output_path}")
        print("Or run with --replace to overwrite original train.jsonl")


if __name__ == "__main__":
    main()
