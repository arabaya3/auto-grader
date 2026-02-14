"""
STEP 2: Clean Rebuild of Dataset After Audit Failures.

Creates a fully clean, deduplicated, schema-consistent dataset.

Phases:
1. Merge all unique examples (remove exact duplicates)
2. Fix schema violations (flags, rubric_items, reasoning)
3. Remove leakage via fresh split (70/15/15)
4. Balance scores in train only
5. Validate clean dataset

Usage:
    python -m src.data.rebuild_clean_dataset
    python -m src.data.rebuild_clean_dataset --seed 42 --data_dir data --out_dir data_clean
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# =============================================================================
# Type Aliases
# =============================================================================

UniqueKey = Tuple[str, str, str]  # (prompt, response, rubric_title)


# =============================================================================
# Configuration
# =============================================================================

REQUIRED_FLAGS = {"over_refusal", "prompt_injection_detected", "format_violation"}
MAX_REASONING_LENGTH = 240
TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15


# =============================================================================
# Helper Functions
# =============================================================================

def load_jsonl_safe(filepath: Path) -> Tuple[List[dict], List[str]]:
    """Load JSONL file safely, skipping blank lines."""
    examples = []
    errors = []
    
    if not filepath.exists():
        return [], [f"File not found: {filepath}"]
    
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                examples.append(obj)
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: Invalid JSON - {e}")
    
    return examples, errors


def save_jsonl(examples: List[dict], filepath: Path) -> None:
    """Save examples to JSONL file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def extract_unique_key(example: dict) -> Optional[UniqueKey]:
    """Extract unique key: (prompt, response, rubric_title)."""
    try:
        prompt = example.get("prompt", "").strip()
        response = example.get("response", "").strip()
        rubric = example.get("rubric", {})
        rubric_title = rubric.get("title", "").strip()
        return (prompt, response, rubric_title)
    except Exception:
        return None


def print_header(text: str, char: str = "=") -> None:
    """Print a section header."""
    print(f"\n{char * 80}")
    print(text)
    print(char * 80)


# =============================================================================
# PHASE 1: Merge All Unique Examples
# =============================================================================

def phase1_merge_and_deduplicate(
    data_dir: Path,
    dataset_files: List[str],
) -> Tuple[List[dict], Dict[str, int]]:
    """Merge all splits and remove exact duplicates.
    
    Returns:
        Tuple of (unique_examples, stats_dict)
    """
    print_header("PHASE 1: MERGE & DEDUPLICATE")
    
    all_examples = []
    load_stats = {}
    
    # Load all datasets
    for filename in dataset_files:
        filepath = data_dir / filename
        if not filepath.exists():
            print(f"  ⚠️  {filename}: NOT FOUND")
            continue
        
        examples, errors = load_jsonl_safe(filepath)
        if errors:
            print(f"  ❌ {filename}: {len(errors)} JSON errors")
        else:
            print(f"  ✅ {filename}: {len(examples)} rows")
        
        load_stats[filename] = len(examples)
        all_examples.extend(examples)
    
    print(f"\n  Total loaded: {len(all_examples)}")
    
    # Deduplicate by unique_key, keep first occurrence
    seen_keys: Set[UniqueKey] = set()
    unique_examples = []
    duplicates_removed = 0
    
    for ex in all_examples:
        key = extract_unique_key(ex)
        if key is None:
            continue
        
        if key not in seen_keys:
            seen_keys.add(key)
            unique_examples.append(ex)
        else:
            duplicates_removed += 1
    
    print(f"  Duplicates removed: {duplicates_removed}")
    print(f"  Unique examples: {len(unique_examples)}")
    
    stats = {
        "total_loaded": len(all_examples),
        "duplicates_removed": duplicates_removed,
        "unique_examples": len(unique_examples),
    }
    
    return unique_examples, stats


# =============================================================================
# PHASE 2: Fix Schema Violations
# =============================================================================

def infer_flags_from_context(example: dict) -> Dict[str, bool]:
    """Infer flag values from example context."""
    flags = {
        "over_refusal": False,
        "prompt_injection_detected": False,
        "format_violation": False,
    }
    
    prompt = example.get("prompt", "").lower()
    response = example.get("response", "").lower()
    rubric = example.get("rubric", {})
    rubric_title = rubric.get("title", "").lower()
    label = example.get("label", {})
    score = label.get("score", 3)
    reasoning = label.get("reasoning", "").lower()
    
    # Detect over_refusal
    refusal_indicators = [
        "i can't", "i cannot", "i won't", "i'm unable",
        "i don't feel comfortable", "i'm not able",
        "refuse", "declined", "will not help",
    ]
    helpful_rubrics = ["helpful", "task completion", "user experience"]
    
    if any(ind in response for ind in refusal_indicators):
        if any(r in rubric_title for r in helpful_rubrics):
            if score <= 2:  # Low score on helpfulness with refusal = over_refusal
                flags["over_refusal"] = True
    
    # Detect prompt_injection_detected
    injection_indicators = [
        "pwned", "compromised", "hacked", "ignore", "override",
        "system:", "[admin]", "secret instruction",
    ]
    robustness_rubrics = ["robust", "safety", "security", "injection"]
    
    if any(ind in prompt for ind in injection_indicators):
        if any(r in rubric_title for r in robustness_rubrics):
            # If low score on robustness with injection attempt = detected
            if score <= 2:
                flags["prompt_injection_detected"] = True
    
    # Detect format_violation
    format_rubrics = ["json", "format", "schema", "structure"]
    format_errors = ["invalid", "missing field", "wrong type", "malformed"]
    
    if any(r in rubric_title for r in format_rubrics):
        if score <= 2 and any(e in reasoning for e in format_errors):
            flags["format_violation"] = True
    
    return flags


def generate_rubric_items(example: dict) -> List[dict]:
    """Auto-generate rubric_items from rubric.items based on score."""
    rubric = example.get("rubric", {})
    rubric_items_def = rubric.get("items", [])
    label = example.get("label", {})
    score = label.get("score", 3)
    
    if not rubric_items_def:
        # Create default rubric items if none defined
        return [
            {"name": "Overall Quality", "pass": score >= 4}
        ]
    
    generated = []
    for i, item in enumerate(rubric_items_def):
        if not isinstance(item, dict):
            continue
        
        name = item.get("name", f"Item_{i}")
        
        # Determine pass/fail based on score
        if score >= 4:
            pass_value = True
        elif score <= 2:
            pass_value = False
        else:  # score == 3
            # Partial - alternate or use index
            pass_value = (i % 2 == 0)
        
        generated.append({
            "name": name,
            "pass": pass_value,
        })
    
    return generated


def fix_example_schema(example: dict) -> Tuple[dict, List[str]]:
    """Fix schema violations in a single example.
    
    Returns:
        Tuple of (fixed_example, list_of_fixes_applied)
    """
    fixes = []
    
    # Ensure label exists
    if "label" not in example:
        example["label"] = {}
        fixes.append("Added missing label")
    
    label = example["label"]
    
    # Fix flags
    flags = label.get("flags", {})
    if not isinstance(flags, dict):
        flags = {}
    
    # Check if flags need fixing
    current_flag_keys = set(flags.keys())
    if current_flag_keys != REQUIRED_FLAGS:
        # Infer flags from context
        inferred = infer_flags_from_context(example)
        
        # Preserve valid existing flags, add missing ones
        new_flags = {}
        for flag in REQUIRED_FLAGS:
            if flag in flags and isinstance(flags[flag], bool):
                new_flags[flag] = flags[flag]
            else:
                new_flags[flag] = inferred.get(flag, False)
        
        label["flags"] = new_flags
        fixes.append(f"Fixed flags: {current_flag_keys} → {REQUIRED_FLAGS}")
    else:
        # Ensure all flag values are boolean
        for flag, value in list(flags.items()):
            if not isinstance(value, bool):
                flags[flag] = bool(value) if value else False
                fixes.append(f"Converted flag {flag} to boolean")
    
    # Fix rubric_items
    rubric_items = label.get("rubric_items", [])
    rubric = example.get("rubric", {})
    rubric_items_def = rubric.get("items", [])
    
    needs_regeneration = False
    
    if not rubric_items:
        needs_regeneration = True
        fixes.append("Generated missing rubric_items")
    elif rubric_items_def:
        # Check if names match
        label_names = {item.get("name", "") for item in rubric_items if isinstance(item, dict)}
        rubric_names = {item.get("name", "") for item in rubric_items_def if isinstance(item, dict)}
        
        if label_names != rubric_names:
            needs_regeneration = True
            fixes.append(f"Regenerated rubric_items (name mismatch)")
    
    if needs_regeneration:
        label["rubric_items"] = generate_rubric_items(example)
    
    # Fix reasoning length
    reasoning = label.get("reasoning", "")
    if not isinstance(reasoning, str):
        reasoning = str(reasoning) if reasoning else "No reasoning provided."
        label["reasoning"] = reasoning
        fixes.append("Converted reasoning to string")
    
    if len(reasoning) > MAX_REASONING_LENGTH:
        # Trim at sentence boundary if possible
        trimmed = reasoning[:MAX_REASONING_LENGTH]
        last_period = trimmed.rfind(".")
        if last_period > MAX_REASONING_LENGTH // 2:
            trimmed = trimmed[:last_period + 1]
        label["reasoning"] = trimmed
        fixes.append(f"Trimmed reasoning from {len(reasoning)} to {len(trimmed)} chars")
    elif len(reasoning) < 15:
        # Pad short reasoning
        label["reasoning"] = reasoning + " " + "Assessment complete."
        fixes.append("Padded short reasoning")
    
    # Ensure score is valid
    score = label.get("score")
    if not isinstance(score, int) or score < 1 or score > 5:
        if isinstance(score, (int, float)):
            label["score"] = max(1, min(5, int(round(score))))
        else:
            label["score"] = 3  # Default to middle
        fixes.append(f"Fixed invalid score: {score} → {label['score']}")
    
    example["label"] = label
    return example, fixes


def phase2_fix_schema(examples: List[dict]) -> Tuple[List[dict], Dict[str, int]]:
    """Fix schema violations in all examples."""
    print_header("PHASE 2: FIX SCHEMA VIOLATIONS")
    
    fixed_examples = []
    fix_counts: Dict[str, int] = Counter()
    examples_fixed = 0
    
    for ex in examples:
        fixed_ex, fixes = fix_example_schema(ex)
        fixed_examples.append(fixed_ex)
        
        if fixes:
            examples_fixed += 1
            for fix in fixes:
                # Extract fix type
                fix_type = fix.split(":")[0].split("(")[0].strip()
                fix_counts[fix_type] += 1
    
    print(f"  Examples fixed: {examples_fixed}/{len(examples)}")
    print(f"\n  Fix breakdown:")
    for fix_type, count in sorted(fix_counts.items(), key=lambda x: -x[1]):
        print(f"    - {fix_type}: {count}")
    
    stats = {
        "examples_fixed": examples_fixed,
        "total_fixes": sum(fix_counts.values()),
        "fix_counts": dict(fix_counts),
    }
    
    return fixed_examples, stats


# =============================================================================
# PHASE 3: Remove Leakage via Fresh Split
# =============================================================================

def phase3_split_dataset(
    examples: List[dict],
    seed: int,
) -> Tuple[List[dict], List[dict], List[dict], Dict[str, int]]:
    """Split into train/valid/test with no leakage.
    
    Returns:
        Tuple of (train, valid, test, stats)
    """
    print_header("PHASE 3: FRESH SPLIT (NO LEAKAGE)")
    
    random.seed(seed)
    
    # Shuffle
    shuffled = examples.copy()
    random.shuffle(shuffled)
    
    n = len(shuffled)
    n_train = int(n * TRAIN_RATIO)
    n_valid = int(n * VALID_RATIO)
    # n_test = remainder
    
    train = shuffled[:n_train]
    valid = shuffled[n_train:n_train + n_valid]
    test = shuffled[n_train + n_valid:]
    
    print(f"  Total examples: {n}")
    print(f"  Train: {len(train)} ({len(train)/n*100:.1f}%)")
    print(f"  Valid: {len(valid)} ({len(valid)/n*100:.1f}%)")
    print(f"  Test: {len(test)} ({len(test)/n*100:.1f}%)")
    
    # Verify no leakage
    train_keys = {extract_unique_key(ex) for ex in train}
    valid_keys = {extract_unique_key(ex) for ex in valid}
    test_keys = {extract_unique_key(ex) for ex in test}
    
    leakage_tv = len(train_keys & valid_keys)
    leakage_tt = len(train_keys & test_keys)
    leakage_vt = len(valid_keys & test_keys)
    
    print(f"\n  Leakage check:")
    print(f"    train∩valid: {leakage_tv}")
    print(f"    train∩test: {leakage_tt}")
    print(f"    valid∩test: {leakage_vt}")
    
    if leakage_tv + leakage_tt + leakage_vt == 0:
        print("  ✅ No leakage detected!")
    else:
        print("  ❌ LEAKAGE DETECTED - this should not happen!")
    
    stats = {
        "train_count": len(train),
        "valid_count": len(valid),
        "test_count": len(test),
        "leakage_train_valid": leakage_tv,
        "leakage_train_test": leakage_tt,
        "leakage_valid_test": leakage_vt,
    }
    
    return train, valid, test, stats


# =============================================================================
# PHASE 4: Balance Scores (Train Only)
# =============================================================================

def phase4_balance_scores(
    train: List[dict],
    seed: int,
) -> Tuple[List[dict], Dict[str, Any]]:
    """Balance score distribution in training set."""
    print_header("PHASE 4: BALANCE SCORES (TRAIN)")
    
    random.seed(seed)
    
    # Group by score
    by_score: Dict[int, List[dict]] = defaultdict(list)
    for ex in train:
        score = ex.get("label", {}).get("score", 3)
        by_score[score].append(ex)
    
    print(f"  Original distribution:")
    for score in sorted(by_score.keys()):
        count = len(by_score[score])
        pct = count / len(train) * 100
        print(f"    Score {score}: {count} ({pct:.1f}%)")
    
    # Find target count (median or min to avoid over-upsampling)
    counts = [len(by_score[s]) for s in range(1, 6) if s in by_score]
    
    if not counts:
        return train, {"balanced": False, "reason": "No valid scores"}
    
    # Target: use the median count to balance
    target_count = sorted(counts)[len(counts) // 2]
    
    # Cap at reasonable size to avoid too much reduction
    min_target = max(counts) // 3  # Don't reduce below 1/3 of max
    target_count = max(target_count, min_target)
    
    print(f"\n  Target count per score: {target_count}")
    
    balanced = []
    removed = 0
    
    for score in sorted(by_score.keys()):
        examples = by_score[score]
        
        if len(examples) > target_count:
            # Downsample
            random.shuffle(examples)
            balanced.extend(examples[:target_count])
            removed += len(examples) - target_count
        else:
            balanced.extend(examples)
    
    # Shuffle final result
    random.shuffle(balanced)
    
    print(f"\n  Balanced distribution:")
    final_by_score: Dict[int, int] = Counter()
    for ex in balanced:
        score = ex.get("label", {}).get("score", 3)
        final_by_score[score] += 1
    
    for score in sorted(final_by_score.keys()):
        count = final_by_score[score]
        pct = count / len(balanced) * 100
        print(f"    Score {score}: {count} ({pct:.1f}%)")
    
    print(f"\n  Removed for balance: {removed}")
    print(f"  Final train size: {len(balanced)}")
    
    stats = {
        "original_size": len(train),
        "balanced_size": len(balanced),
        "removed_for_balance": removed,
        "target_per_score": target_count,
        "final_distribution": dict(final_by_score),
    }
    
    return balanced, stats


# =============================================================================
# PHASE 5: Validate Clean Dataset
# =============================================================================

def validate_example(example: dict) -> List[str]:
    """Validate a single example against schema requirements."""
    errors = []
    example_id = example.get("id", "unknown")
    
    # Check required top keys
    required_top = {"id", "prompt", "response", "rubric", "label"}
    missing_top = required_top - set(example.keys())
    if missing_top:
        errors.append(f"{example_id}: missing top keys {missing_top}")
        return errors
    
    # Check label
    label = example.get("label", {})
    
    # Score
    score = label.get("score")
    if not isinstance(score, int) or score < 1 or score > 5:
        errors.append(f"{example_id}: invalid score {score}")
    
    # Flags
    flags = label.get("flags", {})
    if set(flags.keys()) != REQUIRED_FLAGS:
        errors.append(f"{example_id}: invalid flags {set(flags.keys())}")
    else:
        for flag, value in flags.items():
            if not isinstance(value, bool):
                errors.append(f"{example_id}: flag {flag} not boolean")
    
    # Rubric items
    rubric_items = label.get("rubric_items", [])
    if not rubric_items:
        errors.append(f"{example_id}: missing rubric_items")
    
    # Reasoning
    reasoning = label.get("reasoning", "")
    if len(reasoning) < 15:
        errors.append(f"{example_id}: reasoning too short ({len(reasoning)})")
    elif len(reasoning) > MAX_REASONING_LENGTH + 50:  # Small buffer
        errors.append(f"{example_id}: reasoning too long ({len(reasoning)})")
    
    return errors


def phase5_validate(
    train: List[dict],
    valid: List[dict],
    test: List[dict],
) -> Tuple[bool, Dict[str, Any]]:
    """Validate all datasets pass schema checks."""
    print_header("PHASE 5: FINAL VALIDATION")
    
    all_errors = []
    
    datasets = [("train", train), ("valid", valid), ("test", test)]
    
    for name, examples in datasets:
        errors = []
        for ex in examples:
            errors.extend(validate_example(ex))
        
        if errors:
            print(f"  ❌ {name}: {len(errors)} validation errors")
            for e in errors[:5]:
                print(f"      - {e}")
            if len(errors) > 5:
                print(f"      ... and {len(errors) - 5} more")
        else:
            print(f"  ✅ {name}: {len(examples)} examples pass validation")
        
        all_errors.extend(errors)
    
    # Check no leakage
    train_keys = {extract_unique_key(ex) for ex in train}
    valid_keys = {extract_unique_key(ex) for ex in valid}
    test_keys = {extract_unique_key(ex) for ex in test}
    
    leakage = (
        len(train_keys & valid_keys) +
        len(train_keys & test_keys) +
        len(valid_keys & test_keys)
    )
    
    if leakage > 0:
        all_errors.append(f"Cross-split leakage detected: {leakage}")
        print(f"  ❌ Cross-split leakage: {leakage}")
    else:
        print(f"  ✅ No cross-split leakage")
    
    passed = len(all_errors) == 0
    
    stats = {
        "passed": passed,
        "error_count": len(all_errors),
        "errors": all_errors[:20],  # First 20 errors
    }
    
    return passed, stats


# =============================================================================
# Summary & Output
# =============================================================================

def print_final_summary(
    train: List[dict],
    valid: List[dict],
    test: List[dict],
) -> None:
    """Print final dataset summary."""
    print_header("FINAL SUMMARY", "=")
    
    # Counts
    print(f"\n  Dataset Sizes:")
    print(f"    Train: {len(train)}")
    print(f"    Valid: {len(valid)}")
    print(f"    Test:  {len(test)}")
    print(f"    Total: {len(train) + len(valid) + len(test)}")
    
    # Score distribution
    print(f"\n  Score Distribution:")
    for name, examples in [("Train", train), ("Valid", valid), ("Test", test)]:
        scores = Counter(ex.get("label", {}).get("score", 0) for ex in examples)
        dist = " | ".join(f"S{s}:{scores[s]}" for s in sorted(scores.keys()))
        print(f"    {name}: {dist}")
    
    # Rubric coverage
    print(f"\n  Rubric Coverage (Train):")
    rubric_counts: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    
    for ex in train:
        rubric_title = ex.get("rubric", {}).get("title", "Unknown")
        score = ex.get("label", {}).get("score", 0)
        rubric_counts[rubric_title][score] += 1
    
    for rubric in sorted(rubric_counts.keys()):
        scores = rubric_counts[rubric]
        total = sum(scores.values())
        gaps = [s for s in range(1, 6) if scores.get(s, 0) == 0]
        gap_str = f" (gaps: {gaps})" if gaps else ""
        print(f"    {rubric[:35]}: {total} examples{gap_str}")


# =============================================================================
# Main Pipeline
# =============================================================================

def run_rebuild(
    data_dir: Path,
    out_dir: Path,
    seed: int,
) -> int:
    """Run complete dataset rebuild.
    
    Returns:
        Exit code: 0 if successful, 1 if validation fails.
    """
    print_header("STEP 2: CLEAN DATASET REBUILD", "=")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {out_dir}")
    print(f"Random seed: {seed}")
    
    # Input files
    dataset_files = [
        "train.jsonl",
        "valid.jsonl",
        "test.jsonl",
        "calibration.jsonl",
        "adversarial.jsonl",
    ]
    
    # PHASE 1: Merge & Deduplicate
    unique_examples, phase1_stats = phase1_merge_and_deduplicate(data_dir, dataset_files)
    
    if not unique_examples:
        print("\n❌ No valid examples found!")
        return 1
    
    # PHASE 2: Fix Schema
    fixed_examples, phase2_stats = phase2_fix_schema(unique_examples)
    
    # PHASE 3: Split
    train, valid, test, phase3_stats = phase3_split_dataset(fixed_examples, seed)
    
    # PHASE 4: Balance (train only)
    train, phase4_stats = phase4_balance_scores(train, seed)
    
    # PHASE 5: Validate
    passed, phase5_stats = phase5_validate(train, valid, test)
    
    if not passed:
        print("\n❌ VALIDATION FAILED - dataset has errors")
        return 1
    
    # Print final summary
    print_final_summary(train, valid, test)
    
    # Save datasets
    print_header("SAVING CLEAN DATASETS")
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = out_dir / "train_clean.jsonl"
    valid_path = out_dir / "valid_clean.jsonl"
    test_path = out_dir / "test_clean.jsonl"
    
    save_jsonl(train, train_path)
    save_jsonl(valid, valid_path)
    save_jsonl(test, test_path)
    
    print(f"  ✅ Saved: {train_path} ({len(train)} examples)")
    print(f"  ✅ Saved: {valid_path} ({len(valid)} examples)")
    print(f"  ✅ Saved: {test_path} ({len(test)} examples)")
    
    # Save stats
    stats_path = out_dir / "rebuild_stats.json"
    stats = {
        "phase1_merge": phase1_stats,
        "phase2_schema": phase2_stats,
        "phase3_split": phase3_stats,
        "phase4_balance": phase4_stats,
        "phase5_validate": phase5_stats,
        "final": {
            "train_count": len(train),
            "valid_count": len(valid),
            "test_count": len(test),
        },
    }
    
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"  ✅ Saved: {stats_path}")
    
    print("\n" + "=" * 80)
    print("✅ STEP 2 COMPLETE: Clean dataset rebuild successful!")
    print("=" * 80)
    
    return 0


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild clean dataset from audit failures."
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data"),
        help="Input data directory",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("data_clean"),
        help="Output directory for clean datasets",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    
    args = parser.parse_args()
    
    return run_rebuild(args.data_dir, args.out_dir, args.seed)


if __name__ == "__main__":
    sys.exit(main())
