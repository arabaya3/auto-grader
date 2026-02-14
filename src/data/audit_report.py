"""
Production-Grade Data Audit Report for Judge Dataset.

Performs comprehensive validation:
- Exact duplicate detection
- Cross-split leakage detection
- Score distribution analysis
- Rubric coverage matrix
- Schema sanity checks

Usage:
    python -m src.data.audit_report --data_dir data --out outputs
    python -m src.data.audit_report --data_dir data_clean --out outputs

Exit code 1 if any hard schema check fails, else 0.
"""

import argparse
import json
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
# Data Structures
# =============================================================================

@dataclass
class SchemaFailure:
    """A single schema validation failure."""
    rule: str
    example_id: str
    message: str


@dataclass
class DuplicateGroup:
    """A group of duplicate examples."""
    key: UniqueKey
    ids: List[str]
    count: int


@dataclass
class SplitReport:
    """Audit report for a single dataset split."""
    name: str
    total_rows: int = 0
    unique_rows: int = 0
    exact_duplicates_removed: int = 0
    duplicate_groups: List[DuplicateGroup] = field(default_factory=list)
    score_distribution: Dict[int, int] = field(default_factory=dict)
    unique_keys: Set[UniqueKey] = field(default_factory=set)
    schema_failures: List[SchemaFailure] = field(default_factory=list)


@dataclass
class AuditReport:
    """Complete audit report across all splits."""
    splits: Dict[str, SplitReport] = field(default_factory=dict)
    cross_leakage: Dict[str, List[UniqueKey]] = field(default_factory=dict)
    rubric_coverage: Dict[str, Dict[int, int]] = field(default_factory=dict)
    hard_check_failed: bool = False


# =============================================================================
# Helper Functions
# =============================================================================

def load_jsonl_safe(filepath: Path) -> Tuple[List[dict], List[str]]:
    """Load JSONL file safely, skipping blank lines.
    
    Returns:
        Tuple of (valid_examples, error_messages)
    """
    examples = []
    errors = []
    
    if not filepath.exists():
        return [], [f"File not found: {filepath}"]
    
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue  # Skip blank lines
            
            try:
                obj = json.loads(line)
                examples.append(obj)
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: Invalid JSON - {e}")
    
    return examples, errors


def extract_unique_key(example: dict) -> Optional[UniqueKey]:
    """Extract unique key from example: (prompt, response, rubric_title)."""
    try:
        prompt = example.get("prompt", "").strip()
        response = example.get("response", "").strip()
        rubric = example.get("rubric", {})
        rubric_title = rubric.get("title", "").strip()
        return (prompt, response, rubric_title)
    except Exception:
        return None


def truncate_text(text: str, max_len: int = 80) -> str:
    """Truncate text for display."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def format_key_for_display(key: UniqueKey) -> str:
    """Format unique key for readable display."""
    prompt, response, rubric_title = key
    return (
        f"prompt={truncate_text(prompt, 60)!r}, "
        f"response={truncate_text(response, 60)!r}, "
        f"rubric={rubric_title!r}"
    )


# =============================================================================
# Schema Validation
# =============================================================================

REQUIRED_TOP_KEYS = {"id", "prompt", "response", "rubric", "label"}
REQUIRED_RUBRIC_KEYS = {"title", "items", "scoring_guide"}
REQUIRED_LABEL_KEYS = {"score", "reasoning", "rubric_items", "flags"}
REQUIRED_FLAGS = {"over_refusal", "prompt_injection_detected", "format_violation"}


def validate_schema(example: dict, example_id: str) -> List[SchemaFailure]:
    """Validate example against required schema."""
    failures = []
    
    # Check top-level keys
    missing_top = REQUIRED_TOP_KEYS - set(example.keys())
    if missing_top:
        failures.append(SchemaFailure(
            rule="missing_top_keys",
            example_id=example_id,
            message=f"Missing keys: {missing_top}",
        ))
        return failures  # Can't validate further
    
    # Check rubric keys
    rubric = example.get("rubric", {})
    if not isinstance(rubric, dict):
        failures.append(SchemaFailure(
            rule="invalid_rubric",
            example_id=example_id,
            message="rubric is not a dict",
        ))
    else:
        missing_rubric = REQUIRED_RUBRIC_KEYS - set(rubric.keys())
        if missing_rubric:
            failures.append(SchemaFailure(
                rule="missing_rubric_keys",
                example_id=example_id,
                message=f"Missing rubric keys: {missing_rubric}",
            ))
    
    # Check label keys
    label = example.get("label", {})
    if not isinstance(label, dict):
        failures.append(SchemaFailure(
            rule="invalid_label",
            example_id=example_id,
            message="label is not a dict",
        ))
        return failures
    
    missing_label = REQUIRED_LABEL_KEYS - set(label.keys())
    if missing_label:
        failures.append(SchemaFailure(
            rule="missing_label_keys",
            example_id=example_id,
            message=f"Missing label keys: {missing_label}",
        ))
    
    # Check score in [1..5]
    score = label.get("score")
    if score is None:
        failures.append(SchemaFailure(
            rule="invalid_score",
            example_id=example_id,
            message="score is missing",
        ))
    elif not isinstance(score, int) or score < 1 or score > 5:
        failures.append(SchemaFailure(
            rule="invalid_score",
            example_id=example_id,
            message=f"score={score} not in [1..5]",
        ))
    
    # Check flags
    flags = label.get("flags", {})
    if not isinstance(flags, dict):
        failures.append(SchemaFailure(
            rule="invalid_flags",
            example_id=example_id,
            message="flags is not a dict",
        ))
    else:
        flag_keys = set(flags.keys())
        if flag_keys != REQUIRED_FLAGS:
            missing = REQUIRED_FLAGS - flag_keys
            extra = flag_keys - REQUIRED_FLAGS
            msg_parts = []
            if missing:
                msg_parts.append(f"missing={missing}")
            if extra:
                msg_parts.append(f"extra={extra}")
            failures.append(SchemaFailure(
                rule="invalid_flag_keys",
                example_id=example_id,
                message=", ".join(msg_parts),
            ))
        
        # Check flag values are boolean
        for flag_name, flag_value in flags.items():
            if not isinstance(flag_value, bool):
                failures.append(SchemaFailure(
                    rule="invalid_flag_type",
                    example_id=example_id,
                    message=f"flag {flag_name}={flag_value!r} is not boolean",
                ))
    
    # Check rubric_items match rubric.items
    rubric_items = label.get("rubric_items", [])
    rubric_def_items = rubric.get("items", []) if isinstance(rubric, dict) else []
    
    if rubric_items and rubric_def_items:
        label_item_names = {item.get("name", "") for item in rubric_items if isinstance(item, dict)}
        rubric_item_names = {item.get("name", "") for item in rubric_def_items if isinstance(item, dict)}
        
        if label_item_names != rubric_item_names:
            missing = rubric_item_names - label_item_names
            extra = label_item_names - rubric_item_names
            msg_parts = []
            if missing:
                msg_parts.append(f"missing={missing}")
            if extra:
                msg_parts.append(f"extra={extra}")
            failures.append(SchemaFailure(
                rule="rubric_items_mismatch",
                example_id=example_id,
                message=", ".join(msg_parts),
            ))
    
    # Check reasoning length
    reasoning = label.get("reasoning", "")
    if not isinstance(reasoning, str):
        failures.append(SchemaFailure(
            rule="invalid_reasoning",
            example_id=example_id,
            message="reasoning is not a string",
        ))
    elif len(reasoning) < 15:
        failures.append(SchemaFailure(
            rule="reasoning_too_short",
            example_id=example_id,
            message=f"reasoning length={len(reasoning)} < 15",
        ))
    elif len(reasoning) > 2400:
        failures.append(SchemaFailure(
            rule="reasoning_too_long",
            example_id=example_id,
            message=f"reasoning length={len(reasoning)} > 2400",
        ))
    
    return failures


# =============================================================================
# Split Analysis
# =============================================================================

def analyze_split(name: str, examples: List[dict]) -> SplitReport:
    """Analyze a single dataset split."""
    report = SplitReport(name=name, total_rows=len(examples))
    
    # Track unique keys and duplicates
    key_to_ids: Dict[UniqueKey, List[str]] = defaultdict(list)
    
    for ex in examples:
        example_id = ex.get("id", f"unknown_{id(ex)}")
        
        # Schema validation
        failures = validate_schema(ex, example_id)
        report.schema_failures.extend(failures)
        
        # Unique key tracking
        key = extract_unique_key(ex)
        if key:
            key_to_ids[key].append(example_id)
            report.unique_keys.add(key)
        
        # Score distribution
        label = ex.get("label", {})
        score = label.get("score")
        if isinstance(score, int) and 1 <= score <= 5:
            report.score_distribution[score] = report.score_distribution.get(score, 0) + 1
    
    # Calculate duplicates
    report.unique_rows = len(key_to_ids)
    report.exact_duplicates_removed = report.total_rows - report.unique_rows
    
    # Find duplicate groups (sorted by count descending)
    dup_groups = [
        DuplicateGroup(key=key, ids=ids, count=len(ids))
        for key, ids in key_to_ids.items()
        if len(ids) > 1
    ]
    dup_groups.sort(key=lambda g: -g.count)
    report.duplicate_groups = dup_groups[:10]  # Top 10
    
    return report


# =============================================================================
# Cross-Split Leakage
# =============================================================================

def detect_cross_leakage(
    splits: Dict[str, SplitReport]
) -> Dict[str, List[UniqueKey]]:
    """Detect cross-split leakage."""
    leakage = {}
    
    pairs = [
        ("train", "valid"),
        ("train", "test"),
        ("valid", "test"),
    ]
    
    for split_a, split_b in pairs:
        if split_a not in splits or split_b not in splits:
            continue
        
        keys_a = splits[split_a].unique_keys
        keys_b = splits[split_b].unique_keys
        
        intersection = keys_a & keys_b
        key = f"{split_a}∩{split_b}"
        leakage[key] = list(intersection)[:10]  # Sample 10
    
    return leakage


# =============================================================================
# Rubric Coverage
# =============================================================================

def compute_rubric_coverage(
    splits: Dict[str, SplitReport],
    all_examples: List[dict],
) -> Dict[str, Dict[int, int]]:
    """Compute rubric coverage matrix."""
    coverage: Dict[str, Dict[int, int]] = defaultdict(lambda: {1: 0, 2: 0, 3: 0, 4: 0, 5: 0})
    
    for ex in all_examples:
        rubric = ex.get("rubric", {})
        rubric_title = rubric.get("title", "Unknown").strip()
        label = ex.get("label", {})
        score = label.get("score")
        
        if isinstance(score, int) and 1 <= score <= 5:
            coverage[rubric_title][score] += 1
    
    return dict(coverage)


# =============================================================================
# Report Printing
# =============================================================================

def print_header(text: str, char: str = "=") -> None:
    """Print a section header."""
    print(f"\n{char * 80}")
    print(text)
    print(char * 80)


def print_split_report(report: SplitReport) -> None:
    """Print report for a single split."""
    print(f"\n--- {report.name.upper()} ---")
    print(f"Total rows: {report.total_rows}")
    print(f"Unique rows: {report.unique_rows}")
    print(f"Exact duplicates removed: {report.exact_duplicates_removed}")
    
    if report.duplicate_groups:
        print(f"\nTop duplicate groups:")
        for i, group in enumerate(report.duplicate_groups, 1):
            print(f"  {i}. {group.count}x: ids={group.ids[:3]}{'...' if len(group.ids) > 3 else ''}")
            print(f"     {format_key_for_display(group.key)}")


def print_score_distribution(splits: Dict[str, SplitReport]) -> None:
    """Print score distribution table."""
    print_header("SCORE DISTRIBUTION")
    
    # Header
    header = f"{'Split':<15}" + "".join(f"{'Score ' + str(s):>12}" for s in range(1, 6)) + f"{'Total':>12}"
    print(header)
    print("-" * 80)
    
    for name, report in sorted(splits.items()):
        total = sum(report.score_distribution.values())
        row = f"{name:<15}"
        for s in range(1, 6):
            count = report.score_distribution.get(s, 0)
            pct = (count / total * 100) if total > 0 else 0
            row += f"{count:>6} ({pct:4.1f}%)"
        row += f"{total:>12}"
        print(row)


def print_cross_leakage(leakage: Dict[str, List[UniqueKey]], splits: Dict[str, SplitReport]) -> None:
    """Print cross-split leakage report."""
    print_header("CROSS-SPLIT LEAKAGE")
    
    pairs = [
        ("train", "valid"),
        ("train", "test"),
        ("valid", "test"),
    ]
    
    for split_a, split_b in pairs:
        key = f"{split_a}∩{split_b}"
        if key not in leakage:
            continue
        
        keys_a = splits.get(split_a, SplitReport(split_a)).unique_keys
        keys_b = splits.get(split_b, SplitReport(split_b)).unique_keys
        intersection_count = len(keys_a & keys_b)
        
        print(f"\n{key}: {intersection_count} leaked examples")
        
        if leakage[key]:
            print("  Sample leaked keys:")
            for k in leakage[key][:5]:
                print(f"    - {format_key_for_display(k)}")


def print_rubric_coverage(coverage: Dict[str, Dict[int, int]]) -> None:
    """Print rubric coverage matrix as markdown table."""
    print_header("RUBRIC COVERAGE MATRIX")
    
    # Markdown table
    print(f"\n| {'Rubric':<40} | {'S1':>5} | {'S2':>5} | {'S3':>5} | {'S4':>5} | {'S5':>5} | {'Total':>6} | {'Gaps':>8} |")
    print(f"|{'-'*42}|{'-'*7}|{'-'*7}|{'-'*7}|{'-'*7}|{'-'*7}|{'-'*8}|{'-'*10}|")
    
    rubrics_with_gaps = []
    
    for rubric_title in sorted(coverage.keys()):
        scores = coverage[rubric_title]
        total = sum(scores.values())
        missing = [s for s in range(1, 6) if scores.get(s, 0) == 0]
        gaps = ",".join(map(str, missing)) if missing else "-"
        
        if missing:
            rubrics_with_gaps.append((rubric_title, missing))
        
        row = f"| {rubric_title[:40]:<40} | {scores.get(1, 0):>5} | {scores.get(2, 0):>5} | {scores.get(3, 0):>5} | {scores.get(4, 0):>5} | {scores.get(5, 0):>5} | {total:>6} | {gaps:>8} |"
        print(row)
    
    if rubrics_with_gaps:
        print(f"\n⚠️ {len(rubrics_with_gaps)} rubrics have score gaps:")
        for rubric, missing in rubrics_with_gaps[:10]:
            print(f"  - {rubric}: missing scores {missing}")


def print_schema_failures(splits: Dict[str, SplitReport]) -> Dict[str, List[str]]:
    """Print schema validation failures and return by-rule summary."""
    print_header("SCHEMA VALIDATION")
    
    # Aggregate failures by rule
    failures_by_rule: Dict[str, List[SchemaFailure]] = defaultdict(list)
    
    for report in splits.values():
        for f in report.schema_failures:
            failures_by_rule[f.rule].append(f)
    
    if not failures_by_rule:
        print("\n✅ All examples pass schema validation!")
        return {}
    
    print(f"\n⚠️ Schema failures found:")
    
    summary = {}
    for rule in sorted(failures_by_rule.keys()):
        failures = failures_by_rule[rule]
        summary[rule] = [f.example_id for f in failures]
        
        print(f"\n[{rule}] - {len(failures)} failures")
        for f in failures[:10]:
            print(f"  - {f.example_id}: {f.message}")
        if len(failures) > 10:
            print(f"  ... and {len(failures) - 10} more")
    
    return summary


# =============================================================================
# JSON Output
# =============================================================================

def generate_json_report(
    audit: AuditReport,
    schema_summary: Dict[str, List[str]],
) -> dict:
    """Generate JSON summary report."""
    report = {
        "splits": {},
        "cross_leakage": {},
        "rubric_coverage": audit.rubric_coverage,
        "schema_failures": schema_summary,
        "hard_check_failed": audit.hard_check_failed,
    }
    
    for name, split in audit.splits.items():
        report["splits"][name] = {
            "total_rows": split.total_rows,
            "unique_rows": split.unique_rows,
            "exact_duplicates_removed": split.exact_duplicates_removed,
            "score_distribution": dict(sorted(split.score_distribution.items())),
            "duplicate_groups": [
                {
                    "count": g.count,
                    "ids": g.ids,
                    "prompt_preview": truncate_text(g.key[0], 80),
                    "response_preview": truncate_text(g.key[1], 80),
                    "rubric_title": g.key[2],
                }
                for g in split.duplicate_groups
            ],
        }
    
    for key, leaked_keys in audit.cross_leakage.items():
        splits_involved = key.split("∩")
        if len(splits_involved) == 2:
            split_a, split_b = splits_involved
            keys_a = audit.splits.get(split_a, SplitReport(split_a)).unique_keys
            keys_b = audit.splits.get(split_b, SplitReport(split_b)).unique_keys
            count = len(keys_a & keys_b)
        else:
            count = len(leaked_keys)
        
        report["cross_leakage"][key] = {
            "count": count,
            "samples": [
                {
                    "prompt_preview": truncate_text(k[0], 80),
                    "response_preview": truncate_text(k[1], 80),
                    "rubric_title": k[2],
                }
                for k in leaked_keys[:10]
            ],
        }
    
    return report


# =============================================================================
# Main Pipeline
# =============================================================================

def run_audit(data_dir: Path, out_dir: Path) -> int:
    """Run complete data audit.
    
    Returns:
        Exit code: 0 if all hard checks pass, 1 otherwise.
    """
    print_header("DATA AUDIT REPORT", "=")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {out_dir}")
    
    # Dataset files to audit
    dataset_files = [
        ("train", "train.jsonl"),
        ("valid", "valid.jsonl"),
        ("test", "test.jsonl"),
        ("calibration", "calibration.jsonl"),
        ("adversarial", "adversarial.jsonl"),
        ("gold_tests", "gold_tests.jsonl"),
    ]
    
    audit = AuditReport()
    all_examples = []
    load_errors = []
    
    # Load and analyze each split
    print_header("LOADING DATASETS")
    
    for split_name, filename in dataset_files:
        filepath = data_dir / filename
        
        if not filepath.exists():
            print(f"⚠️  {filename}: NOT FOUND")
            continue
        
        examples, errors = load_jsonl_safe(filepath)
        
        if errors:
            print(f"❌ {filename}: {len(errors)} JSON parse errors")
            load_errors.extend(errors)
            for e in errors[:3]:
                print(f"   {e}")
        else:
            print(f"✅ {filename}: {len(examples)} rows loaded")
        
        if examples:
            report = analyze_split(split_name, examples)
            audit.splits[split_name] = report
            all_examples.extend(examples)
    
    if not audit.splits:
        print("\n❌ No valid datasets found!")
        return 1
    
    # Cross-split leakage
    audit.cross_leakage = detect_cross_leakage(audit.splits)
    
    # Rubric coverage
    audit.rubric_coverage = compute_rubric_coverage(audit.splits, all_examples)
    
    # Print reports
    print_header("DUPLICATE ANALYSIS")
    for name in sorted(audit.splits.keys()):
        print_split_report(audit.splits[name])
    
    print_score_distribution(audit.splits)
    print_cross_leakage(audit.cross_leakage, audit.splits)
    print_rubric_coverage(audit.rubric_coverage)
    schema_summary = print_schema_failures(audit.splits)
    
    # Determine hard check status
    hard_rules = {
        "missing_top_keys",
        "invalid_rubric",
        "invalid_label",
        "missing_rubric_keys",
        "missing_label_keys",
        "invalid_score",
        "invalid_flags",
        "invalid_flag_keys",
        "invalid_flag_type",
    }
    
    for rule in schema_summary:
        if rule in hard_rules:
            audit.hard_check_failed = True
            break
    
    # Generate JSON output
    out_dir.mkdir(parents=True, exist_ok=True)
    json_report = generate_json_report(audit, schema_summary)
    
    json_path = out_dir / "data_audit_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_report, f, indent=2, ensure_ascii=False)
    
    print_header("SUMMARY")
    print(f"\nJSON report saved to: {json_path}")
    
    total_examples = sum(s.total_rows for s in audit.splits.values())
    total_unique = sum(s.unique_rows for s in audit.splits.values())
    total_failures = sum(len(v) for v in schema_summary.values())
    
    print(f"\nTotal examples: {total_examples}")
    print(f"Total unique: {total_unique}")
    print(f"Total duplicates: {total_examples - total_unique}")
    print(f"Schema failures: {total_failures}")
    
    if audit.hard_check_failed:
        print("\n❌ HARD SCHEMA CHECK FAILED - exit code 1")
        return 1
    else:
        print("\n✅ All hard schema checks passed - exit code 0")
        return 0


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Production-grade data audit report for Judge dataset."
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data"),
        help="Directory containing dataset JSONL files",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs"),
        help="Output directory for JSON report",
    )
    
    args = parser.parse_args()
    
    return run_audit(args.data_dir, args.out)


if __name__ == "__main__":
    sys.exit(main())
