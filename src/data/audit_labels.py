"""
Dataset Label Auditing for Auto-Grader Judge Model.

Audits consistency between:
1. Score and rubric_items pass/fail alignment
2. Reasoning content and score level
3. Flag values and rubric context

Usage:
    python -m src.data.audit_labels
    python -m src.data.audit_labels --auto_fix
    python -m src.data.audit_labels --data_dir data_clean
"""

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class AuditIssue:
    """A single audit issue found in an example."""
    example_id: str
    issue_type: str
    description: str
    current_value: str
    expected_value: Optional[str] = None
    auto_fixable: bool = False


@dataclass
class AuditReport:
    """Complete audit report for a dataset."""
    dataset_name: str
    total_examples: int = 0
    issues: List[AuditIssue] = field(default_factory=list)
    
    @property
    def issue_count(self) -> int:
        return len(self.issues)
    
    @property
    def fixable_count(self) -> int:
        return sum(1 for i in self.issues if i.auto_fixable)
    
    def issues_by_type(self) -> Dict[str, List[AuditIssue]]:
        """Group issues by type."""
        by_type = defaultdict(list)
        for issue in self.issues:
            by_type[issue.issue_type].append(issue)
        return dict(by_type)


# =============================================================================
# Audit Functions
# =============================================================================

def audit_score_rubric_consistency(example: dict) -> List[AuditIssue]:
    """Check if score aligns with rubric_items pass/fail values.
    
    Rules:
    - score=5 → all rubric_items should pass
    - score=1 → majority of rubric_items should fail
    - score=2-4 → mixed results expected
    """
    issues = []
    example_id = example.get("id", "unknown")
    label = example.get("label", {})
    score = label.get("score")
    rubric_items = label.get("rubric_items", [])
    
    if not rubric_items or score is None:
        return issues
    
    pass_count = sum(1 for item in rubric_items if item.get("pass", False))
    fail_count = len(rubric_items) - pass_count
    pass_rate = pass_count / len(rubric_items) if rubric_items else 0
    
    # Score 5 should have all passes
    if score == 5 and fail_count > 0:
        issues.append(AuditIssue(
            example_id=example_id,
            issue_type="score_rubric_mismatch",
            description=f"Score=5 but {fail_count}/{len(rubric_items)} rubric items failed",
            current_value=f"score=5, pass_rate={pass_rate:.0%}",
            expected_value="All rubric items should pass for score=5",
            auto_fixable=False,  # Ambiguous which to fix
        ))
    
    # Score 1 should have majority fails
    if score == 1 and pass_rate > 0.5:
        issues.append(AuditIssue(
            example_id=example_id,
            issue_type="score_rubric_mismatch",
            description=f"Score=1 but {pass_count}/{len(rubric_items)} rubric items passed",
            current_value=f"score=1, pass_rate={pass_rate:.0%}",
            expected_value="Majority of rubric items should fail for score=1",
            auto_fixable=False,
        ))
    
    # Score 4-5 should have high pass rate
    if score >= 4 and pass_rate < 0.5:
        issues.append(AuditIssue(
            example_id=example_id,
            issue_type="score_rubric_mismatch",
            description=f"Score={score} but only {pass_rate:.0%} items passed",
            current_value=f"score={score}, pass_rate={pass_rate:.0%}",
            expected_value="High scores should have high pass rates",
            auto_fixable=False,
        ))
    
    # Score 1-2 should have low pass rate
    if score <= 2 and pass_rate > 0.7:
        issues.append(AuditIssue(
            example_id=example_id,
            issue_type="score_rubric_mismatch",
            description=f"Score={score} but {pass_rate:.0%} items passed",
            current_value=f"score={score}, pass_rate={pass_rate:.0%}",
            expected_value="Low scores should have low pass rates",
            auto_fixable=False,
        ))
    
    return issues


def audit_reasoning_consistency(example: dict) -> List[AuditIssue]:
    """Check if reasoning content matches the score level.
    
    Rules:
    - score<=2 → reasoning should mention errors/flaws/issues
    - score>=4 → reasoning should not claim critical failures
    """
    issues = []
    example_id = example.get("id", "unknown")
    label = example.get("label", {})
    score = label.get("score")
    reasoning = label.get("reasoning", "").lower()
    
    if not reasoning or score is None:
        return issues
    
    # Negative indicators (errors, problems)
    negative_patterns = [
        r'\b(error|wrong|incorrect|fail|flaw|mistake|issue|problem|missing|incomplete)\b',
        r'\b(inaccurate|misleading|confusing|unclear|poor|bad|weak)\b',
        r'\b(does not|doesn\'t|cannot|can\'t|unable|refused|refuses)\b',
    ]
    
    # Positive indicators
    positive_patterns = [
        r'\b(correct|accurate|complete|excellent|good|well|clear|helpful)\b',
        r'\b(thorough|comprehensive|appropriate|properly|successfully)\b',
    ]
    
    has_negative = any(re.search(p, reasoning) for p in negative_patterns)
    has_positive = any(re.search(p, reasoning) for p in positive_patterns)
    
    # Low score (1-2) should mention errors
    if score <= 2 and not has_negative:
        issues.append(AuditIssue(
            example_id=example_id,
            issue_type="reasoning_score_mismatch",
            description=f"Score={score} but reasoning doesn't mention errors or issues",
            current_value=f"reasoning: {reasoning[:100]}...",
            expected_value="Low score reasoning should explain what went wrong",
            auto_fixable=False,
        ))
    
    # High score (4-5) shouldn't claim critical failures
    critical_failure_patterns = [
        r'\b(completely wrong|totally incorrect|major error|critical flaw)\b',
        r'\b(fails to|failed to|unable to|cannot|refuses to)\b',
    ]
    has_critical_failure = any(re.search(p, reasoning) for p in critical_failure_patterns)
    
    if score >= 4 and has_critical_failure:
        issues.append(AuditIssue(
            example_id=example_id,
            issue_type="reasoning_score_mismatch",
            description=f"Score={score} but reasoning mentions critical failures",
            current_value=f"reasoning: {reasoning[:100]}...",
            expected_value="High score reasoning shouldn't claim critical failures",
            auto_fixable=False,
        ))
    
    return issues


def audit_flag_consistency(example: dict) -> List[AuditIssue]:
    """Check if flags are consistent with rubric context.
    
    Rules:
    - over_refusal=True → rubric should be about Helpfulness
    - prompt_injection_detected=True → rubric should be about Safety/Robustness
    """
    issues = []
    example_id = example.get("id", "unknown")
    label = example.get("label", {})
    flags = label.get("flags", {})
    rubric = example.get("rubric", {})
    rubric_title = rubric.get("title", "").lower()
    
    # over_refusal flag check
    over_refusal = flags.get("over_refusal", False)
    if over_refusal:
        helpfulness_keywords = ["helpful", "task completion", "user experience", "assist"]
        is_helpfulness_rubric = any(kw in rubric_title for kw in helpfulness_keywords)
        
        if not is_helpfulness_rubric:
            issues.append(AuditIssue(
                example_id=example_id,
                issue_type="flag_rubric_mismatch",
                description=f"over_refusal=True but rubric is '{rubric_title}'",
                current_value=f"flag=over_refusal, rubric={rubric_title}",
                expected_value="over_refusal should be used with Helpfulness rubrics",
                auto_fixable=False,
            ))
    
    # prompt_injection_detected flag check
    injection_detected = flags.get("prompt_injection_detected", False)
    if injection_detected:
        safety_keywords = ["safety", "security", "robustness", "injection", "harmful"]
        is_safety_rubric = any(kw in rubric_title for kw in safety_keywords)
        
        if not is_safety_rubric:
            issues.append(AuditIssue(
                example_id=example_id,
                issue_type="flag_rubric_mismatch",
                description=f"prompt_injection_detected=True but rubric is '{rubric_title}'",
                current_value=f"flag=prompt_injection_detected, rubric={rubric_title}",
                expected_value="prompt_injection_detected should be used with Safety/Robustness rubrics",
                auto_fixable=False,
            ))
    
    return issues


def audit_required_fields(example: dict) -> List[AuditIssue]:
    """Check that all required fields are present and valid."""
    issues = []
    example_id = example.get("id", "unknown")
    label = example.get("label", {})
    
    # Check score range
    score = label.get("score")
    if score is None:
        issues.append(AuditIssue(
            example_id=example_id,
            issue_type="missing_field",
            description="Missing score in label",
            current_value="score=None",
            expected_value="score should be 1-5",
            auto_fixable=False,
        ))
    elif not isinstance(score, int) or score < 1 or score > 5:
        issues.append(AuditIssue(
            example_id=example_id,
            issue_type="invalid_field",
            description=f"Invalid score value: {score}",
            current_value=f"score={score}",
            expected_value="score should be integer 1-5",
            auto_fixable=True,
        ))
    
    # Check reasoning
    reasoning = label.get("reasoning", "")
    if not reasoning or len(reasoning.strip()) < 10:
        issues.append(AuditIssue(
            example_id=example_id,
            issue_type="missing_field",
            description="Missing or too short reasoning",
            current_value=f"reasoning='{reasoning[:50]}'",
            expected_value="reasoning should explain the score",
            auto_fixable=False,
        ))
    
    # Check flags exist
    flags = label.get("flags", {})
    required_flags = {"over_refusal", "prompt_injection_detected", "format_violation"}
    missing_flags = required_flags - set(flags.keys())
    
    if missing_flags:
        issues.append(AuditIssue(
            example_id=example_id,
            issue_type="missing_field",
            description=f"Missing required flags: {missing_flags}",
            current_value=f"flags={list(flags.keys())}",
            expected_value=f"flags should include {required_flags}",
            auto_fixable=True,
        ))
    
    # Check flag values are boolean
    for flag_name, flag_value in flags.items():
        if not isinstance(flag_value, bool):
            issues.append(AuditIssue(
                example_id=example_id,
                issue_type="invalid_field",
                description=f"Flag '{flag_name}' is not boolean",
                current_value=f"{flag_name}={flag_value} ({type(flag_value).__name__})",
                expected_value="flags should be boolean",
                auto_fixable=True,
            ))
    
    return issues


def audit_example(example: dict) -> List[AuditIssue]:
    """Run all audit checks on a single example."""
    issues = []
    
    issues.extend(audit_required_fields(example))
    issues.extend(audit_score_rubric_consistency(example))
    issues.extend(audit_reasoning_consistency(example))
    issues.extend(audit_flag_consistency(example))
    
    return issues


# =============================================================================
# Auto-Fix Functions
# =============================================================================

def auto_fix_example(example: dict) -> Tuple[dict, List[str]]:
    """Attempt to auto-fix simple issues in an example.
    
    Returns:
        Tuple of (fixed_example, list of fixes applied)
    """
    fixes = []
    label = example.get("label", {})
    
    # Fix missing flags
    flags = label.get("flags", {})
    required_flags = {"over_refusal", "prompt_injection_detected", "format_violation"}
    
    for flag in required_flags:
        if flag not in flags:
            flags[flag] = False
            fixes.append(f"Added missing flag: {flag}=False")
    
    # Fix non-boolean flags
    for flag_name, flag_value in list(flags.items()):
        if not isinstance(flag_value, bool):
            if flag_value in (0, "false", "False", "no", "No", None):
                flags[flag_name] = False
                fixes.append(f"Converted {flag_name}={flag_value} to False")
            elif flag_value in (1, "true", "True", "yes", "Yes"):
                flags[flag_name] = True
                fixes.append(f"Converted {flag_name}={flag_value} to True")
    
    label["flags"] = flags
    
    # Fix score if it's a float
    score = label.get("score")
    if isinstance(score, float):
        new_score = int(round(score))
        new_score = max(1, min(5, new_score))
        label["score"] = new_score
        fixes.append(f"Converted score={score} to {new_score}")
    
    example["label"] = label
    return example, fixes


# =============================================================================
# Main Pipeline
# =============================================================================

def load_jsonl(filepath: Path) -> List[dict]:
    """Load examples from JSONL file."""
    if not filepath.exists():
        return []
    
    examples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def save_jsonl(examples: List[dict], filepath: Path) -> None:
    """Save examples to JSONL file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def audit_dataset(
    filepath: Path,
    auto_fix: bool = False,
    output_path: Optional[Path] = None,
) -> AuditReport:
    """Audit a single dataset file.
    
    Args:
        filepath: Path to JSONL file.
        auto_fix: Whether to auto-fix simple issues.
        output_path: Optional path to save fixed dataset.
    
    Returns:
        AuditReport with all issues found.
    """
    dataset_name = filepath.stem
    examples = load_jsonl(filepath)
    
    if not examples:
        return AuditReport(dataset_name=dataset_name)
    
    report = AuditReport(
        dataset_name=dataset_name,
        total_examples=len(examples),
    )
    
    fixed_examples = []
    total_fixes = []
    
    for example in examples:
        # Audit
        issues = audit_example(example)
        report.issues.extend(issues)
        
        # Auto-fix if enabled
        if auto_fix:
            example, fixes = auto_fix_example(example)
            total_fixes.extend(fixes)
        
        fixed_examples.append(example)
    
    # Save fixed dataset if requested
    if auto_fix and output_path and total_fixes:
        save_jsonl(fixed_examples, output_path)
        print(f"Saved fixed dataset to: {output_path}")
        print(f"Applied {len(total_fixes)} auto-fixes")
    
    return report


def print_report(report: AuditReport, verbose: bool = True) -> None:
    """Print audit report for a dataset."""
    print(f"\n{'='*60}")
    print(f"AUDIT: {report.dataset_name}")
    print(f"{'='*60}")
    print(f"Total examples: {report.total_examples}")
    print(f"Issues found: {report.issue_count}")
    print(f"Auto-fixable: {report.fixable_count}")
    
    if report.issues:
        by_type = report.issues_by_type()
        
        print(f"\nIssues by type:")
        for issue_type, issues in by_type.items():
            print(f"  - {issue_type}: {len(issues)}")
        
        if verbose:
            print(f"\nExample issues (first 5 per type):")
            for issue_type, issues in by_type.items():
                print(f"\n  [{issue_type}]")
                for issue in issues[:5]:
                    print(f"    - {issue.example_id}: {issue.description}")
    else:
        print("\n✅ No issues found!")


def print_summary(reports: Dict[str, AuditReport]) -> None:
    """Print summary across all datasets."""
    print("\n" + "=" * 80)
    print("AUDIT SUMMARY")
    print("=" * 80)
    
    header = f"{'Dataset':<20} {'Examples':>10} {'Issues':>10} {'Fixable':>10} {'Status':>10}"
    print(header)
    print("-" * 80)
    
    total_examples = 0
    total_issues = 0
    total_fixable = 0
    
    for name, report in reports.items():
        status = "✅ OK" if report.issue_count == 0 else "⚠️ Issues"
        row = f"{name:<20} {report.total_examples:>10} {report.issue_count:>10} {report.fixable_count:>10} {status:>10}"
        print(row)
        
        total_examples += report.total_examples
        total_issues += report.issue_count
        total_fixable += report.fixable_count
    
    print("-" * 80)
    total_status = "✅" if total_issues == 0 else "⚠️"
    print(f"{'TOTAL':<20} {total_examples:>10} {total_issues:>10} {total_fixable:>10} {total_status:>10}")
    print("=" * 80)
    
    if total_issues > 0:
        print(f"\n⚠️ Found {total_issues} label inconsistencies across datasets.")
        print(f"   {total_fixable} can be auto-fixed with --auto_fix flag.")
    else:
        print("\n✅ All datasets pass label consistency checks!")


def run_audit(
    data_dir: Path,
    auto_fix: bool = False,
    output_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Dict[str, AuditReport]:
    """Run audit on all dataset files.
    
    Args:
        data_dir: Directory containing JSONL files.
        auto_fix: Whether to auto-fix simple issues.
        output_dir: Directory to save fixed datasets (if auto_fix).
        verbose: Print detailed issue examples.
    
    Returns:
        Dictionary mapping dataset name to audit report.
    """
    # Dataset files to audit
    dataset_files = [
        "train.jsonl",
        "valid.jsonl",
        "test.jsonl",
        "calibration.jsonl",
        "adversarial.jsonl",
        "gold_tests.jsonl",
        "train_clean.jsonl",
        "valid_clean.jsonl",
        "test_clean.jsonl",
    ]
    
    reports = {}
    
    for filename in dataset_files:
        filepath = data_dir / filename
        
        if not filepath.exists():
            continue
        
        output_path = None
        if auto_fix and output_dir:
            output_path = output_dir / filename.replace(".jsonl", "_fixed.jsonl")
        
        report = audit_dataset(filepath, auto_fix, output_path)
        reports[report.dataset_name] = report
        
        if verbose:
            print_report(report, verbose=verbose)
    
    return reports


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Audit dataset labels for consistency."
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data"),
        help="Directory containing dataset files",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory to save fixed datasets (with --auto_fix)",
    )
    parser.add_argument(
        "--auto_fix",
        action="store_true",
        help="Automatically fix simple issues (missing flags, type conversions)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print summary, not detailed issues",
    )
    
    args = parser.parse_args()
    
    # Default output dir for auto-fix
    output_dir = args.output_dir
    if args.auto_fix and output_dir is None:
        output_dir = args.data_dir / "fixed"
    
    reports = run_audit(
        data_dir=args.data_dir,
        auto_fix=args.auto_fix,
        output_dir=output_dir,
        verbose=not args.quiet,
    )
    
    print_summary(reports)


if __name__ == "__main__":
    main()
