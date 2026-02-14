"""
Regression tests for flag schema consistency.

Ensures all dataset files use exactly the 3 official flags:
- over_refusal
- prompt_injection_detected
- format_violation
"""

import json
import pytest
from pathlib import Path

# Official flags that must be present in all labels
REQUIRED_FLAGS = {"over_refusal", "prompt_injection_detected", "format_violation"}

DATA_DIR = Path(__file__).parent.parent / "data"


def load_jsonl(filepath: Path) -> list[dict]:
    """Load a JSONL file and return list of records."""
    if not filepath.exists():
        return []
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def validate_flags(record: dict, record_id: str, filename: str) -> list[str]:
    """Validate that a record has exactly the required flags."""
    errors = []
    
    label = record.get("label", {})
    flags = label.get("flags", {})
    
    if not isinstance(flags, dict):
        errors.append(f"{filename}[{record_id}]: flags is not a dict")
        return errors
    
    flag_keys = set(flags.keys())
    
    # Check for missing required flags
    missing = REQUIRED_FLAGS - flag_keys
    if missing:
        errors.append(f"{filename}[{record_id}]: missing required flags: {missing}")
    
    # Check for extra flags that shouldn't be there
    extra = flag_keys - REQUIRED_FLAGS
    if extra:
        errors.append(f"{filename}[{record_id}]: unexpected extra flags: {extra}")
    
    # Check that all flag values are booleans
    for key, value in flags.items():
        if not isinstance(value, bool):
            errors.append(f"{filename}[{record_id}]: flag '{key}' is not boolean: {type(value).__name__}")
    
    return errors


class TestFlagSchema:
    """Test suite for flag schema validation across all datasets."""
    
    @pytest.mark.parametrize("filename", [
        "train.jsonl",
        "valid.jsonl", 
        "test.jsonl",
        "gold_mini.jsonl",
        "gold_tests.jsonl",
        "train_mini.jsonl",
        "valid_mini.jsonl",
        "calibration.jsonl",
        "adversarial.jsonl",
    ])
    def test_dataset_flag_schema(self, filename: str):
        """Verify each dataset file has correct flag schema."""
        filepath = DATA_DIR / filename
        
        if not filepath.exists():
            pytest.skip(f"{filename} not found (may not be generated yet)")
        
        records = load_jsonl(filepath)
        if not records:
            pytest.skip(f"{filename} is empty")
        
        all_errors = []
        for i, record in enumerate(records):
            record_id = record.get("id", f"row_{i}")
            errors = validate_flags(record, record_id, filename)
            all_errors.extend(errors)
        
        if all_errors:
            error_summary = "\n".join(all_errors[:20])  # Show first 20 errors
            if len(all_errors) > 20:
                error_summary += f"\n... and {len(all_errors) - 20} more errors"
            pytest.fail(f"Flag schema violations:\n{error_summary}")
    
    def test_calibration_examples_in_code(self):
        """Test that the CALIBRATION_EXAMPLES in code have correct flags."""
        from src.data.build_calibration_dataset import CALIBRATION_EXAMPLES
        
        all_errors = []
        for score, examples in CALIBRATION_EXAMPLES.items():
            for i, example in enumerate(examples):
                record_id = f"score_{score}_idx_{i}"
                # Wrap in label structure for validation
                mock_record = {"label": {"flags": example.get("flags", {})}}
                errors = validate_flags(mock_record, record_id, "CALIBRATION_EXAMPLES")
                all_errors.extend(errors)
        
        if all_errors:
            pytest.fail(f"CALIBRATION_EXAMPLES flag violations:\n" + "\n".join(all_errors))
    
    def test_adversarial_examples_in_code(self):
        """Test that ADVERSARIAL_EXAMPLES in code have correct flags."""
        from src.data.build_calibration_dataset import ADVERSARIAL_EXAMPLES
        
        all_errors = []
        for i, example in enumerate(ADVERSARIAL_EXAMPLES):
            record_id = f"adversarial_{i}"
            mock_record = {"label": {"flags": example.get("flags", {})}}
            errors = validate_flags(mock_record, record_id, "ADVERSARIAL_EXAMPLES")
            all_errors.extend(errors)
        
        if all_errors:
            pytest.fail(f"ADVERSARIAL_EXAMPLES flag violations:\n" + "\n".join(all_errors))
    
    def test_io_schema_required_flags_match(self):
        """Verify our test uses the same flags validated by io_schema.py."""
        # Read io_schema.py and verify our REQUIRED_FLAGS match
        from src.io_schema import JUDGE_OUTPUT_SCHEMA
        
        schema_flags = JUDGE_OUTPUT_SCHEMA["properties"]["flags"]["properties"].keys()
        schema_required = set(JUDGE_OUTPUT_SCHEMA["properties"]["flags"]["required"])
        
        assert REQUIRED_FLAGS == schema_required, (
            f"Test REQUIRED_FLAGS {REQUIRED_FLAGS} doesn't match "
            f"io_schema required flags {schema_required}"
        )
        assert REQUIRED_FLAGS == set(schema_flags), (
            f"Test REQUIRED_FLAGS {REQUIRED_FLAGS} doesn't match "
            f"io_schema flag properties {set(schema_flags)}"
        )


class TestFlagGeneration:
    """Test that generated examples have correct flag schema."""
    
    def test_generate_calibration_example_flags(self):
        """Test generate_calibration_example includes all required flags."""
        from src.data.build_calibration_dataset import (
            generate_calibration_example,
            CALIBRATION_EXAMPLES,
        )
        
        # Test with an example that has minimal flags
        example_data = CALIBRATION_EXAMPLES[3][0]  # Score 3 example
        result = generate_calibration_example(3, example_data, "test_001")
        
        flags = result["label"]["flags"]
        assert set(flags.keys()) == REQUIRED_FLAGS
        assert all(isinstance(v, bool) for v in flags.values())
    
    def test_generate_adversarial_example_flags(self):
        """Test generate_adversarial_example includes all required flags."""
        from src.data.build_calibration_dataset import (
            generate_adversarial_example,
            ADVERSARIAL_EXAMPLES,
        )
        
        example_data = ADVERSARIAL_EXAMPLES[0]
        result = generate_adversarial_example(example_data, example_id="test_adv_001")
        
        flags = result["label"]["flags"]
        assert set(flags.keys()) == REQUIRED_FLAGS
        assert all(isinstance(v, bool) for v in flags.values())
