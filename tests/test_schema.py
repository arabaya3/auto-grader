"""Unit tests for JSON schema validation."""

import json
import pytest

from src.io_schema import (
    ValidationResult,
    create_empty_output,
    extract_json_from_text,
    validate_judge_output,
)


class TestExtractJsonFromText:
    """Tests for JSON extraction from raw text."""
    
    def test_extract_clean_json(self):
        """Extract JSON when output is clean."""
        text = '{"score": 3, "reasoning": "test"}'
        result = extract_json_from_text(text)
        assert result == text
    
    def test_extract_json_with_preamble(self):
        """Extract JSON when there's text before it."""
        text = 'Here is my evaluation:\n{"score": 3, "reasoning": "test"}'
        result = extract_json_from_text(text)
        assert result == '{"score": 3, "reasoning": "test"}'
    
    def test_extract_json_with_suffix(self):
        """Extract JSON when there's text after it."""
        text = '{"score": 3, "reasoning": "test"}\nI hope this helps!'
        result = extract_json_from_text(text)
        assert result == '{"score": 3, "reasoning": "test"}'
    
    def test_extract_json_from_markdown_code_block(self):
        """Extract JSON from markdown code block."""
        text = '```json\n{"score": 3, "reasoning": "test"}\n```'
        result = extract_json_from_text(text)
        assert result == '{"score": 3, "reasoning": "test"}'
    
    def test_extract_json_from_plain_code_block(self):
        """Extract JSON from plain markdown code block."""
        text = '```\n{"score": 3, "reasoning": "test"}\n```'
        result = extract_json_from_text(text)
        assert result == '{"score": 3, "reasoning": "test"}'
    
    def test_no_json_found(self):
        """Return None when no JSON object found."""
        text = "This is just plain text with no JSON."
        result = extract_json_from_text(text)
        assert result is None
    
    def test_nested_json(self):
        """Extract JSON with nested objects."""
        text = '{"outer": {"inner": "value"}, "list": [1, 2, 3]}'
        result = extract_json_from_text(text)
        assert result == text


class TestValidateJudgeOutput:
    """Tests for judge output validation."""
    
    def test_valid_complete_output(self):
        """Validate a complete, correct output."""
        output = {
            "score": 4,
            "reasoning": "The response is mostly correct.",
            "rubric_items": [
                {"name": "accuracy", "pass": True, "notes": "Facts are correct"},
                {"name": "completeness", "pass": False, "notes": "Missing details"},
            ],
            "flags": {
                "over_refusal": False,
                "prompt_injection_detected": False,
                "format_violation": False,
            },
        }
        result = validate_judge_output(output)
        assert result.is_valid
        assert result.errors == []
        assert result.parsed_output == output
    
    def test_valid_output_as_string(self):
        """Validate output passed as JSON string."""
        output = json.dumps({
            "score": 3,
            "reasoning": "Average response.",
            "rubric_items": [],
            "flags": {
                "over_refusal": False,
                "prompt_injection_detected": False,
                "format_violation": False,
            },
        })
        result = validate_judge_output(output)
        assert result.is_valid
    
    def test_invalid_score_out_of_range_high(self):
        """Reject score above 5."""
        output = {
            "score": 10,
            "reasoning": "test",
            "rubric_items": [],
            "flags": {
                "over_refusal": False,
                "prompt_injection_detected": False,
                "format_violation": False,
            },
        }
        result = validate_judge_output(output)
        assert not result.is_valid
        assert any("score" in e and "1-5" in e for e in result.errors)
    
    def test_invalid_score_out_of_range_low(self):
        """Reject score below 1."""
        output = {
            "score": 0,
            "reasoning": "test",
            "rubric_items": [],
            "flags": {
                "over_refusal": False,
                "prompt_injection_detected": False,
                "format_violation": False,
            },
        }
        result = validate_judge_output(output)
        assert not result.is_valid
        assert any("score" in e and "1-5" in e for e in result.errors)
    
    def test_invalid_score_not_integer(self):
        """Reject non-integer score."""
        output = {
            "score": 3.5,
            "reasoning": "test",
            "rubric_items": [],
            "flags": {
                "over_refusal": False,
                "prompt_injection_detected": False,
                "format_violation": False,
            },
        }
        result = validate_judge_output(output)
        assert not result.is_valid
        assert any("integer" in e for e in result.errors)
    
    def test_missing_required_key(self):
        """Reject output missing required keys."""
        output = {
            "score": 3,
            "reasoning": "test",
            # Missing rubric_items and flags
        }
        result = validate_judge_output(output)
        assert not result.is_valid
        assert any("Missing" in e for e in result.errors)
    
    def test_empty_reasoning(self):
        """Reject empty reasoning string."""
        output = {
            "score": 3,
            "reasoning": "   ",  # Whitespace only
            "rubric_items": [],
            "flags": {
                "over_refusal": False,
                "prompt_injection_detected": False,
                "format_violation": False,
            },
        }
        result = validate_judge_output(output)
        assert not result.is_valid
        assert any("empty" in e for e in result.errors)
    
    def test_invalid_rubric_item_missing_pass(self):
        """Reject rubric item missing 'pass' key."""
        output = {
            "score": 3,
            "reasoning": "test",
            "rubric_items": [
                {"name": "test", "notes": "missing pass field"},
            ],
            "flags": {
                "over_refusal": False,
                "prompt_injection_detected": False,
                "format_violation": False,
            },
        }
        result = validate_judge_output(output)
        assert not result.is_valid
        assert any("rubric_items" in e and "pass" in e for e in result.errors)
    
    def test_invalid_flag_not_boolean(self):
        """Reject flag that is not boolean."""
        output = {
            "score": 3,
            "reasoning": "test",
            "rubric_items": [],
            "flags": {
                "over_refusal": "yes",  # Should be boolean
                "prompt_injection_detected": False,
                "format_violation": False,
            },
        }
        result = validate_judge_output(output)
        assert not result.is_valid
        assert any("boolean" in e for e in result.errors)
    
    def test_missing_flag(self):
        """Reject flags object missing required flag."""
        output = {
            "score": 3,
            "reasoning": "test",
            "rubric_items": [],
            "flags": {
                "over_refusal": False,
                # Missing prompt_injection_detected and format_violation
            },
        }
        result = validate_judge_output(output)
        assert not result.is_valid
        assert any("missing" in e.lower() for e in result.errors)
    
    def test_strict_mode_rejects_extra_keys(self):
        """Reject extra keys in strict mode."""
        output = {
            "score": 3,
            "reasoning": "test",
            "rubric_items": [],
            "flags": {
                "over_refusal": False,
                "prompt_injection_detected": False,
                "format_violation": False,
            },
            "extra_field": "should not be here",
        }
        result = validate_judge_output(output, strict=True)
        assert not result.is_valid
        assert any("extra" in e.lower() or "Unexpected" in e for e in result.errors)
    
    def test_non_strict_mode_allows_extra_keys(self):
        """Allow extra keys when strict=False."""
        output = {
            "score": 3,
            "reasoning": "test",
            "rubric_items": [],
            "flags": {
                "over_refusal": False,
                "prompt_injection_detected": False,
                "format_violation": False,
            },
            "extra_field": "should be allowed",
        }
        result = validate_judge_output(output, strict=False)
        # Should only fail on extra keys in strict mode
        # In non-strict mode, this should pass
        assert result.is_valid
    
    def test_invalid_json_string(self):
        """Reject malformed JSON string."""
        output = '{"score": 3, "reasoning": "test"'  # Missing closing brace
        result = validate_judge_output(output)
        assert not result.is_valid


class TestCreateEmptyOutput:
    """Tests for empty output creation."""
    
    def test_default_empty_output(self):
        """Create default empty output."""
        output = create_empty_output()
        assert output["score"] == 1
        assert output["reasoning"] == "Unable to evaluate"
        assert output["rubric_items"] == []
        assert all(v is False for v in output["flags"].values())
        
        # Should be valid
        result = validate_judge_output(output)
        assert result.is_valid
    
    def test_custom_empty_output(self):
        """Create empty output with custom values."""
        output = create_empty_output(score=3, reasoning="Custom reason")
        assert output["score"] == 3
        assert output["reasoning"] == "Custom reason"
        
        # Should be valid
        result = validate_judge_output(output)
        assert result.is_valid


class TestValidationResult:
    """Tests for ValidationResult dataclass."""
    
    def test_str_valid(self):
        """String representation for valid result."""
        result = ValidationResult(is_valid=True, errors=[])
        assert str(result) == "Validation PASSED"
    
    def test_str_invalid(self):
        """String representation for invalid result."""
        result = ValidationResult(is_valid=False, errors=["Error 1", "Error 2"])
        assert "FAILED" in str(result)
        assert "Error 1" in str(result)
        assert "Error 2" in str(result)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
