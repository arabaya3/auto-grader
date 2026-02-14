"""JSON schema definition and validation for Judge Model output."""

import json
import re
from dataclasses import dataclass
from typing import Any, Optional

# JSON Schema for judge output (JSON Schema Draft 7)
JUDGE_OUTPUT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "JudgeOutput",
    "description": "Output schema for the Auto-Grader Judge Model",
    "type": "object",
    "required": ["score", "reasoning", "rubric_items", "flags"],
    "properties": {
        "score": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5,
            "description": "Overall score from 1 (worst) to 5 (best)"
        },
        "reasoning": {
            "type": "string",
            "minLength": 1,
            "description": "Brief explanation of the score grounded in rubric"
        },
        "rubric_items": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "pass", "notes"],
                "properties": {
                    "name": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Name of the rubric criterion"
                    },
                    "pass": {
                        "type": "boolean",
                        "description": "Whether this criterion was satisfied"
                    },
                    "notes": {
                        "type": "string",
                        "description": "Brief justification for pass/fail"
                    }
                },
                "additionalProperties": False
            },
            "description": "Individual rubric criterion evaluations"
        },
        "flags": {
            "type": "object",
            "required": ["over_refusal", "prompt_injection_detected", "format_violation"],
            "properties": {
                "over_refusal": {
                    "type": "boolean",
                    "description": "True if response unnecessarily refused a benign request"
                },
                "prompt_injection_detected": {
                    "type": "boolean", 
                    "description": "True if candidate response attempted to manipulate scoring"
                },
                "format_violation": {
                    "type": "boolean",
                    "description": "True if response format was problematic"
                }
            },
            "additionalProperties": False,
            "description": "Warning flags for special conditions"
        }
    },
    "additionalProperties": False
}


@dataclass
class ValidationResult:
    """Result of JSON validation."""
    
    is_valid: bool
    errors: list[str]
    parsed_output: Optional[dict[str, Any]] = None
    
    def __str__(self) -> str:
        if self.is_valid:
            return "Validation PASSED"
        return f"Validation FAILED: {'; '.join(self.errors)}"


def extract_json_from_text(text: str) -> Optional[str]:
    """Extract JSON object from text that may contain other content.
    
    Handles cases where the model outputs text before/after JSON,
    or wraps JSON in markdown code blocks.
    
    Args:
        text: Raw text that may contain JSON.
    
    Returns:
        Extracted JSON string or None if not found.
    """
    # Try to find JSON in markdown code blocks first
    code_block_pattern = r"```(?:json)?\s*(\{[\s\S]*?\})\s*```"
    match = re.search(code_block_pattern, text)
    if match:
        return match.group(1)
    
    # Try to find raw JSON object
    # Look for outermost { } pair
    brace_count = 0
    start_idx = None
    
    for i, char in enumerate(text):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx is not None:
                return text[start_idx:i+1]
    
    return None


def validate_judge_output(output: str | dict, strict: bool = True) -> ValidationResult:
    """Validate judge model output against the schema.
    
    Args:
        output: JSON string or already-parsed dictionary.
        strict: If True, enforce all schema requirements.
    
    Returns:
        ValidationResult with status and any errors.
    """
    errors = []
    parsed = None
    
    # Parse JSON if string
    if isinstance(output, str):
        # Try to extract JSON from potentially noisy output
        json_str = extract_json_from_text(output)
        if json_str is None:
            return ValidationResult(
                is_valid=False,
                errors=["No valid JSON object found in output"],
            )
        
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"JSON parse error: {e}"],
            )
    else:
        parsed = output
    
    if not isinstance(parsed, dict):
        return ValidationResult(
            is_valid=False,
            errors=["Output must be a JSON object"],
        )
    
    # Check required top-level keys
    required_keys = {"score", "reasoning", "rubric_items", "flags"}
    missing_keys = required_keys - set(parsed.keys())
    if missing_keys:
        errors.append(f"Missing required keys: {missing_keys}")
    
    # Validate score
    if "score" in parsed:
        score = parsed["score"]
        if not isinstance(score, int):
            errors.append(f"'score' must be integer, got {type(score).__name__}")
        elif score < 1 or score > 5:
            errors.append(f"'score' must be 1-5, got {score}")
    
    # Validate reasoning
    if "reasoning" in parsed:
        reasoning = parsed["reasoning"]
        if not isinstance(reasoning, str):
            errors.append(f"'reasoning' must be string, got {type(reasoning).__name__}")
        elif len(reasoning.strip()) == 0:
            errors.append("'reasoning' cannot be empty")
    
    # Validate rubric_items
    if "rubric_items" in parsed:
        rubric_items = parsed["rubric_items"]
        if not isinstance(rubric_items, list):
            errors.append(f"'rubric_items' must be array, got {type(rubric_items).__name__}")
        else:
            for i, item in enumerate(rubric_items):
                item_errors = _validate_rubric_item(item, i)
                errors.extend(item_errors)
    
    # Validate flags
    if "flags" in parsed:
        flags = parsed["flags"]
        if not isinstance(flags, dict):
            errors.append(f"'flags' must be object, got {type(flags).__name__}")
        else:
            flag_errors = _validate_flags(flags)
            errors.extend(flag_errors)
    
    # Check for unexpected keys in strict mode
    if strict:
        allowed_keys = {"score", "reasoning", "rubric_items", "flags"}
        extra_keys = set(parsed.keys()) - allowed_keys
        if extra_keys:
            errors.append(f"Unexpected keys (strict mode): {extra_keys}")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        parsed_output=parsed if len(errors) == 0 else None,
    )


def _validate_rubric_item(item: Any, index: int) -> list[str]:
    """Validate a single rubric item."""
    errors = []
    prefix = f"rubric_items[{index}]"
    
    if not isinstance(item, dict):
        return [f"{prefix}: must be object, got {type(item).__name__}"]
    
    # Check required keys
    required = {"name", "pass", "notes"}
    missing = required - set(item.keys())
    if missing:
        errors.append(f"{prefix}: missing keys {missing}")
    
    # Validate types
    if "name" in item and not isinstance(item["name"], str):
        errors.append(f"{prefix}.name: must be string")
    
    if "pass" in item and not isinstance(item["pass"], bool):
        errors.append(f"{prefix}.pass: must be boolean")
    
    if "notes" in item and not isinstance(item["notes"], str):
        errors.append(f"{prefix}.notes: must be string")
    
    return errors


def _validate_flags(flags: dict) -> list[str]:
    """Validate the flags object."""
    errors = []
    required_flags = {"over_refusal", "prompt_injection_detected", "format_violation"}
    
    missing = required_flags - set(flags.keys())
    if missing:
        errors.append(f"flags: missing required flags {missing}")
    
    for flag_name in required_flags:
        if flag_name in flags and not isinstance(flags[flag_name], bool):
            errors.append(f"flags.{flag_name}: must be boolean")
    
    return errors


def create_empty_output(score: int = 1, reasoning: str = "Unable to evaluate") -> dict:
    """Create a minimal valid output structure.
    
    Useful for error cases or defaults.
    
    Args:
        score: Default score.
        reasoning: Default reasoning.
    
    Returns:
        Valid judge output dictionary.
    """
    return {
        "score": score,
        "reasoning": reasoning,
        "rubric_items": [],
        "flags": {
            "over_refusal": False,
            "prompt_injection_detected": False,
            "format_violation": False,
        }
    }


def parse_and_validate(raw_text: str, strict: bool = True) -> tuple[Optional[dict], bool, list[str]]:
    """Parse raw model output and validate against schema.
    
    Shared helper for eval and demo scripts.
    
    Args:
        raw_text: Raw model output that may contain JSON.
        strict: If True, enforce all schema requirements.
    
    Returns:
        Tuple of (parsed_dict, is_valid, errors)
        - parsed_dict: Parsed JSON dict or None if extraction failed
        - is_valid: True if output passes full schema validation  
        - errors: List of validation error messages
    """
    # Extract JSON from text
    json_str = extract_json_from_text(raw_text)
    if json_str is None:
        return None, False, ["No valid JSON object found in output"]
    
    # Parse JSON
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as e:
        return None, False, [f"JSON parse error: {e}"]
    
    # Validate against schema  
    result = validate_judge_output(parsed, strict=strict)
    
    return parsed, result.is_valid, result.errors
