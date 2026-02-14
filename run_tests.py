"""Quick test runner that bypasses pytest entry point issues."""
import sys
sys.path.insert(0, '.')

print("=" * 60)
print("Flag Schema Tests")
print("=" * 60)

from tests.test_flag_schema import TestFlagSchema, TestFlagGeneration

t = TestFlagSchema()
print("Testing calibration examples in code...", end=" ")
t.test_calibration_examples_in_code()
print("PASS")

print("Testing adversarial examples in code...", end=" ")
t.test_adversarial_examples_in_code()
print("PASS")

print("Testing io_schema flag match...", end=" ")
t.test_io_schema_required_flags_match()
print("PASS")

t2 = TestFlagGeneration()
print("Testing generate_calibration_example flags...", end=" ")
t2.test_generate_calibration_example_flags()
print("PASS")

print("Testing generate_adversarial_example flags...", end=" ")
t2.test_generate_adversarial_example_flags()
print("PASS")

print("\n" + "=" * 60)
print("Schema Validation Tests")
print("=" * 60)

from tests.test_schema import TestExtractJsonFromText, TestValidateJudgeOutput

t3 = TestExtractJsonFromText()
print("Testing JSON extraction...", end=" ")
t3.test_extract_clean_json()
t3.test_extract_json_with_preamble()
t3.test_extract_json_with_suffix()
t3.test_extract_json_from_markdown_code_block()
t3.test_extract_json_from_plain_code_block()
t3.test_no_json_found()
t3.test_nested_json()
print("PASS (7 tests)")

t4 = TestValidateJudgeOutput()
print("Testing output validation...", end=" ")
t4.test_valid_complete_output()
t4.test_valid_output_as_string()
t4.test_invalid_score_out_of_range_high()
t4.test_invalid_score_out_of_range_low()
t4.test_invalid_score_not_integer()
t4.test_missing_required_key()
t4.test_empty_reasoning()
t4.test_invalid_rubric_item_missing_pass()
t4.test_invalid_flag_not_boolean()
t4.test_missing_flag()
t4.test_strict_mode_rejects_extra_keys()
t4.test_non_strict_mode_allows_extra_keys()
t4.test_invalid_json_string()
print("PASS (13 tests)")

from tests.test_schema import TestCreateEmptyOutput
t5 = TestCreateEmptyOutput()
print("Testing create_empty_output...", end=" ")
t5.test_default_empty_output()
print("PASS")

print("\n" + "=" * 60)
print("ALL TESTS PASSED!")
print("=" * 60)
