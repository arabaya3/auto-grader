"""Evaluation utilities for Auto-Grader Judge Model."""

from .eval_gold import (
    evaluate_gold_tests,
    run_evaluation,
)

__all__ = [
    "evaluate_gold_tests",
    "run_evaluation",
]
