"""Utility functions for reproducibility, logging, and helpers."""

import logging
import random
import sys
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logger(
    name: str = "auto_grader",
    level: str = "INFO",
    to_stderr: bool = True,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Configure logger that outputs to stderr (not stdout) to keep stdout clean for JSON.
    
    Args:
        name: Logger name.
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        to_stderr: If True, log to stderr instead of stdout.
        log_file: Optional file path for logging.
    
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Format with timestamp
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    if to_stderr:
        # Use stderr to keep stdout clean for JSON output
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_device() -> str:
    """Get the best available device for inference.
    
    Returns:
        Device string ('cuda' or 'cpu').
    """
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_gpu_memory_info() -> dict:
    """Get GPU memory information if available.
    
    Returns:
        Dictionary with memory info or empty dict if no GPU.
    """
    if not torch.cuda.is_available():
        return {}
    
    return {
        "device_name": torch.cuda.get_device_name(0),
        "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
        "allocated_memory_gb": torch.cuda.memory_allocated(0) / 1e9,
        "cached_memory_gb": torch.cuda.memory_reserved(0) / 1e9,
    }
