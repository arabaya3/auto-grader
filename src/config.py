"""Configuration dataclasses for the Auto-Grader Judge Model."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for the judge model."""
    
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    # 4-bit quantization for Colab T4 compatibility
    load_in_4bit: bool = True
    # BitsAndBytes config for quantization
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    # Device mapping
    device_map: str = "auto"
    # Trust remote code for Qwen models
    trust_remote_code: bool = True


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    
    max_new_tokens: int = 512
    temperature: float = 0.1  # Low temp for deterministic JSON
    top_p: float = 0.9
    do_sample: bool = True
    # Repetition penalty to avoid loops
    repetition_penalty: float = 1.1
    # Stop generation on these tokens
    stop_strings: list[str] = field(default_factory=lambda: ["```", "\n\n\n"])


@dataclass
class JudgeConfig:
    """Combined configuration for the judge system."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    # Reproducibility
    seed: int = 42
    # Logging
    log_level: str = "INFO"
    log_to_stderr: bool = True
    # Validation
    strict_json_validation: bool = True


def get_default_config() -> JudgeConfig:
    """Get default configuration."""
    return JudgeConfig()


def get_colab_t4_config() -> JudgeConfig:
    """Get configuration optimized for Google Colab T4."""
    return JudgeConfig(
        model=ModelConfig(
            load_in_4bit=True,
            device_map="auto",
        ),
        generation=GenerationConfig(
            max_new_tokens=512,
            temperature=0.1,
        ),
    )
