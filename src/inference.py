"""Inference module for the Auto-Grader Judge Model.

Loads the model and runs single judgments with JSON validation.
Supports CLI usage for quick evaluations.
"""

import argparse
import json
import sys
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Check if bitsandbytes is actually installed (not just if config class exists)
try:
    import bitsandbytes
    from transformers import BitsAndBytesConfig
    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False
    BitsAndBytesConfig = None

from .config import JudgeConfig, get_default_config
from .io_schema import ValidationResult, validate_judge_output
from .prompt_templates import format_chat_messages
from .utils import get_device, set_seed, setup_logger


class JudgeModel:
    """Wrapper for the judge model with inference capabilities."""
    
    def __init__(
        self,
        config: Optional[JudgeConfig] = None,
        logger_name: str = "auto_grader.inference",
    ):
        """Initialize the judge model.
        
        Args:
            config: Configuration object. Uses defaults if None.
            logger_name: Name for the logger.
        """
        self.config = config or get_default_config()
        self.logger = setup_logger(
            name=logger_name,
            level=self.config.log_level,
            to_stderr=self.config.log_to_stderr,
        )
        
        self.model = None
        self.tokenizer = None
        self._is_loaded = False
        
        # Set seed for reproducibility
        set_seed(self.config.seed)
    
    def load_model(self) -> None:
        """Load the model and tokenizer."""
        if self._is_loaded:
            self.logger.info("Model already loaded, skipping.")
            return
        
        self.logger.info(f"Loading model: {self.config.model.model_name}")
        
        # Configure quantization for 4-bit loading
        quantization_config = None
        use_4bit = self.config.model.load_in_4bit
        
        if use_4bit and not HAS_BITSANDBYTES:
            self.logger.warning(
                "4-bit quantization requested but bitsandbytes not available. "
                "Falling back to FP16. Install bitsandbytes for 4-bit support."
            )
            use_4bit = False
        
        if use_4bit:
            self.logger.info("Configuring 4-bit quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, self.config.model.bnb_4bit_compute_dtype),
                bnb_4bit_quant_type=self.config.model.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.config.model.bnb_4bit_use_double_quant,
            )
        
        # Load tokenizer
        self.logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.model_name,
            trust_remote_code=self.config.model.trust_remote_code,
        )
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.logger.info("Loading model weights...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model.model_name,
            quantization_config=quantization_config,
            device_map=self.config.model.device_map,
            trust_remote_code=self.config.model.trust_remote_code,
            torch_dtype=torch.float16 if not self.config.model.load_in_4bit else None,
        )
        
        self._is_loaded = True
        self.logger.info(f"Model loaded successfully on {get_device()}")
    
    def judge(
        self,
        user_prompt: str,
        candidate_response: str,
        rubric: str,
        additional_context: Optional[str] = None,
        validate: bool = True,
    ) -> tuple[str, Optional[ValidationResult]]:
        """Run a single judgment.
        
        Args:
            user_prompt: The original prompt given to the candidate.
            candidate_response: The response to evaluate.
            rubric: The evaluation criteria.
            additional_context: Optional extra context.
            validate: Whether to validate the output JSON.
        
        Returns:
            Tuple of (raw_output, validation_result).
            validation_result is None if validate=False.
        """
        if not self._is_loaded:
            self.load_model()
        
        # Build chat messages
        messages = format_chat_messages(
            user_prompt=user_prompt,
            candidate_response=candidate_response,
            rubric=rubric,
            additional_context=additional_context,
        )
        
        # Apply chat template
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        self.logger.debug(f"Prompt length: {len(prompt_text)} chars")
        
        # Tokenize
        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(self.model.device)
        
        # Generate
        self.logger.info("Generating judgment...")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.generation.max_new_tokens,
                temperature=self.config.generation.temperature,
                top_p=self.config.generation.top_p,
                do_sample=self.config.generation.do_sample,
                repetition_penalty=self.config.generation.repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode only the generated part
        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        raw_output = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        self.logger.debug(f"Raw output: {raw_output[:200]}...")
        
        # Validate if requested
        validation_result = None
        if validate:
            validation_result = validate_judge_output(
                raw_output,
                strict=self.config.strict_json_validation,
            )
            if validation_result.is_valid:
                self.logger.info("Output validation PASSED")
            else:
                self.logger.warning(f"Output validation FAILED: {validation_result.errors}")
        
        return raw_output, validation_result


def run_cli():
    """Run the judge model from command line."""
    parser = argparse.ArgumentParser(
        description="Auto-Grader Judge Model - Evaluate LLM responses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="The original user prompt",
    )
    parser.add_argument(
        "--response",
        type=str,
        required=True,
        help="The candidate response to evaluate",
    )
    parser.add_argument(
        "--rubric",
        type=str,
        required=True,
        help="The evaluation rubric/criteria",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Model name or path (default: Qwen/Qwen2.5-1.5B-Instruct)",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip JSON validation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging to stderr",
    )
    
    args = parser.parse_args()
    
    # Build config
    config = get_default_config()
    config.model.model_name = args.model
    config.model.load_in_4bit = not args.no_4bit
    config.seed = args.seed
    config.log_level = "DEBUG" if args.verbose else "WARNING"
    
    # Initialize and run
    judge = JudgeModel(config=config)
    raw_output, validation = judge.judge(
        user_prompt=args.prompt,
        candidate_response=args.response,
        rubric=args.rubric,
        validate=not args.no_validate,
    )
    
    # Output JSON to stdout (only JSON, nothing else)
    # Try to output clean JSON if validation passed
    if validation and validation.parsed_output:
        print(json.dumps(validation.parsed_output, indent=2))
    else:
        print(raw_output)
    
    # Print validation status to stderr
    if validation:
        if validation.is_valid:
            print("VALIDATION: PASSED", file=sys.stderr)
            sys.exit(0)
        else:
            print(f"VALIDATION: FAILED - {validation.errors}", file=sys.stderr)
            sys.exit(1)
    else:
        sys.exit(0)


# Entry point for: python -m src.inference
if __name__ == "__main__":
    run_cli()
