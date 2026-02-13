"""
Calibration Training Script for Judge Model.

Continues SFT training from existing adapter using calibration dataset
to improve score accuracy on borderline cases (scores 2-5).

Usage:
    python -m src.training.sft_calibration_train \
        --adapter_path outputs/judge_sft_lora/final_adapter \
        --calibration_file data/calibration.jsonl \
        --output_dir outputs/judge_sft_lora_calibrated
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import PeftModel, LoraConfig, get_peft_model

# Suppress verbose logging
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)

# Check for bitsandbytes
try:
    import bitsandbytes
    from transformers import BitsAndBytesConfig
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

# TRL imports
try:
    from trl import SFTTrainer, SFTConfig
    HAS_TRL = True
except ImportError:
    HAS_TRL = False
    print("ERROR: TRL not installed. Run: pip install trl")
    sys.exit(1)

# Import from parent module
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.prompt_templates import JUDGE_SYSTEM_PROMPT


@dataclass
class CalibrationConfig:
    """Configuration for calibration training."""
    # Model
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    adapter_path: str = "outputs/judge_sft_lora/final_adapter"
    
    # Training - Conservative for calibration
    num_train_epochs: int = 1
    learning_rate: float = 5e-5  # Lower LR for fine-tuning
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 1024
    
    # Regularization
    weight_decay: float = 0.05
    warmup_ratio: float = 0.1
    
    # Checkpointing
    logging_steps: int = 5
    save_steps: int = 50
    
    # Output
    output_dir: str = "outputs/judge_sft_lora_calibrated"
    seed: int = 42
    
    # Quantization
    use_4bit: bool = True


def format_example_for_training(example: dict) -> str:
    """Format a calibration example for training."""
    prompt = example["prompt"]
    response = example["response"]
    rubric = example["rubric"]
    label = example["label"]
    
    # Build rubric text
    rubric_items = "\n".join([
        f"- {item['name']}: {item['description']} (weight: {item['weight']})"
        for item in rubric["items"]
    ])
    
    scoring_guide = "\n".join([
        f"  {score}: {desc}"
        for score, desc in rubric["scoring_guide"].items()
    ])
    
    # User message with full context
    user_content = f"""Evaluate this response:

**User Prompt:** {prompt}

**Assistant Response:** {response}

**Rubric: {rubric['title']}**
{rubric_items}

**Scoring Guide:**
{scoring_guide}

Provide your evaluation as JSON with: reasoning, score (1-5), and flags."""

    # Label as JSON
    label_json = json.dumps({
        "reasoning": label["reasoning"],
        "score": label["score"],
        "flags": label.get("flags", {})
    }, indent=2)
    
    # Format as chat
    formatted = f"""<|im_start|>system
{JUDGE_SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
{user_content}<|im_end|>
<|im_start|>assistant
{label_json}<|im_end|>"""
    
    return formatted


def load_calibration_data(file_path: str) -> list:
    """Load calibration dataset."""
    examples = []
    with open(file_path) as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def create_calibration_dataset(
    examples: list,
    tokenizer,
    seed: int = 42,
) -> Dataset:
    """Create HuggingFace Dataset for calibration training."""
    formatted_examples = []
    
    for example in examples:
        text = format_example_for_training(example)
        formatted_examples.append({"text": text})
    
    dataset = Dataset.from_list(formatted_examples)
    
    # Add EOS token
    def add_eos(example):
        if not example["text"].endswith(tokenizer.eos_token):
            example["text"] = example["text"] + tokenizer.eos_token
        return example
    
    dataset = dataset.map(add_eos)
    
    return dataset


def load_model_with_adapter(
    config: CalibrationConfig
) -> tuple:
    """Load base model with existing LoRA adapter."""
    print(f"Loading base model: {config.base_model}")
    print(f"Loading adapter from: {config.adapter_path}")
    
    # Quantization config
    quantization_config = None
    if config.use_4bit and HAS_BNB:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model,
        trust_remote_code=True,
        padding_side="right",
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load existing adapter
    if Path(config.adapter_path).exists():
        print("Loading existing LoRA adapter...")
        model = PeftModel.from_pretrained(
            model,
            config.adapter_path,
            is_trainable=True,  # Enable training
        )
        print("Adapter loaded and set to trainable")
    else:
        print(f"WARNING: Adapter path not found: {config.adapter_path}")
        print("Training from base model (no adapter)")
    
    # Print trainable params
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    return model, tokenizer


def calibration_train(
    config: CalibrationConfig,
    calibration_file: str,
) -> str:
    """Run calibration training."""
    
    # Load model with existing adapter
    model, tokenizer = load_model_with_adapter(config)
    
    # Load and prepare data
    print(f"\nLoading calibration data from: {calibration_file}")
    examples = load_calibration_data(calibration_file)
    print(f"Calibration examples: {len(examples)}")
    
    train_dataset = create_calibration_dataset(examples, tokenizer, config.seed)
    
    # Output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training arguments with TRL 0.8+ API
    import trl
    trl_version = tuple(int(x) for x in trl.__version__.split('.')[:2])
    
    if trl_version >= (0, 8):
        training_args = SFTConfig(
            output_dir=str(output_dir),
            num_train_epochs=config.num_train_epochs,
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            warmup_ratio=config.warmup_ratio,
            weight_decay=config.weight_decay,
            logging_steps=config.logging_steps,
            save_steps=config.save_steps,
            save_total_limit=2,
            report_to="none",
            bf16=True,
            fp16=False,
            gradient_checkpointing=True,
            optim="adamw_torch",
            seed=config.seed,
            max_length=config.max_seq_length,
            dataset_text_field="text",
        )
        
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
        )
    else:
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=config.num_train_epochs,
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            warmup_ratio=config.warmup_ratio,
            weight_decay=config.weight_decay,
            logging_steps=config.logging_steps,
            save_steps=config.save_steps,
            save_total_limit=2,
            report_to="none",
            bf16=True,
            fp16=False,
            gradient_checkpointing=True,
            optim="adamw_torch",
            seed=config.seed,
        )
        
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            max_seq_length=config.max_seq_length,
        )
    
    # Train
    print("\n" + "=" * 60)
    print("Starting Calibration Training")
    print("=" * 60)
    
    trainer.train()
    
    # Save final adapter
    final_adapter_path = output_dir / "final_adapter"
    print(f"\nSaving calibrated adapter to: {final_adapter_path}")
    
    # Save the adapter weights
    model.save_pretrained(final_adapter_path)
    tokenizer.save_pretrained(final_adapter_path)
    
    # Save config
    config_dict = {
        "base_model": config.base_model,
        "source_adapter": config.adapter_path,
        "calibration_file": calibration_file,
        "num_examples": len(examples),
        "num_epochs": config.num_train_epochs,
        "learning_rate": config.learning_rate,
        "seed": config.seed,
    }
    
    with open(output_dir / "calibration_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Calibration Training Complete!")
    print("=" * 60)
    print(f"Adapter saved to: {final_adapter_path}")
    
    return str(final_adapter_path)


def main():
    parser = argparse.ArgumentParser(description="Calibration training for Judge Model")
    
    # Required
    parser.add_argument("--calibration_file", "--calibration", "-c",
                       type=str, default="data/calibration.jsonl",
                       help="Path to calibration dataset")
    
    # Model
    parser.add_argument("--base_model", "--model", "-m",
                       type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                       help="Base model name")
    parser.add_argument("--adapter_path", "--adapter", "-a",
                       type=str, default="outputs/judge_sft_lora/final_adapter",
                       help="Path to existing LoRA adapter")
    
    # Output
    parser.add_argument("--output_dir", "--output", "-o",
                       type=str, default="outputs/judge_sft_lora_calibrated",
                       help="Output directory for calibrated adapter")
    
    # Training
    parser.add_argument("--epochs", type=int, default=1,
                       help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Batch size")
    
    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_4bit", action="store_true", help="Disable 4-bit quantization")
    
    args = parser.parse_args()
    
    # Build config
    config = CalibrationConfig(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        seed=args.seed,
        use_4bit=not args.no_4bit,
    )
    
    # Check files
    if not Path(args.calibration_file).exists():
        print(f"ERROR: Calibration file not found: {args.calibration_file}")
        print("Run: python -m src.data.build_calibration_dataset")
        sys.exit(1)
    
    # Run calibration training
    calibration_train(config, args.calibration_file)


if __name__ == "__main__":
    main()
