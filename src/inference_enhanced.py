"""
Enhanced Inference Module with Winning Features.

Adds:
1. Confidence scoring (0.0-1.0)
2. Explainability mode (detailed reasoning breakdown)
3. Strict JSON enforcement with retry logic
4. Batch inference support

Usage:
    from src.inference_enhanced import EnhancedJudge
    
    judge = EnhancedJudge(adapter_path="outputs/judge_sft_lora/final_adapter")
    judge.load_model()
    
    # With confidence
    result = judge.judge_with_confidence(prompt, response, rubric)
    print(f"Score: {result['score']}, Confidence: {result['confidence']:.2f}")
    
    # With explanation
    result = judge.judge_with_explanation(prompt, response, rubric)
    print(f"Score Breakdown: {result['explanation']}")
"""

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Try bitsandbytes
try:
    from transformers import BitsAndBytesConfig
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

# Try PEFT
try:
    from peft import PeftModel
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.prompt_templates import JUDGE_SYSTEM_PROMPT


@dataclass
class JudgmentResult:
    """Enhanced judgment result with confidence and explanation."""
    # Core output
    raw_response: str
    parsed: Optional[dict]
    json_valid: bool
    
    # Judgment values
    score: Optional[int]
    reasoning: Optional[str]
    flags: Optional[dict]
    
    # Enhanced features
    confidence: float = 0.0
    explanation: Optional[dict] = None
    retry_count: int = 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "score": self.score,
            "reasoning": self.reasoning,
            "flags": self.flags,
            "json_valid": self.json_valid,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "retry_count": self.retry_count,
        }


class EnhancedJudge:
    """Enhanced Judge Model with confidence, explainability, and strict JSON."""
    
    DEFAULT_RUBRIC = {
        "title": "General Quality Assessment",
        "items": [
            {"name": "Accuracy", "description": "Information is factually correct", "weight": 0.35},
            {"name": "Helpfulness", "description": "Addresses the user's request", "weight": 0.30},
            {"name": "Clarity", "description": "Response is clear and well-structured", "weight": 0.20},
            {"name": "Safety", "description": "Response is safe and appropriate", "weight": 0.15},
        ],
        "scoring_guide": {
            "1": "Very poor - major issues",
            "2": "Below average - significant problems",
            "3": "Average - acceptable but with room for improvement",
            "4": "Good - minor issues only",
            "5": "Excellent - meets all criteria",
        }
    }
    
    def __init__(
        self,
        base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        adapter_path: Optional[str] = None,
        use_4bit: bool = True,
        max_retries: int = 3,
    ):
        """Initialize enhanced judge.
        
        Args:
            base_model: Base model name or path
            adapter_path: Path to LoRA adapter (optional)
            use_4bit: Use 4-bit quantization
            max_retries: Max retries for JSON enforcement
        """
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.use_4bit = use_4bit and HAS_BNB
        self.max_retries = max_retries
        
        self.model = None
        self.tokenizer = None
        self._loaded = False
    
    def load_model(self) -> None:
        """Load model and tokenizer."""
        if self._loaded:
            return
        
        print(f"Loading model: {self.base_model}")
        
        # Quantization config
        quantization_config = None
        if self.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            trust_remote_code=True,
            padding_side="left",
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load adapter if provided
        if self.adapter_path and HAS_PEFT:
            if Path(self.adapter_path).exists():
                print(f"Loading adapter: {self.adapter_path}")
                self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
        
        self.model.eval()
        self._loaded = True
        print("Model ready!")
    
    def _build_prompt(
        self,
        user_prompt: str,
        response: str,
        rubric: Optional[dict] = None,
    ) -> str:
        """Build evaluation prompt."""
        rubric = rubric or self.DEFAULT_RUBRIC
        
        rubric_items = "\n".join([
            f"- {item['name']}: {item['description']} (weight: {item['weight']})"
            for item in rubric["items"]
        ])
        
        scoring_guide = "\n".join([
            f"  {score}: {desc}"
            for score, desc in rubric["scoring_guide"].items()
        ])
        
        return f"""Evaluate this response:

**User Prompt:** {user_prompt}

**Assistant Response:** {response}

**Rubric: {rubric['title']}**
{rubric_items}

**Scoring Guide:**
{scoring_guide}

Provide your evaluation as JSON with: reasoning, score (1-5), and flags."""
    
    def _generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate response from model."""
        if not self._loaded:
            self.load_model()
        
        messages = [
            {"role": "system", "content": system_prompt or JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        return response
    
    def _parse_json(self, text: str) -> Optional[dict]:
        """Parse JSON from text with fallback extraction."""
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON object
        json_patterns = [
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Nested objects
            r'```json\s*([\s\S]*?)\s*```',        # Markdown code block
            r'```\s*([\s\S]*?)\s*```',            # Generic code block
        ]
        
        for pattern in json_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    json_str = match.group(1) if match.lastindex else match.group()
                    return json.loads(json_str)
                except (json.JSONDecodeError, IndexError):
                    continue
        
        return None
    
    def _compute_confidence(
        self,
        parsed: Optional[dict],
        raw_response: str,
        retry_count: int,
    ) -> float:
        """Compute confidence score (0.0-1.0)."""
        if not parsed:
            return 0.0
        
        confidence = 0.3  # Base confidence for valid JSON
        
        # Score is valid integer 1-5
        score = parsed.get("score")
        if isinstance(score, int) and 1 <= score <= 5:
            confidence += 0.25
        
        # Has meaningful reasoning
        reasoning = parsed.get("reasoning", "")
        if len(reasoning) > 30:
            confidence += 0.15
        if len(reasoning) > 80:
            confidence += 0.1
        
        # Has flags dictionary
        flags = parsed.get("flags")
        if isinstance(flags, dict):
            confidence += 0.1
        
        # Response structure indicates certainty
        if any(word in reasoning.lower() for word in ["clearly", "definitely", "obviously"]):
            confidence += 0.05
        
        # Penalize retries (less confident if needed retries)
        confidence -= 0.1 * retry_count
        
        return max(0.0, min(1.0, confidence))
    
    def _generate_explanation(
        self,
        parsed: dict,
        rubric: dict,
    ) -> dict:
        """Generate detailed score explanation."""
        if not parsed:
            return {"error": "No valid JSON to explain"}
        
        score = parsed.get("score", 0)
        reasoning = parsed.get("reasoning", "")
        flags = parsed.get("flags", {})
        
        explanation = {
            "overall_score": score,
            "score_meaning": rubric["scoring_guide"].get(str(score), "Unknown"),
            "criteria_analysis": [],
            "flag_analysis": [],
            "summary": "",
        }
        
        # Analyze each rubric criterion (based on reasoning)
        for item in rubric["items"]:
            criteria_entry = {
                "criterion": item["name"],
                "weight": item["weight"],
                "description": item["description"],
            }
            
            # Simple heuristic: check if criterion mentioned in reasoning
            mention_words = item["name"].lower().split()
            mentioned = any(word in reasoning.lower() for word in mention_words)
            criteria_entry["mentioned_in_reasoning"] = mentioned
            
            explanation["criteria_analysis"].append(criteria_entry)
        
        # Analyze flags
        for flag_name, flag_value in flags.items():
            if flag_value:
                flag_display = flag_name.replace("_", " ").title()
                explanation["flag_analysis"].append({
                    "flag": flag_display,
                    "triggered": True,
                    "impact": "May indicate quality issue",
                })
        
        # Generate summary
        if score >= 4:
            summary = "Response meets quality standards"
        elif score >= 3:
            summary = "Response is acceptable but has room for improvement"
        else:
            summary = "Response has significant issues to address"
        
        if explanation["flag_analysis"]:
            flags_str = ", ".join([f["flag"] for f in explanation["flag_analysis"]])
            summary += f". Flagged issues: {flags_str}"
        
        explanation["summary"] = summary
        
        return explanation
    
    def judge(
        self,
        user_prompt: str,
        response: str,
        rubric: Optional[dict] = None,
    ) -> JudgmentResult:
        """Run basic judgment without retries."""
        prompt = self._build_prompt(user_prompt, response, rubric)
        raw_response = self._generate(prompt)
        parsed = self._parse_json(raw_response)
        
        return JudgmentResult(
            raw_response=raw_response,
            parsed=parsed,
            json_valid=parsed is not None,
            score=parsed.get("score") if parsed else None,
            reasoning=parsed.get("reasoning") if parsed else None,
            flags=parsed.get("flags") if parsed else None,
        )
    
    def judge_strict(
        self,
        user_prompt: str,
        response: str,
        rubric: Optional[dict] = None,
    ) -> JudgmentResult:
        """Run judgment with strict JSON enforcement (retries)."""
        rubric = rubric or self.DEFAULT_RUBRIC
        retry_count = 0
        
        for attempt in range(self.max_retries + 1):
            prompt = self._build_prompt(user_prompt, response, rubric)
            
            # Add enforcement hint after first failure
            if attempt > 0:
                prompt += f"\n\nIMPORTANT: Your response MUST be valid JSON only. No text before or after the JSON object. Attempt {attempt + 1}/{self.max_retries + 1}."
            
            raw_response = self._generate(prompt, temperature=0.1 + 0.1 * attempt)
            parsed = self._parse_json(raw_response)
            
            if parsed and isinstance(parsed.get("score"), int):
                return JudgmentResult(
                    raw_response=raw_response,
                    parsed=parsed,
                    json_valid=True,
                    score=parsed.get("score"),
                    reasoning=parsed.get("reasoning"),
                    flags=parsed.get("flags"),
                    retry_count=attempt,
                )
            
            retry_count = attempt
        
        # Return last attempt even if failed
        return JudgmentResult(
            raw_response=raw_response,
            parsed=parsed,
            json_valid=parsed is not None,
            score=parsed.get("score") if parsed else None,
            reasoning=parsed.get("reasoning") if parsed else None,
            flags=parsed.get("flags") if parsed else None,
            retry_count=retry_count,
        )
    
    def judge_with_confidence(
        self,
        user_prompt: str,
        response: str,
        rubric: Optional[dict] = None,
    ) -> JudgmentResult:
        """Run judgment with confidence scoring."""
        result = self.judge_strict(user_prompt, response, rubric)
        
        result.confidence = self._compute_confidence(
            result.parsed,
            result.raw_response,
            result.retry_count,
        )
        
        return result
    
    def judge_with_explanation(
        self,
        user_prompt: str,
        response: str,
        rubric: Optional[dict] = None,
    ) -> JudgmentResult:
        """Run judgment with detailed explanation."""
        rubric = rubric or self.DEFAULT_RUBRIC
        result = self.judge_with_confidence(user_prompt, response, rubric)
        
        if result.parsed:
            result.explanation = self._generate_explanation(result.parsed, rubric)
        
        return result
    
    def batch_judge(
        self,
        examples: List[dict],
        with_confidence: bool = True,
        with_explanation: bool = False,
    ) -> List[JudgmentResult]:
        """Run batch judgment on multiple examples.
        
        Args:
            examples: List of dicts with 'prompt', 'response', and optional 'rubric'
            with_confidence: Include confidence scores
            with_explanation: Include detailed explanations
        
        Returns:
            List of JudgmentResult objects
        """
        results = []
        
        for i, example in enumerate(examples):
            print(f"Processing {i+1}/{len(examples)}...")
            
            if with_explanation:
                result = self.judge_with_explanation(
                    example["prompt"],
                    example["response"],
                    example.get("rubric"),
                )
            elif with_confidence:
                result = self.judge_with_confidence(
                    example["prompt"],
                    example["response"],
                    example.get("rubric"),
                )
            else:
                result = self.judge(
                    example["prompt"],
                    example["response"],
                    example.get("rubric"),
                )
            
            results.append(result)
        
        return results


# Quick usage example
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Judge Demo")
    parser.add_argument("--adapter", "-a", type=str, default=None)
    parser.add_argument("--prompt", "-p", type=str, default="What is 2+2?")
    parser.add_argument("--response", "-r", type=str, default="2+2 equals 4.")
    parser.add_argument("--with_explanation", "-e", action="store_true")
    args = parser.parse_args()
    
    judge = EnhancedJudge(adapter_path=args.adapter)
    judge.load_model()
    
    if args.with_explanation:
        result = judge.judge_with_explanation(args.prompt, args.response)
    else:
        result = judge.judge_with_confidence(args.prompt, args.response)
    
    print("\n" + "=" * 50)
    print("JUDGMENT RESULT")
    print("=" * 50)
    print(f"Score: {result.score}/5")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"JSON Valid: {result.json_valid}")
    print(f"Reasoning: {result.reasoning}")
    
    if result.flags:
        active = [k for k, v in result.flags.items() if v]
        print(f"Flags: {active if active else 'None'}")
    
    if result.explanation:
        print(f"\nExplanation: {result.explanation['summary']}")
