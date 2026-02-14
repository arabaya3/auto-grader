"""
Dataset Cleaning Pipeline for Auto-Grader Judge Model.

Performs:
1. Exact duplicate removal (hash-based)
2. Near-duplicate detection (sentence-transformers embeddings)
3. Cross-split leakage removal
4. Clean dataset export

Usage:
    python -m src.data.clean_dataset
    python -m src.data.clean_dataset --similarity_threshold 0.90
    python -m src.data.clean_dataset --skip_embeddings  # Fast mode, exact only
"""

import argparse
import hashlib
import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# Optional: sentence-transformers for near-duplicate detection
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class CleaningStats:
    """Statistics from the cleaning process."""
    original_count: int = 0
    exact_duplicates_removed: int = 0
    near_duplicates_removed: int = 0
    leakage_removed: int = 0
    final_count: int = 0
    
    def __str__(self) -> str:
        return (
            f"Original: {self.original_count} | "
            f"Exact dups: -{self.exact_duplicates_removed} | "
            f"Near dups: -{self.near_duplicates_removed} | "
            f"Leakage: -{self.leakage_removed} | "
            f"Final: {self.final_count}"
        )


@dataclass
class CleaningConfig:
    """Configuration for dataset cleaning."""
    data_dir: Path = field(default_factory=lambda: Path("data"))
    output_dir: Path = field(default_factory=lambda: Path("data_clean"))
    similarity_threshold: float = 0.92
    embedding_model: str = "all-MiniLM-L6-v2"
    batch_size: int = 64
    seed: int = 42
    skip_embeddings: bool = False


# =============================================================================
# Hashing & Exact Duplicate Detection
# =============================================================================

def compute_content_hash(example: dict) -> str:
    """Compute deterministic hash for duplicate detection.
    
    Hash is based on: prompt + response + rubric.title
    """
    prompt = example.get("prompt", "")
    response = example.get("response", "")
    rubric_title = example.get("rubric", {}).get("title", "")
    
    content = f"{prompt}|||{response}|||{rubric_title}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def remove_exact_duplicates(
    examples: List[dict],
    seen_hashes: Optional[Set[str]] = None,
) -> Tuple[List[dict], int, Set[str]]:
    """Remove exact duplicates based on content hash.
    
    Args:
        examples: List of examples to deduplicate.
        seen_hashes: Optional set of already-seen hashes (for cross-split).
    
    Returns:
        Tuple of (deduplicated examples, count removed, updated hash set)
    """
    if seen_hashes is None:
        seen_hashes = set()
    
    unique_examples = []
    duplicates_removed = 0
    
    for example in examples:
        content_hash = compute_content_hash(example)
        
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_examples.append(example)
        else:
            duplicates_removed += 1
    
    return unique_examples, duplicates_removed, seen_hashes


# =============================================================================
# Near-Duplicate Detection (Embedding-based)
# =============================================================================

def get_text_for_embedding(example: dict) -> str:
    """Extract text to embed for similarity comparison."""
    prompt = example.get("prompt", "")
    response = example.get("response", "")
    return f"{prompt} [SEP] {response}"


def compute_embeddings(
    examples: List[dict],
    model: "SentenceTransformer",
    batch_size: int = 64,
) -> np.ndarray:
    """Compute embeddings for all examples."""
    texts = [get_text_for_embedding(ex) for ex in examples]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # For cosine similarity via dot product
    )
    return embeddings


def find_near_duplicates(
    embeddings: np.ndarray,
    threshold: float = 0.92,
) -> Set[int]:
    """Find indices of near-duplicate examples to remove.
    
    Uses efficient batch cosine similarity computation.
    When duplicates are found, keeps the first occurrence.
    
    Args:
        embeddings: Normalized embeddings (N x D).
        threshold: Cosine similarity threshold for duplicates.
    
    Returns:
        Set of indices to remove.
    """
    n = len(embeddings)
    to_remove = set()
    
    # Process in chunks to avoid memory issues
    chunk_size = 1000
    
    for i in range(0, n, chunk_size):
        end_i = min(i + chunk_size, n)
        chunk = embeddings[i:end_i]
        
        # Compute similarity with all embeddings after this chunk
        for j in range(i, n, chunk_size):
            end_j = min(j + chunk_size, n)
            other_chunk = embeddings[j:end_j]
            
            # Cosine similarity (embeddings are normalized)
            similarities = np.dot(chunk, other_chunk.T)
            
            # Find pairs above threshold
            for ci in range(len(chunk)):
                global_i = i + ci
                if global_i in to_remove:
                    continue
                    
                for cj in range(len(other_chunk)):
                    global_j = j + cj
                    
                    # Skip self-comparison and already-removed
                    if global_i >= global_j or global_j in to_remove:
                        continue
                    
                    if similarities[ci, cj] > threshold:
                        # Mark the later one for removal
                        to_remove.add(global_j)
    
    return to_remove


def remove_near_duplicates(
    examples: List[dict],
    model: "SentenceTransformer",
    threshold: float = 0.92,
    batch_size: int = 64,
) -> Tuple[List[dict], int]:
    """Remove near-duplicate examples using embedding similarity.
    
    Args:
        examples: List of examples.
        model: Sentence transformer model.
        threshold: Similarity threshold (0-1).
        batch_size: Batch size for encoding.
    
    Returns:
        Tuple of (deduplicated examples, count removed)
    """
    if len(examples) < 2:
        return examples, 0
    
    logger.info(f"Computing embeddings for {len(examples)} examples...")
    embeddings = compute_embeddings(examples, model, batch_size)
    
    logger.info(f"Finding near-duplicates (threshold={threshold})...")
    to_remove = find_near_duplicates(embeddings, threshold)
    
    # Keep examples not marked for removal
    clean_examples = [
        ex for i, ex in enumerate(examples) 
        if i not in to_remove
    ]
    
    return clean_examples, len(to_remove)


# =============================================================================
# Cross-Split Leakage Detection
# =============================================================================

def remove_cross_split_leakage(
    train: List[dict],
    valid: List[dict],
    test: List[dict],
) -> Tuple[List[dict], List[dict], List[dict], Dict[str, int]]:
    """Remove examples that appear in multiple splits.
    
    Priority: train > valid > test
    If an example appears in multiple splits, keep it in the higher-priority split.
    
    Args:
        train: Training examples.
        valid: Validation examples.
        test: Test examples.
    
    Returns:
        Tuple of (clean_train, clean_valid, clean_test, leakage_counts)
    """
    leakage_counts = {"valid": 0, "test": 0}
    
    # Build hash set from train
    train_hashes = {compute_content_hash(ex) for ex in train}
    
    # Remove from valid if in train
    clean_valid = []
    for ex in valid:
        if compute_content_hash(ex) not in train_hashes:
            clean_valid.append(ex)
        else:
            leakage_counts["valid"] += 1
    
    # Build combined hash set (train + valid)
    train_valid_hashes = train_hashes | {compute_content_hash(ex) for ex in clean_valid}
    
    # Remove from test if in train or valid
    clean_test = []
    for ex in test:
        if compute_content_hash(ex) not in train_valid_hashes:
            clean_test.append(ex)
        else:
            leakage_counts["test"] += 1
    
    return train, clean_valid, clean_test, leakage_counts


# =============================================================================
# Main Pipeline
# =============================================================================

def load_jsonl(filepath: Path) -> List[dict]:
    """Load examples from JSONL file."""
    if not filepath.exists():
        logger.warning(f"File not found: {filepath}")
        return []
    
    examples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def save_jsonl(examples: List[dict], filepath: Path) -> None:
    """Save examples to JSONL file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    
    logger.info(f"Saved {len(examples)} examples to {filepath}")


def clean_single_dataset(
    examples: List[dict],
    name: str,
    model: Optional["SentenceTransformer"],
    config: CleaningConfig,
    global_hashes: Set[str],
) -> Tuple[List[dict], CleaningStats, Set[str]]:
    """Clean a single dataset file.
    
    Args:
        examples: Raw examples.
        name: Dataset name for logging.
        model: Sentence transformer model (None to skip near-dup).
        config: Cleaning configuration.
        global_hashes: Set of hashes seen across all datasets.
    
    Returns:
        Tuple of (clean examples, stats, updated global hashes)
    """
    stats = CleaningStats(original_count=len(examples))
    
    # Step 1: Exact duplicate removal (within dataset)
    logger.info(f"[{name}] Removing exact duplicates...")
    examples, exact_removed, local_hashes = remove_exact_duplicates(examples)
    stats.exact_duplicates_removed = exact_removed
    
    # Step 2: Near-duplicate removal
    if model is not None and not config.skip_embeddings:
        logger.info(f"[{name}] Removing near-duplicates...")
        examples, near_removed = remove_near_duplicates(
            examples, model, config.similarity_threshold, config.batch_size
        )
        stats.near_duplicates_removed = near_removed
    
    # Step 3: Cross-file exact duplicate check
    clean_examples = []
    for ex in examples:
        h = compute_content_hash(ex)
        if h not in global_hashes:
            global_hashes.add(h)
            clean_examples.append(ex)
        else:
            stats.leakage_removed += 1
    
    stats.final_count = len(clean_examples)
    
    return clean_examples, stats, global_hashes


def run_cleaning_pipeline(config: CleaningConfig) -> Dict[str, CleaningStats]:
    """Run the full dataset cleaning pipeline.
    
    Args:
        config: Cleaning configuration.
    
    Returns:
        Dictionary mapping dataset name to cleaning stats.
    """
    random.seed(config.seed)
    np.random.seed(config.seed)
    
    logger.info("=" * 60)
    logger.info("DATASET CLEANING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Data directory: {config.data_dir}")
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"Similarity threshold: {config.similarity_threshold}")
    logger.info(f"Skip embeddings: {config.skip_embeddings}")
    
    # Load sentence transformer model
    model = None
    if not config.skip_embeddings:
        if HAS_SENTENCE_TRANSFORMERS:
            logger.info(f"Loading embedding model: {config.embedding_model}")
            model = SentenceTransformer(config.embedding_model)
        else:
            logger.warning("sentence-transformers not installed, skipping near-duplicate detection")
            logger.warning("Install with: pip install sentence-transformers")
    
    # Define dataset files to process
    # Order matters: train first, then valid, then test (for leakage priority)
    datasets = [
        ("train", "train.jsonl", "train_clean.jsonl"),
        ("valid", "valid.jsonl", "valid_clean.jsonl"),
        ("test", "test.jsonl", "test_clean.jsonl"),
    ]
    
    # Optional datasets (cleaned independently, no cross-leakage check)
    optional_datasets = [
        ("calibration", "calibration.jsonl", "calibration_clean.jsonl"),
        ("adversarial", "adversarial.jsonl", "adversarial_clean.jsonl"),
    ]
    
    all_stats = {}
    global_hashes: Set[str] = set()  # Track hashes across train/valid/test
    
    # Process main datasets with cross-leakage checking
    logger.info("\n" + "-" * 40)
    logger.info("Processing main datasets (train/valid/test)...")
    logger.info("-" * 40)
    
    for name, input_file, output_file in datasets:
        input_path = config.data_dir / input_file
        output_path = config.output_dir / output_file
        
        examples = load_jsonl(input_path)
        if not examples:
            logger.warning(f"[{name}] No examples found, skipping")
            continue
        
        clean_examples, stats, global_hashes = clean_single_dataset(
            examples, name, model, config, global_hashes
        )
        
        save_jsonl(clean_examples, output_path)
        all_stats[name] = stats
        logger.info(f"[{name}] {stats}")
    
    # Process optional datasets (no cross-leakage with main)
    logger.info("\n" + "-" * 40)
    logger.info("Processing optional datasets...")
    logger.info("-" * 40)
    
    for name, input_file, output_file in optional_datasets:
        input_path = config.data_dir / input_file
        output_path = config.output_dir / output_file
        
        examples = load_jsonl(input_path)
        if not examples:
            logger.info(f"[{name}] Not found, skipping")
            continue
        
        # Clean independently (fresh hash set for each)
        clean_examples, stats, _ = clean_single_dataset(
            examples, name, model, config, set()
        )
        
        save_jsonl(clean_examples, output_path)
        all_stats[name] = stats
        logger.info(f"[{name}] {stats}")
    
    return all_stats


def print_summary(stats: Dict[str, CleaningStats]) -> None:
    """Print cleaning summary table."""
    print("\n" + "=" * 80)
    print("CLEANING SUMMARY")
    print("=" * 80)
    
    header = f"{'Dataset':<15} {'Original':>10} {'Exact Dup':>10} {'Near Dup':>10} {'Leakage':>10} {'Final':>10}"
    print(header)
    print("-" * 80)
    
    total_original = 0
    total_exact = 0
    total_near = 0
    total_leak = 0
    total_final = 0
    
    for name, s in stats.items():
        row = f"{name:<15} {s.original_count:>10} {s.exact_duplicates_removed:>10} {s.near_duplicates_removed:>10} {s.leakage_removed:>10} {s.final_count:>10}"
        print(row)
        
        total_original += s.original_count
        total_exact += s.exact_duplicates_removed
        total_near += s.near_duplicates_removed
        total_leak += s.leakage_removed
        total_final += s.final_count
    
    print("-" * 80)
    print(f"{'TOTAL':<15} {total_original:>10} {total_exact:>10} {total_near:>10} {total_leak:>10} {total_final:>10}")
    print("=" * 80)
    
    removed = total_original - total_final
    pct = (removed / total_original * 100) if total_original > 0 else 0
    print(f"\nTotal removed: {removed} ({pct:.1f}%)")
    print(f"Clean datasets saved to: data_clean/")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Clean dataset by removing duplicates and cross-split leakage."
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data"),
        help="Input data directory",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data_clean"),
        help="Output directory for cleaned datasets",
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.92,
        help="Cosine similarity threshold for near-duplicate detection (default: 0.92)",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence transformer model for embeddings",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for embedding computation",
    )
    parser.add_argument(
        "--skip_embeddings",
        action="store_true",
        help="Skip near-duplicate detection (faster, exact duplicates only)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    
    args = parser.parse_args()
    
    config = CleaningConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        similarity_threshold=args.similarity_threshold,
        embedding_model=args.embedding_model,
        batch_size=args.batch_size,
        skip_embeddings=args.skip_embeddings,
        seed=args.seed,
    )
    
    stats = run_cleaning_pipeline(config)
    print_summary(stats)


if __name__ == "__main__":
    main()
