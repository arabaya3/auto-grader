#!/usr/bin/env python3
"""
Migrate cleaned dataset files from data_clean/ to data/.

This script safely migrates the cleaned dataset to become the official dataset:
1. Validates source files exist
2. Backs up existing files
3. Copies clean files with renamed names
4. Verifies copy integrity
5. Deletes data_clean directory
6. Runs audit to verify data quality
7. Rolls back on any failure

Usage:
    python -m src.data.migrate_clean_to_data
    python -m src.data.migrate_clean_to_data --dry-run
    python -m src.data.migrate_clean_to_data --skip-audit
"""

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class MigrationError(Exception):
    """Raised when migration fails and rollback is needed."""
    pass


class MigrationManager:
    """Manages safe migration of clean dataset files."""
    
    # File mappings: source (in data_clean) -> destination (in data)
    FILE_MAPPINGS = {
        "train_clean.jsonl": "train.jsonl",
        "valid_clean.jsonl": "valid.jsonl",
        "test_clean.jsonl": "test.jsonl",
    }
    
    def __init__(self, project_root: Path, dry_run: bool = False):
        self.project_root = project_root
        self.data_clean_dir = project_root / "data_clean"
        self.data_dir = project_root / "data"
        self.dry_run = dry_run
        self.backups_created: list[tuple[Path, Path]] = []
        self.files_copied: list[Path] = []
    
    def run(self, skip_audit: bool = False) -> bool:
        """Execute the full migration process."""
        try:
            logger.info("=" * 60)
            logger.info("DATASET MIGRATION: data_clean/ -> data/")
            logger.info("=" * 60)
            
            if self.dry_run:
                logger.info("DRY RUN MODE - No changes will be made")
            
            # Step 1: Validate source files
            self._validate_source_files()
            
            # Step 2: Create backups
            self._create_backups()
            
            # Step 3: Copy files
            self._copy_files()
            
            # Step 4: Verify copies
            self._verify_copies()
            
            # Step 5: Delete data_clean directory
            self._delete_data_clean()
            
            # Step 6: Run audit (optional)
            if not skip_audit:
                self._run_audit()
            
            # Step 7: Print final confirmation
            self._print_confirmation()
            
            logger.info("=" * 60)
            logger.info("MIGRATION COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            return True
            
        except MigrationError as e:
            logger.error(f"Migration failed: {e}")
            self._rollback()
            return False
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            self._rollback()
            raise
    
    def _validate_source_files(self) -> None:
        """Step 1: Validate that all required source files exist."""
        logger.info("\n[Step 1/6] Validating source files...")
        
        if not self.data_clean_dir.exists():
            raise MigrationError(f"Source directory not found: {self.data_clean_dir}")
        
        missing = []
        for src_name in self.FILE_MAPPINGS.keys():
            src_path = self.data_clean_dir / src_name
            if not src_path.exists():
                missing.append(src_name)
            else:
                lines = sum(1 for _ in open(src_path, encoding="utf-8"))
                logger.info(f"  ✓ {src_name}: {lines} lines")
        
        if missing:
            raise MigrationError(
                f"Missing required files in data_clean/: {', '.join(missing)}"
            )
        
        logger.info("  All source files validated.")
    
    def _create_backups(self) -> None:
        """Step 2: Backup existing files in data/ directory."""
        logger.info("\n[Step 2/6] Creating backups of existing files...")
        
        if not self.data_dir.exists():
            logger.info("  data/ directory does not exist, creating it.")
            if not self.dry_run:
                self.data_dir.mkdir(parents=True, exist_ok=True)
            return
        
        for _, dest_name in self.FILE_MAPPINGS.items():
            dest_path = self.data_dir / dest_name
            if dest_path.exists():
                backup_path = self.data_dir / f"{dest_path.stem}_backup{dest_path.suffix}"
                logger.info(f"  Backing up: {dest_name} -> {backup_path.name}")
                
                if not self.dry_run:
                    shutil.copy2(dest_path, backup_path)
                    self.backups_created.append((dest_path, backup_path))
            else:
                logger.info(f"  Skip backup: {dest_name} (does not exist)")
        
        if self.backups_created:
            logger.info(f"  Created {len(self.backups_created)} backup(s).")
        else:
            logger.info("  No existing files to backup.")
    
    def _copy_files(self) -> None:
        """Step 3: Copy clean files to data/ directory."""
        logger.info("\n[Step 3/6] Copying clean files to data/...")
        
        for src_name, dest_name in self.FILE_MAPPINGS.items():
            src_path = self.data_clean_dir / src_name
            dest_path = self.data_dir / dest_name
            
            logger.info(f"  Copying: {src_name} -> {dest_name}")
            
            if not self.dry_run:
                shutil.copy2(src_path, dest_path)
                self.files_copied.append(dest_path)
        
        logger.info(f"  Copied {len(self.FILE_MAPPINGS)} file(s).")
    
    def _verify_copies(self) -> None:
        """Step 4: Verify that copies match source files."""
        logger.info("\n[Step 4/6] Verifying copy integrity...")
        
        if self.dry_run:
            logger.info("  Skipping verification in dry-run mode.")
            return
        
        for src_name, dest_name in self.FILE_MAPPINGS.items():
            src_path = self.data_clean_dir / src_name
            dest_path = self.data_dir / dest_name
            
            # Compare file sizes
            src_size = src_path.stat().st_size
            dest_size = dest_path.stat().st_size
            
            if src_size != dest_size:
                raise MigrationError(
                    f"Size mismatch for {dest_name}: "
                    f"source={src_size}, dest={dest_size}"
                )
            
            # Compare line counts
            src_lines = sum(1 for _ in open(src_path, encoding="utf-8"))
            dest_lines = sum(1 for _ in open(dest_path, encoding="utf-8"))
            
            if src_lines != dest_lines:
                raise MigrationError(
                    f"Line count mismatch for {dest_name}: "
                    f"source={src_lines}, dest={dest_lines}"
                )
            
            logger.info(f"  ✓ {dest_name}: {dest_lines} lines, {dest_size} bytes")
        
        logger.info("  All copies verified successfully.")
    
    def _delete_data_clean(self) -> None:
        """Step 5: Delete the data_clean directory."""
        logger.info("\n[Step 5/6] Removing data_clean/ directory...")
        
        if self.dry_run:
            logger.info(f"  Would delete: {self.data_clean_dir}")
            return
        
        # List contents before deletion
        contents = list(self.data_clean_dir.iterdir())
        logger.info(f"  Removing {len(contents)} item(s) from data_clean/")
        
        shutil.rmtree(self.data_clean_dir)
        
        if self.data_clean_dir.exists():
            raise MigrationError("Failed to delete data_clean/ directory")
        
        logger.info("  ✓ data_clean/ directory removed.")
    
    def _run_audit(self) -> None:
        """Step 6: Run audit report to verify data quality."""
        logger.info("\n[Step 6/6] Running data audit...")
        
        if self.dry_run:
            logger.info("  Skipping audit in dry-run mode.")
            return
        
        cmd = [
            sys.executable, "-m", "src.data.audit_report",
            "--data_dir", str(self.data_dir),
            "--out", str(self.project_root / "outputs"),
        ]
        
        logger.info(f"  Command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            cwd=self.project_root,
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            logger.error(f"  Audit failed with exit code {result.returncode}")
            if result.stderr:
                logger.error(f"  stderr: {result.stderr[:500]}")
            raise MigrationError("Audit failed - data quality check did not pass")
        
        logger.info("  ✓ Audit passed successfully.")
    
    def _print_confirmation(self) -> None:
        """Print final confirmation with file statistics."""
        logger.info("\n" + "=" * 60)
        logger.info("FINAL DATASET STATISTICS")
        logger.info("=" * 60)
        
        if self.dry_run:
            logger.info("(Dry run - showing expected results)")
            source_dir = self.data_clean_dir
            file_mappings = {src: src for src in self.FILE_MAPPINGS.keys()}
        else:
            source_dir = self.data_dir
            file_mappings = {dest: dest for dest in self.FILE_MAPPINGS.values()}
        
        for filename in file_mappings.values():
            path = source_dir / filename
            if path.exists():
                lines = sum(1 for _ in open(path, encoding="utf-8"))
                size = path.stat().st_size
                logger.info(f"  {filename}: {lines} examples ({size:,} bytes)")
        
        if not self.dry_run:
            logger.info(f"\n  ✓ data_clean/ directory: REMOVED")
            if self.backups_created:
                logger.info(f"  ✓ Backups created: {len(self.backups_created)}")
                for _, backup in self.backups_created:
                    logger.info(f"      - {backup.name}")
    
    def _rollback(self) -> None:
        """Rollback changes on failure."""
        logger.warning("\n[ROLLBACK] Restoring previous state...")
        
        if self.dry_run:
            logger.info("  Dry run - no changes to rollback.")
            return
        
        rollback_errors = []
        
        # Restore backups
        for original, backup in self.backups_created:
            try:
                if backup.exists():
                    logger.info(f"  Restoring: {backup.name} -> {original.name}")
                    shutil.copy2(backup, original)
            except Exception as e:
                rollback_errors.append(f"Failed to restore {original.name}: {e}")
        
        # Remove copied files that shouldn't exist
        for copied in self.files_copied:
            # Only remove if there was no backup (meaning it's a new file)
            was_backed_up = any(orig == copied for orig, _ in self.backups_created)
            if not was_backed_up and copied.exists():
                try:
                    logger.info(f"  Removing new file: {copied.name}")
                    copied.unlink()
                except Exception as e:
                    rollback_errors.append(f"Failed to remove {copied.name}: {e}")
        
        if rollback_errors:
            logger.error("  Rollback encountered errors:")
            for err in rollback_errors:
                logger.error(f"    - {err}")
        else:
            logger.info("  ✓ Rollback completed successfully.")


def find_project_root() -> Path:
    """Find the project root directory."""
    # Start from this file's location
    current = Path(__file__).resolve().parent
    
    # Walk up looking for markers
    while current != current.parent:
        if (current / "src").is_dir() and (current / "data").is_dir():
            return current
        if (current / "pytest.ini").exists():
            return current
        current = current.parent
    
    # Fallback to cwd
    return Path.cwd()


def main():
    parser = argparse.ArgumentParser(
        description="Migrate cleaned dataset from data_clean/ to data/"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--skip-audit",
        action="store_true",
        help="Skip running the audit report after migration",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root directory (auto-detected if not specified)",
    )
    
    args = parser.parse_args()
    
    project_root = args.project_root or find_project_root()
    logger.info(f"Project root: {project_root}")
    
    manager = MigrationManager(project_root, dry_run=args.dry_run)
    success = manager.run(skip_audit=args.skip_audit)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
