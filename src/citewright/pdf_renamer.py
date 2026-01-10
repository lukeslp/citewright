"""
PDF Renamer for CiteWright.

Handles renaming PDF files based on extracted metadata.
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .aggregator import MetadataAggregator, AggregationResult
from .api.clients import PaperMetadata
from .utils.config import get_config, LOG_DIR
from .utils.extractors import clean_text_for_filename, extract_author_lastname

logger = logging.getLogger(__name__)

# Rename log file
RENAME_LOG_FILE = LOG_DIR / "pdf_rename_log.json"


class RenameOperation:
    """Represents a single rename operation."""
    
    def __init__(
        self,
        original_path: Path,
        new_path: Path,
        metadata: Optional[PaperMetadata],
        strategy: str,
        dry_run: bool = True,
    ):
        self.original_path = original_path
        self.new_path = new_path
        self.metadata = metadata
        self.strategy = strategy
        self.dry_run = dry_run
        self.timestamp = datetime.now().isoformat()
        self.executed = False
        self.error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging."""
        return {
            "original_path": str(self.original_path),
            "new_path": str(self.new_path),
            "metadata": self.metadata.to_dict() if self.metadata else None,
            "strategy": self.strategy,
            "dry_run": self.dry_run,
            "timestamp": self.timestamp,
            "executed": self.executed,
            "error": self.error,
        }
    
    def execute(self) -> bool:
        """Execute the rename operation."""
        if self.dry_run:
            self.executed = False
            return True
        
        try:
            # Ensure target directory exists
            self.new_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Rename the file
            self.original_path.rename(self.new_path)
            self.executed = True
            return True
        except Exception as e:
            self.error = str(e)
            logger.error(f"Failed to rename {self.original_path}: {e}")
            return False
    
    def undo(self) -> bool:
        """Undo the rename operation."""
        if not self.executed or self.dry_run:
            return False
        
        try:
            self.new_path.rename(self.original_path)
            self.executed = False
            return True
        except Exception as e:
            logger.error(f"Failed to undo rename: {e}")
            return False


class PDFRenamer:
    """
    Renames PDF files based on metadata from multiple sources.
    
    Uses the MetadataAggregator to find the best metadata for each file,
    then generates a standardized filename.
    """
    
    def __init__(self):
        self.config = get_config()
        self.aggregator = MetadataAggregator()
        self.operations: List[RenameOperation] = []
        self._load_log()
    
    def _load_log(self) -> None:
        """Load the rename log from disk."""
        if RENAME_LOG_FILE.exists():
            try:
                with open(RENAME_LOG_FILE, "r") as f:
                    data = json.load(f)
                    # We don't restore operations, just keep the file for history
            except (json.JSONDecodeError, IOError):
                pass
    
    def _save_log(self) -> None:
        """Save the rename log to disk."""
        log_data = {
            "last_updated": datetime.now().isoformat(),
            "operations": [op.to_dict() for op in self.operations],
        }
        
        RENAME_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(RENAME_LOG_FILE, "w") as f:
            json.dump(log_data, f, indent=2)
    
    def generate_filename(self, metadata: PaperMetadata, original_path: Path) -> str:
        """
        Generate a standardized filename from metadata.
        
        Format: author_year_title_keywords.pdf
        
        Args:
            metadata: Paper metadata.
            original_path: Original file path (for extension).
            
        Returns:
            Generated filename.
        """
        parts = []
        
        # Author (first author's last name)
        if metadata.authors:
            author = extract_author_lastname(metadata.authors[0])
            if author:
                parts.append(clean_text_for_filename(author, max_words=1))
        
        # Year
        if metadata.year:
            parts.append(str(metadata.year))
        
        # Title (first N words)
        if metadata.title:
            title_clean = clean_text_for_filename(
                metadata.title,
                max_words=self.config.max_title_words
            )
            if title_clean:
                parts.append(title_clean)
        
        # Fallback if we have nothing
        if not parts:
            parts.append(clean_text_for_filename(original_path.stem, max_words=5))
        
        # Join parts and add extension
        filename = "_".join(parts)
        
        # Clean up multiple underscores
        filename = re.sub(r"_+", "_", filename)
        filename = filename.strip("_")
        
        # Add extension
        extension = original_path.suffix.lower()
        if not extension:
            extension = ".pdf"
        
        return f"{filename}{extension}"
    
    def process_file(
        self,
        pdf_path: Path,
        dry_run: bool = True,
    ) -> RenameOperation:
        """
        Process a single PDF file.
        
        Args:
            pdf_path: Path to the PDF file.
            dry_run: If True, don't actually rename.
            
        Returns:
            RenameOperation with the result.
        """
        logger.info(f"Processing: {pdf_path.name}")
        
        # Aggregate metadata
        result = self.aggregator.aggregate(pdf_path)
        
        if not result.success:
            logger.warning(f"No metadata found for: {pdf_path.name}")
            # Create a failed operation
            return RenameOperation(
                original_path=pdf_path,
                new_path=pdf_path,
                metadata=None,
                strategy="failed",
                dry_run=dry_run,
            )
        
        # Generate new filename
        new_filename = self.generate_filename(result.metadata, pdf_path)
        new_path = pdf_path.parent / new_filename
        
        # Handle filename conflicts
        counter = 1
        base_path = new_path
        while new_path.exists() and new_path != pdf_path:
            stem = base_path.stem
            new_path = base_path.parent / f"{stem}_{counter}{base_path.suffix}"
            counter += 1
        
        # Skip if filename unchanged
        if new_path == pdf_path:
            logger.info(f"Filename unchanged: {pdf_path.name}")
            return RenameOperation(
                original_path=pdf_path,
                new_path=pdf_path,
                metadata=result.metadata,
                strategy=result.strategy,
                dry_run=dry_run,
            )
        
        # Create operation
        operation = RenameOperation(
            original_path=pdf_path,
            new_path=new_path,
            metadata=result.metadata,
            strategy=result.strategy,
            dry_run=dry_run,
        )
        
        # Execute if not dry run
        if not dry_run:
            operation.execute()
        
        self.operations.append(operation)
        return operation
    
    def process_directory(
        self,
        directory: Path,
        recursive: bool = False,
        dry_run: bool = True,
    ) -> List[RenameOperation]:
        """
        Process all PDF files in a directory.
        
        Args:
            directory: Directory to process.
            recursive: If True, process subdirectories.
            dry_run: If True, don't actually rename.
            
        Returns:
            List of RenameOperations.
        """
        pattern = "**/*.pdf" if recursive else "*.pdf"
        pdf_files = list(directory.glob(pattern))
        
        logger.info(f"Found {len(pdf_files)} PDF files in {directory}")
        
        operations = []
        for pdf_path in pdf_files:
            operation = self.process_file(pdf_path, dry_run=dry_run)
            operations.append(operation)
        
        # Save log
        self._save_log()
        
        return operations
    
    def undo_last(self) -> int:
        """
        Undo the last batch of rename operations.
        
        Returns:
            Number of operations undone.
        """
        undone = 0
        for operation in reversed(self.operations):
            if operation.executed and operation.undo():
                undone += 1
        
        self._save_log()
        return undone
    
    def get_stats(self) -> Dict:
        """Get statistics about processed files."""
        total = len(self.operations)
        executed = sum(1 for op in self.operations if op.executed)
        failed = sum(1 for op in self.operations if op.error)
        
        strategies = {}
        for op in self.operations:
            strategies[op.strategy] = strategies.get(op.strategy, 0) + 1
        
        return {
            "total": total,
            "executed": executed,
            "failed": failed,
            "strategies": strategies,
        }
