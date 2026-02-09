"""
Media Renamer for CiteWright.

Handles renaming image and video files using metadata extraction
and optional vision-model-based description.
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .utils.config import get_config, LOG_DIR
from .utils.extractors import clean_text_for_filename

logger = logging.getLogger(__name__)

# Rename log file
MEDIA_LOG_FILE = LOG_DIR / "media_rename_log.json"


def extract_image_metadata(image_path: Path) -> Dict:
    """
    Extract metadata from an image file.
    
    Args:
        image_path: Path to the image file.
        
    Returns:
        Dictionary with extracted metadata.
    """
    metadata = {
        "filename": image_path.name,
        "extension": image_path.suffix.lower(),
        "size": image_path.stat().st_size if image_path.exists() else 0,
    }
    
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS
        
        with Image.open(image_path) as img:
            metadata["width"] = img.width
            metadata["height"] = img.height
            metadata["format"] = img.format
            metadata["mode"] = img.mode
            
            # Extract EXIF data
            exif_data = img._getexif()
            if exif_data:
                exif = {}
                for tag_id, value in exif_data.items():
                    tag = TAGS.get(tag_id, tag_id)
                    if isinstance(value, bytes):
                        continue  # Skip binary data
                    exif[tag] = value
                
                metadata["exif"] = exif
                
                # Extract useful fields
                if "DateTimeOriginal" in exif:
                    metadata["date_taken"] = exif["DateTimeOriginal"]
                elif "DateTime" in exif:
                    metadata["date_taken"] = exif["DateTime"]
                
                if "Make" in exif:
                    metadata["camera_make"] = exif["Make"]
                if "Model" in exif:
                    metadata["camera_model"] = exif["Model"]
    except ImportError:
        logger.debug("PIL not available for image metadata extraction")
    except Exception as e:
        logger.debug(f"Failed to extract image metadata: {e}")
    
    return metadata


def extract_video_metadata(video_path: Path) -> Dict:
    """
    Extract metadata from a video file.
    
    Args:
        video_path: Path to the video file.
        
    Returns:
        Dictionary with extracted metadata.
    """
    metadata = {
        "filename": video_path.name,
        "extension": video_path.suffix.lower(),
        "size": video_path.stat().st_size if video_path.exists() else 0,
    }
    
    try:
        import cv2
        
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            metadata["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            metadata["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            metadata["fps"] = cap.get(cv2.CAP_PROP_FPS)
            metadata["frame_count"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if metadata["fps"] > 0:
                metadata["duration"] = metadata["frame_count"] / metadata["fps"]
        
        cap.release()
    except ImportError:
        logger.debug("OpenCV not available for video metadata extraction")
    except Exception as e:
        logger.debug(f"Failed to extract video metadata: {e}")
    
    return metadata


class MediaRenameOperation:
    """Represents a single media rename operation."""
    
    def __init__(
        self,
        original_path: Path,
        new_path: Path,
        metadata: Dict,
        method: str,
        dry_run: bool = True,
    ):
        self.original_path = original_path
        self.new_path = new_path
        self.metadata = metadata
        self.method = method
        self.dry_run = dry_run
        self.timestamp = datetime.now().isoformat()
        self.executed = False
        self.error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging."""
        return {
            "original_path": str(self.original_path),
            "new_path": str(self.new_path),
            "metadata": self.metadata,
            "method": self.method,
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
            self.new_path.parent.mkdir(parents=True, exist_ok=True)
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


class MediaRenamer:
    """
    Renames media files (images, videos) based on metadata.

    Uses EXIF data, file properties, and optionally vision models
    to generate descriptive filenames.
    """
    
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".heic"}
    VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
    
    def __init__(self):
        self.config = get_config()
        self.operations: List[MediaRenameOperation] = []
        self._ai_analyzer = None

    @property
    def ai_analyzer(self):
        """Lazy load LLM analyzer."""
        if self._ai_analyzer is None and self.config.ai_enabled:
            from .ai_analyzer import AIAnalyzer
            self._ai_analyzer = AIAnalyzer()
        return self._ai_analyzer
    
    def _is_image(self, path: Path) -> bool:
        """Check if file is an image."""
        return path.suffix.lower() in self.IMAGE_EXTENSIONS
    
    def _is_video(self, path: Path) -> bool:
        """Check if file is a video."""
        return path.suffix.lower() in self.VIDEO_EXTENSIONS
    
    def generate_filename(
        self,
        metadata: Dict,
        original_path: Path,
        ai_description: Optional[str] = None,
    ) -> str:
        """
        Generate a filename from metadata.
        
        Args:
            metadata: Extracted metadata.
            original_path: Original file path.
            ai_description: Optional vision-model-generated description.
            
        Returns:
            Generated filename.
        """
        parts = []
        
        # Use vision model description if available
        if ai_description:
            # Extract suggested filename from vision model response
            clean_desc = clean_text_for_filename(ai_description, max_words=5)
            if clean_desc:
                parts.append(clean_desc)
        
        # Use date if available
        date_str = metadata.get("date_taken")
        if date_str:
            try:
                # Parse common date formats
                for fmt in ["%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y%m%d"]:
                    try:
                        dt = datetime.strptime(date_str, fmt)
                        parts.insert(0, dt.strftime("%Y%m%d"))
                        break
                    except ValueError:
                        continue
            except Exception:
                pass
        
        # Use camera info if available
        camera = metadata.get("camera_model") or metadata.get("camera_make")
        if camera and not ai_description:
            parts.append(clean_text_for_filename(camera, max_words=2))
        
        # Use dimensions for images/videos
        width = metadata.get("width")
        height = metadata.get("height")
        if width and height and not ai_description:
            parts.append(f"{width}x{height}")
        
        # Fallback to original stem
        if not parts:
            parts.append(clean_text_for_filename(original_path.stem, max_words=5))
        
        filename = "_".join(parts)
        filename = re.sub(r"_+", "_", filename).strip("_")
        
        return f"{filename}{original_path.suffix.lower()}"
    
    def process_file(
        self,
        file_path: Path,
        use_ai: bool = False,
        dry_run: bool = True,
    ) -> MediaRenameOperation:
        """
        Process a single media file.
        
        Args:
            file_path: Path to the media file.
            use_ai: Whether to use vision model for description.
            dry_run: If True, don't actually rename.
            
        Returns:
            MediaRenameOperation with the result.
        """
        logger.info(f"Processing media: {file_path.name}")
        
        # Extract metadata
        if self._is_image(file_path):
            metadata = extract_image_metadata(file_path)
            method = "image_metadata"
        elif self._is_video(file_path):
            metadata = extract_video_metadata(file_path)
            method = "video_metadata"
        else:
            metadata = {"filename": file_path.name}
            method = "fallback"
        
        # Try vision model analysis if enabled
        ai_description = None
        if use_ai and self.ai_analyzer and self._is_image(file_path):
            try:
                result = self.ai_analyzer.analyze_image(file_path)
                if result and "description" in result:
                    ai_description = result["description"]
                    method = "vision_model"
            except Exception as e:
                logger.debug(f"vision model analysis failed: {e}")
        
        # Generate new filename
        new_filename = self.generate_filename(metadata, file_path, ai_description)
        new_path = file_path.parent / new_filename
        
        # Handle conflicts
        counter = 1
        base_path = new_path
        while new_path.exists() and new_path != file_path:
            stem = base_path.stem
            new_path = base_path.parent / f"{stem}_{counter}{base_path.suffix}"
            counter += 1
        
        # Create operation
        operation = MediaRenameOperation(
            original_path=file_path,
            new_path=new_path,
            metadata=metadata,
            method=method,
            dry_run=dry_run,
        )
        
        if not dry_run:
            operation.execute()
        
        self.operations.append(operation)
        return operation
    
    def process_directory(
        self,
        directory: Path,
        recursive: bool = False,
        use_ai: bool = False,
        dry_run: bool = True,
    ) -> List[MediaRenameOperation]:
        """
        Process all media files in a directory.
        
        Args:
            directory: Directory to process.
            recursive: If True, process subdirectories.
            use_ai: Whether to use vision model for descriptions.
            dry_run: If True, don't actually rename.
            
        Returns:
            List of MediaRenameOperations.
        """
        all_extensions = self.IMAGE_EXTENSIONS | self.VIDEO_EXTENSIONS
        
        media_files = []
        if recursive:
            for ext in all_extensions:
                media_files.extend(directory.rglob(f"*{ext}"))
                media_files.extend(directory.rglob(f"*{ext.upper()}"))
        else:
            for ext in all_extensions:
                media_files.extend(directory.glob(f"*{ext}"))
                media_files.extend(directory.glob(f"*{ext.upper()}"))
        
        logger.info(f"Found {len(media_files)} media files in {directory}")
        
        operations = []
        for file_path in media_files:
            operation = self.process_file(file_path, use_ai=use_ai, dry_run=dry_run)
            operations.append(operation)
        
        return operations
    
    def undo_last(self) -> int:
        """Undo the last batch of operations."""
        undone = 0
        for operation in reversed(self.operations):
            if operation.executed and operation.undo():
                undone += 1
        return undone
