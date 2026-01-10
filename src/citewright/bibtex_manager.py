"""
BibTeX Manager for CiteWright.

Handles generating and managing BibTeX entries for processed papers.
"""

import re
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from .api.clients import PaperMetadata

logger = logging.getLogger(__name__)


def generate_cite_key(metadata: PaperMetadata) -> str:
    """
    Generate a BibTeX citation key from metadata.
    
    Format: author_year_firstword
    
    Args:
        metadata: Paper metadata.
        
    Returns:
        Citation key string.
    """
    parts = []
    
    # First author's last name
    if metadata.authors:
        author = metadata.authors[0]
        # Extract last name
        if "," in author:
            last_name = author.split(",")[0].strip()
        else:
            last_name = author.split()[-1] if author.split() else "unknown"
        # Clean and lowercase
        last_name = re.sub(r"[^a-zA-Z]", "", last_name).lower()
        parts.append(last_name)
    else:
        parts.append("unknown")
    
    # Year
    if metadata.year:
        parts.append(str(metadata.year))
    
    # First significant word of title
    if metadata.title:
        # Remove common words
        stopwords = {"a", "an", "the", "of", "in", "on", "for", "to", "and", "with"}
        words = metadata.title.lower().split()
        for word in words:
            clean_word = re.sub(r"[^a-zA-Z]", "", word)
            if clean_word and clean_word not in stopwords:
                parts.append(clean_word)
                break
    
    return "_".join(parts)


def escape_bibtex(text: str) -> str:
    """
    Escape special characters for BibTeX.
    
    Args:
        text: Text to escape.
        
    Returns:
        Escaped text.
    """
    if not text:
        return ""
    
    # Replace special characters
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    
    return text


def format_authors_bibtex(authors: List[str]) -> str:
    """
    Format author list for BibTeX.
    
    Args:
        authors: List of author names.
        
    Returns:
        BibTeX-formatted author string.
    """
    if not authors:
        return ""
    
    formatted = []
    for author in authors:
        author = author.strip()
        if not author:
            continue
        
        # If already in "Last, First" format, keep it
        if "," in author:
            formatted.append(author)
        else:
            # Convert "First Last" to "Last, First"
            parts = author.split()
            if len(parts) >= 2:
                formatted.append(f"{parts[-1]}, {' '.join(parts[:-1])}")
            else:
                formatted.append(author)
    
    return " and ".join(formatted)


def metadata_to_bibtex(metadata: PaperMetadata, cite_key: Optional[str] = None) -> str:
    """
    Convert PaperMetadata to a BibTeX entry.
    
    Args:
        metadata: Paper metadata.
        cite_key: Optional citation key (auto-generated if not provided).
        
    Returns:
        BibTeX entry string.
    """
    if cite_key is None:
        cite_key = generate_cite_key(metadata)
    
    # Determine entry type
    if metadata.isbn:
        entry_type = "book"
    elif metadata.arxiv_id:
        entry_type = "article"
    elif metadata.venue and any(kw in metadata.venue.lower() for kw in ["conference", "proceedings", "workshop"]):
        entry_type = "inproceedings"
    else:
        entry_type = "article"
    
    # Build fields
    fields = []
    
    # Title (required)
    title = escape_bibtex(metadata.title) if metadata.title else "Unknown Title"
    fields.append(f'  title = {{{title}}}')
    
    # Authors
    if metadata.authors:
        authors = format_authors_bibtex(metadata.authors)
        fields.append(f'  author = {{{authors}}}')
    
    # Year
    if metadata.year:
        fields.append(f'  year = {{{metadata.year}}}')
    
    # DOI
    if metadata.doi:
        fields.append(f'  doi = {{{metadata.doi}}}')
    
    # arXiv
    if metadata.arxiv_id:
        fields.append(f'  eprint = {{{metadata.arxiv_id}}}')
        fields.append('  archiveprefix = {arXiv}')
    
    # ISBN
    if metadata.isbn:
        fields.append(f'  isbn = {{{metadata.isbn}}}')
    
    # PMID
    if metadata.pmid:
        fields.append(f'  pmid = {{{metadata.pmid}}}')
    
    # Venue/Journal
    if metadata.venue:
        venue = escape_bibtex(metadata.venue)
        if entry_type == "inproceedings":
            fields.append(f'  booktitle = {{{venue}}}')
        else:
            fields.append(f'  journal = {{{venue}}}')
    
    # URL
    if metadata.url:
        fields.append(f'  url = {{{metadata.url}}}')
    
    # Abstract
    if metadata.abstract:
        abstract = escape_bibtex(metadata.abstract[:500])  # Truncate long abstracts
        fields.append(f'  abstract = {{{abstract}}}')
    
    # Keywords
    if metadata.keywords:
        keywords = ", ".join(metadata.keywords[:10])  # Limit keywords
        fields.append(f'  keywords = {{{escape_bibtex(keywords)}}}')
    
    # Build entry
    entry = f"@{entry_type}{{{cite_key},\n"
    entry += ",\n".join(fields)
    entry += "\n}"
    
    return entry


class BibTeXManager:
    """
    Manages a BibTeX database file.
    
    Handles adding entries, avoiding duplicates, and writing to file.
    """
    
    def __init__(self, bib_file: Path):
        """
        Initialize BibTeX manager.
        
        Args:
            bib_file: Path to the .bib file.
        """
        self.bib_file = Path(bib_file)
        self.entries: Dict[str, str] = {}
        self._load()
    
    def _load(self) -> None:
        """Load existing entries from the BibTeX file."""
        if not self.bib_file.exists():
            return
        
        try:
            content = self.bib_file.read_text(encoding="utf-8")
            
            # Simple parser to extract cite keys
            pattern = r'@\w+\{([^,]+),'
            for match in re.finditer(pattern, content):
                cite_key = match.group(1).strip()
                # Find the full entry
                start = match.start()
                brace_count = 0
                end = start
                for i, char in enumerate(content[start:], start):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end = i + 1
                            break
                
                self.entries[cite_key] = content[start:end]
            
            logger.info(f"Loaded {len(self.entries)} entries from {self.bib_file}")
        except Exception as e:
            logger.error(f"Failed to load BibTeX file: {e}")
    
    def add(self, metadata: PaperMetadata) -> str:
        """
        Add a paper to the BibTeX database.
        
        Args:
            metadata: Paper metadata.
            
        Returns:
            The citation key used.
        """
        cite_key = generate_cite_key(metadata)
        
        # Handle duplicate keys
        original_key = cite_key
        counter = 1
        while cite_key in self.entries:
            cite_key = f"{original_key}_{chr(ord('a') + counter - 1)}"
            counter += 1
            if counter > 26:
                cite_key = f"{original_key}_{counter}"
        
        entry = metadata_to_bibtex(metadata, cite_key)
        self.entries[cite_key] = entry
        
        logger.debug(f"Added BibTeX entry: {cite_key}")
        return cite_key
    
    def remove(self, cite_key: str) -> bool:
        """
        Remove an entry by citation key.
        
        Args:
            cite_key: Citation key to remove.
            
        Returns:
            True if removed, False if not found.
        """
        if cite_key in self.entries:
            del self.entries[cite_key]
            return True
        return False
    
    def get(self, cite_key: str) -> Optional[str]:
        """
        Get an entry by citation key.
        
        Args:
            cite_key: Citation key.
            
        Returns:
            BibTeX entry string or None.
        """
        return self.entries.get(cite_key)
    
    def save(self) -> None:
        """Save all entries to the BibTeX file."""
        # Create header
        header = f"% BibTeX database generated by CiteWright\n"
        header += f"% Last updated: {datetime.now().isoformat()}\n"
        header += f"% Total entries: {len(self.entries)}\n\n"
        
        # Combine all entries
        content = header + "\n\n".join(self.entries.values())
        
        # Write to file
        self.bib_file.parent.mkdir(parents=True, exist_ok=True)
        self.bib_file.write_text(content, encoding="utf-8")
        
        logger.info(f"Saved {len(self.entries)} entries to {self.bib_file}")
    
    def __len__(self) -> int:
        """Return number of entries."""
        return len(self.entries)
    
    def __contains__(self, cite_key: str) -> bool:
        """Check if a citation key exists."""
        return cite_key in self.entries
