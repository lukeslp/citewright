"""
Text and identifier extraction utilities for CiteWright.

Handles extracting text, DOIs, arXiv IDs, ISBNs, and metadata from PDF files.
"""

import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class ExtractedIdentifiers:
    """Container for extracted identifiers from a document."""
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    isbn: Optional[str] = None
    pmid: Optional[str] = None


@dataclass
class ExtractedMetadata:
    """Container for extracted metadata from a document."""
    title: Optional[str] = None
    authors: List[str] = None
    year: Optional[int] = None
    abstract: Optional[str] = None
    
    def __post_init__(self):
        if self.authors is None:
            self.authors = []


# Regex patterns for identifier extraction
DOI_PATTERNS = [
    r'(?:doi[:\s]*)?(?:https?://(?:dx\.)?doi\.org/)?([10]\.\d{4,}/[^\s\]>"\']+)',
    r'DOI[:\s]+([10]\.\d{4,}/[^\s\]>"\']+)',
]

ARXIV_PATTERNS = [
    r'arXiv[:\s]*(\d{4}\.\d{4,5}(?:v\d+)?)',
    r'arxiv\.org/abs/(\d{4}\.\d{4,5}(?:v\d+)?)',
    r'arxiv\.org/pdf/(\d{4}\.\d{4,5}(?:v\d+)?)',
    # Old-style arXiv IDs
    r'arXiv[:\s]*([a-z-]+/\d{7}(?:v\d+)?)',
]

ISBN_PATTERNS = [
    r'ISBN[:\s-]*(?:13[:\s-]*)?(\d{3}[-\s]?\d[-\s]?\d{3}[-\s]?\d{5}[-\s]?\d)',
    r'ISBN[:\s-]*(?:10[:\s-]*)?(\d[-\s]?\d{3}[-\s]?\d{5}[-\s]?[\dX])',
    r'978[-\s]?\d[-\s]?\d{3}[-\s]?\d{5}[-\s]?\d',
]

PMID_PATTERNS = [
    r'PMID[:\s]*(\d{7,8})',
    r'PubMed[:\s]*ID[:\s]*(\d{7,8})',
]

YEAR_PATTERN = r'\b(19\d{2}|20[0-3]\d)\b'


def extract_text_from_pdf(pdf_path: Path, max_pages: int = 3, max_chars: int = 5000) -> Optional[str]:
    """
    Extract text from the first few pages of a PDF.
    
    Args:
        pdf_path: Path to the PDF file.
        max_pages: Maximum number of pages to extract.
        max_chars: Maximum characters to return.
        
    Returns:
        Extracted text or None if extraction fails.
    """
    try:
        import fitz  # PyMuPDF
        
        doc = fitz.open(pdf_path)
        text_parts = []
        
        for page_num in range(min(max_pages, len(doc))):
            page = doc[page_num]
            text_parts.append(page.get_text())
            if len("".join(text_parts)) >= max_chars:
                break
        
        doc.close()
        text = "".join(text_parts)[:max_chars]
        return text.strip() if text.strip() else None
        
    except Exception:
        return None


def extract_pdf_metadata(pdf_path: Path) -> ExtractedMetadata:
    """
    Extract embedded metadata from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file.
        
    Returns:
        ExtractedMetadata object with any found metadata.
    """
    metadata = ExtractedMetadata()
    
    try:
        import fitz  # PyMuPDF
        
        doc = fitz.open(pdf_path)
        pdf_meta = doc.metadata
        doc.close()
        
        if not pdf_meta:
            return metadata
        
        # Extract title
        title = pdf_meta.get("title", "").strip()
        if title and len(title) > 5:  # Ignore very short/empty titles
            metadata.title = title
        
        # Extract author
        author = pdf_meta.get("author", "").strip()
        if author:
            # Split on common separators
            authors = re.split(r'[,;&]|\band\b', author)
            metadata.authors = [a.strip() for a in authors if a.strip()]
        
        # Extract year from various fields
        for field in ["creationDate", "modDate", "subject"]:
            field_value = pdf_meta.get(field, "")
            if field_value:
                year = extract_year(field_value)
                if year:
                    metadata.year = year
                    break
        
    except Exception:
        pass
    
    return metadata


def extract_identifiers(text: str) -> ExtractedIdentifiers:
    """
    Extract all identifiers (DOI, arXiv ID, ISBN, PMID) from text.
    
    Args:
        text: Text to search for identifiers.
        
    Returns:
        ExtractedIdentifiers object with any found identifiers.
    """
    identifiers = ExtractedIdentifiers()
    
    # Extract DOI
    for pattern in DOI_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            doi = match.group(1)
            # Clean trailing punctuation
            doi = re.sub(r'[,;.\s\]>"\')]+$', '', doi)
            identifiers.doi = doi
            break
    
    # Extract arXiv ID
    for pattern in ARXIV_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            identifiers.arxiv_id = match.group(1)
            break
    
    # Extract ISBN
    for pattern in ISBN_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            isbn = match.group(1) if match.lastindex else match.group(0)
            # Normalize ISBN (remove hyphens/spaces)
            isbn = re.sub(r'[-\s]', '', isbn)
            identifiers.isbn = isbn
            break
    
    # Extract PMID
    for pattern in PMID_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            identifiers.pmid = match.group(1)
            break
    
    return identifiers


def extract_year(text: str) -> Optional[int]:
    """
    Extract a 4-digit year from text.
    
    Args:
        text: Text to search for a year.
        
    Returns:
        Year as integer or None if not found.
    """
    years = re.findall(YEAR_PATTERN, text)
    if years:
        # Prefer years in reasonable range (1990-2030)
        for year in years:
            year_int = int(year)
            if 1990 <= year_int <= 2030:
                return year_int
        # Fall back to first found year
        return int(years[0])
    return None


def clean_text_for_filename(text: str, max_words: int = 5) -> str:
    """
    Clean and format text for use in a filename.
    
    Args:
        text: Text to clean.
        max_words: Maximum number of words to include.
        
    Returns:
        Cleaned text suitable for a filename.
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters, keep alphanumeric and spaces
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Take first N words
    words = text.split()[:max_words]
    
    # Join with underscores
    return '_'.join(words)


def extract_author_lastname(author_string: str) -> Optional[str]:
    """
    Extract the last name from an author string.
    
    Args:
        author_string: Author name in various formats.
        
    Returns:
        Last name or None if extraction fails.
    """
    if not author_string:
        return None
    
    author_string = author_string.strip()
    
    # Handle "Last, First" format
    if ',' in author_string:
        parts = author_string.split(',')
        return parts[0].strip()
    
    # Handle "First Last" format
    parts = author_string.split()
    if parts:
        return parts[-1].strip()
    
    return None


def guess_title_from_filename(filename: str) -> Optional[str]:
    """
    Attempt to extract a title from a filename.
    
    Args:
        filename: The filename to parse.
        
    Returns:
        Guessed title or None.
    """
    # Remove extension
    name = Path(filename).stem
    
    # Replace common separators with spaces
    name = re.sub(r'[-_.]', ' ', name)
    
    # Remove common prefixes like numbers, dates
    name = re.sub(r'^\d+\s*', '', name)
    name = re.sub(r'^\d{4}[-_\s]', '', name)
    
    # Clean up
    name = re.sub(r'\s+', ' ', name).strip()
    
    if len(name) > 10:  # Reasonable title length
        return name
    
    return None
