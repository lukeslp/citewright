"""
CiteWright - Intelligent academic paper and media renaming tool.

A comprehensive tool for renaming academic papers and media files using
multi-source metadata extraction with optional AI enhancement.
"""

__version__ = "0.1.0"
__author__ = "Luke Steuber"

from .aggregator import MetadataAggregator
from .pdf_renamer import PDFRenamer
from .bibtex_manager import BibTeXManager

__all__ = [
    "MetadataAggregator",
    "PDFRenamer", 
    "BibTeXManager",
    "__version__",
]
