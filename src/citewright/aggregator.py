"""
Metadata Aggregator for CiteWright.

Orchestrates the multi-source metadata extraction pipeline, querying
multiple APIs in a prioritized order to find the best metadata for a document.
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from .api.clients import (
    PaperMetadata,
    CrossrefClient,
    SemanticScholarClient,
    ArxivClient,
    PubMedClient,
    OpenLibraryClient,
    UnpaywallClient,
)
from .utils.extractors import (
    extract_text_from_file,
    extract_pdf_metadata,
    extract_identifiers,
    ExtractedIdentifiers,
    ExtractedMetadata,
)
from .utils.config import get_config

logger = logging.getLogger(__name__)


@dataclass
class AggregationResult:
    """Result of metadata aggregation."""
    metadata: Optional[PaperMetadata]
    strategy: str
    identifiers: ExtractedIdentifiers
    local_metadata: ExtractedMetadata
    all_results: List[PaperMetadata]
    
    @property
    def success(self) -> bool:
        """Check if aggregation was successful."""
        return self.metadata is not None
    
    @property
    def confidence(self) -> float:
        """Get confidence score of the result."""
        return self.metadata.confidence if self.metadata else 0.0


class MetadataAggregator:
    """
    Aggregates metadata from multiple sources using a tiered strategy.
    
    Tier 1: Local metadata extraction (PDF embedded metadata)
    Tier 2: Identifier-based lookup (DOI, arXiv ID, ISBN, PMID)
    Tier 3: Content-based search (title/author search)
    Tier 4: AI analysis (optional, requires configuration)
    """
    
    def __init__(self):
        config = get_config()
        
        # Initialize API clients
        self.crossref = CrossrefClient()
        self.semantic_scholar = SemanticScholarClient(
            api_key=config.semantic_scholar_api_key
        )
        self.arxiv = ArxivClient()
        self.pubmed = PubMedClient()
        self.openlibrary = OpenLibraryClient()
        self.unpaywall = UnpaywallClient(email=config.unpaywall_email)
        
        # Cache for API results
        self._cache: Dict[str, PaperMetadata] = {}
    
    def aggregate(self, pdf_path: Path) -> AggregationResult:
        """
        Aggregate metadata for a PDF file using multiple strategies.
        
        Args:
            pdf_path: Path to the PDF file.
            
        Returns:
            AggregationResult with the best metadata found.
        """
        logger.info(f"Aggregating metadata for: {pdf_path.name}")
        
        all_results: List[PaperMetadata] = []
        
        # Step 1: Extract local metadata and text
        local_metadata = extract_pdf_metadata(pdf_path)
        text = extract_text_from_file(pdf_path)
        
        # Step 2: Extract identifiers from text
        identifiers = ExtractedIdentifiers()
        if text:
            identifiers = extract_identifiers(text)
            logger.debug(f"Extracted identifiers: DOI={identifiers.doi}, "
                        f"arXiv={identifiers.arxiv_id}, ISBN={identifiers.isbn}, "
                        f"PMID={identifiers.pmid}")
        
        # Strategy 1: DOI lookup (highest confidence)
        if identifiers.doi:
            logger.info(f"Found DOI: {identifiers.doi}")
            result = self._lookup_by_doi(identifiers.doi)
            if result:
                all_results.append(result)
                if result.confidence >= 0.9:
                    return AggregationResult(
                        metadata=result,
                        strategy="doi_lookup",
                        identifiers=identifiers,
                        local_metadata=local_metadata,
                        all_results=all_results,
                    )
        
        # Strategy 2: arXiv ID lookup
        if identifiers.arxiv_id:
            logger.info(f"Found arXiv ID: {identifiers.arxiv_id}")
            result = self._lookup_by_arxiv(identifiers.arxiv_id)
            if result:
                all_results.append(result)
                if result.confidence >= 0.9:
                    return AggregationResult(
                        metadata=result,
                        strategy="arxiv_lookup",
                        identifiers=identifiers,
                        local_metadata=local_metadata,
                        all_results=all_results,
                    )
        
        # Strategy 3: PMID lookup
        if identifiers.pmid:
            logger.info(f"Found PMID: {identifiers.pmid}")
            result = self._lookup_by_pmid(identifiers.pmid)
            if result:
                all_results.append(result)
                if result.confidence >= 0.9:
                    return AggregationResult(
                        metadata=result,
                        strategy="pmid_lookup",
                        identifiers=identifiers,
                        local_metadata=local_metadata,
                        all_results=all_results,
                    )
        
        # Strategy 4: ISBN lookup (for books)
        if identifiers.isbn:
            logger.info(f"Found ISBN: {identifiers.isbn}")
            result = self._lookup_by_isbn(identifiers.isbn)
            if result:
                all_results.append(result)
                if result.confidence >= 0.9:
                    return AggregationResult(
                        metadata=result,
                        strategy="isbn_lookup",
                        identifiers=identifiers,
                        local_metadata=local_metadata,
                        all_results=all_results,
                    )
        
        # Strategy 5: Title/author search using local metadata
        if local_metadata.title:
            logger.info(f"Searching by title: {local_metadata.title[:50]}...")
            results = self._search_by_title(local_metadata.title)
            all_results.extend(results)
            
            # Find best match
            best = self._select_best_result(results, local_metadata)
            if best and best.confidence >= 0.7:
                return AggregationResult(
                    metadata=best,
                    strategy="title_search",
                    identifiers=identifiers,
                    local_metadata=local_metadata,
                    all_results=all_results,
                )
        
        # Strategy 6: Use local metadata as fallback
        if local_metadata.title or local_metadata.authors:
            logger.info("Using local PDF metadata as fallback")
            fallback = PaperMetadata(
                title=local_metadata.title or pdf_path.stem,
                authors=local_metadata.authors,
                year=local_metadata.year,
                source="pdf_metadata",
                confidence=0.5,
            )
            all_results.append(fallback)
            return AggregationResult(
                metadata=fallback,
                strategy="local_metadata",
                identifiers=identifiers,
                local_metadata=local_metadata,
                all_results=all_results,
            )
        
        # No metadata found
        logger.warning(f"Could not find metadata for: {pdf_path.name}")
        return AggregationResult(
            metadata=None,
            strategy="failed",
            identifiers=identifiers,
            local_metadata=local_metadata,
            all_results=all_results,
        )
    
    def _lookup_by_doi(self, doi: str) -> Optional[PaperMetadata]:
        """Look up metadata by DOI from multiple sources."""
        cache_key = f"doi:{doi}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Try Crossref first (authoritative for DOIs)
        result = self.crossref.get_by_doi(doi)
        if result:
            self._cache[cache_key] = result
            return result
        
        # Try Semantic Scholar
        result = self.semantic_scholar.get_by_doi(doi)
        if result:
            self._cache[cache_key] = result
            return result
        
        # Try Unpaywall
        result = self.unpaywall.get_by_doi(doi)
        if result:
            self._cache[cache_key] = result
            return result
        
        return None
    
    def _lookup_by_arxiv(self, arxiv_id: str) -> Optional[PaperMetadata]:
        """Look up metadata by arXiv ID."""
        cache_key = f"arxiv:{arxiv_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Try arXiv first
        result = self.arxiv.get_by_id(arxiv_id)
        if result:
            self._cache[cache_key] = result
            return result
        
        # Try Semantic Scholar
        result = self.semantic_scholar.get_by_arxiv_id(arxiv_id)
        if result:
            self._cache[cache_key] = result
            return result
        
        return None
    
    def _lookup_by_pmid(self, pmid: str) -> Optional[PaperMetadata]:
        """Look up metadata by PubMed ID."""
        cache_key = f"pmid:{pmid}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        result = self.pubmed.get_by_pmid(pmid)
        if result:
            self._cache[cache_key] = result
            return result
        
        return None
    
    def _lookup_by_isbn(self, isbn: str) -> Optional[PaperMetadata]:
        """Look up metadata by ISBN."""
        cache_key = f"isbn:{isbn}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        result = self.openlibrary.get_by_isbn(isbn)
        if result:
            self._cache[cache_key] = result
            return result
        
        return None
    
    def _search_by_title(self, title: str, limit: int = 3) -> List[PaperMetadata]:
        """Search for papers by title across multiple sources."""
        results: List[PaperMetadata] = []
        
        # Search Semantic Scholar
        ss_results = self.semantic_scholar.search(title, limit=limit)
        results.extend(ss_results)
        
        # Search arXiv
        arxiv_results = self.arxiv.search(title, limit=limit)
        results.extend(arxiv_results)
        
        # Search Crossref
        crossref_results = self.crossref.search(title, limit=limit)
        results.extend(crossref_results)
        
        return results
    
    def _select_best_result(
        self,
        results: List[PaperMetadata],
        local_metadata: ExtractedMetadata
    ) -> Optional[PaperMetadata]:
        """Select the best result from a list based on match quality."""
        if not results:
            return None
        
        def score_result(result: PaperMetadata) -> float:
            score = result.confidence
            
            # Boost if title matches
            if local_metadata.title and result.title:
                title_lower = local_metadata.title.lower()
                result_lower = result.title.lower()
                if title_lower in result_lower or result_lower in title_lower:
                    score += 0.2
            
            # Boost if year matches
            if local_metadata.year and result.year:
                if local_metadata.year == result.year:
                    score += 0.1
            
            # Boost if has DOI
            if result.doi:
                score += 0.1
            
            return min(score, 1.0)
        
        # Sort by score and return best
        scored = [(score_result(r), r) for r in results]
        scored.sort(key=lambda x: x[0], reverse=True)
        
        return scored[0][1] if scored else None
