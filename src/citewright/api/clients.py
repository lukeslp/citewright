"""
API clients for academic metadata retrieval.

Provides unified interfaces for querying multiple academic data sources:
- Crossref (DOI resolution)
- Semantic Scholar
- arXiv
- PubMed
- OpenLibrary (ISBN lookup)
- Unpaywall (open access)
"""

import logging
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from urllib.parse import quote

import requests

logger = logging.getLogger(__name__)


@dataclass
class PaperMetadata:
    """Standardized paper metadata from any source."""
    title: str
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    abstract: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    isbn: Optional[str] = None
    pmid: Optional[str] = None
    venue: Optional[str] = None
    url: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    source: str = "unknown"
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "abstract": self.abstract,
            "doi": self.doi,
            "arxiv_id": self.arxiv_id,
            "isbn": self.isbn,
            "pmid": self.pmid,
            "venue": self.venue,
            "url": self.url,
            "keywords": self.keywords,
            "source": self.source,
            "confidence": self.confidence,
        }


class BaseClient:
    """Base class for API clients."""
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "CiteWright/0.1.0 (https://github.com/lukeslp/citewright)"
        })
    
    def _get(self, url: str, params: Dict = None, headers: Dict = None) -> Optional[Dict]:
        """Make a GET request and return JSON response."""
        try:
            response = self.session.get(
                url,
                params=params,
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.debug(f"Request failed: {e}")
            return None
        except ValueError as e:
            logger.debug(f"JSON decode failed: {e}")
            return None


class CrossrefClient(BaseClient):
    """Client for Crossref API (DOI resolution)."""
    
    BASE_URL = "https://api.crossref.org"
    
    def get_by_doi(self, doi: str) -> Optional[PaperMetadata]:
        """
        Get paper metadata by DOI.
        
        Args:
            doi: The DOI to look up.
            
        Returns:
            PaperMetadata or None if not found.
        """
        url = f"{self.BASE_URL}/works/{quote(doi, safe='')}"
        data = self._get(url)
        
        if not data or "message" not in data:
            return None
        
        return self._parse_work(data["message"])
    
    def search(self, query: str, limit: int = 5) -> List[PaperMetadata]:
        """
        Search Crossref for papers.
        
        Args:
            query: Search query.
            limit: Maximum results.
            
        Returns:
            List of PaperMetadata objects.
        """
        url = f"{self.BASE_URL}/works"
        params = {"query": query, "rows": limit}
        data = self._get(url, params)
        
        if not data or "message" not in data:
            return []
        
        items = data["message"].get("items", [])
        return [self._parse_work(item) for item in items if self._parse_work(item)]
    
    def _parse_work(self, work: Dict) -> Optional[PaperMetadata]:
        """Parse Crossref work into PaperMetadata."""
        title_list = work.get("title", [])
        if not title_list:
            return None
        
        title = title_list[0] if isinstance(title_list, list) else title_list
        
        # Extract authors
        authors = []
        for author in work.get("author", []):
            given = author.get("given", "")
            family = author.get("family", "")
            if family:
                authors.append(f"{given} {family}".strip())
        
        # Extract year
        year = None
        date_parts = work.get("published-print", {}).get("date-parts", [[]])
        if not date_parts[0]:
            date_parts = work.get("published-online", {}).get("date-parts", [[]])
        if date_parts and date_parts[0]:
            year = date_parts[0][0]
        
        return PaperMetadata(
            title=title,
            authors=authors,
            year=year,
            abstract=work.get("abstract"),
            doi=work.get("DOI"),
            venue=work.get("container-title", [""])[0] if work.get("container-title") else None,
            url=work.get("URL"),
            source="crossref",
            confidence=0.95 if work.get("DOI") else 0.7,
        )


class SemanticScholarClient(BaseClient):
    """Client for Semantic Scholar API."""
    
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    
    DEFAULT_FIELDS = [
        "title", "authors", "year", "abstract", "doi",
        "venue", "url", "paperId", "citationCount"
    ]
    
    def __init__(self, api_key: Optional[str] = None, timeout: int = 30):
        super().__init__(timeout)
        self.api_key = api_key
        if api_key:
            self.session.headers["x-api-key"] = api_key
    
    def get_by_doi(self, doi: str) -> Optional[PaperMetadata]:
        """Get paper by DOI."""
        return self._get_paper(doi)
    
    def get_by_arxiv_id(self, arxiv_id: str) -> Optional[PaperMetadata]:
        """Get paper by arXiv ID."""
        return self._get_paper(f"arXiv:{arxiv_id}")
    
    def _get_paper(self, paper_id: str) -> Optional[PaperMetadata]:
        """Get paper by any identifier."""
        url = f"{self.BASE_URL}/paper/{paper_id}"
        params = {"fields": ",".join(self.DEFAULT_FIELDS)}
        data = self._get(url, params)
        
        if not data:
            return None
        
        return self._parse_paper(data)
    
    def search(self, query: str, limit: int = 5) -> List[PaperMetadata]:
        """Search for papers."""
        url = f"{self.BASE_URL}/paper/search"
        params = {
            "query": query,
            "limit": limit,
            "fields": ",".join(self.DEFAULT_FIELDS)
        }
        data = self._get(url, params)
        
        if not data:
            return []
        
        papers = data.get("data", [])
        return [self._parse_paper(p) for p in papers if self._parse_paper(p)]
    
    def _parse_paper(self, paper: Dict) -> Optional[PaperMetadata]:
        """Parse Semantic Scholar paper into PaperMetadata."""
        if not paper.get("title"):
            return None
        
        authors = [a.get("name", "") for a in paper.get("authors", []) if a.get("name")]
        
        return PaperMetadata(
            title=paper["title"],
            authors=authors,
            year=paper.get("year"),
            abstract=paper.get("abstract"),
            doi=paper.get("doi"),
            venue=paper.get("venue"),
            url=paper.get("url"),
            source="semantic_scholar",
            confidence=0.9 if paper.get("doi") else 0.75,
        )


class ArxivClient(BaseClient):
    """Client for arXiv API."""
    
    BASE_URL = "http://export.arxiv.org/api/query"
    
    def get_by_id(self, arxiv_id: str) -> Optional[PaperMetadata]:
        """Get paper by arXiv ID."""
        # Clean the ID
        arxiv_id = arxiv_id.replace("arXiv:", "").replace("arxiv:", "")
        
        params = {"id_list": arxiv_id}
        
        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=self.timeout)
            response.raise_for_status()
            papers = self._parse_response(response.text)
            return papers[0] if papers else None
        except Exception as e:
            logger.debug(f"arXiv request failed: {e}")
            return None
    
    def search(self, query: str, limit: int = 5) -> List[PaperMetadata]:
        """Search arXiv for papers."""
        search_query = f'all:"{query}" OR abs:"{query}" OR ti:"{query}"'
        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": limit,
            "sortBy": "relevance",
        }
        
        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=self.timeout)
            response.raise_for_status()
            return self._parse_response(response.text)
        except Exception as e:
            logger.debug(f"arXiv search failed: {e}")
            return []
    
    def _parse_response(self, xml_content: str) -> List[PaperMetadata]:
        """Parse arXiv XML response."""
        import xml.etree.ElementTree as ET
        
        try:
            root = ET.fromstring(xml_content)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            
            papers = []
            for entry in root.findall("atom:entry", ns):
                title_elem = entry.find("atom:title", ns)
                if title_elem is None or not title_elem.text:
                    continue
                
                title = " ".join(title_elem.text.split())
                
                # Authors
                authors = []
                for author in entry.findall("atom:author", ns):
                    name = author.find("atom:name", ns)
                    if name is not None and name.text:
                        authors.append(name.text)
                
                # Year from published date
                year = None
                published = entry.find("atom:published", ns)
                if published is not None and published.text:
                    match = re.search(r"(\d{4})", published.text)
                    if match:
                        year = int(match.group(1))
                
                # Abstract
                summary = entry.find("atom:summary", ns)
                abstract = " ".join(summary.text.split()) if summary is not None and summary.text else None
                
                # arXiv ID from entry ID
                entry_id = entry.find("atom:id", ns)
                arxiv_id = None
                if entry_id is not None and entry_id.text:
                    match = re.search(r"(\d{4}\.\d{4,5}(?:v\d+)?)", entry_id.text)
                    if match:
                        arxiv_id = match.group(1)
                
                # Categories as keywords
                keywords = []
                for cat in entry.findall("atom:category", ns):
                    term = cat.get("term")
                    if term:
                        keywords.append(term)
                
                papers.append(PaperMetadata(
                    title=title,
                    authors=authors,
                    year=year,
                    abstract=abstract,
                    arxiv_id=arxiv_id,
                    url=f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None,
                    keywords=keywords,
                    source="arxiv",
                    confidence=0.95,
                ))
            
            return papers
            
        except ET.ParseError as e:
            logger.debug(f"XML parse error: {e}")
            return []


class PubMedClient(BaseClient):
    """Client for PubMed API."""
    
    SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    SUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    
    def get_by_pmid(self, pmid: str) -> Optional[PaperMetadata]:
        """Get paper by PubMed ID."""
        pmid = pmid.replace("PMID:", "").replace("pmid:", "").strip()
        
        params = {
            "db": "pubmed",
            "id": pmid,
            "retmode": "json",
        }
        
        data = self._get(self.SUMMARY_URL, params)
        if not data or "result" not in data:
            return None
        
        result = data["result"]
        if pmid not in result:
            return None
        
        return self._parse_summary(pmid, result[pmid])
    
    def search(self, query: str, limit: int = 5) -> List[PaperMetadata]:
        """Search PubMed for papers."""
        # First, search for IDs
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmax": limit,
            "retmode": "json",
        }
        
        search_data = self._get(self.SEARCH_URL, search_params)
        if not search_data:
            return []
        
        id_list = search_data.get("esearchresult", {}).get("idlist", [])
        if not id_list:
            return []
        
        # Then get summaries
        summary_params = {
            "db": "pubmed",
            "id": ",".join(id_list),
            "retmode": "json",
        }
        
        summary_data = self._get(self.SUMMARY_URL, summary_params)
        if not summary_data or "result" not in summary_data:
            return []
        
        result = summary_data["result"]
        papers = []
        for pmid in id_list:
            if pmid in result:
                paper = self._parse_summary(pmid, result[pmid])
                if paper:
                    papers.append(paper)
        
        return papers
    
    def _parse_summary(self, pmid: str, data: Dict) -> Optional[PaperMetadata]:
        """Parse PubMed summary into PaperMetadata."""
        title = data.get("title", "").strip()
        if not title:
            return None
        
        # Authors
        authors = []
        for author in data.get("authors", []):
            name = author.get("name")
            if name:
                authors.append(name)
        
        # Year from pubdate
        year = None
        pubdate = data.get("pubdate", "")
        match = re.search(r"(\d{4})", pubdate)
        if match:
            year = int(match.group(1))
        
        # DOI from elocationid
        doi = None
        elocationid = data.get("elocationid", "")
        if "doi:" in elocationid.lower():
            doi = elocationid.replace("doi:", "").strip()
        
        return PaperMetadata(
            title=title,
            authors=authors,
            year=year,
            doi=doi,
            pmid=pmid,
            venue=data.get("fulljournalname") or data.get("source"),
            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            source="pubmed",
            confidence=0.9,
        )


class OpenLibraryClient(BaseClient):
    """Client for OpenLibrary API (ISBN lookup)."""
    
    BASE_URL = "https://openlibrary.org"
    
    def get_by_isbn(self, isbn: str) -> Optional[PaperMetadata]:
        """Get book metadata by ISBN."""
        # Normalize ISBN
        isbn = re.sub(r"[-\s]", "", isbn)
        
        url = f"{self.BASE_URL}/api/books"
        params = {
            "bibkeys": f"ISBN:{isbn}",
            "format": "json",
            "jscmd": "data",
        }
        
        data = self._get(url, params)
        if not data:
            return None
        
        book_data = data.get(f"ISBN:{isbn}")
        if not book_data:
            return None
        
        # Authors
        authors = [a.get("name", "") for a in book_data.get("authors", []) if a.get("name")]
        
        # Year from publish_date
        year = None
        publish_date = book_data.get("publish_date", "")
        match = re.search(r"(\d{4})", publish_date)
        if match:
            year = int(match.group(1))
        
        return PaperMetadata(
            title=book_data.get("title", ""),
            authors=authors,
            year=year,
            isbn=isbn,
            url=book_data.get("url"),
            source="openlibrary",
            confidence=0.95,
        )


class UnpaywallClient(BaseClient):
    """Client for Unpaywall API (open access lookup)."""
    
    BASE_URL = "https://api.unpaywall.org/v2"
    
    def __init__(self, email: str = "anonymous@example.com", timeout: int = 30):
        super().__init__(timeout)
        self.email = email
    
    def get_by_doi(self, doi: str) -> Optional[PaperMetadata]:
        """Get paper metadata by DOI."""
        url = f"{self.BASE_URL}/{quote(doi, safe='')}"
        params = {"email": self.email}
        
        data = self._get(url, params)
        if not data:
            return None
        
        # Authors
        authors = []
        for author in data.get("z_authors", []):
            given = author.get("given", "")
            family = author.get("family", "")
            if family:
                authors.append(f"{given} {family}".strip())
        
        return PaperMetadata(
            title=data.get("title", ""),
            authors=authors,
            year=data.get("year"),
            doi=data.get("doi"),
            venue=data.get("journal_name"),
            url=data.get("best_oa_location", {}).get("url") if data.get("best_oa_location") else None,
            source="unpaywall",
            confidence=0.95,
        )
