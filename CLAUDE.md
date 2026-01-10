# CiteWright Development Notes

## Project Overview

CiteWright is an intelligent academic paper and media renaming tool that uses a multi-tier metadata extraction strategy. It prioritizes free, deterministic data sources and only uses AI as an optional final step.

## Architecture

```
src/citewright/
├── __init__.py          # Package initialization
├── cli.py               # Click-based CLI interface
├── aggregator.py        # Multi-source metadata aggregation pipeline
├── pdf_renamer.py       # PDF file renaming logic
├── media_renamer.py     # Image/video renaming logic
├── bibtex_manager.py    # BibTeX database management
├── ai_analyzer.py       # Optional LLM-based analysis
├── api/
│   ├── __init__.py
│   └── clients.py       # API clients (Crossref, Semantic Scholar, arXiv, etc.)
└── utils/
    ├── __init__.py
    ├── config.py        # Configuration management
    └── extractors.py    # Text and identifier extraction utilities
```

## Key Design Decisions

1. **Non-LLM First**: The tool exhausts all free, deterministic methods before using AI.
2. **Multi-Source Aggregation**: Queries multiple APIs in parallel for best results.
3. **Safe by Default**: Dry-run mode is the default; user must explicitly enable execution.
4. **Extensible AI**: Supports multiple AI providers (Ollama, OpenAI, Anthropic, Gemini).

## Data Sources

| Source | Purpose | API Key Required |
|--------|---------|------------------|
| Crossref | DOI resolution | No |
| Semantic Scholar | Paper search | Optional |
| arXiv | Preprint search | No |
| PubMed | Biomedical papers | No |
| OpenLibrary | ISBN lookup | No |
| Unpaywall | Open access | Email only |

## Development Commands

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/
ruff check src/

# Run the CLI
citewright --help
```

## Configuration

Configuration is stored at `~/.config/citewright/config.json`. Key settings:

- `ai_provider`: Which AI provider to use (ollama, openai, anthropic, gemini)
- `ai_enabled`: Whether to use AI features
- `max_title_words`: Number of title words in filename
- `unpaywall_email`: Email for Unpaywall API

## Future Enhancements

- Add more data sources (DBLP, Google Scholar via SerpAPI)
- Implement batch processing with progress bars
- Add support for EPUB and other document formats
- Create a web interface
