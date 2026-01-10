# CiteWright: The Intelligent Academic Renamer

Anybody else have a huge folder full of files with names like `235680_download.PDF` and `smith_et_al_2008_full.pdf(2)`?

... yeah.

**CiteWright** is a powerful Python command-line tool that brings order to your digital library. It intelligently renames academic papers, books, and even media files using a sophisticated multi-source metadata extraction strategy. It prioritizes free, high-quality, non-AI data sources and only uses AI as a final, optional step.

## Key Features

- **Multi-Tier Metadata Extraction**: Uses a 4-tier strategy to find the best metadata, from local files to powerful APIs.
- **Comprehensive API Support**: Pulls data from **Crossref, Semantic Scholar, arXiv, PubMed, OpenLibrary, and Unpaywall**—all for free.
- **Standardized Renaming**: Renames files to a clean, consistent format (e.g., `Author_Year_Title.pdf`).
- **BibTeX Generation**: Automatically creates and maintains a `.bib` file for your collection.
- **Optional AI Enhancement**: For the toughest files, it can optionally use **Ollama, OpenAI, Anthropic, or Gemini** to analyze content.
- **Media Renaming**: Can also organize images and videos using EXIF data and AI vision.
- **Safe by Default**: Runs in "dry run" mode to let you preview all changes before they happen.
- **Undo Functionality**: Easily revert the last batch of renames if you make a mistake.

## The CiteWright Philosophy

CiteWright is built on a hierarchical, "non-LLM first" approach. It exhausts all possible deterministic methods for identifying a file before resorting to probabilistic AI analysis. This ensures maximum accuracy, speed, and cost-effectiveness.

| Tier | Method | Description |
| :--- | :--- | :--- |
| **Tier 1** | **Local Metadata** | Extract embedded metadata directly from the file (e.g., PDF info dictionary). Fastest, but often unreliable. |
| **Tier 2** | **Identifier Lookup** | Extract persistent identifiers (DOI, ArXiv ID, ISBN) from the file's text and query authoritative public APIs. Highly accurate. |
| **Tier 3** | **Content-Based Search** | Use the file's title and author information (extracted from text) to search academic APIs. Good for finding papers without clear identifiers. |
| **Tier 4** | **LLM Analysis (Optional)** | As a last resort, use an LLM to analyze the raw text and infer metadata. Powerful but slower and potentially costly. |

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/lukeslp/citewright.git
    cd citewright
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**

    ```bash
    pip install .
    ```

    For optional AI and media processing features, install the extras:

    ```bash
    pip install ".[all]"
    ```

## Usage

### Renaming PDFs

- **Preview renames in a directory:**

  ```bash
  citewright pdf ~/path/to/your/papers
  ```

- **Execute the renames:**

  ```bash
  citewright pdf ~/path/to/your/papers --execute
  ```

- **Process recursively and generate a BibTeX file:**

  ```bash
  citewright pdf ~/path/to/your/papers -r --execute --bibtex library.bib
  ```

- **Use AI for difficult files:**

  ```bash
  citewright pdf ~/path/to/your/papers --ai --execute
  ```

### Renaming Media Files

- **Preview image and video renames:**

  ```bash
  citewright media ~/path/to/your/photos
  ```

- **Use AI Vision to generate descriptive filenames:**

  ```bash
  citewright media ~/path/to/your/photos --ai --execute
  ```

### Configuration

CiteWright is configurable via a JSON file located at `~/.config/citewright/config.json`. You can also manage settings from the command line.

- **Show current configuration:**

  ```bash
  citewright config --show
  ```

- **Set your default AI provider:**

  ```bash
  citewright config --ai-provider openai
  citewright config --ai-enabled
  ```

- **Set your email for the Unpaywall API (for better service):**

  ```bash
  citewright config --unpaywall-email "your.email@example.com"
  ```

### Undo

Made a mistake? Easily undo the last batch of renames:

```bash
citewright undo
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

