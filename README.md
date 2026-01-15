# CiteWright

Anybody else have a huge folder full of files with names like `235680_download.PDF` and `smith_et_al_2008_full.pdf(2)`?

... yeah.

I wrote this because I got mass-downloading papers from Sci-Hub and then staring at a folder of cryptic filenames wondering which one was the paper about transformer attention mechanisms and which one was about soil bacteria. Life's too short.

## What It Does

- Strips text from documents and uses **arXiv, Semantic Scholar, Crossref, PubMed, OpenLibrary, and Unpaywall** to find the actual source
- Renames files to `Author_Year_Title.ext` like a civilized person
- Handles **PDF, TXT, Markdown, DOC/DOCX, and Python files** - throw it at it, let's find out
- Maintains a BibTeX database so you don't have to
- Logs everything, doesn't break anything, asks before doing anything destructive
- **Optionally** uses a local LLM (Ollama) or cloud providers (OpenAI, Anthropic, Gemini) if the free APIs come up empty

## The Philosophy

I built this with a "try the free stuff first" approach. Why pay for API calls when CrossRef is right there?

| Tier | What Happens |
| :--- | :--- |
| **1** | Check if the PDF already has metadata embedded. Usually garbage, but sometimes you get lucky. |
| **2** | Extract DOIs, arXiv IDs, ISBNs from the text and look them up. This is where the magic happens. |
| **3** | Search academic APIs using whatever title/author text it can scrape. Works more often than you'd think. |
| **4** | (Optional) Throw the text at an LLM and ask nicely. Costs money unless you're running Ollama locally. |

## Installation

```bash
git clone https://github.com/lukeslp/citewright.git
cd citewright
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install .
```

Want the AI features and media processing?

```bash
pip install ".[all]"
```

## Usage

**Preview what would happen (dry run, safe):**
```bash
citewright pdf ~/papers
```

**Actually rename things:**
```bash
citewright pdf ~/papers --execute
```

**Go recursive and spit out a BibTeX file:**
```bash
citewright pdf ~/papers -r --execute --bibtex library.bib
```

**Let AI take a crack at the stubborn ones:**
```bash
citewright pdf ~/papers --ai --execute
```

**Rename photos and videos too (uses EXIF data):**
```bash
citewright media ~/photos --execute
```

**Use AI vision to describe images:**
```bash
citewright media ~/photos --ai --execute
```

**Oh no go back:**
```bash
citewright undo
```

## Configuration

Config lives at `~/.config/citewright/config.json`, or use the CLI:

```bash
citewright config --show
citewright config --ai-provider openai
citewright config --ai-enabled
citewright config --unpaywall-email "you@example.com"
```

The Unpaywall email is optional but they appreciate it. Be cool.

## License

MIT. Do whatever.

## Author

Luke Steuber  
https://github.com/lukeslp  
luke@actuallyuseful.ai
