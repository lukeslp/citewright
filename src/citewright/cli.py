"""
Command Line Interface for CiteWright.

Provides commands for renaming PDFs, media files, and managing configuration.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.text import Text

from . import __version__
from .pdf_renamer import PDFRenamer
from .media_renamer import MediaRenamer
from .bibtex_manager import BibTeXManager
from .utils.config import get_config, reload_config, Config, CONFIG_FILE

console = Console()


def setup_logging(verbose: bool = False) -> None:
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
    )


@click.group()
@click.version_option(version=__version__, prog_name="citewright")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """
    CiteWright - Intelligent academic paper and media renaming tool.
    
    Renames files using metadata from multiple academic sources (arXiv,
    Semantic Scholar, Crossref, PubMed, OpenLibrary) with optional AI enhancement.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    setup_logging(verbose)


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("-r", "--recursive", is_flag=True, help="Process subdirectories")
@click.option("-n", "--dry-run", is_flag=True, default=True, help="Preview changes without renaming (default)")
@click.option("--execute", is_flag=True, help="Actually perform the renames")
@click.option("--ai", is_flag=True, help="Enable AI analysis for difficult files")
@click.option("-b", "--bibtex", type=click.Path(), help="Output BibTeX file path")
@click.pass_context
def pdf(
    ctx: click.Context,
    path: str,
    recursive: bool,
    dry_run: bool,
    execute: bool,
    ai: bool,
    bibtex: Optional[str],
) -> None:
    """
    Rename PDF files based on academic metadata.
    
    PATH can be a single PDF file or a directory containing PDFs.
    
    By default, runs in dry-run mode (preview only). Use --execute to perform renames.
    
    \b
    Examples:
        citewright pdf ~/papers/                    # Preview renames
        citewright pdf ~/papers/ --execute          # Actually rename
        citewright pdf ~/papers/ -r --execute       # Recursive
        citewright pdf paper.pdf --execute          # Single file
        citewright pdf ~/papers/ --ai --execute     # Use AI for difficult files
    """
    target = Path(path)
    
    # Determine actual dry_run state
    actual_dry_run = not execute
    
    # Update config if AI requested
    if ai:
        config = get_config()
        config.ai_enabled = True
    
    renamer = PDFRenamer()
    bibtex_manager = None
    
    if bibtex:
        bibtex_manager = BibTeXManager(Path(bibtex))
    
    console.print(Panel.fit(
        f"[bold blue]CiteWright PDF Renamer[/bold blue]\n"
        f"Target: {target}\n"
        f"Mode: {'[yellow]DRY RUN[/yellow]' if actual_dry_run else '[green]EXECUTE[/green]'}\n"
        f"AI: {'[green]Enabled[/green]' if ai else '[dim]Disabled[/dim]'}",
        title="Configuration"
    ))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        if target.is_file():
            task = progress.add_task("Processing file...", total=1)
            operations = [renamer.process_file(target, dry_run=actual_dry_run)]
            progress.update(task, completed=1)
        else:
            task = progress.add_task("Scanning directory...", total=None)
            operations = renamer.process_directory(
                target,
                recursive=recursive,
                dry_run=actual_dry_run,
            )
            progress.update(task, completed=len(operations))
    
    # Display results
    table = Table(title="Rename Operations")
    table.add_column("Original", style="dim")
    table.add_column("New Name", style="green")
    table.add_column("Strategy", style="cyan")
    table.add_column("Status")
    
    success_count = 0
    for op in operations:
        if op.new_path == op.original_path:
            status = "[dim]unchanged[/dim]"
        elif op.error:
            status = f"[red]error: {op.error}[/red]"
        elif actual_dry_run:
            status = "[yellow]preview[/yellow]"
            success_count += 1
        else:
            status = "[green]renamed[/green]"
            success_count += 1
        
        table.add_row(
            op.original_path.name[:40],
            op.new_path.name[:40],
            op.strategy,
            status,
        )
        
        # Add to BibTeX if we have metadata
        if bibtex_manager and op.metadata:
            bibtex_manager.add(op.metadata)
    
    console.print(table)
    
    # Save BibTeX
    if bibtex_manager:
        bibtex_manager.save()
        console.print(f"\n[green]BibTeX saved to: {bibtex}[/green]")
    
    # Summary
    console.print(f"\n[bold]Summary:[/bold] {success_count}/{len(operations)} files processed")
    
    if actual_dry_run:
        console.print("\n[yellow]This was a dry run. Use --execute to perform actual renames.[/yellow]")


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("-r", "--recursive", is_flag=True, help="Process subdirectories")
@click.option("-n", "--dry-run", is_flag=True, default=True, help="Preview changes without renaming (default)")
@click.option("--execute", is_flag=True, help="Actually perform the renames")
@click.option("--ai", is_flag=True, help="Use AI vision for image descriptions")
@click.pass_context
def media(
    ctx: click.Context,
    path: str,
    recursive: bool,
    dry_run: bool,
    execute: bool,
    ai: bool,
) -> None:
    """
    Rename media files (images, videos) based on metadata.
    
    PATH can be a single file or a directory.
    
    Uses EXIF data, file properties, and optionally AI vision to generate
    descriptive filenames.
    
    \b
    Examples:
        citewright media ~/photos/                  # Preview renames
        citewright media ~/photos/ --execute        # Actually rename
        citewright media ~/photos/ --ai --execute   # Use AI vision
    """
    target = Path(path)
    actual_dry_run = not execute
    
    if ai:
        config = get_config()
        config.ai_enabled = True
    
    renamer = MediaRenamer()
    
    console.print(Panel.fit(
        f"[bold blue]CiteWright Media Renamer[/bold blue]\n"
        f"Target: {target}\n"
        f"Mode: {'[yellow]DRY RUN[/yellow]' if actual_dry_run else '[green]EXECUTE[/green]'}\n"
        f"AI Vision: {'[green]Enabled[/green]' if ai else '[dim]Disabled[/dim]'}",
        title="Configuration"
    ))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        if target.is_file():
            task = progress.add_task("Processing file...", total=1)
            operations = [renamer.process_file(target, use_ai=ai, dry_run=actual_dry_run)]
            progress.update(task, completed=1)
        else:
            task = progress.add_task("Scanning directory...", total=None)
            operations = renamer.process_directory(
                target,
                recursive=recursive,
                use_ai=ai,
                dry_run=actual_dry_run,
            )
            progress.update(task, completed=len(operations))
    
    # Display results
    table = Table(title="Rename Operations")
    table.add_column("Original", style="dim")
    table.add_column("New Name", style="green")
    table.add_column("Method", style="cyan")
    table.add_column("Status")
    
    for op in operations:
        if op.new_path == op.original_path:
            status = "[dim]unchanged[/dim]"
        elif op.error:
            status = f"[red]error[/red]"
        elif actual_dry_run:
            status = "[yellow]preview[/yellow]"
        else:
            status = "[green]renamed[/green]"
        
        table.add_row(
            op.original_path.name[:40],
            op.new_path.name[:40],
            op.method,
            status,
        )
    
    console.print(table)
    
    if actual_dry_run:
        console.print("\n[yellow]This was a dry run. Use --execute to perform actual renames.[/yellow]")


@main.command()
@click.pass_context
def undo(ctx: click.Context) -> None:
    """
    Undo the last batch of rename operations.
    
    Works for both PDF and media renames.
    """
    pdf_renamer = PDFRenamer()
    media_renamer = MediaRenamer()
    
    pdf_undone = pdf_renamer.undo_last()
    media_undone = media_renamer.undo_last()
    
    total = pdf_undone + media_undone
    
    if total > 0:
        console.print(f"[green]Successfully undid {total} rename operations.[/green]")
    else:
        console.print("[yellow]No operations to undo.[/yellow]")


@main.command()
@click.option("--show", is_flag=True, help="Show current configuration")
@click.option("--ai-provider", type=click.Choice(["ollama", "openai", "anthropic", "gemini"]), help="Set AI provider")
@click.option("--ai-enabled/--ai-disabled", default=None, help="Enable/disable AI features")
@click.option("--max-title-words", type=int, help="Maximum words in title for filename")
@click.option("--unpaywall-email", type=str, help="Email for Unpaywall API")
@click.pass_context
def config(
    ctx: click.Context,
    show: bool,
    ai_provider: Optional[str],
    ai_enabled: Optional[bool],
    max_title_words: Optional[int],
    unpaywall_email: Optional[str],
) -> None:
    """
    View or modify CiteWright configuration.
    
    \b
    Examples:
        citewright config --show
        citewright config --ai-provider ollama
        citewright config --ai-enabled
        citewright config --max-title-words 5
    """
    cfg = get_config()
    
    # Apply changes
    changed = False
    if ai_provider is not None:
        cfg.ai_provider = ai_provider
        changed = True
    if ai_enabled is not None:
        cfg.ai_enabled = ai_enabled
        changed = True
    if max_title_words is not None:
        cfg.max_title_words = max_title_words
        changed = True
    if unpaywall_email is not None:
        cfg.unpaywall_email = unpaywall_email
        changed = True
    
    if changed:
        cfg.save()
        console.print("[green]Configuration saved.[/green]")
    
    if show or not changed:
        table = Table(title="CiteWright Configuration")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Config file", str(CONFIG_FILE))
        table.add_row("AI Provider", cfg.ai_provider)
        table.add_row("AI Enabled", str(cfg.ai_enabled))
        table.add_row("Max Title Words", str(cfg.max_title_words))
        table.add_row("Filename Format", cfg.filename_format)
        table.add_row("Generate BibTeX", str(cfg.generate_bibtex))
        table.add_row("Unpaywall Email", cfg.unpaywall_email)
        table.add_row("Log Level", cfg.log_level)
        
        console.print(table)


@main.command()
@click.pass_context
def sources(ctx: click.Context) -> None:
    """
    List available metadata sources and their status.
    """
    table = Table(title="Metadata Sources")
    table.add_column("Source", style="cyan")
    table.add_column("Type", style="dim")
    table.add_column("API Key", style="yellow")
    table.add_column("Status")
    
    sources_info = [
        ("Crossref", "DOI lookup", "Not required", "[green]Available[/green]"),
        ("Semantic Scholar", "Paper search", "Optional", "[green]Available[/green]"),
        ("arXiv", "Preprint search", "Not required", "[green]Available[/green]"),
        ("PubMed", "Biomedical papers", "Not required", "[green]Available[/green]"),
        ("OpenLibrary", "ISBN lookup", "Not required", "[green]Available[/green]"),
        ("Unpaywall", "Open access", "Email only", "[green]Available[/green]"),
    ]
    
    for name, type_, api_key, status in sources_info:
        table.add_row(name, type_, api_key, status)
    
    console.print(table)
    
    # AI providers
    console.print("\n")
    ai_table = Table(title="AI Providers (Optional)")
    ai_table.add_column("Provider", style="cyan")
    ai_table.add_column("Env Variable", style="dim")
    ai_table.add_column("Status")
    
    import os
    ai_providers = [
        ("Ollama", "N/A (local)", "[green]Available[/green]" if True else "[red]Not running[/red]"),
        ("OpenAI", "OPENAI_API_KEY", "[green]Configured[/green]" if os.getenv("OPENAI_API_KEY") else "[dim]Not configured[/dim]"),
        ("Anthropic", "ANTHROPIC_API_KEY", "[green]Configured[/green]" if os.getenv("ANTHROPIC_API_KEY") else "[dim]Not configured[/dim]"),
        ("Google Gemini", "GEMINI_API_KEY", "[green]Configured[/green]" if os.getenv("GEMINI_API_KEY") else "[dim]Not configured[/dim]"),
    ]
    
    for name, env_var, status in ai_providers:
        ai_table.add_row(name, env_var, status)
    
    console.print(ai_table)


if __name__ == "__main__":
    main()
