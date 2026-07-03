"""
scripts/ingest_laws.py
-----------------------
RESUMABLE ingestion pipeline for 1500+ law files.

Features:
  ✓ Tracks every file in SQLite — safe to stop/restart anytime
  ✓ Skips already-processed files automatically
  ✓ Handles .docx, .pdf, .txt
  ✓ Memory-safe batched embedding
  ✓ Detailed progress bar + live stats
  ✓ Failed files logged for easy retry

Usage:
  # First run (or add new files):
  python scripts/ingest_laws.py

  # Retry only failed files:
  python scripts/ingest_laws.py --retry-failed

  # See current progress without processing:
  python scripts/ingest_laws.py --status

  # Point to a custom folder:
  python scripts/ingest_laws.py --dir /path/to/my/laws
"""

import sys
import os
import argparse
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from dotenv import load_dotenv

from src.parser import parse_file
from src.embedder import embed_batch
from src.qdrant_store import get_client, ensure_collection, upsert_chunks, collection_stats
from src.progress_db import (
    init_db, is_already_done, mark_started, mark_done,
    mark_failed, get_summary, get_failed_files, reset_failed,
)

load_dotenv()
console = Console()

SUPPORTED = {".docx", ".pdf", ".txt"}
EMBED_BATCH = int(os.getenv("INGEST_BATCH_SIZE", "32"))


def print_status():
    """Print current ingestion progress from the SQLite tracker."""
    init_db()
    summary = get_summary()

    table = Table(title="📊 Ingestion Progress", border_style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Count", justify="right")

    done    = summary.get("done", 0)
    failed  = summary.get("failed", 0)
    pending = summary.get("pending", 0)
    in_prog = summary.get("in_progress", 0)
    chunks  = summary.get("total_chunks_uploaded", 0)

    table.add_row("[green]✅ Done[/green]",        str(done))
    table.add_row("[red]❌ Failed[/red]",          str(failed))
    table.add_row("[yellow]⏳ Pending[/yellow]",   str(pending))
    table.add_row("[blue]🔄 In Progress[/blue]",   str(in_prog))
    table.add_row("[cyan]📦 Total Chunks[/cyan]",  str(chunks))

    console.print(table)

    if failed:
        console.print(f"\n[yellow]Re-run with --retry-failed to retry {failed} failed files.[/yellow]")


def discover_files(laws_dir: Path) -> list[Path]:
    """Find all supported law files recursively."""
    files = []
    for ext in SUPPORTED:
        files.extend(laws_dir.rglob(f"*{ext}"))
    return sorted(files)


def process_file(filepath: Path, client, embed_batch_size: int) -> int:
    """
    Parse → embed → upload one file.
    Returns number of chunks uploaded, or raises on error.
    """
    law_name, chunks = parse_file(str(filepath))

    if not chunks:
        return 0

    mark_started(str(filepath), law_name)

    texts = [c["text"] for c in chunks]
    vectors = embed_batch(texts, batch_size=embed_batch_size)
    upsert_chunks(client, chunks, vectors)

    return len(chunks)


def run_ingestion(laws_dir: Path, retry_failed: bool = False):
    init_db()

    files = discover_files(laws_dir)
    if not files:
        console.print(f"[red]No supported files found in {laws_dir}[/red]")
        console.print("Supported formats: .docx, .pdf, .txt")
        return

    # If retrying failures, reset them to pending
    if retry_failed:
        n = reset_failed()
        console.print(f"[yellow]Reset {n} failed files to pending.[/yellow]")

    # Filter to only unprocessed files
    to_process = [f for f in files if not is_already_done(str(f))]
    already_done = len(files) - len(to_process)

    console.print(Panel.fit(
        f"[bold green]🇧🇩 Bangladesh Law Ingestion Pipeline[/bold green]\n\n"
        f"  Total files found:  [cyan]{len(files)}[/cyan]\n"
        f"  Already ingested:   [green]{already_done}[/green]\n"
        f"  To process now:     [yellow]{len(to_process)}[/yellow]\n"
        f"  Embedding batch:    [dim]{EMBED_BATCH} chunks at a time[/dim]",
        border_style="green",
    ))

    if not to_process:
        console.print("[green]✅ All files already ingested! Nothing to do.[/green]")
        print_status()
        return

    # Connect to Qdrant
    console.print("\n[dim]Connecting to Qdrant Cloud...[/dim]")
    client = get_client()
    ensure_collection(client)

    # ── Main ingestion loop ────────────────────────────────────
    success_count = 0
    fail_count = 0
    total_chunks = 0
    start_time = time.time()

    with tqdm(
        to_process,
        desc="Ingesting",
        unit="file",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
    ) as pbar:
        for filepath in pbar:
            short_name = filepath.name[:45]
            pbar.set_postfix_str(f"✓{success_count} ✗{fail_count} | {short_name}", refresh=True)

            try:
                n_chunks = process_file(filepath, client, EMBED_BATCH)
                mark_done(str(filepath), n_chunks)
                success_count += 1
                total_chunks += n_chunks

            except Exception as e:
                mark_failed(str(filepath), str(e))
                fail_count += 1
                # Log to file without stopping the pipeline
                with open("logs/ingest_errors.log", "a") as log:
                    log.write(f"[FAILED] {filepath}\n{e}\n\n")

    # ── Final summary ──────────────────────────────────────────
    elapsed = time.time() - start_time
    mins, secs = divmod(int(elapsed), 60)

    stats = collection_stats(client)

    console.print()
    console.print(Panel.fit(
        f"[bold green]✅ Ingestion Complete![/bold green]\n\n"
        f"  Processed:   [green]{success_count}[/green] files succeeded\n"
        f"  Failed:      [red]{fail_count}[/red] files (see logs/ingest_errors.log)\n"
        f"  New chunks:  [cyan]{total_chunks}[/cyan] uploaded\n"
        f"  DB total:    [cyan]{stats['vectors_count']}[/cyan] vectors\n"
        f"  Time taken:  {mins}m {secs}s",
        border_style="green" if fail_count == 0 else "yellow",
    ))

    if fail_count:
        console.print(
            f"\n[yellow]⚠️  {fail_count} files failed. "
            f"Run again with --retry-failed to retry them.[/yellow]"
        )

    console.print("\n[bold]Next step:[/bold] python src/api.py")


# ── CLI ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ingest Bangladeshi law files into Qdrant Cloud"
    )
    parser.add_argument(
        "--dir", default="data/laws",
        help="Folder containing law files (default: data/laws)"
    )
    parser.add_argument(
        "--retry-failed", action="store_true",
        help="Retry files that previously failed"
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Show ingestion progress and exit"
    )
    args = parser.parse_args()

    if args.status:
        print_status()
        return

    laws_dir = Path(args.dir)
    if not laws_dir.exists():
        laws_dir.mkdir(parents=True)
        console.print(f"[yellow]Created directory: {laws_dir}[/yellow]")
        console.print("Place your .docx / .pdf / .txt law files there and re-run.")
        return

    run_ingestion(laws_dir, retry_failed=args.retry_failed)


if __name__ == "__main__":
    main()
