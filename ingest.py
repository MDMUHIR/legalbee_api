"""
ingest.py — Production ingestion pipeline for Bangladeshi law documents.

Uses the new ingestion/ package:
  - PyMuPDF (fitz) for PDF text extraction
  - Structure-aware legal section detection (Bengali + English)
  - Header/footer/noise removal
  - BGE-M3 embeddings (1024-dim, multilingual)
  - Qdrant vector store with hybrid search support
  - SQLite tracking for resumable ingestion

Usage:
  python ingest.py                  # process all PDFs in ./data
  python ingest.py --status         # show progress
  python ingest.py --no-resume      # re-process all files
  python ingest.py --file data/act-print-1630.pdf   # single file
"""

import sys
import argparse
import logging
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.stdout.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

from ingestion.pipeline import IngestionPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Bangladesh Law Ingestion Pipeline — PDF → Qdrant"
    )
    parser.add_argument(
        "--dir", default="./data", help="Directory containing PDF files (default: ./data)"
    )
    parser.add_argument(
        "--file", default=None, help="Process a single PDF file"
    )
    parser.add_argument(
        "--no-resume", action="store_true", help="Re-process already ingested files"
    )
    parser.add_argument(
        "--status", action="store_true", help="Show ingestion status and exit"
    )
    parser.add_argument(
        "--collection",
        default="bangladesh_laws",
        help="Qdrant collection name (default: bangladesh_laws)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch size (default: 32)",
    )
    args = parser.parse_args()

    pipeline = IngestionPipeline(
        data_dir=args.dir,
        collection_name=args.collection,
        embedding_batch_size=args.batch_size,
        resume=not args.no_resume,
    )

    if args.status:
        import json

        print(json.dumps(pipeline.status(), indent=2, ensure_ascii=False))
        return

    if args.file:
        result = pipeline.run_file(args.file)
        status = "SUCCESS" if result.success else "FAILED"
        print(f"\n{status}: {result.filename}")
        print(f"  Chunks:   {result.chunks_count}")
        print(f"  Pages:    {result.pages}")
        print(f"  Sections: {result.sections}")
        print(f"  Time:     {result.elapsed_seconds:.2f}s")
        if result.metadata:
            print(f"  Act:      {result.metadata.act_name or result.metadata.bangla_name}")
            print(f"  Year:     {result.metadata.act_year}")
            print(f"  Type:     {result.metadata.document_type}")
        if result.error:
            print(f"  Error:    {result.error}")
    else:
        result = pipeline.run()
        print(
            f"\nDone. {result.succeeded}/{result.total_files} files succeeded, "
            f"{result.failed} failed, {result.total_chunks} chunks."
        )

    pipeline.qdrant_store.client.close()


if __name__ == "__main__":
    main()
