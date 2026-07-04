"""Full ingestion pipeline: Load → Clean → Parse → Chunk → Embed → Store.

Orchestrates all components for processing Bangladeshi law PDFs into Qdrant.
Supports single-file and batch processing with resumability via SQLite tracking.
"""

import os
import time
import logging
import sqlite3
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

from ingestion.loader import PDFLoader
from ingestion.cleaner import TextCleaner
from ingestion.metadata import MetadataExtractor, LawMetadata
from ingestion.structure_parser import StructureParser
from ingestion.chunker import LegalChunker, Chunk
from ingestion.embeddings import EmbeddingGenerator
from ingestion.qdrant_store import QdrantStore

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """Result of processing a single document."""

    filepath: str
    filename: str
    success: bool = False
    chunks_count: int = 0
    error: Optional[str] = None
    metadata: Optional[LawMetadata] = None
    elapsed_seconds: float = 0.0
    pages: int = 0
    sections: int = 0


@dataclass
class BatchResult:
    """Aggregate result from batch ingestion."""

    total_files: int = 0
    succeeded: int = 0
    failed: int = 0
    total_chunks: int = 0
    elapsed_seconds: float = 0.0
    results: list[IngestionResult] = field(default_factory=list)
    errors_log_path: str = ""


class IngestionPipeline:
    """Production-grade pipeline for ingesting Bangladeshi law PDFs into Qdrant.

    Usage:
        pipeline = IngestionPipeline(
            data_dir="./data",
            collection_name="bangladesh_laws",
        )
        pipeline.run()           # Process all PDFs in data_dir
        pipeline.run_file("act-print-1630.pdf")  # Process a single file
    """

    def __init__(
        self,
        data_dir: str = "./data",
        collection_name: str = "bangladesh_laws",
        embedding_batch_size: int = 32,
        qdrant_batch_size: int = 64,
        target_tokens_min: int = 600,
        target_tokens_max: int = 900,
        max_tokens: int = 1000,
        resume: bool = True,
        logs_dir: str = "logs",
        db_path: Optional[str] = None,
    ):
        self.data_dir = Path(data_dir)
        self.collection_name = collection_name
        self.embedding_batch_size = embedding_batch_size
        self.qdrant_batch_size = qdrant_batch_size
        self.resume = resume

        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        if db_path is None:
            db_path = str(self.logs_dir / "ingestion_state.db")
        self.db_path = db_path

        self.loader = PDFLoader()
        self.cleaner = TextCleaner(preserve_first_header=True)
        self.metadata_extractor = MetadataExtractor()
        self.structure_parser = StructureParser()
        self.chunker = LegalChunker(
            target_min=target_tokens_min,
            target_max=target_tokens_max,
            max_tokens=max_tokens,
        )

        self._embedding_generator: Optional[EmbeddingGenerator] = None
        self._qdrant_store: Optional[QdrantStore] = None

        self._init_tracking_db()

    @property
    def embedding_generator(self) -> EmbeddingGenerator:
        if self._embedding_generator is None:
            self._embedding_generator = EmbeddingGenerator(
                model_name="BAAI/bge-m3",
                batch_size=self.embedding_batch_size,
            )
        return self._embedding_generator

    @property
    def qdrant_store(self) -> QdrantStore:
        if self._qdrant_store is None:
            self._qdrant_store = QdrantStore(collection_name=self.collection_name)
            self._qdrant_store.ensure_collection()
        return self._qdrant_store

    # ------------------------------------------------------------------
    # Tracking database (SQLite) for resumable ingestion
    # ------------------------------------------------------------------

    def _init_tracking_db(self) -> None:
        """Initialize SQLite database for tracking processed files."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """CREATE TABLE IF NOT EXISTS ingestion_state (
                filepath TEXT PRIMARY KEY,
                filename TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                chunks_count INTEGER DEFAULT 0,
                error TEXT,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        conn.commit()
        conn.close()

    def _is_processed(self, filepath: str) -> bool:
        """Check if a file has already been successfully processed."""
        if not self.resume:
            return False
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT status FROM ingestion_state WHERE filepath = ?",
            (str(filepath),),
        ).fetchone()
        conn.close()
        return row is not None and row[0] == "done"

    def _mark_processed(self, filepath: str, filename: str, chunks_count: int) -> None:
        """Mark a file as successfully processed."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """INSERT OR REPLACE INTO ingestion_state
               (filepath, filename, status, chunks_count, processed_at)
               VALUES (?, ?, 'done', ?, CURRENT_TIMESTAMP)""",
            (str(filepath), filename, chunks_count),
        )
        conn.commit()
        conn.close()

    def _mark_failed(self, filepath: str, filename: str, error: str) -> None:
        """Mark a file as failed with error message."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """INSERT OR REPLACE INTO ingestion_state
               (filepath, filename, status, error, processed_at)
               VALUES (?, ?, 'failed', ?, CURRENT_TIMESTAMP)""",
            (str(filepath), filename, str(error)[:500]),
        )
        conn.commit()
        conn.close()

    def _get_pending_files(self) -> list[str]:
        """Get list of filepaths that are pending or failed (for retry)."""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT filepath FROM ingestion_state WHERE status IN ('pending', 'failed')"
        ).fetchall()
        conn.close()
        return [r[0] for r in rows]

    def _get_tracking_summary(self) -> dict:
        """Return summary of the tracking database."""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT status, COUNT(*) FROM ingestion_state GROUP BY status"
        ).fetchall()
        conn.close()
        return {r[0]: r[1] for r in rows}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, file_pattern: str = "*.pdf") -> BatchResult:
        """Process all PDF files in the data directory.

        Args:
            file_pattern: Glob pattern for files to process.

        Returns:
            BatchResult with aggregate statistics.
        """
        pdf_files = sorted(self.data_dir.glob(file_pattern))
        if not pdf_files:
            logger.warning("No PDF files found in %s matching '%s'", self.data_dir, file_pattern)
            return BatchResult(total_files=0)

        start_time = time.time()
        result = BatchResult(
            total_files=len(pdf_files),
            errors_log_path=str(self.logs_dir / "ingest_errors.log"),
        )

        self.qdrant_store.ensure_collection()

        for pdf_path in pdf_files:
            filepath_str = str(pdf_path)

            if self._is_processed(filepath_str):
                logger.info("Skipping already-processed: %s", pdf_path.name)
                result.total_files -= 1
                continue

            logger.info("Processing: %s", pdf_path.name)
            single_result = self.run_file(filepath_str)

            result.results.append(single_result)

            if single_result.success:
                result.succeeded += 1
                result.total_chunks += single_result.chunks_count
            else:
                result.failed += 1
                self._log_error(single_result)

        result.elapsed_seconds = time.time() - start_time
        self._log_summary(result)
        return result

    def run_file(self, filepath: str) -> IngestionResult:
        """Process a single PDF file through the full pipeline.

        Steps:
          1. Load PDF via PyMuPDF
          2. Clean text (remove headers, footers, noise)
          3. Extract metadata
          4. Parse legal structure
          5. Chunk into semantic units
          6. Generate BGE-M3 embeddings
          7. Upsert into Qdrant

        Args:
            filepath: Path to a PDF file.

        Returns:
            IngestionResult with processing status and stats.
        """
        filename = Path(filepath).name
        result = IngestionResult(filepath=filepath, filename=filename)
        start = time.time()

        try:
            loaded = self.loader.load(filepath)
            if not loaded.success:
                result.error = loaded.error
                self._mark_failed(filepath, filename, loaded.error or "Unknown load error")
                result.elapsed_seconds = time.time() - start
                return result

            result.pages = loaded.total_pages

            first_page_text = loaded.pages[0] if loaded.pages else ""

            cleaned = self.cleaner.clean(loaded.pages)
            if not cleaned.cleaned_text.strip():
                result.error = "No text remaining after cleaning"
                self._mark_failed(filepath, filename, result.error)
                result.elapsed_seconds = time.time() - start
                return result

            metadata = self.metadata_extractor.extract(
                text=cleaned.cleaned_text,
                filename=loaded.filename,
                first_page_text=first_page_text,
            )
            result.metadata = metadata

            parsed = self.structure_parser.parse(cleaned.cleaned_text)
            if not parsed.sections:
                logger.warning("No legal sections detected in '%s'", filename)

            result.sections = len(parsed.sections)

            chunks: list[Chunk] = self.chunker.chunk(
                parsed_doc=parsed,
                metadata=metadata,
                page_count=loaded.total_pages,
            )

            if not chunks:
                logger.warning("No chunks generated for '%s'", filename)
                result.success = True
                result.chunks_count = 0
                self._mark_processed(filepath, filename, 0)
                result.elapsed_seconds = time.time() - start
                return result

            texts = [chunk.text for chunk in chunks]
            metadata_list = [chunk.metadata for chunk in chunks]
            chunk_types = [chunk.chunk_type.value for chunk in chunks]
            citations = [chunk.citation for chunk in chunks]
            references_list = [chunk.references for chunk in chunks]
            validations = [chunk.validation for chunk in chunks]

            logger.info(
                "Embedding %d chunks for '%s' (%d chars avg)",
                len(texts),
                filename,
                sum(len(t) for t in texts) // max(1, len(texts)),
            )

            embeddings = list(
                self.embedding_generator.embed_generator(texts)
            )

            embedding_vectors = [emb.tolist() for emb in embeddings]

            self.qdrant_store.upsert_chunks(
                chunk_texts=texts,
                chunk_metadata=metadata_list,
                chunk_types=chunk_types,
                citations=citations,
                references_list=references_list,
                validations=validations,
                embeddings=embedding_vectors,
                batch_size=self.qdrant_batch_size,
            )

            result.chunks_count = len(chunks)
            result.success = True
            self._mark_processed(filepath, filename, len(chunks))

        except Exception as e:
            logger.exception("Failed to process '%s'", filename)
            result.error = str(e)
            self._mark_failed(filepath, filename, str(e))

        result.elapsed_seconds = time.time() - start
        return result

    # ------------------------------------------------------------------
    # Status / Info
    # ------------------------------------------------------------------

    def status(self) -> dict:
        """Return current pipeline status."""
        info = self._get_tracking_summary()
        try:
            qdrant_info = self.qdrant_store.collection_info()
        except Exception as e:
            qdrant_info = {"error": str(e)}

        return {
            "tracking_db": info,
            "collection": qdrant_info,
            "data_dir": str(self.data_dir),
            "resume": self.resume,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log_error(self, result: IngestionResult) -> None:
        """Append error to the error log file."""
        error_path = self.logs_dir / "ingest_errors.log"
        with open(error_path, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {result.filename}\n")
            f.write(f"  Error: {result.error}\n")
            f.write(f"  File: {result.filepath}\n\n")

    def _log_summary(self, result: BatchResult) -> None:
        """Log batch processing summary."""
        mins, secs = divmod(int(result.elapsed_seconds), 60)
        logger.info(
            "BATCH COMPLETE | %d/%d files | %d chunks | %dm %ds",
            result.succeeded,
            result.total_files,
            result.total_chunks,
            mins,
            secs,
        )
        if result.failed:
            logger.warning(
                "%d files failed. See %s for details.",
                result.failed,
                result.errors_log_path,
            )


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Bangladesh Law Ingestion Pipeline")
    parser.add_argument("--dir", default="./data", help="Directory with PDF files")
    parser.add_argument("--file", default=None, help="Process a single PDF file")
    parser.add_argument("--no-resume", action="store_true", help="Re-process all files")
    parser.add_argument("--status", action="store_true", help="Show ingestion status")
    parser.add_argument(
        "--collection", default="bangladesh_laws", help="Qdrant collection name"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Embedding batch size"
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
        sys.exit(0)

    if args.file:
        result = pipeline.run_file(args.file)
        status = "SUCCESS" if result.success else "FAILED"
        print(f"\n{status}: {result.filename}")
        print(f"  Chunks: {result.chunks_count}")
        print(f"  Pages:  {result.pages}")
        print(f"  Sections: {result.sections}")
        print(f"  Time:   {result.elapsed_seconds:.2f}s")
        if result.error:
            print(f"  Error:  {result.error}")
    else:
        result = pipeline.run()
        print(f"\nDone. {result.succeeded}/{result.total_files} files, {result.total_chunks} chunks.")

    pipeline.qdrant_store.client.close()
