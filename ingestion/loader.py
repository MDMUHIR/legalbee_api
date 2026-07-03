"""PDF loader using PyMuPDF (fitz) for high-quality text extraction with page-level granularity."""

import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class LoadedDocument:
    """Result of loading a single PDF document."""

    filepath: str
    filename: str
    pages: list[str] = field(default_factory=list)
    total_pages: int = 0
    raw_text: str = ""
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None


class PDFLoader:
    """Load Bangladeshi law PDFs using PyMuPDF with per-page text extraction.

    PyMuPDF preserves text positioning and Unicode fidelity better than PyPDF2,
    which is critical for Bangla text extraction.
    """

    def __init__(self, password: Optional[str] = None):
        self.password = password

    def load(self, filepath: str) -> LoadedDocument:
        """Load a single PDF and return page-level text plus raw concatenated text.

        Args:
            filepath: Absolute or relative path to a PDF file.

        Returns:
            LoadedDocument with page texts and metadata.
        """
        path = Path(filepath)
        doc = LoadedDocument(
            filepath=str(path),
            filename=path.name,
        )

        try:
            import fitz  # PyMuPDF

            pdf = fitz.open(
                str(path),
                filetype="pdf",
            )
            if self.password and pdf.needs_pass:
                pdf.authenticate(self.password)

            pages_text: list[str] = []
            for page_num in range(pdf.page_count):
                page = pdf[page_num]
                text = page.get_text("text", sort=True)  # sort=True preserves reading order
                pages_text.append(text)

            doc.pages = pages_text
            doc.total_pages = len(pages_text)
            doc.raw_text = "\n".join(pages_text)
            pdf.close()

            logger.info(
                "Loaded PDF '%s': %d pages, %d characters",
                doc.filename,
                doc.total_pages,
                len(doc.raw_text),
            )

        except ImportError:
            doc.error = "PyMuPDF (fitz) not installed. Run: pip install pymupdf"
            logger.error(doc.error)
        except Exception as e:
            doc.error = f"Failed to load PDF: {e}"
            logger.error("Failed to load '%s': %s", filepath, e)

        return doc

    def load_batch(self, filepaths: list[str]) -> list[LoadedDocument]:
        """Load multiple PDFs, collecting errors individually so one failure
        does not stop the batch."""
        results: list[LoadedDocument] = []
        for fp in filepaths:
            result = self.load(fp)
            results.append(result)
            if not result.success:
                logger.warning("Skipping failed file: %s — %s", fp, result.error)
        return results

    @staticmethod
    def load_from_bytes(data: bytes, filename: str = "stream.pdf") -> LoadedDocument:
        """Load PDF from bytes (e.g., from an HTTP download)."""
        doc = LoadedDocument(filepath=f"<bytes>:{filename}", filename=filename)
        try:
            import fitz

            pdf = fitz.open(stream=data, filetype="pdf")
            pages_text = []
            for page_num in range(pdf.page_count):
                page = pdf[page_num]
                text = page.get_text("text", sort=True)
                pages_text.append(text)

            doc.pages = pages_text
            doc.total_pages = len(pages_text)
            doc.raw_text = "\n".join(pages_text)
            pdf.close()
        except Exception as e:
            doc.error = str(e)
        return doc
