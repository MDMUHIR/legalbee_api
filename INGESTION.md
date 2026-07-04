# Ingestion Pipeline Documentation

## Overview

The ingestion pipeline processes Bangladeshi law PDFs and stores them as searchable chunks in Qdrant. It handles Bangla, English, and mixed-language documents.

**Pipeline flow**: `Load PDF → Clean text → Extract metadata → Parse structure → Chunk → Embed → Store in Qdrant`

---

## Quick Start

```bash
# Process all PDFs in ./data
uv run ingest.py

# Process a single file
uv run ingest.py --file data/act-print-1630.pdf

# Check progress
uv run ingest.py --status

# Re-process everything (ignore SQLite resume tracker)
uv run ingest.py --no-resume

# Use a custom data directory
uv run ingest.py --dir /path/to/laws
```

---

## Architecture

```
ingestion/
├── __init__.py            # Package exports
├── loader.py              # PDF loading via PyMuPDF (fitz)
├── cleaner.py             # Header/footer/noise removal
├── metadata.py            # Act metadata extraction + reference detection
├── structure_parser.py    # Legal hierarchy parsing
├── chunker.py             # Structure-aware chunking + validation
├── embeddings.py          # BGE-M3 embedding via sentence-transformers
├── qdrant_store.py        # Qdrant collection + upsert
├── pipeline.py            # Full orchestration + SQLite tracking
└── utils.py               # Bengali digit conversion, token estimation, Unicode
```

### Module Responsibilities

| Module | Input | Output |
|---|---|---|
| `loader.py` | PDF filepath | Per-page text strings |
| `cleaner.py` | List of page texts | Cleaned concatenated text |
| `metadata.py` | Cleaned text | `LawMetadata` dataclass |
| `structure_parser.py` | Cleaned text | `ParsedDocument` (sections + clauses tree) |
| `chunker.py` | `ParsedDocument` + `LawMetadata` | `List[Chunk]` with embeddings-ready text + metadata |
| `embeddings.py` | List of text strings | 1024-dim BGE-M3 vectors |
| `qdrant_store.py` | Chunk data + vectors | Qdrant points |
| `pipeline.py` | data directory | Orchestrates all modules |

---

## Chunk Structure

Each chunk stored in Qdrant has this payload:

```json
{
  "text": "১। (১) এই আইন সরকারি চাকরি (সংশোধন) আইন, ২০২৬ নামে অভিহিত হইবে...",
  "chunk_type": "section",
  "citation": "সরকারি চাকরি (সংশোধন) আইন, ২০২৬, Section ১ (সংক্ষিপ্ত শিরোনাম)",
  "metadata": {
    "act_name": "Government Service (Amendment) Act, 2026",
    "bangla_name": "সরকারি চাকরি (সংশোধন) আইন, ২০২৬",
    "act_number": "1",
    "year": 2026,
    "publication_date": "১০ এপ্রিল, ২০২৬",
    "document_type": "Amendment Act",
    "source_pdf": "act-print-1630.pdf",
    "volume": "1630",
    "pdf_page": 1,
    "language": "bn",
    "section_title": "সংক্ষিপ্ত শিরোনাম",
    "hierarchy": {
      "part": "",
      "chapter": "",
      "article": "",
      "section": "১",
      "clause": "১",
      "sub_clause": ""
    },
    "chunk_index": 1
  },
  "validation": {
    "validated": true,
    "starts_at_boundary": true,
    "ends_at_boundary": true,
    "is_complete_chunk": true
  },
  "references": [
    {"type": "amends", "target": "Government Service Act, 2018"},
    {"type": "act", "target": "2018 সনের 57 নং আইন"},
    {"type": "ordinance", "target": "2025 সনের 26 নং অধ্যাদেশ"}
  ]
}
```

### Key Fields

| Field | Description |
|---|---|
| `text` | The chunk text embedded by BGE-M3 |
| `chunk_type` | One of: `act`, `chapter`, `part`, `article`, `section`, `clause`, `sub_clause`, `schedule`, `appendix`, `explanation`, `definition`, `preamble`, `amendment` |
| `citation` | Auto-generated legal citation. Ready for LLM to use without hallucinating |
| `metadata.hierarchy` | Nested legal position. Part → Chapter → Article → Section → Clause → Sub-clause. Only populated fields exist |
| `metadata.act_name` | Canonical act name (English if available, otherwise Bangla) |
| `validation` | Chunk quality indicators. `is_complete_chunk=false` means the chunk was force-split |
| `references` | Structured cross-references with types: `amends`, `act`, `ordinance`, `article`, `section`, `rule`, `order` |

---

## Chunking Rules

The pipeline follows strict rules to optimize retrieval quality:

1. **One legal concept per chunk**. A chunk never mixes two different sections.
2. **Clause boundaries are atomic**. Sections are split at clause/sub-clause markers only — never mid-sentence.
3. **Self-contained**. Every chunk includes the section header so it's independently retrievable. A chunk from Section 2, Clause (11) still starts with `Section 2:` context.
4. **Token targets**: 600–900 tokens, maximum 1000.
5. **Amendment acts**: Citations include the inserted provision. Section 2 becomes `Section 2 (inserted Section 37A)`.

### When sections are too long

Long sections are split at clause boundaries `(1)`, `(ক)`, `(a)` etc. If a single clause is still too large, it splits at Bangla sentence boundaries (`।`).

**Never split**: Sections, articles, clauses, or sub-clauses across chunks. Each chunk represents one complete legal provision or a contiguous group of clauses within the same section.

---

## Legal Structure Parsing

The parser detects hierarchical structure in Bangladeshi law documents:

```
Act
 ├── Part
 │    └── Chapter (অধ্যায়)
 │         └── Section (ধারা)
 │              ├── Sub-section (উপ-ধারা) — (১), (2)
 │              ├── Clause (দফা) — (ক), (a)
 │              └── Sub-clause — (অ), (aa), (ii), (xial)
 └── Schedule (তফসিল)
```

### Amendment Act Handling

When an amendment act inserts a new section into the target law, the parser detects it:

```
Pattern: "ধারা ৩৭ এর পর নিম্নরূপ নূতন ধারা ৩৭ক সন্নিবেশিত"
Captures: inserted_section_number = "৩৭ক"
```

The citation reflects this: `Section 2 (inserted Section 37A)`

### False Positive Prevention

The clause identifier validator excludes non-legal parenthetical text:
- `(review)` — English word, not a sub-clause
- `(GEMS)`, `(PMIS)` — acronyms, not clause markers
- `(Amendment)` — descriptive text, not a clause

Only legal identifiers are accepted: digits (1-3 chars), Bangla consonants, 1-3 lowercase ASCII letters, and specials (`xial`, `xiaa`).

---

## Metadata Extraction

### Act-level metadata

| Field | Example | Source |
|---|---|---|
| `act_name` | `Representation of the People (Amendment) Act, 2026` | First page header lines |
| `bangla_name` | `সরকারি চাকরি (সংশোধন) আইন, ২০২৬` | First page header lines |
| `act_number` | `3` | `( ২০২৬ সনের ০৩ নং আইন )` |
| `year` | `2026` | Year from act number pattern |
| `publication_date` | `১০ এপ্রিল, ২০২৬` | Date in brackets |
| `document_type` | `Amendment Act` | Detected from text (Act, Ordinance, Rule, Amendment) |
| `language` | `bn` / `en` / `mixed` | Character ratio analysis |
| `amendment_of` | `Government Service Act, 2018` | Pattern: `... সংশোধনকল্পে প্রণীত আইন` |

### Cross-references

References are extracted with types:

| Type | Pattern | Example |
|---|---|---|
| `act` | `XXXX সনের XX নং আইন` | `2018 সনের 57 নং আইন` |
| `ordinance` | `XXXX সনের XX নং অধ্যাদেশ` | `2025 সনের 26 নং অধ্যাদেশ` |
| `article` | `Article X` | `Article 90E` |
| `section` | `Section X` | `Section 5` |
| `order` | `P.O. No. X of XXXX` | `P.O. No. 155 of 1972` |
| `rule` | `Rules` / `Regulations` | `বিধিমালা` |
| `amends` | Detected from amendment_of | The target law being amended |

---

## Text Cleaning

The cleaner removes artifacts common to bdlaws PDFs:

| Artifact | Detection Method |
|---|---|
| Page headers (date, act name) | Cross-page frequency analysis — lines appearing on >60% of pages near the top |
| Page footers (URL, page X/Y) | URL pattern `bdlaws.minlaw.gov.bd` + page number pattern `X/Y` |
| Isolated Bangla diacritics | Lines where >50% of characters are vowel signs with length ≤4 |
| Copyright notices | Keyword match: `Copyright`, `Legislative and Parliamentary Affairs Division` |
| Access date prefix | Regex `\d{2}/\d{2}/\d{4}` stripped from every line |
| Duplicate whitespace | Collapsed to single spaces, paragraphs preserved |
| Hyphenated line-break words | `(\w)-\n(\w)` reunited |

---

## Qdrant Configuration

### Collection

- **Name**: `bangladesh_laws` (configurable)
- **Vector size**: 1024 dimensions (BGE-M3 output)
- **Distance**: Cosine
- **Mode**: Dense vectors only

### Payload Indexes

Indexed for filtered search:

| Index Field | Type |
|---|---|
| `metadata.act_name` | Keyword |
| `metadata.year` | Integer |
| `metadata.document_type` | Keyword |
| `metadata.source_pdf` | Keyword |
| `metadata.language` | Keyword |
| `metadata.hierarchy.section` | Keyword |
| `metadata.hierarchy.article` | Keyword |
| `metadata.hierarchy.chapter` | Keyword |
| `metadata.hierarchy.clause` | Keyword |
| `metadata.hierarchy.sub_clause` | Keyword |
| `chunk_type` | Keyword |

### Query Examples

```python
# All chunks from a specific act
Filter(must=[FieldCondition(key="metadata.act_name", match=MatchValue(value="Penal Code"))])

# All section-type chunks from the Penal Code
Filter(must=[
    FieldCondition(key="metadata.act_name", match=MatchValue(value="Penal Code")),
    FieldCondition(key="chunk_type", match=MatchValue(value="section"))
])

# Only validated, complete chunks
Filter(must=[FieldCondition(key="validation.is_complete_chunk", match=MatchValue(value=True))])

# All amendment sections that modify Article 90E
Filter(must=[
    FieldCondition(key="metadata.hierarchy.article", match=MatchValue(value="90E")),
    FieldCondition(key="chunk_type", match=MatchValue(value="article"))
])
```

---

## Resumability

The pipeline uses SQLite (`logs/ingestion_state.db`) to track every file. If interrupted mid-batch, re-running skips already-processed files.

```sql
-- Schema
CREATE TABLE ingestion_state (
    filepath TEXT PRIMARY KEY,
    filename TEXT,
    status TEXT NOT NULL DEFAULT 'pending',   -- pending | done | failed
    chunks_count INTEGER DEFAULT 0,
    error TEXT,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

Failed files are logged to `logs/ingest_errors.log` and can be retried with `--no-resume`.

---

## Programmatic Usage

```python
from ingestion import IngestionPipeline

# Create pipeline
pipeline = IngestionPipeline(
    data_dir="./data",
    collection_name="bangladesh_laws",
    embedding_batch_size=32,
    max_tokens=1000,
    resume=True,
)

# Process all PDFs
result = pipeline.run()
print(f"{result.succeeded}/{result.total_files} files, {result.total_chunks} chunks")

# Process a single file
single = pipeline.run_file("data/act-print-1630.pdf")
if single.success:
    print(f"Act: {single.metadata.act_name}")
    print(f"Sections: {single.sections}, Chunks: {single.chunks_count}")

# Check status
status = pipeline.status()
print(status["collection"])

# Use individual components
from ingestion.loader import PDFLoader
from ingestion.cleaner import TextCleaner
from ingestion.metadata import MetadataExtractor
from ingestion.structure_parser import StructureParser
from ingestion.chunker import LegalChunker

loader = PDFLoader()
doc = loader.load("data/act-print-1630.pdf")

cleaner = TextCleaner(preserve_first_header=True)
cleaned = cleaner.clean(doc.pages)

meta_extractor = MetadataExtractor()
meta = meta_extractor.extract(cleaned.cleaned_text, doc.filename, doc.pages[0])

parser = StructureParser()
parsed = parser.parse(cleaned.cleaned_text)

chunker = LegalChunker(max_tokens=1000)
chunks = chunker.chunk(parsed, meta, doc.total_pages)

for chunk in chunks:
    print(f"[{chunk.chunk_type.value}] {chunk.citation}")
    print(f"  hierarchy: {chunk.hierarchy}")
    print(f"  validation: {chunk.validation}")
```

---

## Environment Configuration

Required in `.env`:

```
QDRANT_URL=https://your-cluster-id.qdrant.io
QDRANT_API_KEY=your_api_key
```

Optional:

```
INGEST_BATCH_SIZE=32     # Embedding batch size (default: 32)
```

---

## CLI Reference

```
usage: ingest.py [-h] [--dir DIR] [--file FILE] [--no-resume] [--status]
                 [--collection COLLECTION] [--batch-size BATCH_SIZE]

Options:
  --dir DIR               Directory containing PDF files (default: ./data)
  --file FILE             Process a single PDF file
  --no-resume             Re-process already ingested files
  --status                Show ingestion status and exit
  --collection COLLECTION Qdrant collection name (default: bangladesh_laws)
  --batch-size BATCH_SIZE Embedding batch size (default: 32)
```

---

## Unit Tests

Run the test suite:

```bash
python tests/test_ingestion.py
```

33 tests across 6 classes:
- `TestClauseIdentifier` — 11 tests verifying clause validation logic
- `TestHierarchyParsing` — 7 tests for section/clause/article parsing
- `TestCitationGeneration` — 5 tests for citation format
- `TestReferenceExtraction` — 3 tests for structured references
- `TestChunkBoundaries` — 3 tests for boundary detection
- `TestChunkValidation` — 6 tests for chunk quality validation

---

## Token Estimation

The pipeline uses the BGE-M3 tokenizer (`XLM-RoBERTa` SentencePiece) for accurate token counting. If unavailable (e.g., first run without cache), it falls back to a character estimate:

- Bangla characters: 1 token ≈ 2.5 characters
- Other characters: 1 token ≈ 4.0 characters

The tokenizer is lazily loaded once and cached via `@lru_cache`.

---

## Performance Notes

- **Embedding model**: BGE-M3 via sentence-transformers, runs on CPU by default. First run downloads the model from HuggingFace (~2.2 GB).
- **Batch embedding**: Configurable via `embedding_batch_size` (default 32). Larger batches improve throughput but use more memory.
- **Resumability**: SQLite tracking means restarts skip processed files. No re-embedding overhead.
- **Memory**: Each page is loaded and cleaned separately before concatenation. No memory issues even for 100+ page documents.
- **1500+ documents**: The pipeline works without code changes. Processing time scales linearly with the number of documents and pages.

---

## Troubleshooting

### "Collection 'bangladesh_laws' already exists"

Normal. The pipeline reuses existing collections. Delete via `qdrant_management/delete_old_data.py` if you need a fresh start.

### "No legal sections detected"

The document may use a non-standard structure. Check the cleaned text in `debug_cleaned_*.txt`. The parser expects Bengali numerals followed by danda (`১।`, `২।`) for section markers.

### "Chunks dropped by validation"

Check `logs/ingest_errors.log`. Common causes:
- Sections too small (<20 chars of useful text)
- Missing year metadata (check the PDF's first page for act number patterns)
- Chunks starting with `)` — indicates a paragraph split inside a provision

### Model download timeout

Set `HF_HUB_DOWNLOAD_TIMEOUT=300` in `.env` for slow connections. The model is cached after first download.

### Out of memory

Reduce `--batch-size` to 8 or 16. Each 1024-dim vector uses ~4 KB.
