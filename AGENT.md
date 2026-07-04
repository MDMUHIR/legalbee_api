# Legal Bee RAG Agent — Documentation

## Overview

**LEEGAL BEE** is a production-grade Retrieval-Augmented Generation (RAG) system for Bangladeshi law. It answers legal questions strictly from a vector database of Bangladeshi Acts, Ordinances, Rules, and Amendments — never hallucinating or fabricating information.

The system is built with **LangChain**, **LangGraph**, **FastAPI**, **Qdrant**, and **BAAI/bge-m3** embeddings. It supports both **English** and **Bengali** queries.

---

## Quick Start

```bash
# Start the server
python -m app.main
# or
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Open interactive API docs
http://localhost:8000/api/docs
```

Test from the command line:

```powershell
Invoke-RestMethod -Uri http://localhost:8000/api/chat -Method Post `
  -ContentType "application/json" `
  -Body '{"question":"What is the punishment for theft under Bangladeshi law?","language":"en"}'
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        USER QUERY                            │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  1. detect_intent                                            │
│     - Language detection (en / bn)                          │
│     - Query type (legal_question, law_search, etc.)         │
│     - Legal domain (criminal, election, civil, etc.)        │
│     - Rewrite check                                         │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  2. rewrite_query (conditional)                              │
│     - LLM expands vague queries                             │
│     - "What is nomination?" → "What are the nomination      │
│       requirements under the Representation of the People   │
│       Order, 1972?"                                         │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  3. retrieve                                                 │
│     - BGE-M3 query embedding                                │
│     - Dense search in Qdrant                                │
│     - Optional metadata filtering                           │
│     - Score threshold                                       │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  4. build_context                                            │
│     - Format chunks with citations + relevance scores       │
│     - Deduplicate                                            │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  5. generate_answer                                          │
│     - System prompt with strict grounding rules             │
│     - LLM generates structured markdown answer              │
│     - Confidence scoring based on retrieval count            │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  6. build_citations                                          │
│     - Extract citations from chunk metadata                 │
│     - Extract structured references (amends, acts, etc.)    │
│     - Never generates citations manually                    │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                      STRUCTURED RESPONSE                      │
│  { question, answer, answer_markdown, confidence,            │
│    citations, references, retrieved_chunks,                  │
│    execution_time_ms, token_usage }                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
app/
├── main.py                    # FastAPI entry point
├── config.py                  # Environment configuration
├── models/
│   └── schemas.py             # Pydantic request/response models
├── agents/
│   ├── legal_agent.py         # LegalBeeAgent class (chat, search, analyze, summarize)
│   └── workflow.py            # LangGraph StateGraph (6-node pipeline)
├── retrieval/
│   ├── retriever.py           # Qdrant dense search + metadata filtering
│   ├── query_rewriter.py      # LLM-based query expansion
│   ├── citation_builder.py    # Extract citations from chunk metadata
│   └── intent_detector.py     # Query type, legal domain, language detection
├── services/
│   ├── qdrant_service.py      # QdrantClient wrapper (singleton)
│   ├── embedding_service.py   # BGE-M3 query embedding
│   └── llm_service.py         # Provider-agnostic LLM (Groq / Gemini)
├── prompts/
│   ├── system_prompt.py       # EN/BN system prompts with grounding rules
│   └── answer_prompt.py       # Summary + fact analysis prompts
└── api/
    └── routes.py              # FastAPI route handlers

tests/
└── test_agent.py              # 24 unit tests
```

---

## API Endpoints

### `POST /api/chat` — Legal Question Answering

Ask any legal question. The agent retrieves relevant law and generates a grounded answer.

**Request:**

```json
{
  "question": "What is the punishment for theft under Bangladeshi law?",
  "language": "en",
  "user_type": "general",
  "conversation_id": null
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `question` | string | Yes | Legal question (1-4000 chars) |
| `language` | string | No | `"en"` or `"bn"` (auto-detected) |
| `user_type` | string | No | `"lawyer"` for detailed analysis, `"general"` for plain language (default) |
| `conversation_id` | string | No | For multi-turn conversations (future use) |

**Response:**

```json
{
  "question": "What is the punishment for theft under Bangladeshi law?",
  "answer": "## Answer\n\nAccording to the Penal Code, 1860, Section 379...",
  "answer_markdown": "## Answer\n\n...",
  "language_detected": "en",
  "query_type": "legal_question",
  "confidence": "high",
  "citations": [
    "Penal Code, 1860, Section 379",
    "Penal Code, 1860, Section 380"
  ],
  "retrieved_chunks": [
    {
      "text": "Whoever commits theft shall be punished...",
      "chunk_id": "uuid",
      "score": 0.87,
      "chunk_type": "section",
      "citation": "Penal Code, 1860, Section 379",
      "act_name": "Penal Code",
      "year": 1860,
      "hierarchy": {"section": "379", "clause": "", "sub_clause": ""},
      "references": []
    }
  ],
  "references": [
    {"type": "act", "target": "Penal Code, 1860"}
  ],
  "execution_time_ms": 1234,
  "token_usage": {"prompt_tokens": 500, "completion_tokens": 200},
  "timestamp": "2026-01-01T00:00:00"
}
```

---

### `POST /api/search` — Law Search

Search the legal database with optional metadata filters. No LLM generation — returns raw retrieved context.

**Request:**

```json
{
  "query": "Article 90E",
  "filters": {
    "act_name": "Representation of the People Order, 1972"
  },
  "language": "en",
  "top_k": 8
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `query` | string | Yes | Search query |
| `filters` | object | No | Metadata filters: `act_name`, `year`, `document_type`, `language`, `chunk_type`, `hierarchy.section`, `hierarchy.article`, `hierarchy.chapter` |
| `language` | string | No | Language hint |
| `top_k` | int | No | Results count (1-50, default 8) |

**Example filters:**

```json
{"act_name": "Penal Code", "year": 1860}
{"document_type": "Amendment Act"}
{"hierarchy.section": "302"}
{"chunk_type": "section", "language": "bn"}
```

---

### `POST /api/analyze` — Fact Analysis

Describe a real-world scenario. The agent retrieves relevant law and analyzes your situation.

**Request:**

```json
{
  "facts": "My landlord has increased the rent by 50% without any written notice. What are my rights?",
  "language": "en"
}
```

**Response:** Structured markdown analysis with legal basis, party positions, and potential remedies.

---

### `POST /api/summary` — Act Summary

Get a structured summary of any Bangladeshi legal act in the database.

**Request:**

```json
{
  "act_name": "Representation of the People (Amendment) Act, 2026",
  "language": "en"
}
```

**Response:** Markdown summary with Purpose, Key Provisions, Amendments Made, and Important Sections — each with section citations.

---

### `GET /api/health` — Health Check

```json
{
  "status": "healthy",
  "service": "Legal Bee API",
  "version": "1.0.0",
  "timestamp": "2026-01-01T00:00:00"
}
```

### `GET /api/languages` — Supported Languages

```json
{
  "supported_languages": [
    {"code": "en", "name": "English"},
    {"code": "bn", "name": "Bengali (বাংলা)"}
  ]
}
```

---

## Answer Format

Every chat answer follows this structured markdown format:

```markdown
# Answer
[Concise, direct answer]

## Legal Basis
- Act: [name], [year]
- Section: [number]
- Article: [number]
- Clause: [letter/number]

## Relevant Legal Text
> [Quoted excerpt from the law]

## Explanation
[How the law applies to the question]

## References
[List of all cited legal provisions]

## Confidence
[High / Medium / Low]

---
⚠️ This is for informational purposes only. Please consult a licensed lawyer for legal advice.
```

---

## Anti-Hallucination Rules

The system prompt enforces six strict rules that the LLM cannot violate:

1. **Answer ONLY from retrieved context** — never use training knowledge
2. **If context is empty, say so** — never fill gaps
3. **Always cite** — Act name, year, section, article, clause
4. **No fabrication** — never invent laws, sections, or citations
5. **No speculation** — do not infer or extrapolate beyond context
6. **Citations are pre-built** — the LLM never generates them from scratch

**When no results are found:**

```
I could not find sufficient legal information in the available
Bangladeshi law database for this query.

Please try rephrasing your question or ask about a different
legal topic.
```

---

## Confidence Scoring

| Level | Chunks Retrieved | Meaning |
|---|---|---|
| **High** | ≥5 | Strong retrieval coverage; answer is well-grounded |
| **Medium** | 2-4 | Some relevant context found; answer may be partial |
| **Low** | 0-1 | Limited or no context; answer is a disclaimer |

---

## Language Detection

Automatic detection based on Unicode character ratio:

- If >30% of alphabetic characters are Bengali (U+0980-U+09FF) → **Bengali**
- Otherwise → **English**

Users can override by passing `"language": "bn"` or `"language": "en"`.

---

## Query Types

| Type | Description | Example |
|---|---|---|
| `legal_question` | General law question | "What is the punishment for theft?" |
| `law_search` | Specific provision lookup | "Show me Article 90E" |
| `act_summary` | Request to summarize an act | "Summarize the Penal Code" |
| `amendment_question` | Question about an amendment | "What changed in the 2026 Amendment Act?" |
| `fact_analysis` | Real-world scenario | "My landlord evicted me illegally" |
| `no_results` | No retrieval results | — |

## Legal Domains

| Domain | Keywords |
|---|---|
| `criminal` | punishment, crime, theft, murder, offence |
| `election` | election, vote, candidate, nomination |
| `employment` | employment, service, employee, government servant |
| `constitution` | constitution, fundamental right |
| `administrative` | government, authority, administrative |
| `tax` | tax, income, revenue, VAT |
| `civil` | property, contract, land, family |

---

## Query Rewriting

Short, vague queries are expanded by the LLM before retrieval for better search results.

**Trigger:** Question has <5 words AND contains no section/article reference.

**Example:**

| Original | Rewritten |
|---|---|
| "What is nomination?" | "What are the legal requirements for nomination of candidates under the Representation of the People Order, 1972 in Bangladesh election law?" |
| "চুরির শাস্তি" | "বাংলাদেশের দণ্ডবিধি অনুযায়ী চুরির অপরাধের শাস্তি কী?" |

---

## Configuration

All settings via `.env` file:

```env
# Required
QDRANT_URL=https://your-cluster-id.qdrant.io
QDRANT_API_KEY=your_api_key

# Optional — with defaults
COLLECTION_NAME=bangladesh_laws
DENSE_VECTOR_NAME=law_dense_vector
EMBEDDING_MODEL=BAAI/bge-m3

# LLM
LLM_PROVIDER=groq                                          # groq | gemini
LLM_MODEL=llama-3.3-70b-versatile                          # or gemini-2.5-pro
LLM_TEMPERATURE=0
LLM_MAX_TOKENS=2048

# API keys
GROQ_API_KEY=gsk_xxx                                       # for Groq
GOOGLE_API_KEY=xxx                                         # for Gemini

# Retrieval
TOP_K=8
SCORE_THRESHOLD=0.3
USE_RERANKER=false
USE_HYBRID_SEARCH=false

# Resilience
MAX_RETRIES=2
REQUEST_TIMEOUT=60
```

### Switching to Gemini 2.5 Pro

```env
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.5-pro
GOOGLE_API_KEY=your_google_api_key
```

---

## Programmatic Usage

```python
from app.agents.legal_agent import LegalBeeAgent

agent = LegalBeeAgent()

# Chat
response = agent.chat(
    question="What is the punishment for theft?",
    language="en",
    user_type="lawyer",
)
print(response.answer)
print(response.citations)

# Search
response = agent.search(
    query="Article 90E",
    filters={"act_name": "Representation of the People Order, 1972"},
    top_k=10,
)

# Fact analysis
response = agent.analyze(
    facts="My landlord increased rent without notice.",
    language="en",
)

# Act summary
response = agent.summarize(
    act_name="Penal Code, 1860",
    language="en",
)
```

### Using Individual Components

```python
from app.retrieval.retriever import LegalRetriever
from app.retrieval.intent_detector import IntentDetector
from app.retrieval.citation_builder import CitationBuilder

# Language detection
detector = IntentDetector()
lang = detector.detect_language("আইন কী")
domain = detector.detect_legal_domain("What is the punishment?")

# Direct retrieval
retriever = LegalRetriever()
chunks = retriever.retrieve("theft punishment", top_k=5)
chunks = retriever.retrieve_by_act("Penal Code")
chunks = retriever.retrieve_by_section("379", act_name="Penal Code")

# Citations from chunks
builder = CitationBuilder()
citations = builder.build(chunks)
references = builder.build_references(chunks)
context = builder.build_context_string(chunks)
```

---

## Qdrant Payload Structure

The agent reads from the `bangladesh_laws` collection. Chunks have this structure:

```json
{
  "text": "The actual legal provision text...",
  "chunk_type": "section",
  "citation": "Penal Code, 1860, Section 379",
  "metadata": {
    "act_name": "Penal Code",
    "bangla_name": "দণ্ডবিধি",
    "act_number": "45",
    "year": 1860,
    "document_type": "Act",
    "source_pdf": "act-print-0045.pdf",
    "volume": "45",
    "language": "en",
    "hierarchy": {
      "part": "",
      "chapter": "XVII",
      "article": "",
      "section": "379",
      "clause": "",
      "sub_clause": ""
    }
  },
  "validation": {
    "validated": true,
    "starts_at_boundary": true,
    "ends_at_boundary": true,
    "is_complete_chunk": true
  },
  "references": [
    {"type": "act", "target": "Indian Penal Code, 1860"}
  ]
}
```

### Filtering by Metadata

The `/search` endpoint supports these filter keys:

| Key | Example |
|---|---|
| `act_name` | `"Penal Code"` |
| `year` | `1860` |
| `document_type` | `"Amendment Act"` |
| `language` | `"bn"` |
| `source_pdf` | `"act-print-1630.pdf"` |
| `chunk_type` | `"section"` |
| `hierarchy.section` | `"379"` |
| `hierarchy.article` | `"90E"` |
| `hierarchy.chapter` | `"XVII"` |
| `hierarchy.clause` | `"2"` |
| `hierarchy.sub_clause` | `"aa"` |

---

## Tests

57 unit tests (33 ingestion + 24 agent). Run:

```bash
python tests/test_agent.py
python tests/test_ingestion.py
```

### Test Coverage

| Module | Tests | Coverage |
|---|---|---|
| `IntentDetector` | 15 | Language, query type, domain, rewrite check |
| `CitationBuilder` | 6 | Pre-built citations, hierarchy fallback, dedup, context format, references |
| `ClauseIdentifier` | 11 | Digits, consonants, ASCII, special IDs, word/acronym rejection |
| `HierarchyParsing` | 7 | Section detection, clauses, article extraction, inserted section |
| `CitationGeneration` | 5 | Section, full hierarchy, amendment, preamble, empty |
| `ReferenceExtraction` | 3 | Amendment refs, empty, typed references |
| `Config` | 2 | Loading, validation |
| `Schemas` | 2 | Request defaults, response defaults |
| `ChunkBoundaries` | 3 | Parenthesis detection, boundary starts |
| `ChunkValidation` | 6 | Empty/short text, missing fields, preamble, validation metadata |

---

## Error Handling

| Scenario | Behavior |
|---|---|
| Qdrant unavailable | Returns 500 with error detail; logged |
| LLM failure | Returns error message in answer; `AgentResponse.error` populated |
| No retrieval results | Returns grounded disclaimer (EN/BN) |
| Invalid request | Returns 400 with validation detail |
| Timeout | Handled by FastAPI; configurable via `REQUEST_TIMEOUT` |

---

## Logging

All operations log to stdout with timestamps:

```
HH:MM:SS | INFO    | app.agents.workflow | Intent: type=legal_question domain=criminal lang=en rewrite=False
HH:MM:SS | INFO    | app.retrieval.retriever | Retrieved 8 chunks for query '...' in 245 ms
HH:MM:SS | INFO    | app.retrieval.query_rewriter | Query rewritten: '...' -> '...'
HH:MM:SS | ERROR   | app.agents.workflow | LLM generation failed: ...
```

---

## Ingestion Pipeline

The RAG agent reads from the database populated by the ingestion pipeline. See `INGESTION.md` for full documentation.

```bash
# Ingest PDFs
uv run ingest.py

# Check status
uv run ingest.py --status

# Re-process
uv run ingest.py --no-resume
```

The ingestion pipeline handles:
- PDF loading via PyMuPDF
- Header/footer/noise removal
- Structure-aware parsing (sections, articles, clauses, sub-clauses)
- Legal-aware chunking (one concept per chunk)
- BGE-M3 1024-dim embeddings
- Qdrant storage with nested metadata

---

## Deployment

### Development

```bash
python -m app.main
```

### Production (uvicorn)

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker

```dockerfile
FROM python:3.13
WORKDIR /app
COPY . .
RUN pip install .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Troubleshooting

### "QDRANT_URL is required"

Set `QDRANT_URL` and `QDRANT_API_KEY` in `.env`.

### "No module named 'app'"

Run from the project root: `cd "E:\Legal Bee"`.

### No results for valid queries

- Check the collection has ingested data: `uv run ingest.py --status`
- Lower the score threshold: `SCORE_THRESHOLD=0.15`
- Increase retrieval size: `TOP_K=15`

### LLM errors

- Verify API key in `.env`
- Check provider: `LLM_PROVIDER=groq` requires `GROQ_API_KEY`
- Try a different model: `LLM_MODEL=llama-3.3-70b-versatile`

### Slow first query

BGE-M3 model is loaded on first use (~2.2 GB download, cached after). Subsequent queries are fast.
