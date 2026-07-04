# Legal Bee

A production-grade **Retrieval-Augmented Generation (RAG)** system for **Bangladeshi law**. Answers legal questions strictly from a vector database of Bangladeshi Acts, Ordinances, Rules, and Amendments — never hallucinates. Supports **English** and **Bengali (বাংলা)** queries.

---

## Features

- **Multi-Agent LangGraph Workflow** — 5 specialized agents (Law Search, Legal Analysis, Act Summary, Amendment Compare, Legal QA) behind a query router
- **Anti-Hallucination Guardrails** — 6 strict rules: answers only from retrieved context, mandatory citations, no fabrication or speculation
- **Bilingual** — automatic language detection (English / Bengali) with native Bangla system prompts
- **Structure-Aware Ingestion** — PDFs parsed into legal hierarchy (Part → Chapter → Section → Clause → Sub-clause), chunked at semantic boundaries
- **BGE-M3 Embeddings** — 1024-dim multilingual dense vectors, optimized for crossover between English and Bangla legal text
- **Rich API** — `/chat`, `/search`, `/analyze`, `/summary` endpoints with metadata filtering, confidence scoring, and structured citations
- **Resumable Pipeline** — SQLite-tracked ingestion tolerates interruptions without re-processing

---

## Architecture

```
User Query
    │
    ▼
Intent Detection + Language Detection
    │
    ▼
Query Rewriter (expands vague queries)
    │
    ▼
┌─────────────── Query Router ───────────────┐
│    Law Search │ Analysis │ Summary │ Amend │ QA    │
└──────────────────┬─────────────────────────┘
                   ▼
    Qdrant Dense Search (BGE-M3, 1024-dim)
                   ▼
    Citation Verification → Structured Response
```

**Ingestion Pipeline**: `PDF → PyMuPDF → Clean → Extract Metadata → Parse Hierarchy → Legal Chunking → BGE-M3 Embed → Qdrant`

---

## Tech Stack

| Component | Technology |
|---|---|
| Framework | FastAPI + LangChain + LangGraph |
| Vector DB | Qdrant Cloud (Cosine distance, HNSW index) |
| Embeddings | `BAAI/bge-m3` (1024-dim, multilingual) |
| LLM | Groq (`llama-3.3-70b-versatile`) or Google Gemini (`gemini-2.5-pro`) |
| PDF | PyMuPDF (fitz) |
| Tracking | SQLite (resumable ingestion state) |
| Python | 3.13+, managed with UV |

---

## Quick Start

### Prerequisites

- Python 3.13+
- [UV](https://docs.astral.sh/uv/) package manager
- Qdrant Cloud account (or self-hosted Qdrant)

### Setup

```bash
# Clone & enter
cd "E:\Legal Bee"

# Install dependencies
uv sync

# Copy and fill in your credentials
cp .env.example .env
```

### Configure `.env`

```env
QDRANT_URL=https://your-cluster-id.qdrant.io
QDRANT_API_KEY=your_api_key
COLLECTION_NAME=bangladesh_laws
LLM_PROVIDER=groq                          # groq | gemini
LLM_MODEL=llama-3.3-70b-versatile
GROQ_API_KEY=gsk_xxx                       # for Groq
# GOOGLE_API_KEY=xxx                       # for Gemini
```

### Ingest Laws

```bash
# Process all PDFs in ./data
uv run ingest.py

# Check status
uv run ingest.py --status
```

### Start the API

```bash
python -m app.main
# or
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000/api/docs** for interactive Swagger docs.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/chat` | Legal Q&A with grounded answers and citations |
| `POST` | `/api/search` | Raw law search with metadata filters (no LLM) |
| `POST` | `/api/analyze` | Fact scenario analysis against relevant laws |
| `POST` | `/api/summary` | Structured summary of a legal act |
| `GET` | `/api/health` | Health check |
| `GET` | `/api/languages` | Supported languages |

### Example

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the punishment for theft under Bangladeshi law?","language":"en"}'
```

```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query":"Article 90E","filters":{"act_name":"Representation of the People Order, 1972"},"top_k":10}'
```

---

## Answer Format

Every response includes:

```markdown
# Answer
[Concise, direct answer]

## Legal Basis
[Act name, year, section, article, clause]

## Relevant Legal Text
> [Quoted excerpt from the law]

## Explanation
[How the law applies]

## References
[Deduplicated legal citations]

## Confidence
[High / Medium / Low]
```

---

## Anti-Hallucination Rules

1. Answer **ONLY** from retrieved context — never use training knowledge
2. If context is empty, say so — never fill gaps
3. Always cite — Act name, year, section, article, clause
4. No fabrication — laws, sections, or citations
5. No speculation — do not infer beyond context
6. Citations are **pre-built** from ingestion metadata, never LLM-generated

---

## Project Structure

```
E:\Legal Bee\
├── app/                          # FastAPI RAG application
│   ├── main.py                   # Entry point
│   ├── config.py                 # Environment configuration
│   ├── api/routes.py             # Route handlers
│   ├── agents/
│   │   ├── legal_agent.py        # LegalBeeAgent class
│   │   └── workflow.py           # LangGraph StateGraph
│   ├── retrieval/
│   │   ├── retriever.py          # Qdrant dense search
│   │   ├── query_rewriter.py     # LLM query expansion
│   │   ├── citation_builder.py   # Citation extraction
│   │   └── intent_detector.py    # Intent + language detection
│   ├── services/
│   │   ├── qdrant_service.py     # Qdrant client singleton
│   │   ├── embedding_service.py  # BGE-M3 query embedding
│   │   └── llm_service.py        # Groq/Gemini abstraction
│   ├── prompts/
│   │   ├── system_prompt.py      # EN/BN prompts per agent
│   │   └── answer_prompt.py      # Summary + analysis prompts
│   └── models/schemas.py         # Pydantic request/response models
│
├── ingestion/                    # PDF → Qdrant pipeline
│   ├── loader.py                 # PyMuPDF loading
│   ├── cleaner.py                # Header/footer/noise removal
│   ├── metadata.py               # Act metadata + references
│   ├── structure_parser.py       # Legal hierarchy parsing
│   ├── chunker.py                # Legal-aware chunking
│   ├── embeddings.py             # BGE-M3 embedding
│   ├── qdrant_store.py           # Qdrant collection management
│   ├── pipeline.py               # Orchestrator + SQLite tracking
│   └── utils.py                  # Token estimation, Unicode helpers
│
├── data/                         # PDF law files
├── tests/                        # 57 unit tests
│   ├── test_agent.py             # 24 agent tests
│   └── test_ingestion.py         # 33 ingestion tests
├── frontend.html                 # Web UI (4 tabs)
├── pyproject.toml                # Dependencies
├── .env.example                  # Environment template
├── AGENT.md                      # Agent documentation
├── API_DOCUMENTATION.md          # API reference
└── INGESTION.md                  # Ingestion pipeline docs
```

---

## Programmatic Usage

```python
from app.agents.legal_agent import LegalBeeAgent

agent = LegalBeeAgent()

# Chat
response = agent.chat(question="What is the punishment for theft?", language="en")
print(response.answer)
print(response.citations)

# Search
response = agent.search(query="Article 90E", filters={"act_name": "RPO, 1972"}, top_k=10)

# Fact analysis
response = agent.analyze(facts="My landlord evicted me without notice.", language="en")

# Act summary
response = agent.summarize(act_name="Penal Code, 1860", language="en")
```

---

## Testing

```bash
python tests/test_agent.py       # 24 tests (intent, citations, config, schemas)
python tests/test_ingestion.py   # 33 tests (clauses, hierarchy, chunking, validation)
```

---

## Deployment

### Local

```bash
# Development
python -m app.main

# Production (4 workers)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Hugging Face Spaces (Docker)

1. Create a new Space at https://huggingface.co/new-space with **Docker** as the SDK
2. Clone the Space repo and copy this project into it (or push directly)
3. Set the following **Secrets** in your Space settings (Settings → Repository secrets):

   | Secret | Description |
   |---|---|
   | `QDRANT_URL` | Your Qdrant Cloud cluster URL |
   | `QDRANT_API_KEY` | Your Qdrant Cloud API key |
   | `GROQ_API_KEY` | Your Groq API key (or `GOOGLE_API_KEY` for Gemini) |
   | `LLM_PROVIDER` | `groq` or `gemini` |
   | `LLM_MODEL` | e.g. `llama-3.3-70b-versatile` |

4. The Space auto-builds from the `Dockerfile`. The app starts on port `7860` — the frontend UI is served at `/` and the API docs at `/api/docs`.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `QDRANT_URL is required` | Set `QDRANT_URL` and `QDRANT_API_KEY` in `.env` |
| No results for valid queries | Lower `SCORE_THRESHOLD=0.15`, increase `TOP_K=15` |
| First query is slow | BGE-M3 downloads on first use (~2.2 GB, cached after) |
| Ingestion interrupted | Re-run — SQLite tracker skips processed files |
| LLM errors | Verify API key and `LLM_PROVIDER` setting |
| "No legal sections detected" | Document may use non-standard structure; check debug output |

---

For detailed documentation, see [`AGENT.md`](AGENT.md), [`API_DOCUMENTATION.md`](API_DOCUMENTATION.md), and [`INGESTION.md`](INGESTION.md).
