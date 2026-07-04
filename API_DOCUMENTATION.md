# Legal Bee API Documentation

## Overview

**LEEGAL BEE** is a multi-agent Legal RAG system for Bangladeshi law. It uses a LangGraph workflow with 5 specialized agents behind a query router. All answers are grounded in a Qdrant vector database of Bangladeshi Acts, Ordinances, Rules, and Amendments.

**Base URL:** `http://localhost:8000`

**Interactive Docs:** `http://localhost:8000/api/docs` (Swagger) | `http://localhost:8000/api/redoc` (ReDoc)

---

## Architecture

```
User Query
    │
    ▼
Query Router (intent detection + language detection)
    │
    ├── law_search_agent         ← "Show me Article 90E"
    ├── legal_analysis_agent     ← "My landlord evicted me"
    ├── act_summary_agent        ← "Summarize the Penal Code"
    ├── amendment_compare_agent  ← "What changed in the 2026 Amendment?"
    └── legal_qa_agent           ← "What is the punishment for theft?"
            │
            ▼
    Qdrant Retriever (BGE-M3 dense search)
            │
            ▼
    Citation Verification → Structured Response
```

---

## Endpoints

### 1. `POST /api/chat` — Legal Question Answering

Ask any legal question. The system auto-detects intent, routes to the appropriate specialized agent, retrieves relevant law, and generates a grounded answer with citations.

**Request:**

```json
{
  "question": "What is the punishment for theft under Bangladeshi law?",
  "language": "en",
  "user_type": "general"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `question` | string | Yes | Legal question (1–4000 characters) |
| `language` | string | No | `"en"` or `"bn"` — auto-detected if omitted |
| `user_type` | string | No | `"lawyer"` for detailed analysis, `"general"` for plain language (default) |

**Response:**

```json
{
  "question": "What is the punishment for theft?",
  "answer": "# Answer\n\nAccording to the Penal Code...",
  "answer_markdown": "# Answer\n\n...",
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
      "chunk_id": "act-print-0045.pdf:s379:c5",
      "score": 0.87,
      "chunk_type": "section",
      "citation": "Penal Code, 1860, Section 379",
      "act_name": "Penal Code",
      "year": 1860,
      "hierarchy": {"part": "", "chapter": "", "article": "", "section": "379", "clause": "", "sub_clause": ""},
      "references": [{"type": "act", "target": "1860 সনের 45 নং আইন"}]
    }
  ],
  "references": [
    {"type": "act", "target": "Penal Code, 1860"}
  ],
  "execution_time_ms": 2341,
  "token_usage": {"prompt_tokens": 850, "completion_tokens": 320},
  "timestamp": "2026-07-04T22:30:00",
  "error": null
}
```

| Response Field | Type | Description |
|---|---|---|
| `answer` | string | Plain text answer |
| `answer_markdown` | string | Markdown formatted answer |
| `language_detected` | string | `"en"` or `"bn"` |
| `query_type` | string | Routed query type — see table below |
| `confidence` | string | `"high"` (5+ chunks), `"medium"` (2–4), `"low"` (0–1) |
| `citations` | string[] | Deduplicated legal citations from chunk metadata |
| `retrieved_chunks` | object[] | Full chunks with text, scores, hierarchy, references |
| `references` | object[] | Structured cross-references with `type` and `target` |
| `execution_time_ms` | float | Total wall-clock time in milliseconds |
| `token_usage` | object | LLM token usage (`prompt_tokens`, `completion_tokens`) |
| `error` | string \| null | Error message if the workflow failed |

**Query Type Values:**

| Value | Routing | Description |
|---|---|---|
| `law_search` | Law Search Agent | Specific provision lookup |
| `fact_analysis` | Legal Analysis Agent | Real-world scenario analysis |
| `act_summary` | Act Summary Agent | Act/document summarization |
| `amendment_question` | Amendment Compare Agent | What changed in an amendment |
| `legal_question` | Legal QA Agent (default) | General legal question |

---

### 2. `POST /api/search` — Law Search

Search the legal database with optional metadata filters. No LLM generation — returns raw retrieved chunks.

**Request:**

```json
{
  "query": "nomination requirements election",
  "filters": {
    "act_name": "Representation of the People Order, 1972"
  },
  "top_k": 10
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `query` | string | Yes | Search query (1–2000 characters) |
| `filters` | object | No | Metadata key-value pairs (see filter reference below) |
| `language` | string | No | `"en"` or `"bn"` |
| `top_k` | int | No | Results count (1–50, default 8) |

**Example filters:**

```json
{"act_name": "Penal Code"}
{"year": 1860}
{"document_type": "Amendment Act"}
{"chunk_type": "section"}
{"language": "bn"}
{"hierarchy.section": "302"}
{"hierarchy.article": "90E"}
{"hierarchy.chapter": "XVII"}
{"hierarchy.clause": "2"}
{"hierarchy.sub_clause": "aa"}
```

Filters can be combined:

```json
{
  "act_name": "Penal Code",
  "hierarchy.section": "379",
  "chunk_type": "section"
}
```

The `answer` field in the response contains formatted retrieved chunks with citations and relevance scores. No LLM is called.

---

### 3. `POST /api/analyze` — Fact Analysis

Describe a real-world scenario. The agent retrieves relevant law and analyzes your situation.

**Request:**

```json
{
  "facts": "My landlord increased the rent by 50% without any written notice. What are my rights?",
  "language": "en"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `facts` | string | Yes | Factual scenario (5–5000 characters) |
| `language` | string | No | `"en"` or `"bn"` |

**Response format highlights:**

```markdown
# Legal Analysis

## Applicable Laws
[List each applicable act, section, and why it applies]

## Legal Position
[Explain the legal standing of each party]

## Potential Remedies
[What legal actions can be taken]

## Important Considerations
[Deadlines, procedural requirements, risks]
```

Routed through the **Legal Analysis Agent** which uses `top_k=12` retrieval for broader coverage.

---

### 4. `POST /api/summary` — Act Summary

Get a structured summary of any Bangladeshi legal act in the database.

**Request:**

```json
{
  "act_name": "Representation of the People (Amendment) Act, 2026",
  "language": "en"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `act_name` | string | Yes | Act name (1–500 characters) |
| `language` | string | No | `"en"` or `"bn"` |

**Response format highlights:**

```markdown
# Summary: [Act Name]

## Purpose
[Why this act was enacted]

## Key Provisions
[List of the most important sections]

## Amendments Made
[What this act changed, if it is an amendment]

## Important Sections
[Critical sections with citations]
```

Routed through the **Act Summary Agent** which uses `top_k=20` retrieval and act-name metadata filtering.

---

### 5. `GET /api/health` — Health Check

```json
{
  "status": "healthy",
  "service": "Legal Bee API",
  "version": "1.0.0",
  "timestamp": "2026-07-04T22:30:00"
}
```

---

### 6. `GET /api/languages` — Supported Languages

```json
{
  "supported_languages": [
    {"code": "en", "name": "English"},
    {"code": "bn", "name": "Bengali (বাংলা)"}
  ],
  "note": "Language is auto-detected if not specified"
}
```

---

## Answer Format

Every `/chat` answer follows this structured markdown template:

```markdown
# Answer
[Concise, direct answer]

## Legal Basis
- Act: [name], [year]
- Section: [number]
- Article: [number, if applicable]
- Clause: [number/letter, if applicable]

## Relevant Legal Text
> [Quoted excerpt from the law]

## Explanation
[How the law applies to the question]

## References
[List all cited legal provisions]

## Confidence
[High / Medium / Low]

---
⚠️ This is for informational purposes only. Please consult a licensed lawyer.
```

---

## Anti-Hallucination Rules

The system enforces six strict rules:

1. **Answer ONLY from retrieved context** — never use training knowledge
2. **If context is empty, say so** — never fill gaps
3. **Always cite** — Act name, year, section, article, clause
4. **Never fabricate** — laws, sections, or citations
5. **Never speculate** — do not infer beyond context
6. **Citations are pre-built** — extracted from ingestion metadata, never LLM-generated

---

## Metadata Filter Reference

Supported filter keys for `/search` and programmatic retrieval:

| Key | Type | Example | Description |
|---|---|---|---|
| `act_name` | text | `"Penal Code"` | Full or partial act name |
| `year` | integer | `1860` | Act year |
| `document_type` | text | `"Amendment Act"` | Act, Ordinance, Rule, Amendment Act |
| `language` | text | `"bn"` | Document language |
| `source_pdf` | text | `"act-print-1630.pdf"` | Source file |
| `chunk_type` | text | `"section"` | Provision type |
| `hierarchy.section` | text | `"379"` | Section number |
| `hierarchy.article` | text | `"90E"` | Article number |
| `hierarchy.chapter` | text | `"XVII"` | Chapter identifier |
| `hierarchy.clause` | text | `"2"` | Clause identifier |
| `hierarchy.sub_clause` | text | `"aa"` | Sub-clause identifier |

---

## Language Detection

Automatic detection based on Unicode character ratio:

- \>30% alphabetic characters are Bengali (U+0980–U+09FF) → `"bn"`
- Otherwise → `"en"`

Override by passing `"language": "en"` or `"language": "bn"` in the request.

---

## Error Responses

**400 — Bad Request:**

```json
{
  "error": "Question must be at least 3 characters",
  "status_code": 400,
  "timestamp": "2026-07-04T22:30:00"
}
```

**500 — Internal Error:**

```json
{
  "error": "Internal error",
  "detail": "Qdrant search failed: connection timeout",
  "timestamp": "2026-07-04T22:30:00"
}
```

---

## Programmatic Usage

### Python

```python
import requests

BASE = "http://localhost:8000/api"

# Chat
r = requests.post(f"{BASE}/chat", json={
    "question": "What is the punishment for theft?",
    "language": "en",
    "user_type": "lawyer"
})
print(r.json()["answer"])

# Search with filters
r = requests.post(f"{BASE}/search", json={
    "query": "Article 90E",
    "filters": {"act_name": "Representation of the People Order, 1972"}
})
for chunk in r.json()["retrieved_chunks"]:
    print(f"{chunk['citation']}: {chunk['text'][:100]}")

# Analyze
r = requests.post(f"{BASE}/analyze", json={
    "facts": "My employer terminated me without notice.",
    "language": "en"
})
print(r.json()["answer"])

# Summary
r = requests.post(f"{BASE}/summary", json={
    "act_name": "Penal Code, 1860",
    "language": "en"
})
print(r.json()["answer"])
```

### JavaScript

```javascript
const BASE = "http://localhost:8000/api";

async function ask(question, userType = "general") {
  const r = await fetch(`${BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, user_type: userType }),
  });
  return r.json();
}

// Usage
const result = await ask("What is the punishment for theft?");
document.getElementById("answer").innerHTML = result.answer_markdown;
```

### cURL

```bash
# Chat
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the punishment for theft?","language":"en"}'

# Search
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query":"Article 90E","top_k":5}'

# Bengali
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"চুরির শাস্তি কী?"}'

# Health
curl http://localhost:8000/api/health
```

---

## Configuration

All settings in `.env`:

```env
# Required
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your_key

# Optional (with defaults)
COLLECTION_NAME=bangladesh_laws
EMBEDDING_MODEL=BAAI/bge-m3
LLM_PROVIDER=groq
LLM_MODEL=llama-3.3-70b-versatile
LLM_TEMPERATURE=0
LLM_MAX_TOKENS=2048
TOP_K=8
SCORE_THRESHOLD=0.3
```

---

## Deployment

```bash
# Development
python -m app.main

# Production
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```
