# Legal Bee API Documentation

## Quick Start

### Install Dependencies

```bash
uv sync
```

### Run the API Server

```bash
python app.py
```

The API will be available at: **http://localhost:8000**

## API Endpoints

### 1. Health Check

**GET** `/api/health`

Check if the API is running and connected to Qdrant Cloud.

**Response:**

```json
{
  "status": "healthy",
  "service": "Legal Bee API",
  "version": "1.0.0",
  "timestamp": "2024-01-15T10:30:45.123456"
}
```

---

### 2. Ask Legal Question

**POST** `/api/ask`

Ask a legal question about Bangladeshi law in Bengali or English.

**Request Body:**

```json
{
  "question": "What is the punishment for theft under Bangladeshi law?",
  "language": "en"
}
```

**Parameters:**

- `question` (string, required): Legal question (5-1000 characters)
- `language` (string, optional): Language hint - "en" or "bn" (auto-detected if not provided)

**Response:**

```json
{
  "question": "What is the punishment for theft?",
  "answer": "According to the Penal Code, Section 379: Punishment for theft is imprisonment for a term which may extend to three years...",
  "language_detected": "en",
  "timestamp": "2024-01-15T10:30:45.123456",
  "source": "qdrant_cloud_database"
}
```

---

### 3. Batch Questions

**POST** `/api/ask-batch`

Ask multiple legal questions in one request.

**Request Body:**

```json
[
  {
    "question": "What is the punishment for theft?",
    "language": "en"
  },
  {
    "question": "সম্পত্তির অধিকার কী?",
    "language": "bn"
  }
]
```

**Response:**

```json
{
  "count": 2,
  "results": [
    {
      "question": "What is the punishment for theft?",
      "answer": "...",
      "language_detected": "en",
      "timestamp": "2024-01-15T10:30:45.123456"
    },
    {
      "question": "সম্পত্তির অধিকার কী?",
      "answer": "...",
      "language_detected": "bn",
      "timestamp": "2024-01-15T10:30:45.123456"
    }
  ]
}
```

---

### 4. Supported Languages

**GET** `/api/languages`

Get list of supported languages.

**Response:**

```json
{
  "supported_languages": [
    {
      "code": "en",
      "name": "English",
      "example": "What is the punishment for theft under Bangladeshi law?"
    },
    {
      "code": "bn",
      "name": "Bengali (বাংলা)",
      "example": "বাংলাদেশে চুরির শাস্তি কী?"
    }
  ],
  "note": "Language is auto-detected if not specified"
}
```

---

### 5. About API

**GET** `/api/about`

Get information about Legal Bee.

**Response:**

```json
{
  "name": "Legal Bee",
  "description": "Bangladeshi Legal RAG System - Agentic Retrieval Augmented Generation",
  "version": "1.0.0",
  "features": [
    "Bengali & English support",
    "Legal database retrieval",
    "Anti-hallucination mode (database-only answers)",
    "Citation of Acts and Sections",
    "Real-time legal queries"
  ],
  "database": "Qdrant Cloud Vector Database",
  "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
  "llm_model": "Groq Llama 3.1 8B Instant"
}
```

---

## Interactive Documentation

- **Swagger UI**: http://localhost:8000/api/docs
- **ReDoc**: http://localhost:8000/api/redoc

---

## Frontend Integration Examples

### Python

```python
import requests

BASE_URL = "http://localhost:8000"

# Ask a legal question
response = requests.post(
    f"{BASE_URL}/api/ask",
    json={
        "question": "What is the punishment for theft under Bangladeshi law?",
        "language": "en"
    }
)

answer = response.json()
print(answer["answer"])
```

### JavaScript/Fetch

```javascript
const BASE_URL = "http://localhost:8000";

async function askLegalQuestion(question, language = null) {
  const response = await fetch(`${BASE_URL}/api/ask`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      question: question,
      language: language,
    }),
  });

  if (!response.ok) {
    throw new Error(`API Error: ${response.status}`);
  }

  return await response.json();
}

// Usage
askLegalQuestion("What is the punishment for theft?", "en")
  .then((data) => console.log(data.answer))
  .catch((error) => console.error(error));
```

### JavaScript/Axios

```javascript
import axios from "axios";

const BASE_URL = "http://localhost:8000";
const api = axios.create({ baseURL: BASE_URL });

async function askQuestion(question, language = null) {
  try {
    const response = await api.post("/api/ask", {
      question: question,
      language: language,
    });
    return response.data;
  } catch (error) {
    console.error("Error:", error.response?.data || error.message);
    throw error;
  }
}

// Usage
const result = await askQuestion("বাংলাদেশে চুরির শাস্তি কী?");
console.log(result.answer);
```

### React Component Example

```jsx
import React, { useState } from "react";
import axios from "axios";

function LegalBeeChat() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleAsk = async () => {
    if (!question.trim()) return;

    setLoading(true);
    setError("");

    try {
      const response = await axios.post("http://localhost:8000/api/ask", {
        question: question,
      });
      setAnswer(response.data.answer);
    } catch (err) {
      setError(err.response?.data?.detail || "Error fetching answer");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="legal-bee-chat">
      <input
        type="text"
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        placeholder="Ask a legal question..."
      />
      <button onClick={handleAsk} disabled={loading}>
        {loading ? "Searching..." : "Ask"}
      </button>

      {error && <div className="error">{error}</div>}
      {answer && <div className="answer">{answer}</div>}
    </div>
  );
}

export default LegalBeeChat;
```

### cURL

```bash
# Single question
curl -X POST "http://localhost:8000/api/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the punishment for theft?", "language": "en"}'

# Bengali question
curl -X POST "http://localhost:8000/api/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "বাংলাদেশে চুরির শাস্তি কী?"}'

# Health check
curl "http://localhost:8000/api/health"
```

---

## Error Handling

### Common Error Responses

**400 - Bad Request**

```json
{
  "error": "Question must be at least 3 characters long",
  "detail": null,
  "timestamp": "2024-01-15T10:30:45.123456"
}
```

**503 - Service Unavailable**

```json
{
  "error": "Qdrant Cloud credentials not configured",
  "detail": "Missing QDRANT_URL or QDRANT_API_KEY",
  "timestamp": "2024-01-15T10:30:45.123456"
}
```

---

## Environment Configuration

Create a `.env` file with:

```
QDRANT_URL=https://your-cluster-id.qdrant.io
QDRANT_API_KEY=your_api_key_here
GROQ_API_KEY=your_groq_api_key
```

---

## Performance Tips

1. **Batch Processing**: Use `/api/ask-batch` for multiple questions
2. **Language Specification**: Provide `language` hint for faster processing
3. **Question Length**: Keep questions between 10-500 characters for best results
4. **Rate Limiting**: Groq free tier has ~14,000 tokens/minute limit

---

## Deployment

### Production (with Gunicorn)

```bash
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app --bind 0.0.0.0:8000
```

### Docker

Create `Dockerfile`:

```dockerfile
FROM python:3.13
WORKDIR /app
COPY . .
RUN pip install -e .
EXPOSE 8000
CMD ["python", "app.py"]
```

Build and run:

```bash
docker build -t legal-bee .
docker run -p 8000:8000 -e QDRANT_URL=$QDRANT_URL -e QDRANT_API_KEY=$QDRANT_API_KEY legal-bee
```

---

## Support

For issues or questions, consult the interactive documentation at:

- **Swagger UI**: http://localhost:8000/api/docs
- **ReDoc**: http://localhost:8000/api/redoc
