# Multi-Agent RAG System

A multi-agent Retrieval-Augmented Generation (RAG) system built with LangChain, LangGraph, Milvus, and Ollama. It exposes a REST API to query a document knowledge base and upload PDF documents for embedding.

## Architecture

```
Client
  │
  ▼
FastAPI (api.py)
  │
  ▼
SupervisorAgent          ← Orchestrates queries, manages memory & conversation history
  │
  ▼
RagAgent                 ← Retrieves relevant document chunks from the vector store
  │
  ▼
VectorStoreManager       ← Manages Milvus vector store, PDF loading & embedding
```

- **SupervisorAgent** — Main LLM agent (LangGraph) that decides when to delegate to the RAG agent. Maintains per-user conversation history via SQLite.
- **RagAgent** — Searches the Milvus vector store for relevant document chunks and returns them to the supervisor.
- **VectorStoreManager** — Handles PDF ingestion, text cleaning, embedding via Ollama, and similarity search using Milvus Lite.

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) running locally on `http://localhost:11434`
- The following Ollama models pulled:

```bash
ollama pull granite4:latest
ollama pull qwen3-embedding:0.6b
```

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Models can be configured via environment variables:

| Variable          | Default                | Description                  |
|-------------------|------------------------|------------------------------|
| `LLM_MODEL`       | `granite4:latest`      | Ollama chat model            |
| `EMBEDDING_MODEL` | `qwen3-embedding:0.6b` | Ollama embedding model       |

## Running the API Server

```bash
cd src
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

## API Endpoints

### POST /query

Send a natural language query to the agent. The agent will search the knowledge base if needed.

```http
POST http://localhost:8000/query
Content-Type: application/json
X-User-ID: your-user-id

{
  "query": "What is the topic of the uploaded document?"
}
```

- `X-User-ID` header is optional. When provided, conversation history is maintained per user across requests.

### POST /upload

Upload a PDF file to be embedded into the knowledge base.

```http
POST http://localhost:8000/upload
Content-Type: multipart/form-data

file: <your-file.pdf>
```

Only `.pdf` files are supported. The file is processed, chunked, embedded, and stored in the Milvus vector store.

## Running from CLI

You can also query the agent directly without starting the API server:

```bash
cd src
python query_caller.py --query "What is a Llama?"
```

Optional arguments:

```bash
python query_caller.py \
  --query "Your question here" \
  --model granite4:latest \
  --embedding_model qwen3-embedding:0.6b
```

## Testing with test.http

The `test.http` file contains ready-to-use HTTP requests compatible with tools like [REST Client (VS Code)](https://marketplace.visualstudio.com/items?itemName=humao.rest-client) or IntelliJ HTTP Client.

```bash
# Query the agent
POST http://localhost:8000/query

# Upload a PDF
POST http://localhost:8000/upload
```

## Data Storage

| File                    | Purpose                              |
|-------------------------|--------------------------------------|
| `src/milvus_demo.db`    | Milvus Lite vector store             |
| `src/agent_memory.db`   | LangGraph conversation memory (SQLite) |
| `src/agent_memory_cache.db` | LLM response cache (SQLite)      |
| `/tmp/uploads/`         | Temporary storage for uploaded PDFs  |
