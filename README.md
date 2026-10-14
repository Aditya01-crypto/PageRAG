# PageRAG

A document question-answering API. Upload PDF or DOCX files, then ask questions and get answers grounded **only** in your documents, with the source files listed.

Built with FastAPI, Chroma, sentence-transformers and Gemini. Retrieval can run in three modes so the strategies can be compared on the same pipeline:

| Mode | How it retrieves |
|---|---|
| `vector_only` | Dense vector similarity search (semantic meaning) |
| `hybrid` | Vector search + BM25 keyword search, merged with Reciprocal Rank Fusion (RRF) |
| `hybrid_rerank` | Hybrid candidates re-scored by a cross-encoder, then the best `top_k` are kept |

If the retrieved context doesn't contain the answer, the model is instructed to say so instead of guessing.

## How it works

```
upload ─► extract text ─► chunk (165 words, 33 overlap) ─► embed ─► Chroma
                                                              └─► BM25 index (in memory)

query ─► vector search (top fetch_k) ─┐
         BM25 search   (top fetch_k) ─┴─► RRF fusion ─► cross-encoder rerank ─► top_k chunks ─► Gemini ─► answer + sources
```

- **Extraction:** `pypdf` and `python-docx`, run off the event loop. A single `EXTRACTORS` mapping decides which file types are supported.
- **Chunking:** word-based windows of 165 words with 33 words of overlap. Chunk IDs are `{filename}_chunk_{n}`.
- **Embeddings:** `all-MiniLM-L6-v2` (384 dimensions, runs locally on CPU).
- **Vector store:** Chroma `PersistentClient`, cosine distance, stored in `./doc_db`.
- **Keyword search:** `rank-bm25` (BM25Okapi) with a lowercasing tokenizer that strips edge punctuation. The index is kept in memory and rebuilt after every upload or delete.
- **Fusion:** RRF (`k = 60`) merges the two ranked lists by chunk ID, so scores from different scales never need to be compared.
- **Reranking:** `cross-encoder/ms-marco-MiniLM-L-6-v2` scores each (query, chunk) pair.
- **Generation:** Gemini through plain `httpx` calls, including a streaming endpoint (server-sent events).

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | List of endpoints |
| `POST` | `/documents/upload` | Upload a PDF or DOCX (max 20 MB). Re-uploading a file with the same name replaces its chunks |
| `GET` | `/documents/` | List uploaded documents (`skip`, `limit`) |
| `DELETE` | `/documents/{doc_name}` | Remove a document and its chunks |
| `POST` | `/query/` | Ask a question, get an answer and its sources |
| `POST` | `/query/stream` | Same, streamed as server-sent events |
| `POST` | `/query/chunks` | Debug: show the retrieved chunks without calling the LLM |

Interactive docs are at `/docs`.

### Query request

```json
{
  "query": "How many layers are in the encoder stack?",
  "mode": "hybrid_rerank",
  "fetch_k": 10,
  "top_k": 5,
  "token_limit": 2000
}
```

| Field | Default | Notes |
|---|---|---|
| `query` | required | At least 5 characters |
| `mode` | required | `vector_only`, `hybrid` or `hybrid_rerank` |
| `fetch_k` | 10 | Candidates fetched from each retriever |
| `top_k` | 5 | Chunks passed to the LLM |
| `token_limit` | 2000 | Max output tokens, minimum 500 |

### Example

```bash
curl -F "file=@Attention.pdf" http://127.0.0.1:8000/documents/upload

curl -X POST http://127.0.0.1:8000/query/ \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the feed-forward inner-layer dimension?", "mode": "hybrid_rerank"}'
```

Response:

```json
{
  "answer": "...",
  "sources": ["Attention.pdf"],
  "question": "What is the feed-forward inner-layer dimension?"
}
```

Query endpoints are rate limited to 10 requests per minute per client.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone "https://github.com/Aditya01-crypto/PageRAG.git"
cd pagerag
uv sync
```

Create a `.env` file in the project root:

```
API_KEY=your_gemini_api_key
MODEL=your_gemini_model_id
DEBUG=False
RELOAD=False
```

Run the server:

```bash
uv run pagerag
```

The API is served at `http://127.0.0.1:8000`. The first start downloads the embedding and reranker models from Hugging Face, so it takes longer than later runs.

## Project structure

```
src/pagerag/
├── routers/
│   ├── documents.py   # upload, list, delete
│   └── query.py       # query, stream, debug chunks
├── chunker.py         # word-window chunking, chunk IDs
├── config.py          # upload directory, rate limiter
├── embeddings.py      # sentence-transformers model
├── extract.py         # PDF/DOCX extraction, supported types
├── hybrid.py          # BM25 index + search, RRF fusion
├── llm.py             # prompt, Gemini calls, streaming
├── main.py            # app, lifespan (Chroma, HTTP client, BM25 state)
├── reranker.py        # cross-encoder reranking
├── retrieve.py        # retrieval modes, no FastAPI dependency
├── schemas.py         # Pydantic models, Hit dataclass
└── vectorstore.py     # Chroma add, delete, vector search
```

## Design notes

- **Retrieval is separate from FastAPI.** `retrieve.py` takes the collection and BM25 state as arguments, so it can be called from scripts and tests as well as the routes.
- **Every search hit carries its own ID, text and source.** Fusion matches chunks by ID, and sources are read from the hits rather than recomputed.
- **Blocking work runs in threads.** Embedding, Chroma writes and BM25 rebuilds run through `asyncio.to_thread` so one upload doesn't stall other requests.
- **BM25 state is swapped atomically.** The index and its chunks live in one frozen dataclass assigned in a single step, so a query never sees a mismatched pair.
- **Uploads are cleaned up.** A context manager removes the temporary file whether processing succeeds or fails, and filenames are reduced to their final path component.

## Lessons from building it

- **Chunk size must fit the embedding model.** `all-MiniLM-L6-v2` reads only the first 256 tokens. With the original 367-word chunks, the end of each chunk was never embedded, and a test question about the Transformer's feed-forward dimension failed in `vector_only` mode. After reducing chunks to 165 words it answered correctly.
- **Hybrid search needs consistent text handling.** BM25 tokens, stored text and chunk identity must agree, or fusion treats the same chunk as two different ones.

## Known limitations

- The BM25 index is in memory and per process, so run a single worker. Multiple workers would go stale after uploads.
- The index is rebuilt in full after each upload or delete, which is fine for small collections but doesn't scale.
- PDFs are read as text only, with no OCR for scanned documents.
- No authentication.
- The Chroma client is synchronous and is used from worker threads.
- Retrieval quality has been checked on spot queries only. A systematic evaluation is planned.

## Roadmap

- Evaluation with RAGAS: a question set scored for faithfulness, answer relevance and context precision, comparing the three retrieval modes
- Configurable chunk size and overlap
