# Changes Since Last Push

This document summarizes all the new features, bug fixes, and architectural improvements that have been made in the local repository but not yet pushed to GitHub.

## 1. Core Architecture: MongoDB Migration
- **Shared Connection (`mongo_client.py`)**: Centralized the MongoDB connection logic to prevent multiple open connections. Added a robust exponential backoff retry system to gracefully handle initial TLS handshake timeouts on flaky networks.
- **Metadata & Feedback Tracking (`db.py`)**: Created collections for tracking document metadata (chunk count, ingestion time) and user feedback on AI responses.
- **GridFS Storage (`mongo_storage.py`)**: Migrated away from local disk storage. 
  - Uploaded PDFs are now chunked into GridFS so they survive server restarts and ephemeral filesystems (like Render's free tier). 
  - Matplotlib charts are generated purely in-memory and saved to GridFS instead of writing loose PNG files to disk, stopping the `uvicorn` file-watcher from constantly reloading the server during testing.

## 2. RAG & AI Agent Improvements
- **Multi-Model Provider Fallbacks (`llm_config.py`)**: Implemented a resilient fallback chain for different LLM roles (Grader, Synthesis, Vision). If an API hits a rate limit or quota (e.g., Gemini's strict vision quota), the system automatically falls back to Claude or Groq.
- **Vision Tool (`vision_tool.py`)**: Built a tool that renders specific PDF pages to images using PyMuPDF and sends them to a vision model (like local `llava` via Ollama) to read charts, tables, and figures that regular text extraction mangles. Optimized the DPI to speed up local inference.
- **Map-Reduce Summarization (`agent.py`)**: Added a highly concurrent `summarize_document` tool. Instead of failing on large documents, it pulls all chunks from ChromaDB, batches them, summarizes them in parallel using Groq, and reduces them into a unified executive summary.

## 3. Concurrency & Performance Fixes
- **FastAPI Thread-pool Fix (`api.py`)**: Converted endpoints from `async def` to synchronous `def`. This allows FastAPI to run long, blocking LangGraph agent tasks in its thread pool rather than freezing the main asyncio event loop.
- **Matplotlib Thread-Safety (`agent.py`)**: Migrated the `plot_stock_chart` tool to use Matplotlib's Object-Oriented API (`fig, ax = plt.subplots()`) to prevent state leaks and crashes when multiple users request charts simultaneously.
- **ChromaDB Connection Reuse (`ingest.py`)**: Modified the ingestion script to reuse the global `get_retriever_instance()` instead of opening a conflicting SQLite lock.

## 4. Caching & Infrastructure
- **Redis Integration**: Added Redis caching for `yfinance` API calls in the agent (fundamentals and charts) to drastically speed up repeated queries.
- **Dockerization**: 
  - Created a local `docker-compose.yml` to spin up the Redis container.
  - Added `Dockerfile`s and `.dockerignore`s for both the frontend and backend to prepare for remote deployment.
- **Updated `requirements.txt`**: Added `pymongo`, `gridfs`, `redis`, `tenacity`, `fitz` (PyMuPDF), and other newly required dependencies.
- **Updated `README.md`**: Added architecture diagrams, badges, and documentation reflecting all the new Docker, MongoDB, and Vision configurations.
