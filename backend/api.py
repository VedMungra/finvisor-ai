"""
HTTP layer for the Finvisor RAG financial advisor.

This module is deliberately thin: it owns request/response shapes, input validation, and
*failure translation* -- turning "a dependency is down" or "the model produced nothing
useful" into something the React frontend can render. The actual work (retrieval, relevance
grading, web search, tool calls, answer synthesis) all lives behind agent.py's compiled
LangGraph app.

Three conventions in here are load-bearing and easy to "fix" by accident:

1. Every endpoint is a plain `def`, never `async def`. The agent invocation, the embedding /
   cross-encoder models, ChromaDB, PyMongo and every LLM SDK call in this codebase are
   blocking. FastAPI runs plain `def` endpoints in a worker threadpool, so one slow chat
   request doesn't stall the event loop for every other connected client. Converting these
   to `async def` without making every call inside them genuinely awaitable would serialize
   the entire server behind the slowest request -- strictly worse than what's here.

2. The frontend treats a handful of response shapes as a hard contract (see App.jsx):
   /chat -> {response, chart_base64}, /documents -> {documents}, /suggest_questions ->
   {questions: [q1, q2]}, /ingest -> {message, chunk_count}. Fields may be *added*; nothing
   may be renamed or removed. Several of the guarantees below (always a non-empty
   `response`, always exactly two questions) exist specifically so the frontend never has to
   defend against a half-shaped payload.

3. Nothing in this file should be able to take the process down. Startup tolerates an
   unreachable MongoDB, /health never raises, and any unhandled error is converted to clean
   JSON. A RAG app whose vector store still works is far more useful than one that refused
   to boot because a metadata database was briefly unreachable.
"""

import os
import logging

# Logging is configured *before* the heavy application imports below, and the ordering is
# deliberate. agent.py, ingest.py and retriever.py each call logging.basicConfig() at import
# time; basicConfig is a no-op once the root logger already owns a handler, so whichever
# module happens to import first wins the format/level and the other calls silently do
# nothing. api.py is the process entrypoint, so configuring here -- before those imports --
# makes this the configuration that actually takes effect and reduces the other three calls
# to the no-ops they were always assumed to be.
#
# force=True is intentionally NOT used: it would also tear down handlers installed by
# uvicorn's own logging config when running under `uvicorn api:app`, which is how this is
# actually deployed.
LOG_LEVEL = (os.getenv("LOG_LEVEL") or "INFO").strip().upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("API")

# Suppress standard HTTP request logs to keep terminal clean
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

from dotenv import load_dotenv

# agent.py also calls load_dotenv(), but api.py reads its own configuration (CORS origins,
# upload limits, timeouts) at import time and can't rely on a transitive import having
# already populated os.environ. load_dotenv() is idempotent and never overrides variables
# that are already set, so calling it here as well is free.
load_dotenv()

import base64
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from contextlib import asynccontextmanager
from typing import Any, Callable, List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from pymongo.errors import PyMongoError

from langchain_core.messages import HumanMessage

from agent import app as agent_app, get_retriever_instance, get_primary_llm
from ingest import process_and_ingest, extract_text_from_pdf
from mongo_storage import save_pdf, load_chart
from auth import get_current_user, get_optional_user
from auth_routes import router as auth_router
import db


# ---------------------------------------------------------------------------
# Configuration
#
# Every tunable is read from the environment with a safe default and a defensive parse: a
# typo'd numeric env var should log a warning and fall back, not crash the import and take
# the whole service down before it ever binds a port.
# ---------------------------------------------------------------------------

def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(float(raw.strip()))
    except ValueError:
        logger.warning(f"Invalid value for {name}={raw!r}; falling back to {default}.")
        return default


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError:
        logger.warning(f"Invalid value for {name}={raw!r}; falling back to {default}.")
        return default


# Vite's dev server; the two spellings are distinct origins to a browser, so both are
# allowed by default or local testing breaks depending on which URL you happen to open.
DEFAULT_CORS_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"

# Uploads are read into memory for chunking/embedding, so an unbounded body is a trivial
# way to OOM the process. 25 MB comfortably covers a long 10-K.
MAX_UPLOAD_MB = _int_env("MAX_UPLOAD_MB", 25)
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

# The ingestion pipeline only understands text: PDFs go through pypdf, everything else is
# decoded and fed to the Markdown splitter. Anything else (images, archives, office
# documents) would either explode or silently ingest garbage, so reject it up front.
ALLOWED_UPLOAD_EXTENSIONS = {".pdf", ".md", ".txt"}

# A chat request can legitimately take a long time (retrieval + grading + optional web
# search + tool calls + synthesis, each with provider fallback). These caps exist so a hung
# provider surfaces as a clear 504 instead of a browser tab spinning forever.
CHAT_TIMEOUT_SECONDS = _float_env("CHAT_TIMEOUT_SECONDS", 180.0)
SUGGEST_TIMEOUT_SECONDS = _float_env("SUGGEST_TIMEOUT_SECONDS", 90.0)

# Blocking work is submitted to a *dedicated* pool rather than run inline in FastAPI's own
# threadpool. When a request times out, the underlying LLM/network call cannot actually be
# cancelled -- the thread keeps running until the provider gives up. Isolating those
# orphaned threads here means they can't slowly exhaust the threadpool FastAPI needs to
# serve everything else (including /health).
API_WORKER_THREADS = max(1, _int_env("API_WORKER_THREADS", 8))

# /health must answer quickly even when MongoDB is unreachable, so its probe uses its own
# short timeout instead of mongo_client.py's deliberately patient retry-with-backoff.
HEALTH_PROBE_TIMEOUT_MS = 1500

_blocking_executor = ThreadPoolExecutor(
    max_workers=API_WORKER_THREADS, thread_name_prefix="finvisor-work"
)


def _run_with_timeout(fn: Callable[[], Any], timeout: float, what: str) -> Any:
    """
    Runs a blocking callable with a wall-clock deadline, raising a 504 when it expires.

    Note the honest limitation: `future.cancel()` cannot stop work that has already started,
    so the abandoned call keeps running to completion in the background. That's an
    acceptable trade -- the client gets a clear, timely answer instead of an open socket,
    and the leaked work is confined to the dedicated executor above.
    """
    future = _blocking_executor.submit(fn)
    try:
        return future.result(timeout=timeout)
    except FutureTimeoutError:
        future.cancel()
        logger.error(f"{what} exceeded its {timeout:g}s timeout.")
        raise HTTPException(
            status_code=504,
            detail=f"{what} took longer than {timeout:g} seconds. Please try again, "
                   f"or simplify the request.",
        )


def _dependency_error(exc: Exception, what: str) -> HTTPException:
    """
    Maps an exception from a downstream dependency onto a status code the frontend can
    reason about.

    The distinction that matters to a user is "the service is temporarily unable to do this"
    (503 -- retrying later might work, and the message says what to check) versus "something
    is actually broken in the code" (500). Rate limits, exhausted quotas, missing API keys
    and unreachable databases are all the former, and they are by far the most common
    failure modes in this app -- free-tier LLM quotas run out constantly.
    """
    message = str(exc)
    lowered = message.lower()

    if isinstance(exc, PyMongoError):
        return HTTPException(
            status_code=503,
            detail=f"{what}: MongoDB is unavailable. Check MONGODB_URI and that the "
                   f"database is running.",
        )
    if any(token in lowered for token in ("429", "quota", "rate limit", "resource_exhausted")):
        return HTTPException(
            status_code=503,
            detail=f"{what}: every configured AI provider is currently rate-limited or out "
                   f"of quota. Please try again shortly.",
        )
    if any(token in lowered for token in ("api key", "api_key", "401", "403", "authentication",
                                          "unauthorized", "permission_denied")):
        return HTTPException(
            status_code=503,
            detail=f"{what}: no AI provider accepted the request (missing or invalid API "
                   f"key). Check the *_API_KEY variables in your .env.",
        )
    if any(token in lowered for token in ("connection refused", "connect call failed",
                                          "name or service not known", "timed out",
                                          "temporarily unavailable", "503", "overloaded")):
        return HTTPException(
            status_code=503,
            detail=f"{what}: an upstream dependency is unreachable. Please try again shortly.",
        )

    # Genuinely unexpected: keep the original message (truncated) because this is a
    # self-hosted developer tool and losing the error text makes debugging much harder.
    return HTTPException(status_code=500, detail=f"{what}: {message[:500]}")


# ---------------------------------------------------------------------------
# CORS
#
# The previous configuration was allow_origins=["*"] together with
# allow_credentials=True. That combination is invalid per the Fetch/CORS spec -- a browser
# refuses to honour a wildcard Access-Control-Allow-Origin on a credentialed request -- so
# it fails in exactly the situation it was meant to cover. The fix is to echo back an
# explicit allowlist, which is also what any real deployment wants.
# ---------------------------------------------------------------------------

def _resolve_cors_config() -> tuple:
    """
    Reads the comma-separated CORS_ORIGINS variable and returns (origins, allow_credentials).

    "*" is still supported as an explicit opt-in for throwaway/demo deployments, but it
    forces allow_credentials=False, because that pairing is the only spec-legal way to use a
    wildcard. Nothing in this app authenticates via cookies today, so wildcard mode is
    functional -- it just can't silently pretend to support credentials.
    """
    raw = os.getenv("CORS_ORIGINS")
    if raw is None or not raw.strip():
        raw = DEFAULT_CORS_ORIGINS

    # Browsers send Origin without a trailing slash, so normalise to avoid the classic
    # "http://localhost:5173/" entry that silently matches nothing.
    origins = [o.strip().rstrip("/") for o in raw.split(",") if o.strip()]
    if not origins:
        origins = [o.strip() for o in DEFAULT_CORS_ORIGINS.split(",")]

    if "*" in origins:
        return ["*"], False
    return origins, True


CORS_ORIGINS, CORS_ALLOW_CREDENTIALS = _resolve_cors_config()


# ---------------------------------------------------------------------------
# Application lifespan
#
# Replaces the deprecated @app.on_event("startup") hook, which recent Starlette releases
# have removed outright.
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup is intentionally *best-effort*.

    Previously db.init_db() ran unguarded in a startup event, so a transiently unreachable
    MongoDB took down the entire API -- including /documents and /chat, neither of which
    needs Mongo to answer, since retrieval is served by ChromaDB. Now a failure here is
    logged loudly and the server carries on; the endpoints that genuinely need Mongo fail
    individually with a 503 that says so, and mongo_client.py will reconnect lazily the
    moment the database comes back.

    The cheap reachability probe in front of init_db() matters for boot time:
    mongo_client.get_db() deliberately retries with backoff (up to ~50s) because a flaky
    network is usually worth waiting out on a real query. It is *not* worth blocking the
    port on at boot, so we only pay for init_db() when a 1.5s ping says the database is
    actually there.
    """
    logger.info(
        f"CORS allowed origins: {CORS_ORIGINS} (credentials={'on' if CORS_ALLOW_CREDENTIALS else 'off'})"
    )

    if _mongodb_status() == "up":
        try:
            db.init_db()
        except Exception as e:
            logger.warning(
                f"MongoDB index setup failed at startup ({e}). Continuing without it -- "
                f"document metadata, feedback and file storage will return 503 until "
                f"MongoDB recovers."
            )
    else:
        logger.warning(
            "MongoDB is not reachable at startup. Continuing anyway: chat and retrieval "
            "only need ChromaDB. Endpoints backed by MongoDB (/feedback, /documents/stats, "
            "PDF storage, chart images) will return 503 until it recovers."
        )

    logger.info("Finvisor API ready.")
    yield

    # Nothing here owns an event loop or a socket; the executor is the only resource worth
    # draining, and only so in-flight blocking work isn't killed mid-write to MongoDB.
    _blocking_executor.shutdown(wait=False, cancel_futures=True)
    logger.info("Finvisor API shutting down.")


app = FastAPI(title="Financial Advisor AI API", lifespan=lifespan)

# Mount the auth sub-router at /auth
app.include_router(auth_router, prefix="/auth")


# ---------------------------------------------------------------------------
# Middleware
#
# Ordering here is subtle and deliberate: Starlette applies user middleware in reverse
# registration order, so the LAST one added is the outermost. The error trap is registered
# first (making it inner) and CORS second (outermost) so that CORS headers are still
# attached to the JSON error responses the trap produces. With the reverse order, a browser
# would see a CORS failure instead of the actual error message -- which is precisely how
# "the backend is returning nothing" bugs become unreadable in the frontend console.
# ---------------------------------------------------------------------------

@app.middleware("http")
async def error_trap(request: Request, call_next):
    """Converts any unhandled exception into the same {"detail": ...} JSON shape FastAPI
    already uses for HTTPException, while logging the full traceback server-side."""
    try:
        return await call_next(request)
    except HTTPException as exc:
        # Should already have been handled further in; converting here as well guarantees
        # the response shape no matter where it was raised from.
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
            headers=getattr(exc, "headers", None),
        )
    except Exception as exc:
        logger.error(
            f"Unhandled error on {request.method} {request.url.path}: {exc}", exc_info=True
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error. Check the backend logs for details."},
        )


app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=CORS_ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Backstop for anything raised outside the error_trap middleware (e.g. inside another
    middleware). Same clean JSON contract, same server-side traceback."""
    logger.error(
        f"Unhandled error on {request.method} {request.url.path}: {exc}", exc_info=exc
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Check the backend logs for details."},
    )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

_health_probe_client = None


def _mongodb_status() -> str:
    """
    Returns "up"/"down" for MongoDB without ever raising and without borrowing
    mongo_client.py's retrying connection (which can block for the better part of a minute
    by design). The probe client is created once and reused -- constructing a MongoClient
    does no I/O, and reusing it lets PyMongo's topology monitor notice on its own when the
    server comes back.
    """
    global _health_probe_client
    try:
        if _health_probe_client is None:
            from pymongo import MongoClient
            uri = os.getenv("MONGODB_URI") or "mongodb://localhost:27017"
            _health_probe_client = MongoClient(
                uri,
                serverSelectionTimeoutMS=HEALTH_PROBE_TIMEOUT_MS,
                connectTimeoutMS=HEALTH_PROBE_TIMEOUT_MS,
                socketTimeoutMS=HEALTH_PROBE_TIMEOUT_MS,
                appname="finvisor-healthcheck",
            )
        _health_probe_client.admin.command("ping")
        return "up"
    except Exception as e:
        logger.debug(f"MongoDB health probe failed: {e}")
        return "down"


def _chroma_status() -> str:
    """
    Returns "up"/"down" for the vector store without loading any models.

    ChromaDB here is embedded and persisted to a local directory, so there is no server to
    ping. If the retriever has already been constructed by an earlier request we ask it for
    a single row (cheap, no embedding work). If it hasn't, we deliberately do NOT construct
    it: get_retriever_instance() loads the bge embedding model *and* the cross-encoder
    reranker, which takes seconds and hundreds of MB -- exactly what a health check must
    never do. In that case the presence of a readable persistence directory is the best
    honest signal available.
    """
    try:
        import agent as _agent
        instance = getattr(_agent, "_retriever_instance", None)
        if instance is not None:
            instance.vectorstore.get(limit=1, include=[])
            return "up"

        candidates = [
            os.path.abspath("./chroma_db"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db"),
        ]
        for path in candidates:
            if os.path.isdir(path) and os.access(path, os.R_OK):
                return "up"
        return "down"
    except Exception as e:
        logger.debug(f"ChromaDB health probe failed: {e}")
        return "down"


@app.get("/health")
def health():
    """
    Liveness/readiness probe for container orchestrators, uptime monitors and the developer
    who just wants to know which dependency is broken.

    Contract: this must never raise and must never be slow. All three sub-probes swallow
    their own exceptions and are individually bounded, so the endpoint always returns 200
    with a per-dependency verdict rather than failing outright -- "the API is alive but
    MongoDB is down" is a far more actionable answer than a connection error.

    `providers` reports, per role, which model provider is currently usable and why the
    others were skipped. This is deliberately part of the health payload rather than a
    separate endpoint: by far the most common way this app "breaks" is not a crash but a
    provider silently dropping out of a chain -- an exhausted free-tier quota, an empty
    ANTHROPIC_API_KEY, an Ollama model that was configured but never pulled. Those all
    degrade quietly via the fallback chain, so without somewhere to look you only find out
    by reading logs. llm_config.get_provider_status() makes no network calls to hosted
    providers and is itself wrapped so it cannot throw, which keeps the "never slow, never
    raises" contract intact.
    """
    try:
        from llm_config import get_provider_status
        providers = get_provider_status()
    except Exception as e:  # pragma: no cover - defensive; get_provider_status never raises
        logger.debug(f"Provider status probe failed: {e}")
        providers = {"error": "unavailable"}

    return {
        "status": "ok",
        "mongodb": _mongodb_status(),
        "chroma": _chroma_status(),
        "providers": providers,
    }


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=16000)
    thread_id: str = Field(..., min_length=1, max_length=200)
    source_filename: Optional[str] = None
    context_key: Optional[str] = None  # For chat history persistence

    @field_validator("source_filename")
    @classmethod
    def _blank_scope_means_global(cls, value: Optional[str]) -> Optional[str]:
        """An empty string is not the same as "no document scope": it would be passed
        straight into ChromaDB as filter {"source": ""}, which matches nothing and makes
        every question look unanswerable. Normalise it to None (= global agent)."""
        if value is None:
            return None
        value = value.strip()
        return value or None


# Permissive on whitespace/casing because this tag is produced by an LLM copying an
# instruction, and models are inconsistent about exactly how they reproduce markup.
_CHART_TAG_RE = re.compile(r"<\s*chart\s*>(.*?)<\s*/\s*chart\s*>", re.IGNORECASE | re.DOTALL)
# Any leftover half-tag (model wrote an opening tag but never closed it) must still be
# scrubbed, otherwise raw markup leaks into the chat bubble.
_STRAY_CHART_TAG_RE = re.compile(r"<\s*/?\s*chart\s*>", re.IGNORECASE)
# plot_stock_chart builds ids as "<ticker>_<uuid8>"; recovering the ticker lets the
# fallback sentence below say something specific instead of something generic.
_CHART_ID_TICKER_RE = re.compile(r"^([A-Za-z][A-Za-z0-9.\-]{0,9})_[0-9a-fA-F]{6,}$")


def _message_text(content: Any) -> str:
    """
    Flattens a LangChain message's `content` into plain text.

    Content is not reliably a string. Anthropic returns a list of typed blocks (text,
    thinking, tool_use); Gemini and Groq sometimes return a list of bare strings; a
    tool-calling turn can carry no text at all. The original code assumed every list element
    was a dict with a "type" key and crashed with AttributeError on the string case, so this
    handles all four shapes and ignores any block that isn't user-facing prose.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        text = content.get("text")
        return text if isinstance(text, str) else ""
    if isinstance(content, (list, tuple)):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                # Explicit text blocks first; the second branch catches providers that emit
                # a "text" key without the canonical type marker. Thinking/tool_use blocks
                # have no "text" key and are correctly skipped.
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append(block["text"])
                elif isinstance(block.get("text"), str):
                    parts.append(block["text"])
        return "\n".join(part for part in parts if part.strip())
    return str(content)


def _find_chart_ids(text: str) -> List[str]:
    """Returns every chart id mentioned in `text`, in order. The original code used
    re.search and therefore only ever saw the first tag -- when a model plotted two tickers
    the second tag was left in the prose as literal markup."""
    ids = []
    for match in _CHART_TAG_RE.finditer(text or ""):
        chart_id = match.group(1).strip()
        # A well-formed id has no whitespace; anything else is the model hallucinating
        # prose inside the tag rather than a real reference.
        if chart_id and not re.search(r"\s", chart_id):
            ids.append(chart_id)
    return ids


def _strip_chart_tags(text: str) -> str:
    """
    Removes all <chart> tags (matched or stray) and tidies the hole they leave behind.

    Removing a tag from the middle of a sentence otherwise leaves "and  for Microsoft." or a
    dangling " ." -- and a tag on its own line leaves a triple blank line that markdown
    renders as a gap. The cleanup passes are skipped entirely when there was no tag to
    remove, so an ordinary answer is returned byte-for-byte as the model wrote it and
    markdown indentation (code blocks, nested lists) can't be disturbed.
    """
    text = text or ""
    if "<" not in text or "chart" not in text.lower():
        return text.strip()

    cleaned = _CHART_TAG_RE.sub("", text)
    cleaned = _STRAY_CHART_TAG_RE.sub("", cleaned)
    if cleaned == text:
        return text.strip()

    cleaned = re.sub(r"(?<=\S)[ \t]{2,}(?=\S)", " ", cleaned)  # "and  for" -> "and for"
    cleaned = re.sub(r"[ \t]+([.,;:!?])", r"\1", cleaned)      # "chart ." -> "chart."
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _current_turn_messages(messages: List[Any]) -> List[Any]:
    """
    Narrows the checkpointed message history down to just this turn: everything after the
    most recent human message.

    This matters because the graph is compiled with a MemorySaver checkpointer, so
    final_state["messages"] contains the *entire* conversation for that thread_id. Scanning
    all of it for a chart tag would happily resurface a chart generated three questions ago
    and attach it to an unrelated answer.
    """
    last_human_index = -1
    for index, message in enumerate(messages):
        if getattr(message, "type", None) == "human":
            last_human_index = index
    if last_human_index >= 0:
        return list(messages[last_human_index + 1:])
    return list(messages)


def _chart_fallback_text(chart_id: str) -> str:
    """Prose to return when a chart was produced but the model wrote nothing around it."""
    match = _CHART_ID_TICKER_RE.match(chart_id)
    if match:
        return (
            f"Here is the requested price chart for {match.group(1).upper()}. "
            f"Ask a follow-up if you'd like the trend, volatility, or key support and "
            f"resistance levels interpreted."
        )
    return (
        "Here is the chart you requested. Ask a follow-up if you'd like me to walk through "
        "what it shows."
    )


NO_ANSWER_FALLBACK = (
    "I wasn't able to produce an answer for that request. Please try rephrasing it, or ask "
    "about a specific document or ticker."
)


@app.post("/chat")
def chat(request: ChatRequest, req: Request):
    """
    Runs one turn of the agent graph and returns {"response": str, "chart_base64": str|None}.

    `response` is guaranteed to be a non-empty string. That guarantee is not cosmetic: when
    a question is answered purely by the charting tool, the model frequently emits nothing
    but the `<chart>ID</chart>` tag it was instructed to include. Stripping the tag then left
    an empty string, and the frontend rendered an empty chat bubble above the image -- the
    answer looked broken even though the chart was perfect. The tag is now stripped, the
    chart is still returned, and a useful sentence is substituted in place of the prose the
    model didn't write.
    """
    logger.info(
        f"Received chat request for thread '{request.thread_id}' with scope "
        f"'{request.source_filename}'"
    )

    initial_state = {
        "messages": [HumanMessage(content=request.prompt)],
        "source_filename": request.source_filename,
    }
    config = {"configurable": {"thread_id": request.thread_id}}

    try:
        final_state = _run_with_timeout(
            lambda: agent_app.invoke(initial_state, config=config),
            CHAT_TIMEOUT_SECONDS,
            "The assistant",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing chat request: {e}", exc_info=True)
        raise _dependency_error(e, "Could not complete this request")

    messages = (final_state or {}).get("messages") or []
    turn_messages = _current_turn_messages(messages)

    # Collect chart ids from the whole turn, not just the final message: the tool's own
    # output carries the id too, which is what lets us still return the image when the model
    # forgets to echo the tag it was told to include.
    chart_ids: List[str] = []
    for message in turn_messages:
        chart_ids.extend(_find_chart_ids(_message_text(getattr(message, "content", None))))

    # Walk backwards for the last AI message that actually contains prose. The final message
    # in the state can be a tool result, or an AI message whose content is nothing but tool
    # calls / a chart tag, in which case the real answer is the message before it.
    ai_text = ""
    for message in reversed(turn_messages):
        if getattr(message, "type", None) != "ai":
            continue
        candidate = _strip_chart_tags(_message_text(getattr(message, "content", None)))
        if candidate:
            ai_text = candidate
            break

    chart_base64 = None
    if chart_ids:
        # Last id wins: if a turn produced several charts, the most recent one is the one
        # the closing answer is talking about.
        chart_id = chart_ids[-1]
        try:
            # Charts are generated in-memory and stored in MongoDB (see agent.py's
            # plot_stock_chart tool + mongo_storage.save_chart) rather than written to local
            # disk, so they survive an ephemeral filesystem and don't get picked up as a
            # source-file change by uvicorn's --reload watcher.
            chart_bytes = load_chart(chart_id)
            if chart_bytes:
                chart_base64 = base64.b64encode(chart_bytes).decode("utf-8")
            else:
                logger.warning(f"Chart '{chart_id}' was referenced but not found in storage.")
        except Exception as e:
            # A missing chart must never fail the whole answer -- the text is the primary
            # product, the image is an enhancement.
            logger.warning(f"Could not load chart '{chart_id}' from MongoDB: {e}")

    if not ai_text:
        ai_text = _chart_fallback_text(chart_ids[-1]) if chart_ids else NO_ANSWER_FALLBACK
        logger.warning(
            f"Agent returned no usable text for thread '{request.thread_id}'; substituted a "
            f"fallback response (chart present: {chart_base64 is not None})."
        )

    # Persist chat messages if user is authenticated
    user = get_optional_user(req)
    if user and request.context_key:
        ctx = request.context_key
        try:
            db.save_chat_message(
                user_id=user["user_id"],
                thread_id=request.thread_id,
                context_key=ctx,
                role="user",
                content=request.prompt,
            )
            extra = {}
            if chart_base64:
                extra["has_chart"] = True
                if chart_ids:
                    extra["chart_id"] = chart_ids[-1]
            db.save_chat_message(
                user_id=user["user_id"],
                thread_id=request.thread_id,
                context_key=ctx,
                role="agent",
                content=ai_text,
                extra=extra if extra else None,
            )
        except Exception as e:
            # Chat persistence failure must never break the response
            logger.warning(f"Could not persist chat history: {e}")

    logger.info(f"Successfully processed chat response for thread '{request.thread_id}'")
    return {"response": ai_text, "chart_base64": chart_base64}


# ---------------------------------------------------------------------------
# Documents (ChromaDB-backed)
# ---------------------------------------------------------------------------

def _retriever():
    """
    Returns the shared retriever, translating construction failures into a 503.

    The first call is slow by design -- it loads the embedding model and the cross-encoder
    reranker lazily (see agent.py) rather than at import time, so uvicorn can bind its port
    immediately. If that construction fails (corrupt Chroma directory, model download
    blocked) the honest answer is "this dependency is unavailable", not a 500.
    """
    try:
        return get_retriever_instance()
    except Exception as e:
        logger.error(f"Vector store unavailable: {e}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail=f"The vector store (ChromaDB) is unavailable: {str(e)[:300]}",
        )


@app.get("/documents")
def get_documents():
    """Lists the distinct source filenames currently present in the vector store. Sorted so
    the sidebar ordering is stable between reloads -- a Python set's iteration order is not."""
    try:
        data = _retriever().vectorstore.get(include=["metadatas"])
        metadatas = data.get("metadatas") or []
        sources = {
            m.get("source")
            for m in metadatas
            if isinstance(m, dict) and isinstance(m.get("source"), str) and m.get("source")
        }
        return {"documents": sorted(sources)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching documents: {e}", exc_info=True)
        raise _dependency_error(e, "Could not list documents")


@app.delete("/documents/{filename}")
def delete_document(filename: str):
    """
    Deletes a specific document across all storage layers: MongoDB metadata,
    GridFS raw bytes, and ChromaDB vector chunks.
    """
    try:
        # 1. Delete MongoDB metadata
        db.delete_document_metadata(filename)
        
        # 2. Delete GridFS raw bytes
        import mongo_storage
        mongo_storage.delete_pdf(filename)
        
        # 3. Delete ChromaDB vector chunks
        retriever_instance = _retriever()
        data = retriever_instance.vectorstore.get(where={"source": filename})
        ids = data.get("ids") or []
        if ids:
            retriever_instance.vectorstore.delete(ids=ids)
            
        logger.info(f"Successfully deleted document '{filename}' (purged {len(ids)} chunks).")
        return {"message": f"Document '{filename}' deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document '{filename}': {e}", exc_info=True)
        raise _dependency_error(e, f"Could not delete document '{filename}'")


@app.delete("/documents")
def clear_database():
    """Deletes every chunk from the vector store. Deliberately leaves MongoDB's metadata and
    GridFS copies alone -- see the note in the response message."""
    try:
        retriever_instance = _retriever()
        data = retriever_instance.vectorstore.get()
        ids = data.get("ids") or []
        if ids:
            retriever_instance.vectorstore.delete(ids=ids)
        logger.info(f"Successfully cleared {len(ids)} chunk(s) from the vector store.")
        return {"message": "Database cleared successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing database: {e}", exc_info=True)
        raise _dependency_error(e, "Could not clear the database")


class SuggestedQuestions(BaseModel):
    questions: List[str] = Field(description="A list of 2 insightful questions")


def _normalise_questions(result: Any, filename: str) -> List[str]:
    """
    Coerces whatever the structured-output call returned into exactly two non-empty strings.

    The frontend renders `questions[0]` and `questions[1]` unconditionally, so a model that
    returns one question (or three, or a dict instead of the pydantic object -- all of which
    happen across the provider chain in llm_config.py) would break the UI. Padding with
    document-specific generics keeps the contract intact and still gives the user something
    worth clicking.
    """
    raw = getattr(result, "questions", None)
    if raw is None and isinstance(result, dict):
        raw = result.get("questions")

    questions: List[str] = []
    for item in raw or []:
        text = str(item).strip()
        if text and text not in questions:
            questions.append(text)

    defaults = [
        f"What are the most important financial takeaways in {filename}?",
        f"What risks or headwinds does {filename} flag for investors?",
    ]
    for default in defaults:
        if len(questions) >= 2:
            break
        if default not in questions:
            questions.append(default)

    return questions[:2]


@app.get("/suggest_questions")
def suggest_questions(
    filename: str = Query(..., min_length=1, description="The name of the document")
):
    """
    Suggests two starter questions for a specific uploaded document.

    Note the `except HTTPException: raise` before the generic handler. Without it, the 404
    raised below for an unknown document was caught by the bare `except Exception` and
    re-raised as a 500 -- so "you asked about a document that isn't ingested" was reported
    to the client as "the server is broken". Every endpoint in this module now re-raises
    HTTPException first for the same reason.
    """
    try:
        data = _retriever().vectorstore.get(where={"source": filename}, include=["documents"])
        docs = data.get("documents") or []
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reading chunks for {filename}: {e}", exc_info=True)
        raise _dependency_error(e, "Could not read that document")

    if not docs:
        raise HTTPException(status_code=404, detail="Document not found or has no content.")

    # Take the first 3 chunks to give the LLM some context
    context_text = "\n\n".join(str(doc) for doc in docs[:3])

    prompt = f"""You are a senior financial advisor. Based on the following excerpts from '{filename}', generate exactly 2 highly insightful, strategic questions that an investor might want to ask about this document.
The questions should be specific to the data or topics mentioned in the text.
Do not ask generic questions.

Context:
{context_text}"""

    try:
        structured_llm = get_primary_llm().with_structured_output(SuggestedQuestions)
        result = _run_with_timeout(
            lambda: structured_llm.invoke(prompt),
            SUGGEST_TIMEOUT_SECONDS,
            "Question suggestion",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error suggesting questions for {filename}: {e}", exc_info=True)
        raise _dependency_error(e, "Could not suggest questions")

    return {"questions": _normalise_questions(result, filename)}


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def _sanitise_upload_filename(raw_name: Optional[str]) -> str:
    """
    Validates the client-supplied filename and returns a safe basename.

    The filename is not decoration: it becomes the ChromaDB `source` metadata value, the
    GridFS key for the original PDF bytes, and the unique key of the Mongo metadata
    document. `UploadFile.filename` is also Optional -- a multipart part without a filename
    parameter yields None, and the old code called `.lower()` on it immediately, so a
    slightly malformed upload produced an AttributeError and a 500.

    Path separators are rejected rather than silently stripped. Nothing here writes uploads
    to disk today, so `../` isn't an exploit *yet*; but a filename that quietly changes
    meaning between what the user sent and what is stored is a bug waiting to happen the
    moment someone reintroduces a filesystem path, so it's refused explicitly.
    """
    name = (raw_name or "").strip()
    if not name:
        raise HTTPException(
            status_code=422,
            detail="The upload is missing a filename. Please re-select the file and try again.",
        )
    if "\x00" in name or any(ord(ch) < 32 for ch in name):
        raise HTTPException(status_code=400, detail="Filename contains invalid characters.")
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(
            status_code=400,
            detail="Filename must not contain path separators or '..'.",
        )

    name = os.path.basename(name)
    if not name or name in (".", ".."):
        raise HTTPException(status_code=400, detail="Invalid filename.")
    if len(name) > 200:
        raise HTTPException(status_code=400, detail="Filename is too long (max 200 characters).")

    extension = os.path.splitext(name)[1].lower()
    if extension not in ALLOWED_UPLOAD_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_UPLOAD_EXTENSIONS))
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{extension or 'unknown'}'. Allowed types: {allowed}.",
        )
    return name


def _decode_text_upload(content: bytes, filename: str) -> str:
    """
    Decodes a non-PDF upload to text without ever raising UnicodeDecodeError.

    A stray binary or Windows-encoded file used to reach `content.decode("utf-8")` unguarded
    and surface as a 500, which tells the user nothing. Real-world .md/.txt files exported
    from Windows tooling are frequently cp1252 rather than UTF-8, so those are decoded rather
    than refused; genuinely binary payloads (detected via NUL bytes, which never appear in
    text) are refused with a message that explains what to do instead.
    """
    if b"\x00" in content:
        raise HTTPException(
            status_code=415,
            detail=f"'{filename}' looks like a binary file, not text. Upload a PDF, "
                   f"Markdown or plain-text document.",
        )

    for encoding in ("utf-8", "utf-8-sig", "cp1252"):
        try:
            text = content.decode(encoding)
            if encoding != "utf-8":
                logger.warning(f"'{filename}' was not valid UTF-8; decoded as {encoding}.")
            return text
        except UnicodeDecodeError:
            continue

    raise HTTPException(
        status_code=415,
        detail=f"'{filename}' could not be decoded as text. Re-save it as UTF-8 and try again.",
    )


@app.post("/ingest")
def ingest_document(file: UploadFile = File(...)):
    """
    Chunks, embeds and stores an uploaded document, returning {"message", "chunk_count"}.

    Failure policy: ChromaDB is the only hard dependency here, because it's what retrieval
    actually reads. MongoDB stores the original PDF bytes (for vision_tool.py) and the
    metadata row behind /documents/stats -- valuable, but losing them doesn't make the
    document unsearchable. So a MongoDB outage degrades this endpoint (with the caveat
    surfaced in `message`) instead of failing an ingestion that otherwise fully succeeded.
    """
    filename = _sanitise_upload_filename(getattr(file, "filename", None))
    logger.info(f"Received document for ingestion: {filename}")

    # Some Starlette versions leave the spooled file's cursor wherever the parser left it.
    try:
        file.file.seek(0)
    except Exception:
        pass

    # Reject oversized bodies before pulling them into memory. `size` is populated by
    # Starlette for most clients; the capped read below is the authoritative check for the
    # ones where it isn't.
    declared_size = getattr(file, "size", None)
    if isinstance(declared_size, int) and declared_size > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"'{filename}' is larger than the {MAX_UPLOAD_MB} MB upload limit.",
        )

    try:
        content = file.file.read(MAX_UPLOAD_BYTES + 1)
    except Exception as e:
        logger.error(f"Could not read uploaded file {filename}: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail="Could not read the uploaded file.")

    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"'{filename}' is larger than the {MAX_UPLOAD_MB} MB upload limit.",
        )
    if not content:
        raise HTTPException(status_code=422, detail=f"'{filename}' is empty.")

    is_pdf = filename.lower().endswith(".pdf")
    warnings: List[str] = []

    if is_pdf:
        try:
            text_content = extract_text_from_pdf(content)
        except HTTPException:
            raise
        except Exception as e:
            logger.warning(f"Could not parse PDF {filename}: {e}")
            raise HTTPException(
                status_code=422,
                detail=f"Could not read '{filename}' as a PDF (it may be corrupt or "
                       f"password-protected).",
            )
        if not text_content.strip():
            raise HTTPException(
                status_code=422,
                detail="No extractable text found in PDF. Scanned/image-only PDFs are not "
                       "supported by the text pipeline.",
            )

        # Keep the original bytes in MongoDB (not local disk -- see mongo_storage.py)
        # so vision_tool.py can re-render specific pages later, and so uploads survive
        # a Render redeploy or running more than one backend instance.
        try:
            save_pdf(filename, content)
        except Exception as e:
            logger.warning(f"Could not store original PDF bytes for {filename}: {e}")
            warnings.append("the original PDF could not be archived, so page-level visual "
                            "analysis will be unavailable")
    else:
        text_content = _decode_text_upload(content, filename)
        if not text_content.strip():
            raise HTTPException(status_code=422, detail=f"'{filename}' contains no text.")

    source_meta = {"source": filename}

    try:
        _, chunk_count = process_and_ingest(text_content, source_meta)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ingesting document {filename}: {e}", exc_info=True)
        raise _dependency_error(e, f"Could not ingest '{filename}'")

    try:
        db.record_document(
            filename=filename,
            content_type="pdf" if is_pdf else "markdown",
            chunk_count=chunk_count,
            has_original_file=is_pdf,
        )
    except Exception as e:
        # The chunks are already searchable at this point; failing the request now would be
        # actively misleading.
        logger.warning(f"Could not record document metadata for {filename}: {e}")
        warnings.append("document metadata could not be saved, so it won't appear in "
                        "/documents/stats")

    message = f"Successfully ingested {filename}"
    if warnings:
        message += f" (note: {'; '.join(warnings)})"

    logger.info(f"Successfully ingested document: {filename} ({chunk_count} chunks)")
    return {"message": message, "chunk_count": chunk_count}


# ---------------------------------------------------------------------------
# Feedback + stats (MongoDB-backed)
# ---------------------------------------------------------------------------

class FeedbackRequest(BaseModel):
    thread_id: str = Field(..., min_length=1)
    source_filename: Optional[str] = None
    question: str
    answer: str
    rating: str  # "up" or "down"
    comment: Optional[str] = None


@app.post("/feedback")
def submit_feedback(request: FeedbackRequest, req: Request):
    """
    Records a thumbs up/down on a specific answer. This is the feedback loop referenced in
    the README: reviewing a stream of (question, answer, rating, comment) rows over time is
    how you'd spot systematic weak points -- e.g. consistently poor ratings on questions
    about a specific document or ticker -- rather than relying on anecdote.
    """
    if request.rating not in ("up", "down"):
        raise HTTPException(status_code=422, detail="rating must be 'up' or 'down'")

    user = get_optional_user(req)
    user_id = user["user_id"] if user else None

    try:
        db.record_feedback(
            thread_id=request.thread_id,
            source_filename=request.source_filename,
            question=request.question,
            answer=request.answer,
            rating=request.rating,
            comment=request.comment,
            user_id=user_id,
        )
        return {"message": "Feedback recorded"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recording feedback: {e}", exc_info=True)
        raise _dependency_error(e, "Could not record feedback")


@app.get("/documents/stats")
def document_stats():
    """Returns per-document metadata (chunk counts, upload times, whether vision analysis
    is available) from MongoDB -- distinct from /documents, which only lists filenames
    present in ChromaDB. FastAPI's default JSON encoding handles the datetime objects Mongo
    returns automatically (converts to ISO 8601 strings), so no manual conversion is needed
    here the way the old SQLAlchemy version required.

    Unlike /documents this endpoint has no fallback: MongoDB *is* the data source, so an
    outage is reported as a 503 rather than an empty list, which would otherwise read as
    "you have no documents".
    """
    try:
        return {"documents": db.get_all_documents()}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching document stats: {e}", exc_info=True)
        raise _dependency_error(e, "Could not fetch document stats")


# ---------------------------------------------------------------------------
# Portfolio (authenticated)
# ---------------------------------------------------------------------------

class PortfolioAddRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=20)


@app.get("/portfolio")
def get_portfolio(user: dict = Depends(get_current_user)):
    """Returns the authenticated user's ticker watchlist."""
    try:
        tickers = db.get_portfolio(user["user_id"])
        return {"tickers": tickers}
    except Exception as e:
        logger.error(f"Error fetching portfolio: {e}", exc_info=True)
        raise _dependency_error(e, "Could not fetch portfolio")


@app.post("/portfolio")
def add_to_portfolio(request: PortfolioAddRequest, user: dict = Depends(get_current_user)):
    """Adds a ticker to the authenticated user's watchlist."""
    try:
        tickers = db.add_to_portfolio(user["user_id"], request.ticker)
        return {"tickers": tickers}
    except Exception as e:
        logger.error(f"Error adding to portfolio: {e}", exc_info=True)
        raise _dependency_error(e, "Could not update portfolio")


@app.delete("/portfolio/{ticker}")
def remove_from_portfolio(ticker: str, user: dict = Depends(get_current_user)):
    """Removes a ticker from the authenticated user's watchlist."""
    try:
        tickers = db.remove_from_portfolio(user["user_id"], ticker)
        return {"tickers": tickers}
    except Exception as e:
        logger.error(f"Error removing from portfolio: {e}", exc_info=True)
        raise _dependency_error(e, "Could not update portfolio")


# ---------------------------------------------------------------------------
# Chat history (authenticated)
# ---------------------------------------------------------------------------

@app.get("/chat-history")
def get_chat_threads(user: dict = Depends(get_current_user)):
    """Returns all conversation contexts for the authenticated user."""
    try:
        threads = db.get_user_threads(user["user_id"])
        return {"threads": threads}
    except Exception as e:
        logger.error(f"Error fetching chat threads: {e}", exc_info=True)
        raise _dependency_error(e, "Could not fetch chat history")


@app.get("/chat-history/{context_key:path}")
def get_chat_history(context_key: str, user: dict = Depends(get_current_user)):
    """Returns all messages for a specific conversation context."""
    try:
        messages = db.get_chat_history(user["user_id"], context_key)
        for m in messages:
            if m.get("extra") and m["extra"].get("chart_id"):
                import base64
                from mongo_storage import load_chart
                chart_bytes = load_chart(m["extra"]["chart_id"])
                if chart_bytes:
                    m["extra"]["chart_base64"] = base64.b64encode(chart_bytes).decode("utf-8")
        return {"messages": messages}
    except Exception as e:
        logger.error(f"Error fetching chat history: {e}", exc_info=True)
        raise _dependency_error(e, "Could not fetch chat history")


@app.delete("/chat-history/{context_key:path}")
def delete_chat_history(context_key: str, user: dict = Depends(get_current_user)):
    """Clears all messages for a specific conversation context."""
    try:
        deleted = db.delete_chat_history(user["user_id"], context_key)
        return {"message": f"Deleted {deleted} messages.", "deleted_count": deleted}
    except Exception as e:
        logger.error(f"Error deleting chat history: {e}", exc_info=True)
        raise _dependency_error(e, "Could not delete chat history")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
