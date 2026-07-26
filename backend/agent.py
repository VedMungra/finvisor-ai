"""
Self-Reflective LangGraph Agent with Web Search and Memory.

Graph shape (and why it looks like this):

    START -> prepare -> [retrieve -> grade_documents -> (web_search)?]? -> generate
                            ^                                              |  ^
                            |                                              v  |
                            +------ (fast-path recovery) ------------  tools --+
                                                                           |
                                                                    force_answer -> END

`prepare` exists for two reasons. First, MemorySaver keeps a checkpoint per thread_id, so a
second turn on the same conversation starts with the *previous* turn's `context`,
`web_search_required` and tool-loop counter still in state; every run must reset those or a
follow-up question silently answers itself out of stale context. Second, it is the only place
that can both write state and be followed by a conditional edge, which is what the
market-data fast path (see should_take_fast_path) needs.

The `tools -> generate` cycle is bounded (see MAX_TOOL_LOOPS): an unbounded cycle is how a
model that keeps re-requesting the same tool turns into an uncaught GraphRecursionError and a
500 from /chat. When the budget runs out the graph routes to `force_answer`, which produces a
real answer from whatever the tools already returned rather than raising.
"""
import os
import time
import logging
from typing import Annotated, Any, Optional, Sequence, TypedDict, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, RemoveMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.errors import GraphRecursionError
from langgraph.checkpoint.memory import MemorySaver
from retriever import get_retriever
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Configure logging. This lives at the very top of the module (rather than halfway down, where
# it used to sit) because module-level helpers below log during import-time fallbacks.
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Agent")


class AgentState(TypedDict):
    # The message history
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # The retrieved context (either from local vector DB or web)
    context: str
    # Flag to indicate if web search is needed
    web_search_required: bool
    # The current question being processed
    question: str
    # The document scope filter (if any)
    source_filename: str
    # How many times the tools node has run during *this* invocation. Bounds the
    # generate <-> tools cycle; reset by prepare_node on every run so it can't accumulate
    # across turns that share a thread_id (and therefore a checkpoint).
    tool_loops: int
    # True while this run is on the retrieval-skipping market-data fast path. Cleared as soon
    # as we fall back to retrieval so the fallback can only ever happen once.
    fast_path: bool

from llm_config import get_llm, get_llm_for_role

# For grading we want structured output.
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")


# ---------------------------------------------------------------------------
# Message-content normalisation
#
# `.content` is NOT reliably a string. Anthropic (and Gemini, and any provider returning
# multi-part responses through langchain-core's content-block support) return a *list* of
# blocks -- e.g. [{"type": "text", "text": "..."}, {"type": "tool_use", ...}]. Calling
# .strip() / .lower() on that raises AttributeError, which is exactly how the grader's
# fallback path used to crash the whole graph run whenever structured output failed on a
# Claude-backed provider. Every read of `.content` in this module goes through _as_text().
# ---------------------------------------------------------------------------

def _as_text(content: Any) -> str:
    """Flattens any LangChain message content shape into a plain string.

    Handles the three shapes seen in practice: a plain string, a list of dict content
    blocks, and a list of block objects exposing a `.text` attribute. Anything else is
    stringified rather than raising -- a grader that mis-parses a weird response is
    recoverable, a grader that throws takes down the request.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
            else:
                text = getattr(block, "text", None)
                if isinstance(text, str) and text:
                    parts.append(text)
        return "\n".join(parts).strip()
    return str(content)


def _message_text(message: Any) -> str:
    """_as_text() for a message object (or anything with a .content)."""
    return _as_text(getattr(message, "content", message))


def _latest_question(messages: Sequence[BaseMessage]) -> str:
    """Returns the most recent human turn as plain text.

    Deliberately scans backwards for a HumanMessage instead of blindly taking
    messages[-1]: on the fast-path recovery route (see should_continue) the graph re-enters
    retrieve_node *after* the model has already appended an AIMessage, so messages[-1] is no
    longer the user's question at that point.
    """
    for message in reversed(list(messages or [])):
        if isinstance(message, HumanMessage):
            return _message_text(message).strip()
    if messages:
        return _message_text(messages[-1]).strip()
    return ""


# Heavy clients (LLM, embedding/reranker models, search tool) are constructed lazily on first
# use instead of at import time. Loading the embedding + cross-encoder models eagerly blocks
# uvicorn from binding its port until they finish, which trips Render's port-scan timeout on
# slower/free instances. Deferring construction lets the server start immediately and pay that
# cost on the first real request instead.
def get_primary_llm():
    """Backward-compatible alias for the 'synthesis' role (see llm_config.py). Used by
    get_retriever_instance() (an unused constructor param on the retriever) and by
    api.py's /suggest_questions endpoint. grade_documents_node and generate_node use
    get_llm_for_role() directly for the 'grader' and 'synthesis' roles respectively --
    see llm_config.py's module docstring for why these are split into separate roles with
    their own provider fallback chains."""
    return get_llm()

_retriever_instance = None

def get_retriever_instance():
    """Returns the Stage-2 Reranked Retriever, initializing it on first call."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = get_retriever(llm=get_primary_llm(), k_initial=10, k_final=3)
    return _retriever_instance

_web_search_tool = None
_web_search_unavailable = False

def get_web_search_tool():
    """Returns the Tavily web search tool, or None if it cannot be constructed.

    TavilySearch raises at *construction* time when TAVILY_API_KEY is missing, so the old
    eager `TavilySearch(max_results=3)` turned "no web-search key configured" into a hard
    500 on every query that fell through to web search. Web search is a fallback path, not a
    hard requirement: if it is unavailable we log once, return None, and let web_search_node
    hand generate() an empty context so the model can honestly say it lacks the information.
    """
    global _web_search_tool, _web_search_unavailable
    if _web_search_tool is not None:
        return _web_search_tool
    if _web_search_unavailable:
        return None
    if not os.getenv("TAVILY_API_KEY"):
        logger.warning("TAVILY_API_KEY is not set -- web search is disabled for this process.")
        _web_search_unavailable = True
        return None
    try:
        from langchain_tavily import TavilySearch
        _web_search_tool = TavilySearch(max_results=3)
        return _web_search_tool
    except Exception as e:
        logger.warning(f"Could not initialise Tavily web search ({e}); continuing without web search.")
        _web_search_unavailable = True
        return None


# ---------------------------------------------------------------------------
# Market-data fast path
#
# The full pipeline (ChromaDB similarity search -> cross-encoder rerank -> grader LLM call)
# runs before generate() on *every* query, including ones that obviously cannot be answered
# from an uploaded document -- "plot Apple's 6-month chart" has no answer in a PDF, it needs
# the yfinance tool. Paying for a vector search, a rerank and a grader round-trip first was
# measured at roughly a minute on the market-data path, and it burns grader quota for nothing.
#
# The classifier below is deliberately lopsided: a false negative just means the query takes
# the old (correct, slower) route, while a false positive would send a document question to
# generate() with no context. So it requires an affirmative market-data signal AND the absence
# of any word suggesting the user means an uploaded document, and it never fires when the
# question is scoped to a source_filename -- scoped questions keep the exact routing they had
# (see decide_to_generate for why that matters). As a final safety net, should_continue routes
# a fast-pathed query that produced no tool call at all back into retrieval, so even a false
# positive still gets a correct answer, just one extra LLM call slower.
# ---------------------------------------------------------------------------

import re

ENABLE_MARKET_FAST_PATH = os.getenv("ENABLE_MARKET_FAST_PATH", "1").strip().lower() not in ("0", "false", "no")

# Charting is only possible via the plot_stock_chart tool, so an unscoped "plot/chart/graph"
# request is by construction a market-data request.
_CHART_INTENT_RE = re.compile(r"\b(plot|chart|charts|graph|candlestick)\b")

# Metrics that only ever come from a live quote feed, not from a static uploaded document.
_MARKET_METRIC_RE = re.compile(
    r"\b("
    r"stock price|share price|current price|price today|today'?s price|live price|"
    r"market cap|market capitali[sz]ation|p/e|pe ratio|price[- ]to[- ]earnings|forward p/e|"
    r"beta|dividend yield|52[- ]?week|moving average|intrinsic value|dcf|discounted cash flow|"
    r"fundamental|fundamentals|valuation|debt[- ]to[- ]equity|short ratio|"
    r"ticker|stock quote|analyst rating"
    r")\b"
)

# Anything hinting the user is asking about an uploaded document vetoes the fast path.
_DOCUMENT_SIGNAL_RE = re.compile(
    r"\b("
    r"document|documents|report|reports|filing|filings|pdf|10[- ]?k|10[- ]?q|8[- ]?k|"
    r"transcript|earnings call|press release|slide|slides|appendix|footnote|"
    r"page \d+|this file|the file|uploaded|attachment|"
    r"summar(?:y|ise|ize|ised|ized|ising|izing)|according to|as stated|in the text|"
    r"q[1-4]\b|fiscal|quarter|quarterly|revenue|net sales|net income|guidance|segment|"
    r"management|md&a|risk factor"
    r")\b"
)


def should_take_fast_path(question: str, source_filename: Optional[str]) -> bool:
    """True when a question is unscoped and unambiguously about live market data."""
    if not ENABLE_MARKET_FAST_PATH:
        return False
    # A question scoped to an uploaded document ALWAYS goes through retrieval. Skipping it
    # would be the same failure decide_to_generate() guards against, one node earlier.
    if source_filename:
        return False
    text = (question or "").lower()
    if not text:
        return False
    if _DOCUMENT_SIGNAL_RE.search(text):
        return False
    return bool(_CHART_INTENT_RE.search(text) or _MARKET_METRIC_RE.search(text))


def prepare_node(state: AgentState):
    """Normalises the incoming question and resets all per-run state.

    Every key the downstream nodes read is written here, so nothing reads a key that was
    never initialised (the old code did `state["question"]` / `state["context"]` on paths
    where neither had been written yet) and nothing inherits a value from the previous turn
    on the same thread_id -- MemorySaver would otherwise carry the last turn's retrieved
    context and tool-loop count straight into this one.
    """
    question = _latest_question(state.get("messages", []))
    source_filename = state.get("source_filename")
    fast_path = should_take_fast_path(question, source_filename)
    if fast_path:
        logger.info(f"Fast path: '{question}' looks like a live market-data query -- skipping retrieval + grading.")
    return {
        "question": question,
        "context": "",
        "web_search_required": False,
        "tool_loops": 0,
        "fast_path": fast_path,
    }


def route_after_prepare(state: AgentState) -> Literal["retrieve", "generate"]:
    """Sends clearly-quantitative unscoped queries straight to the tool-calling generator."""
    return "generate" if state.get("fast_path") else "retrieve"


def retrieve_node(state: AgentState):
    """Retrieves context from ChromaDB based on the current question.

    Returns an empty context instead of raising if the vector store is unavailable: an empty
    context is a state the rest of the graph already understands (grade_documents_node routes
    it to web search, or generate() honestly reports it has nothing), whereas an exception
    here would surface as a 500.
    """
    question = state.get("question") or _latest_question(state.get("messages", []))
    source_filename = state.get("source_filename")
    logger.info(f"Retrieving local context for: '{question}' in scope: '{source_filename}'...")

    update: dict = {"question": question, "fast_path": False}

    # If we got here via the fast-path fallback, the model has already appended a "no data"
    # AIMessage that is about to be superseded. Drop it so the final history contains one
    # answer rather than a retracted non-answer followed by the real one.
    if state.get("fast_path"):
        messages = list(state.get("messages", []))
        if messages:
            last = messages[-1]
            if isinstance(last, AIMessage) and not getattr(last, "tool_calls", None) and getattr(last, "id", None):
                update["messages"] = [RemoveMessage(id=last.id)]

    try:
        docs = get_retriever_instance().invoke(question, source_filename)
    except Exception as e:
        logger.error(f"Retrieval failed for '{question}': {e}", exc_info=True)
        docs = []

    formatted_context = "\n\n".join([f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}" for doc in docs])

    update["context"] = formatted_context
    return update

def grade_documents_node(state: AgentState):
    """
    Determines whether the retrieved documents are relevant to the question.

    Uses the 'grader' role (fast/cheap provider chain -- Groq by default, falling back to
    Gemini) rather than the 'synthesis' role, since this is a simple, high-frequency binary
    classification that runs on every single query. Reserving the higher-quality/rate-limited
    synthesis model for the answer generation step (where reasoning quality actually matters)
    means the two tasks don't compete for the same provider's quota.
    """
    logger.info("Grading document relevance...")
    question = state.get("question", "")
    context = state.get("context", "") or ""

    if not context.strip():
        logger.info("No local documents found. Falling back to web search.")
        return {"web_search_required": True}

    prompt = f"""You are a grader assessing relevance of a retrieved document to a user question.
    Here is the retrieved document:
    \n ------- \n
    {context}
    \n ------- \n
    Here is the user question: {question}
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

    grader_llm = get_llm_for_role("grader")
    try:
        score = grader_llm.with_structured_output(GradeDocuments).invoke(prompt)
        grade = _as_text(getattr(score, "binary_score", "")).strip().lower()
    except Exception as e:
        logger.warning(f"Structured output failed ({e}), using fallback prompt...")
        fallback_prompt = prompt + "\nRespond ONLY with 'yes' or 'no'."
        try:
            res = grader_llm.invoke(fallback_prompt)
            # _as_text(), not .strip() on .content directly: providers that return content
            # blocks (Claude in particular) hand back a list here, and the bare .strip()
            # this used to do raised AttributeError and killed the whole run.
            grade = _as_text(getattr(res, "content", res)).strip().lower()
        except Exception as inner:
            # Both grading attempts failed -- every provider in the grader chain is down or
            # rate-limited. Treat the retrieved documents as relevant rather than dropping
            # the user into a web search they didn't ask for: we already have context in
            # hand, and generate() can decide for itself whether it answers the question.
            logger.error(f"Grader unavailable ({inner}); assuming retrieved documents are relevant.")
            return {"web_search_required": False}

    if "yes" in grade:
        logger.info("Documents are relevant.")
        return {"web_search_required": False}
    else:
        logger.info("Documents are NOT relevant. Falling back to web search.")
        return {"web_search_required": True}

def web_search_node(state: AgentState):
    """Searches the web for the answer when local documents fail.

    Every failure mode here (no API key, invalid key, network error, rate limit) degrades to
    an empty context rather than an exception. generate() reads an empty context as "I have
    nothing to work from" and says so, which is a far better outcome than a 500.
    """
    question = state.get("question", "")
    search_tool = get_web_search_tool()
    if search_tool is None:
        logger.warning("Web search requested but unavailable; continuing with no additional context.")
        return {"context": ""}

    logger.info("Searching the web using Tavily...")
    try:
        docs = search_tool.invoke({"query": question})
    except Exception as e:
        logger.warning(f"Tavily web search failed ({e}); continuing with no additional context.")
        return {"context": ""}

    # TavilySearch returns a dict with a 'results' key in newer versions
    results_list = docs.get('results', []) if isinstance(docs, dict) else docs
    if not isinstance(results_list, list):
        results_list = []
    # Tavily also returns its own AI-synthesized short answer under 'answer' when available --
    # this is often more directly on-topic than the raw scraped results, which can occasionally
    # match on the wrong sense of a query's keywords (e.g. a generic word in the question
    # matching an unrelated page) and return irrelevant junk. Prioritizing the synthesized
    # answer, and skipping any raw result with empty content, keeps the context focused.
    tavily_answer = docs.get('answer') if isinstance(docs, dict) else None
    logger.info(f"Tavily returned {len(results_list)} result(s) for query: '{question}'. Has synthesized answer: {bool(tavily_answer)}")

    context_parts = []
    if tavily_answer:
        context_parts.append(f"Tavily AI Summary: {tavily_answer}")
    for d in results_list:
        if not isinstance(d, dict):
            continue
        content = (d.get('content') or "").strip()
        if content:
            context_parts.append(f"Source: {d.get('url', 'Unknown')}\n{content}")

    web_results = "\n\n".join(context_parts)
    logger.info(f"Web results context length: {len(web_results)} chars. Preview: {web_results[:500]!r}")

    # Replace existing context with web results
    return {"context": web_results}

from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
import numexpr

@tool
def calculator(expression: str) -> str:
    """
    Evaluates a mathematical expression using numexpr.
    Useful for calculating percentages, margins, or basic arithmetic based on financial data.
    Input must be a valid mathematical string (e.g., '119.6 / 117.2 - 1').
    """
    logger.info(f"Calculating: {expression}")
    try:
        # Empty local/global dicts: without them numexpr resolves bare names against the
        # *calling frame's* namespace, so a malformed expression can silently pick up one of
        # this module's globals instead of failing.
        result = numexpr.evaluate(expression, local_dict={}, global_dict={}).item()
        if isinstance(result, bool):
            # numexpr happily evaluates comparisons, so prose like "this is not math" comes
            # back as True rather than an error. A boolean is never a useful answer here.
            return (f"Error: '{expression}' is a comparison, not a numeric expression. "
                    f"Provide arithmetic such as '119.6 / 117.2 - 1'.")
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"

import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import uuid
import io
import json
import redis

# ---------------------------------------------------------------------------
# Redis cache: optional, lazily connected, never fatal.
#
# This used to call redis_client.ping() at module scope. That makes `import agent` -- and
# therefore uvicorn's startup -- block for the full socket timeout whenever Redis is
# unreachable but not actively refusing connections (a hung container, a firewalled host,
# a DNS black hole). Connecting on first *use* with an explicit short timeout keeps the
# existing "cache is optional" semantics while moving the cost off the import path. A failed
# connection is remembered for a cooldown window so we don't re-pay the timeout on every
# single tool call, but it is retried afterwards so a Redis that comes up later starts
# getting used again without a restart.
# ---------------------------------------------------------------------------

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
REDIS_TIMEOUT_SECONDS = float(os.getenv("REDIS_TIMEOUT_SECONDS", "1.0"))
_REDIS_RETRY_COOLDOWN_SECONDS = 60.0

_redis_client = None
_redis_last_failure_at = 0.0


def get_redis_client():
    """Returns a live Redis client, or None if the cache is unavailable. Never raises."""
    global _redis_client, _redis_last_failure_at
    if _redis_client is not None:
        return _redis_client
    if _redis_last_failure_at and (time.monotonic() - _redis_last_failure_at) < _REDIS_RETRY_COOLDOWN_SECONDS:
        return None
    try:
        client = redis.Redis.from_url(
            REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=REDIS_TIMEOUT_SECONDS,
            socket_timeout=REDIS_TIMEOUT_SECONDS,
        )
        client.ping()
        _redis_client = client
        logger.info(f"Connected to Redis at {REDIS_URL} (cache enabled).")
        return _redis_client
    except Exception as e:
        _redis_last_failure_at = time.monotonic()
        logger.warning(f"Could not connect to Redis: {e}. Continuing without cache.")
        return None


def _cache_get(key: str) -> Optional[str]:
    client = get_redis_client()
    if client is None:
        return None
    try:
        return client.get(key)
    except Exception as e:
        logger.warning(f"Redis cache error reading '{key}': {e}")
        return None


def _cache_set(key: str, value: str, ttl_seconds: int = 600) -> None:
    client = get_redis_client()
    if client is None:
        return
    try:
        client.setex(key, ttl_seconds, value)
    except Exception as e:
        logger.warning(f"Failed to write to Redis cache key '{key}': {e}")


def _cache_delete(key: str) -> None:
    client = get_redis_client()
    if client is None:
        return
    try:
        client.delete(key)
    except Exception as e:
        logger.warning(f"Failed to delete Redis cache key '{key}': {e}")


# ---------------------------------------------------------------------------
# yfinance helpers.
#
# yfinance scrapes a public endpoint; it is rate-limited, and its response shapes have moved
# between versions. Every tool below funnels through these so a data-source hiccup produces a
# useful string for the model rather than an exception -- a raised exception inside a @tool
# aborts the tool call and, without handling, the graph run.
# ---------------------------------------------------------------------------

def _safe_ticker_info(ticker: str) -> dict:
    """Fetches yfinance's .info dict, returning {} instead of raising.

    `.info` performs a live HTTP fetch on attribute access and can raise (rate limit,
    HTML instead of JSON, delisted symbol) or return an unusable stub.
    """
    try:
        info = yf.Ticker(ticker).info
    except Exception as e:
        logger.warning(f"yfinance .info lookup failed for '{ticker}': {e}")
        return {}
    return info if isinstance(info, dict) else {}


def _extract_close_series(hist):
    """Pulls a usable closing-price series out of a yfinance history frame, or None.

    'Close' is not guaranteed: some periods/versions return 'Adj Close' only, and a
    MultiIndex column layout shows up when yfinance decides the request was multi-ticker.
    Indexing hist['Close'] blindly raises KeyError inside the tool.
    """
    if hist is None:
        return None
    try:
        if getattr(hist.columns, "nlevels", 1) > 1:
            hist = hist.droplevel(-1, axis=1)
        columns = list(hist.columns)
    except Exception:
        return None
    for name in ("Close", "Adj Close", "close", "adjclose"):
        if name in columns:
            try:
                series = hist[name].dropna()
            except Exception:
                continue
            if len(series) > 0:
                return series
    return None


def _news_item_fields(item: Any) -> tuple:
    """Returns (title, summary) for a yfinance news item across its various shapes.

    Older yfinance put 'title'/'summary' at the top level of each item; newer versions nest
    them under 'content'. `item["content"]["title"]` raises KeyError/TypeError on whichever
    shape the installed version doesn't use, so try both.
    """
    if not isinstance(item, dict):
        return "", ""
    title = ""
    summary = ""
    content = item.get("content")
    if isinstance(content, dict):
        title = content.get("title") or ""
        summary = content.get("summary") or content.get("description") or ""
    if not title:
        title = item.get("title") or ""
    if not summary:
        summary = item.get("summary") or item.get("description") or ""
    return str(title).strip(), str(summary).strip()


def _summarize_price_series(ticker: str, period: str, closes) -> str:
    """Condenses a closing-price series into the handful of numbers an analyst would quote.

    The tool used to hand the model nothing but a chart id. Since the model cannot see the
    PNG, being told to "write an analysis of this chart" left it with no data, so it produced
    hedged filler ("if the chart shows an upward trend, that may indicate..."). Returning the
    actual start/end/high/low figures alongside the id means the required prose is grounded in
    real numbers -- and the numbers come from the same series the chart was drawn from, so the
    text and the image can never disagree.
    """
    try:
        first = float(closes.iloc[0])
        last = float(closes.iloc[-1])
        high = float(closes.max())
        low = float(closes.min())
        change_pct = ((last - first) / first * 100) if first else 0.0
        start_date = str(closes.index[0])[:10]
        end_date = str(closes.index[-1])[:10]
        return (
            f"Price data underlying the chart ({ticker.upper()}, {period}, "
            f"{start_date} to {end_date}, {len(closes)} sessions):\n"
            f"- Starting close: ${first:,.2f}\n"
            f"- Latest close: ${last:,.2f}\n"
            f"- Change over period: {change_pct:+.2f}%\n"
            f"- Period high: ${high:,.2f}\n"
            f"- Period low: ${low:,.2f}\n"
        )
    except Exception as e:
        logger.warning(f"Could not summarize price series for {ticker} ({period}): {e}")
        return ""


# Instruction appended to a successful chart generation. Deliberately verbose: with the old
# one-line "You MUST include this exact tag" some providers replied with the tag and nothing
# else, and since api.py strips the tag out before returning, the user saw a completely empty
# message next to the image.
def _chart_success_message(chart_id: str, price_summary: str = "") -> str:
    return (
        f"Chart successfully generated (chart id: {chart_id}).\n"
        f"{price_summary}"
        f"Your final answer MUST contain BOTH of the following:\n"
        f"  1. A full written analysis in prose, using the actual figures above -- quote the "
        f"start and latest close, the percentage move, and the period high/low, then explain "
        f"the trend, the likely drivers, and what an investor should take away. Several "
        f"sentences at minimum, and do NOT hedge with 'if the chart shows...' -- you have the "
        f"numbers, state what they say.\n"
        f"  2. This exact tag, copied verbatim, on its own line at the end: <chart>{chart_id}</chart>\n"
        f"Returning the tag on its own (or with only a one-line caption) is NOT an acceptable "
        f"answer -- the tag is stripped before the user sees the reply, so a tag-only response "
        f"reaches them as a blank message. Never mention the tag itself to the user."
    )


def _chart_still_exists(chart_id: str) -> bool:
    """True if the cached chart id still resolves to a stored chart.

    The chart id is cached in Redis for 600s but the PNG it points at lives in MongoDB
    GridFS. Those two stores have independent lifetimes: wipe Mongo (fresh container,
    dropped database) while Redis still holds the key and the cached id resolves to nothing,
    so api.py's load_chart() returns None and the frontend renders a broken image. Verifying
    before serving from cache turns that into a cheap regeneration instead.
    """
    try:
        import mongo_storage
        checker = getattr(mongo_storage, "chart_exists", None)
        if callable(checker):
            return bool(checker(chart_id))
        return mongo_storage.load_chart(chart_id) is not None
    except Exception as e:
        logger.warning(f"Could not verify cached chart '{chart_id}' in MongoDB: {e}")
        return False


_VALID_CHART_PERIODS = ("1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max")


@tool
def plot_stock_chart(ticker: str, period: str = "1mo") -> str:
    """
    Fetches historical stock prices using yfinance and generates a matplotlib chart. The
    chart is built in memory and stored in MongoDB (see mongo_storage.save_chart) rather
    than written to local disk -- see mongo_storage.py's module docstring for why.
    Period options: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max.
    """
    logger.info(f"Plotting chart for {ticker} ({period})...")

    if not ticker or not str(ticker).strip():
        return "Error: a ticker symbol is required to plot a chart."
    ticker = str(ticker).strip()
    period = (period or "1mo").strip().lower()
    if period not in _VALID_CHART_PERIODS:
        return (f"Error: '{period}' is not a supported period. "
                f"Valid options are: {', '.join(_VALID_CHART_PERIODS)}.")

    cache_key = f"chart_{ticker.upper()}_{period}"
    cached = _cache_get(cache_key)
    if cached:
        # Entries are JSON {"chart_id": ..., "price_summary": ...}. A bare string is an entry
        # written by an older build still inside its TTL -- treat it as the chart id alone.
        try:
            payload = json.loads(cached)
            cached_chart_id = payload.get("chart_id")
            cached_summary = payload.get("price_summary", "")
        except (ValueError, AttributeError):
            cached_chart_id, cached_summary = cached, ""

        if cached_chart_id and _chart_still_exists(cached_chart_id):
            logger.info(f"Serving chart for {ticker} ({period}) from Redis cache.")
            return _chart_success_message(cached_chart_id, cached_summary)
        logger.warning(
            f"Cached chart id '{cached_chart_id}' is no longer in MongoDB (stale cache entry); "
            f"dropping it and regenerating."
        )
        _cache_delete(cache_key)

    fig = None
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if hist is None or hist.empty:
            return (f"Error: No price history found for ticker {ticker} with period {period}. "
                    f"The symbol may be wrong, delisted, or the data provider may be rate-limiting.")

        closes = _extract_close_series(hist)
        if closes is None:
            return (f"Error: price history for {ticker} came back without a usable closing-price "
                    f"column (got: {list(getattr(hist, 'columns', []))}).")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(closes.index, closes.values, marker='', color='blue', linewidth=2)
        ax.set_title(f"{ticker.upper()} Stock Price - Last {period}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Closing Price (USD)")
        ax.grid(True)

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)

        chart_id = f"{ticker.lower()}_{uuid.uuid4().hex[:8]}"
        from mongo_storage import save_chart
        save_chart(chart_id, buf.getvalue())

        price_summary = _summarize_price_series(ticker, period, closes)
        _cache_set(cache_key, json.dumps({"chart_id": chart_id, "price_summary": price_summary}), ttl_seconds=600)

        return _chart_success_message(chart_id, price_summary)
    except Exception as e:
        logger.warning(f"Chart generation failed for {ticker} ({period}): {e}")
        return f"Error generating chart for {ticker}: {e}"
    finally:
        # Always release the figure -- Agg keeps every un-closed figure alive, so an
        # exception between subplots() and close() leaks memory on each failed call.
        if fig is not None:
            plt.close(fig)

@tool
def get_stock_fundamentals(ticker: str) -> str:
    """
    Fetches fundamental financial metrics for a given stock ticker using yfinance.
    Useful for answering questions about a company's valuation, profitability, and risk.
    """
    logger.info(f"Fetching fundamentals for {ticker}...")

    if not ticker or not str(ticker).strip():
        return "Error: a ticker symbol is required to fetch fundamentals."
    ticker = str(ticker).strip()

    cache_key = f"fundamentals_{ticker.upper()}"
    cached_data = _cache_get(cache_key)
    if cached_data:
        logger.info(f"Serving fundamentals for {ticker} from Redis cache.")
        return cached_data

    try:
        info = _safe_ticker_info(ticker)

        # Explicit parentheses: `and` binds tighter than `or`, so the un-parenthesised
        # original read as `not info or (A and B)` even though it scans left-to-right as
        # `(not info or A) and B`. The parenthesised form below is the intended meaning --
        # bail out only when there is no data at all, or when *neither* price field is
        # present -- and it no longer depends on the reader knowing Python's precedence.
        if not info or ('regularMarketPrice' not in info and 'previousClose' not in info):
            return (f"Error: Could not retrieve fundamental data for ticker {ticker}. "
                    f"The symbol may be invalid, or the data provider may be rate-limiting.")

        metrics = {
            "Current Price": info.get("currentPrice", info.get("previousClose", "N/A")),
            "Beta (Volatility vs Market)": info.get("beta", "N/A"),
            "Trailing P/E Ratio": info.get("trailingPE", "N/A"),
            "Forward P/E Ratio": info.get("forwardPE", "N/A"),
            "Debt-to-Equity Ratio": info.get("debtToEquity", "N/A"),
            "52-Week High": info.get("fiftyTwoWeekHigh", "N/A"),
            "52-Week Low": info.get("fiftyTwoWeekLow", "N/A"),
            "50-Day Moving Avg": info.get("fiftyDayAverage", "N/A"),
            "200-Day Moving Avg": info.get("twoHundredDayAverage", "N/A"),
            "Profit Margin": info.get("profitMargins", "N/A"),
            "Short Ratio": info.get("shortRatio", "N/A")
        }

        report = f"Fundamental Risk Metrics for {ticker.upper()}:\n"
        for key, value in metrics.items():
            report += f"- {key}: {value}\n"

        _cache_set(cache_key, report, ttl_seconds=600)

        return report
    except Exception as e:
        return f"Error fetching stock fundamentals: {e}"

@tool
def compare_stocks(ticker1: str, ticker2: str) -> str:
    """
    Fetches fundamental metrics for two stock tickers and formats them side-by-side for comparison.
    Useful for determining which of two companies is a better investment or fundamentally stronger.
    """
    logger.info(f"Comparing fundamentals for {ticker1} vs {ticker2}...")
    try:
        if not ticker1 or not ticker2:
            return "Error: two ticker symbols are required for a comparison."

        def get_metrics(ticker):
            info = _safe_ticker_info(ticker)
            if not info or ('currentPrice' not in info and 'previousClose' not in info):
                return None
            return {
                "Price": info.get("currentPrice", info.get("previousClose", "N/A")),
                "Beta": info.get("beta", "N/A"),
                "P/E Ratio": info.get("trailingPE", "N/A"),
                "Forward P/E": info.get("forwardPE", "N/A"),
                "Debt-to-Equity": info.get("debtToEquity", "N/A"),
                "Profit Margin": info.get("profitMargins", "N/A"),
                "Revenue Growth": info.get("revenueGrowth", "N/A"),
                "Return on Equity": info.get("returnOnEquity", "N/A")
            }

        data1 = get_metrics(ticker1)
        data2 = get_metrics(ticker2)

        if not data1: return f"Error: Could not retrieve data for {ticker1}."
        if not data2: return f"Error: Could not retrieve data for {ticker2}."

        report = f"Comparison: {ticker1.upper()} vs {ticker2.upper()}\n"
        report += "-" * 40 + "\n"
        for key in data1.keys():
            report += f"{key}: {ticker1.upper()}={data1[key]} | {ticker2.upper()}={data2[key]}\n"

        return report
    except Exception as e:
        return f"Error comparing stocks: {e}"

@tool
def get_news_sentiment(ticker: str) -> str:
    """
    Fetches the most recent news headlines for a given stock ticker.
    Use this to determine if the current media sentiment is bullish or bearish.
    """
    logger.info(f"Fetching recent news for {ticker}...")
    try:
        if not ticker or not str(ticker).strip():
            return "Error: a ticker symbol is required to fetch news."

        ticker_obj = yf.Ticker(str(ticker).strip())
        try:
            # `.news` is a live fetch on attribute access and can raise on rate limits.
            news = ticker_obj.news
        except Exception as e:
            logger.warning(f"yfinance .news lookup failed for '{ticker}': {e}")
            return f"Error: could not retrieve news for {ticker} ({e})."

        if not news:
            return f"No recent news found for {ticker}."

        headlines = []
        for item in news[:10]:
            title, summary = _news_item_fields(item)
            if title:
                headlines.append(f"- {title}: {summary}" if summary else f"- {title}")

        if not headlines:
            return (f"News was returned for {ticker} but none of the items had a readable "
                    f"headline (the data provider's response shape may have changed).")

        return f"Recent News for {ticker.upper()}:\n" + "\n".join(headlines)
    except Exception as e:
        return f"Error fetching news for {ticker}: {e}"

@tool
def calculate_intrinsic_value(ticker: str, growth_rate: float = 0.10, discount_rate: float = 0.10, terminal_growth_rate: float = 0.025) -> str:
    """
    Calculates the intrinsic value of a stock using a Discounted Cash Flow (DCF) model.
    By default, assumes a 10% growth rate, 10% discount rate, and 2.5% terminal growth rate.
    Rates are decimals (0.10 == 10%). The discount rate must exceed the terminal growth rate.
    """
    logger.info(f"Calculating DCF Intrinsic Value for {ticker}...")
    try:
        if not ticker or not str(ticker).strip():
            return "Error: a ticker symbol is required to run a DCF."

        try:
            growth_rate = float(growth_rate)
            discount_rate = float(discount_rate)
            terminal_growth_rate = float(terminal_growth_rate)
        except (TypeError, ValueError):
            return ("Error: growth_rate, discount_rate and terminal_growth_rate must all be "
                    "numbers expressed as decimals (e.g. 0.10 for 10%).")

        # The Gordon-growth terminal value divides by (discount_rate - terminal_growth_rate).
        # Equal rates are a ZeroDivisionError; a terminal growth rate above the discount rate
        # is worse than an error -- it silently flips the denominator negative and produces a
        # confidently-stated negative "intrinsic value". Both are modelling mistakes, not
        # data problems, so reject them with an explanation the model can relay.
        if discount_rate <= 0:
            return f"Error: discount_rate must be greater than 0 (got {discount_rate}). Rates are decimals, e.g. 0.10 for 10%."
        if terminal_growth_rate < 0:
            return f"Error: terminal_growth_rate cannot be negative (got {terminal_growth_rate})."
        if discount_rate <= terminal_growth_rate:
            return (f"Error: a DCF requires discount_rate ({discount_rate}) to be strictly greater than "
                    f"terminal_growth_rate ({terminal_growth_rate}). Otherwise the terminal value is "
                    f"infinite or negative and the result is meaningless. Try a higher discount rate "
                    f"(e.g. 0.10) or a lower terminal growth rate (e.g. 0.025).")

        info = _safe_ticker_info(ticker)
        if not info:
            return (f"Error: Could not retrieve financial data for {ticker}. The symbol may be "
                    f"invalid, or the data provider may be rate-limiting.")

        fcf = info.get("freeCashflow")
        shares = info.get("sharesOutstanding")
        current_price = info.get("currentPrice", info.get("previousClose"))

        if not fcf or not shares:
            return f"Error: Free Cash Flow or Shares Outstanding data not available for {ticker}. Cannot run DCF."

        try:
            fcf = float(fcf)
            shares = float(shares)
        except (TypeError, ValueError):
            return f"Error: Free Cash Flow or Shares Outstanding for {ticker} were not numeric. Cannot run DCF."

        if shares <= 0:
            return f"Error: Shares Outstanding for {ticker} is not positive ({shares}). Cannot run DCF."
        if fcf <= 0:
            return (f"Error: {ticker.upper()} reports negative or zero free cash flow ({fcf:,.0f}). "
                    f"A standard DCF cannot value a company with no positive cash flow to discount.")

        # Project 5 years
        val = 0
        curr_fcf = fcf
        for i in range(1, 6):
            curr_fcf *= (1 + growth_rate)
            val += curr_fcf / ((1 + discount_rate) ** i)

        # Terminal Value
        tv = (curr_fcf * (1 + terminal_growth_rate)) / (discount_rate - terminal_growth_rate)
        val += tv / ((1 + discount_rate) ** 5)

        intrinsic_value_per_share = val / shares

        try:
            current_price = float(current_price)
            price_line = f"Current Price: ${current_price:,.2f}\n"
            assessment = "UNDERVALUED (Bullish)" if intrinsic_value_per_share > current_price else "OVERVALUED (Bearish)"
        except (TypeError, ValueError):
            price_line = "Current Price: unavailable\n"
            assessment = "Cannot compare to market price (current price unavailable)"

        return (f"DCF Analysis for {ticker.upper()}:\n"
                f"--------------------------\n"
                f"{price_line}"
                f"Calculated Intrinsic Value: ${intrinsic_value_per_share:,.2f}\n"
                f"Assessment: {assessment}\n"
                f"(Assumptions: {growth_rate*100}% Growth, {discount_rate*100}% Discount Rate, {terminal_growth_rate*100}% Terminal Growth)")
    except Exception as e:
        return f"Error calculating DCF for {ticker}: {str(e)}"

import concurrent.futures

@tool
def summarize_document(source_filename: str) -> str:
    """
    Generates a comprehensive executive summary of an entire document using a Map-Reduce strategy.
    Use this tool ONLY when the user explicitly asks for a general summary of the whole report.
    """
    logger.info(f"Generating comprehensive summary for {source_filename}...")

    try:
        # 1. Fetch all chunks for this document from ChromaDB
        data = get_retriever_instance().vectorstore.get(where={"source": source_filename}, include=["documents"])
        docs = data.get("documents", [])
        if not docs:
            return f"Error: Document {source_filename} not found in the database."

        # 2. Batch chunks (approx 20 chunks per batch)
        batch_size = 20
        batches = [docs[i:i + batch_size] for i in range(0, len(docs), batch_size)]

        llm = get_llm_for_role("synthesis")

        # 3. Map Step: Summarize each batch concurrently
        def summarize_batch(batch, batch_index):
            text = "\n\n".join(batch)
            prompt = f"Extract the key financial metrics, risks, and strategic updates from the following text excerpt of a larger document. Be concise but comprehensive:\n\n{text}"
            logger.info(f"Summarizing batch {batch_index+1}/{len(batches)}...")
            res = llm.invoke([HumanMessage(content=prompt)])
            return _as_text(getattr(res, "content", res))

        # Results are collected into a pre-sized list indexed by batch number, not appended in
        # completion order. as_completed() yields whichever batch finishes first, so appending
        # scrambled the document: the reduce step below was being handed section summaries in
        # essentially random order, and it dutifully wrote an "executive summary" whose
        # narrative jumped around the report.
        #
        # Each future is also resolved inside its own try/except. future.result() re-raises
        # whatever the worker threw, so one rate-limited batch out of twenty used to abort the
        # entire summary. A summary missing one section (and saying so) beats no summary.
        results: list = [None] * len(batches)
        failed_batches = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # max_workers=3 to avoid rate limits while still being fast
            future_to_index = {
                executor.submit(summarize_batch, batch, i): i
                for i, batch in enumerate(batches)
            }
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    summary = future.result()
                except Exception as e:
                    failed_batches += 1
                    logger.warning(f"Batch {index+1}/{len(batches)} of '{source_filename}' failed to summarize: {e}")
                    continue
                if summary and summary.strip():
                    results[index] = summary

        chunk_summaries = [summary for summary in results if summary]

        if failed_batches:
            logger.warning(
                f"{failed_batches}/{len(batches)} batches of '{source_filename}' failed; "
                f"summarizing from the {len(chunk_summaries)} that succeeded."
            )

        # 4. Reduce Step: Final Summary
        if len(chunk_summaries) > 1:
            logger.info("Combining batch summaries into final executive summary...")
            combined_summaries = "\n\n--- Next Section ---\n\n".join(chunk_summaries)
            coverage_note = ""
            if failed_batches:
                coverage_note = (
                    f"\n\nNote: {failed_batches} of {len(batches)} sections of this document could not be "
                    f"processed, so state plainly at the end of your summary that it covers "
                    f"{len(chunk_summaries)} of {len(batches)} sections of the document."
                )
            final_prompt = f"You are an elite financial advisor. Based on the following section summaries of a large document, provide a comprehensive, unified executive summary highlighting the most critical strategic and financial insights. Structure it professionally with clear headings:\n\n{combined_summaries}{coverage_note}"
            final_res = llm.invoke([HumanMessage(content=final_prompt)])
            return _as_text(getattr(final_res, "content", final_res))
        elif len(chunk_summaries) == 1:
            return chunk_summaries[0]

        if failed_batches:
            return (f"Error: every section of {source_filename} failed to summarize "
                    f"({failed_batches}/{len(batches)} batches errored). The language-model "
                    f"providers may be rate-limited -- try again shortly.")
        return "Error: Could not generate summary."
    except Exception as e:
        logger.error(f"Error summarizing document {source_filename}: {e}", exc_info=True)
        return f"Error summarizing document: {e}"

from vision_tool import analyze_document_visually

tools = [calculator, plot_stock_chart, get_stock_fundamentals, compare_stocks, get_news_sentiment,
         calculate_intrinsic_value, analyze_document_visually, summarize_document]

# handle_tool_errors is set explicitly rather than left to the default so it can't be lost to a
# library default change, and so tools defined in *other* modules are covered too:
# analyze_document_visually lives in vision_tool.py and lets llm_config's vision fallback chain
# raise when every provider fails. Without this, that exception would escape the tools node and
# abort the whole run; with it, the model receives the error as a ToolMessage and can explain
# the failure or try another approach.
def _format_tool_error(e: Exception) -> str:
    logger.warning(f"Tool call raised: {e}")
    return (f"Error: that tool failed with: {e}. Do not retry it with identical arguments -- "
            f"either try a different approach or tell the user what could not be determined.")

_tool_node = ToolNode(tools, handle_tool_errors=_format_tool_error)


# ---------------------------------------------------------------------------
# Tool-loop budget.
#
# `generate -> tools -> generate` is a cycle with no natural termination: a model that keeps
# emitting tool calls (a common failure mode when a tool returns an error string it doesn't
# know how to recover from) runs until LangGraph's recursion limit trips and raises
# GraphRecursionError, which nothing caught, so /chat returned a 500 and a stack trace.
#
# Two independent guards now exist. The explicit budget below stops the cycle *inside* the
# graph and routes to force_answer_node, which turns whatever the tools already produced into
# a real answer. The recursion limit set on the compiled graph (and the GraphRecursionError
# handler around invoke) is the backstop for anything the budget doesn't anticipate.
# ---------------------------------------------------------------------------

MAX_TOOL_LOOPS = int(os.getenv("AGENT_MAX_TOOL_LOOPS", "5"))
# Worst case node count for one run: prepare, generate (fast path), retrieve, grade,
# web_search, generate, then MAX_TOOL_LOOPS x (tools, generate), then force_answer. The
# headroom below covers that with room to spare.
GRAPH_RECURSION_LIMIT = int(os.getenv("AGENT_RECURSION_LIMIT", str(2 * MAX_TOOL_LOOPS + 15)))


def tools_node(state: AgentState, config=None):
    """Runs the prebuilt ToolNode and increments the per-run tool-loop counter.

    `config` is accepted and forwarded rather than dropped: LangGraph threads its runtime
    (store, injected arguments, callbacks) through the config, and ToolNode raises
    "Missing required config key" if it is invoked without it.
    """
    result = _tool_node.invoke(state, config)
    update = dict(result) if isinstance(result, dict) else {"messages": result}
    update["tool_loops"] = state.get("tool_loops", 0) + 1
    return update


from langchain_core.messages import SystemMessage

# The base prompt below is deliberately unchanged: the [Source: ...] citation discipline and
# the "if it isn't in the context, say you don't know" rule are verified working behaviour and
# regressions there are the expensive kind (silent hallucination). Additional guidance is
# appended as separate blocks rather than woven into it.
_BASE_SYSTEM_PROMPT = """You are an elite Senior Financial Advisor and Wealth Manager. Your job is not just to extract numbers, but to provide strategic, forward-looking financial advice and actionable insights based on the documents.
Synthesize the data to explain *why* it matters to an investor, identify potential risks, and suggest strategic implications.

Use the following pieces of retrieved context to answer the question.
If the answer is not contained in the context, just say that you don't know. Do not hallucinate.

When answering, ALWAYS provide comprehensive, detailed explanations and maintain a highly professional, advisory tone.
If the question involves math or financial metrics, explicitly show the formula you used, the exact numbers you extracted from the text, and the step-by-step calculation.

CRITICAL: You MUST cite the source of your information.
For every sentence that relies on the context, append a strict citation in the format [Source: <source_name>]."""

# Clarifies the boundary between "don't know" (a document fact you weren't given) and "go
# fetch it" (live market data a tool can retrieve). Without this the don't-know rule above
# makes the model refuse market-data questions on the fast path, where context is empty by
# design.
_TOOL_GUIDANCE = """
LIVE MARKET DATA: The retrieved context is static text from uploaded documents. Anything that requires *current* market data -- prices, price charts, valuation multiples, fundamentals, news sentiment, or a DCF valuation -- must be fetched by calling the appropriate tool rather than guessed at or refused. The "say you don't know" rule above governs facts that should have come from the documents; it does not apply to data you can simply look up with a tool.

CHARTS: When a chart tool reports success, your final answer MUST contain BOTH a substantial written analysis in prose AND the exact <chart>...</chart> tag it gave you, verbatim, on its own line at the end. The tag is stripped out before the user sees your reply, so an answer consisting of only the tag arrives as a blank message. Never return the tag alone, and never describe or explain the tag to the user."""

_NO_CONTEXT_NOTE = """
NOTE: No document context was retrieved for this question -- either nothing relevant was found, or the question was routed as a live market-data query. Answer it with your tools where they apply. If it needs information from an uploaded document that you have not been given, say so plainly rather than inventing an answer."""


def _build_system_prompt(context: str) -> str:
    prompt = _BASE_SYSTEM_PROMPT + "\n" + _TOOL_GUIDANCE
    if not (context or "").strip():
        prompt += "\n" + _NO_CONTEXT_NOTE
    return prompt + f"\n\nContext:\n{context}"


def generate_node(state: AgentState):
    """
    Generates the final answer using the retrieved context.

    Uses the 'synthesis' role (Gemini by default, falling back to Claude) -- this is the
    task where reasoning quality, citation discipline, and financial-advisor tone actually
    matter, as opposed to grade_documents_node's simple binary classification. Keeping this
    on its own provider chain, separate from the 'grader' role, means a burst of grading
    calls can't exhaust the quota this step needs.
    """
    logger.info("Synthesizing answer...")
    context = state.get("context", "") or ""

    prompt = _build_system_prompt(context)

    # Prepend the system prompt to the message history so the LLM retains context of tool calls
    messages = [SystemMessage(content=prompt)] + list(state.get("messages", []))

    try:
        response = get_llm_for_role("synthesis").bind_tools(tools).invoke(messages)
    except Exception as e:
        # Every provider in the synthesis chain failed (see llm_config.FallbackLLM, which
        # re-raises the last error once the chain is exhausted). Returning an honest message
        # keeps /chat at 200 with an explanation instead of a 500 with a stack trace.
        logger.error(f"Synthesis failed on every configured provider: {e}", exc_info=True)
        return {"messages": [AIMessage(content=(
            "I'm unable to generate an answer right now -- every configured language-model "
            "provider rejected the request (this is usually a rate limit or an expired API "
            "key). Please try again in a moment."
        ))]}

    return {"messages": [response]}


# ---------------------------------------------------------------------------
# Tool-budget exhaustion
# ---------------------------------------------------------------------------

def _collect_tool_findings(messages: Sequence[BaseMessage], limit_chars: int = 12000) -> str:
    """Flattens every ToolMessage in the history into plain text for a final synthesis pass."""
    findings = []
    for message in messages or []:
        if isinstance(message, ToolMessage):
            text = _message_text(message).strip()
            if text:
                name = getattr(message, "name", None) or "tool"
                findings.append(f"[{name}]\n{text}")
    joined = "\n\n".join(findings)
    return joined[:limit_chars]


def _answer_from_findings(messages: Sequence[BaseMessage], question: str, context: str, reason: str) -> str:
    """Produces a final answer from data the tools already returned, without calling tools.

    Rebuilt as a fresh two-message conversation (system + question) rather than replaying the
    accumulated history on purpose. At the point this runs, the history usually ends with an
    assistant message carrying unanswered tool_calls, and most providers reject a request
    whose tool_use block has no matching tool_result -- replaying it would swap one crash for
    another. Flattening the tool output into text sidesteps the tool-call protocol entirely.
    """
    findings = _collect_tool_findings(messages)

    prompt = (
        _BASE_SYSTEM_PROMPT
        + "\n\nIMPORTANT: You have run out of tool budget for this request and CANNOT call any "
          "more tools. Write the best possible final answer using only the data below. If it is "
          "not enough to answer fully, say clearly and specifically what you could not determine "
          "rather than inventing it.\n"
        + "If any of the data below contains a tag of the form <chart>SOME_ID</chart>, reproduce "
          "that tag verbatim on its own line at the end of your answer, alongside your written "
          "analysis.\n\n"
        + f"Context from documents/web:\n{context or '(none)'}\n\n"
        + f"Data already gathered from tools:\n{findings or '(no tool results were returned)'}"
    )

    try:
        response = get_llm_for_role("synthesis").invoke(
            [SystemMessage(content=prompt), HumanMessage(content=question or "Please answer my question.")]
        )
        text = _as_text(getattr(response, "content", response)).strip()
        if text:
            return text
        logger.warning("Final synthesis returned empty content; falling back to raw tool findings.")
    except Exception as e:
        logger.error(f"Final synthesis after {reason} failed: {e}", exc_info=True)

    # Last resort: hand back the raw tool output. It is not polished, but it is real data --
    # and if a chart was generated its <chart> tag survives, so api.py can still render it.
    if findings:
        return (
            "I wasn't able to finish composing a full analysis for this request, but here is the "
            "data I gathered:\n\n" + findings
        )
    return (
        "I wasn't able to complete this request -- I kept needing more data and couldn't converge "
        "on an answer. Please try rephrasing the question, or ask about one thing at a time."
    )


def force_answer_node(state: AgentState):
    """Terminates the tool loop with a real answer once the tool budget is exhausted."""
    logger.warning(
        f"Tool budget of {MAX_TOOL_LOOPS} loop(s) exhausted; forcing a final answer from "
        f"the results gathered so far."
    )
    answer = _answer_from_findings(
        state.get("messages", []),
        state.get("question", ""),
        state.get("context", "") or "",
        reason="tool budget exhaustion",
    )
    return {"messages": [AIMessage(content=answer)]}


# Conditional routing logic for generation to tools
def should_continue(state: AgentState) -> Literal["tools", "retrieve", "force_answer", END]:
    """Routes after generation: another tool round, a fast-path rescue, a forced answer, or done."""
    messages = list(state.get("messages", []))
    last_message = messages[-1] if messages else None
    tool_calls = getattr(last_message, "tool_calls", None) if last_message else None

    if tool_calls:
        # Only used for logging, so never let an unexpected tool_call shape break routing.
        first = tool_calls[0]
        name = (first.get("name") if isinstance(first, dict) else getattr(first, "name", None)) or "unknown"
        tool_loops = state.get("tool_loops", 0)
        if tool_loops >= MAX_TOOL_LOOPS:
            logger.warning(
                f"Model requested '{name}' after {tool_loops} tool loop(s) -- "
                f"budget exhausted, forcing a final answer."
            )
            return "force_answer"
        logger.info(f"Calling tool: {name} (loop {tool_loops + 1}/{MAX_TOOL_LOOPS})...")
        return "tools"

    # Safety net for the market-data fast path: if we skipped retrieval on a hunch and the
    # model then answered without touching a single tool, the hunch was probably wrong (this
    # was a document question after all). Fall back into the full retrieval pipeline rather
    # than returning a context-free answer. retrieve_node clears `fast_path`, so this can
    # only ever fire once per run.
    if state.get("fast_path") and state.get("tool_loops", 0) == 0:
        logger.info("Fast path produced no tool call -- falling back to the full retrieval pipeline.")
        return "retrieve"

    return END

# Conditional routing logic for web search
def decide_to_generate(state: AgentState) -> Literal["generate", "web_search"]:
    """
    Determines whether to generate an answer, or route to web search.

    Web search is intentionally skipped whenever the question is scoped to a specific
    uploaded document (source_filename is set). If a user has explicitly selected "ask about
    this PDF," silently substituting an unrelated web search result for that document's own
    content is a worse failure mode than the model saying "I don't have that information in
    this document" -- e.g. a vague "summarize this report" query can fail the relevance
    grader (broad summary requests don't always match narrow retrieved chunks well) and
    previously fell through to a generic web search that returned completely unrelated
    articles. Scoped questions always go straight to generate() with whatever was retrieved.
    """
    if state.get("web_search_required", False) and not state.get("source_filename"):
        return "web_search"
    else:
        return "generate"

# Build the LangGraph Workflow
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("prepare", prepare_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade_documents", grade_documents_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("generate", generate_node)
workflow.add_node("tools", tools_node)
workflow.add_node("force_answer", force_answer_node)

# Add edges
workflow.add_edge(START, "prepare")
# Conditional edge after preparation: the market-data fast path skips retrieval + grading
workflow.add_conditional_edges(
    "prepare",
    route_after_prepare,
    {
        "retrieve": "retrieve",
        "generate": "generate",
    }
)
workflow.add_edge("retrieve", "grade_documents")
# Conditional edge after grading
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "web_search": "web_search",
        "generate": "generate",
    }
)
workflow.add_edge("web_search", "generate")

# Conditional edge after generation
workflow.add_conditional_edges(
    "generate",
    should_continue,
    {
        "tools": "tools",
        "retrieve": "retrieve",
        "force_answer": "force_answer",
        END: END
    }
)
# Edge from tools back to generate
workflow.add_edge("tools", "generate")
workflow.add_edge("force_answer", END)

# Compile graph with memory checkpointer
memory = MemorySaver()
_compiled_app = workflow.compile(checkpointer=memory)


class _ResilientAgentApp:
    """Thin wrapper around the compiled graph that bounds and contains runaway runs.

    It does exactly two things beyond delegating to the graph:

      1. Applies GRAPH_RECURSION_LIMIT to every invocation, so the limit is a property of the
         agent rather than something each caller (api.py, benchmark_latency.py) has to
         remember to pass in its config.
      2. Converts a GraphRecursionError into a real answer. The in-graph tool budget should
         reach force_answer_node long before the recursion limit trips, but "should" is doing
         a lot of work in a graph with cycles -- and the failure mode without this is a
         stack trace surfacing to the user as an HTTP 500.

    Every other attribute (.stream, .get_state, .get_graph, ...) passes straight through, so
    the object remains a drop-in for the compiled graph it wraps.
    """

    def __init__(self, graph):
        self._graph = graph

    def __getattr__(self, name):
        # __getattr__ only fires for attributes normal lookup missed. Guard `_graph` itself
        # so a partially-constructed instance raises AttributeError rather than recursing
        # infinitely into this method.
        if name == "_graph":
            raise AttributeError(name)
        return getattr(self._graph, name)

    def __repr__(self):
        return f"<ResilientAgentApp recursion_limit={GRAPH_RECURSION_LIMIT} max_tool_loops={MAX_TOOL_LOOPS} wrapping {self._graph!r}>"

    def invoke(self, input, config=None, **kwargs):
        config = dict(config or {})
        config.setdefault("recursion_limit", GRAPH_RECURSION_LIMIT)
        try:
            return self._graph.invoke(input, config=config, **kwargs)
        except GraphRecursionError as e:
            logger.error(f"Graph hit its recursion limit of {GRAPH_RECURSION_LIMIT}: {e}")
            messages = []
            question = ""
            context = ""
            try:
                snapshot = self._graph.get_state(config)
                values = getattr(snapshot, "values", None) or {}
                messages = list(values.get("messages", []))
                question = values.get("question", "")
                context = values.get("context", "") or ""
            except Exception as snapshot_error:
                logger.warning(f"Could not read graph state after recursion limit: {snapshot_error}")

            if not question:
                question = _latest_question(messages) or _latest_question((input or {}).get("messages", []) if isinstance(input, dict) else [])

            answer = _answer_from_findings(messages, question, context, reason="recursion limit")
            # Drop any trailing assistant turn with unanswered tool calls so the caller's
            # messages[-1] is the answer, not a dangling tool request.
            cleaned = [
                m for m in messages
                if not (isinstance(m, AIMessage) and getattr(m, "tool_calls", None))
            ]
            return {"messages": cleaned + [AIMessage(content=answer)]}


# `app` is what api.py imports (`from agent import app as agent_app`). It keeps the compiled
# graph's interface; `graph` below exposes the raw CompiledStateGraph for anything that needs
# it directly (visualisation, tests).
graph = _compiled_app
app = _ResilientAgentApp(_compiled_app)
