"""
Vision Tool: lets the agent "look at" a page of an uploaded PDF instead of only reading
pypdf's plain-text extraction of it.

Why this is needed:
ingest.py's extract_text_from_pdf() pulls raw text per page. That's fine for prose (MD&A
sections, earnings-call transcript paragraphs), but it silently mangles or drops anything
that lives in a table or chart -- e.g. a segment-revenue table in a 10-K, or a bar chart of
quarterly guidance. pypdf has no concept of a table; it just emits whatever text tokens it
finds in whatever order the PDF's content stream places them, which for tables is often
column-by-column garbage.

Rather than trying to fix that with more text-layer heuristics, this tool fetches the
original PDF bytes from MongoDB (mongo_storage.py), renders the requested page to an image
(via PyMuPDF/fitz -- pure Python binding, no poppler/system dependency), and hands that image
to a vision-capable model. The model reads the table/chart the way a human would: by looking
at it.

Provider selection, message-format differences (Claude vs Gemini vs Ollama expect different
image content-block schemas), and automatic fallback across the configured vision provider
chain are all handled by llm_config.invoke_vision_with_fallback() -- this module stays focused
on PDF-page-to-image rendering and doesn't need to know which provider ends up serving the
request.

This is registered as an additional LangGraph tool alongside the existing yfinance/calculator
tools in agent.py, so the router can call it whenever the retrieved text context looks
incomplete for a numeric/tabular question.

Two rules govern everything below, and both come from how LangGraph executes tools:

  1. A @tool must RETURN its errors as a string, never raise. A raised exception propagates
     out of ToolNode and kills the whole graph run, so the user gets a 500 instead of an
     answer -- even though the agent could easily have recovered by answering from the text
     context, or by asking about a different page. Every failure path here therefore ends in
     `return "Error: ..."`, phrased for the model as much as for the human reading the logs,
     so the agent can decide what to do next.

  2. The rendered image has to fit through the provider's request-size limit. A page image is
     an inline base64 payload, and base64 inflates bytes by ~33%; providers cap the total
     request (Anthropic ~5MB per image, Gemini ~20MB per request, Ollama's limit is whatever
     the local box can hold in VRAM). A 100-DPI render of a normal A4 page is ~100-300KB and
     nobody thinks about it -- but PDFs are not all A4. Financial filings ship fold-out
     schedules, and posters/plans can be metres across; at 100 DPI a 1x1.5m page renders to
     roughly 4000x6000px, which is a multi-megabyte PNG that gets rejected *after* paying the
     upload cost. _render_pdf_page_to_png_bytes() therefore caps the render, downscaling by
     page geometry up front and re-rendering smaller if the encoded PNG still comes out too
     big.
"""
import logging

logger = logging.getLogger("VisionTool")

from langchain_core.tools import tool

# Render caps. DPI is the *requested* resolution; the pixel and byte caps override it for
# oversized pages. 100 DPI is enough for a vision model to read filing-sized type, and the
# 2.2 megapixel cap keeps a full-page render around 1400x1800 -- comfortably legible while
# leaving plenty of headroom under every provider's per-image limit.
DEFAULT_DPI = 100
MAX_RENDER_PIXELS = 2_200_000
MAX_PNG_BYTES = 3_500_000
MIN_ZOOM = 0.2


def _render_pdf_page_to_png_bytes(
    pdf_bytes: bytes,
    page_number: int,
    dpi: int = DEFAULT_DPI,
    max_pixels: int = MAX_RENDER_PIXELS,
    max_bytes: int = MAX_PNG_BYTES,
) -> bytes:
    """Renders a single 1-indexed page of an in-memory PDF to PNG bytes using PyMuPDF.

    The output is bounded twice, because the two things that blow a request size limit are
    independent:
      * Geometry: a poster-size page at the requested DPI would be tens of megapixels. The
        zoom is reduced up front so width*height stays under `max_pixels`, which costs nothing
        (we never render the huge version at all).
      * Content: a page that is mostly a photograph or a dense scan can still encode large
        even at a modest pixel count, since PNG is lossless. If the encoded bytes exceed
        `max_bytes` the page is re-rendered at ~70% zoom, up to three times.
    Both caps are floored by MIN_ZOOM so we can never shrink a page into unreadability while
    chasing a byte target."""
    import fitz  # PyMuPDF

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        if page_number < 1 or page_number > doc.page_count:
            raise ValueError(
                f"Page {page_number} out of range (document has {doc.page_count} pages)."
            )
        page = doc.load_page(page_number - 1)

        zoom = max(dpi, 1) / 72  # PDF default is 72 DPI
        rect = page.rect
        width_pt = max(float(rect.width), 1.0)
        height_pt = max(float(rect.height), 1.0)

        projected_pixels = (width_pt * zoom) * (height_pt * zoom)
        if projected_pixels > max_pixels:
            scale = (max_pixels / projected_pixels) ** 0.5
            new_zoom = max(zoom * scale, MIN_ZOOM)
            logger.info(
                f"Page {page_number} is {width_pt:.0f}x{height_pt:.0f}pt; capping render at "
                f"{max_pixels / 1e6:.1f}MP (zoom {zoom:.2f} -> {new_zoom:.2f}) to stay inside "
                f"provider request-size limits."
            )
            zoom = new_zoom

        png = None
        for attempt in range(4):
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            png = pix.tobytes("png")
            if len(png) <= max_bytes or zoom <= MIN_ZOOM:
                break
            zoom = max(zoom * 0.7, MIN_ZOOM)
            logger.info(
                f"Rendered page {page_number} is {len(png) / 1e6:.1f}MB (> "
                f"{max_bytes / 1e6:.1f}MB cap); re-rendering at zoom {zoom:.2f} "
                f"(attempt {attempt + 2})."
            )
        return png
    finally:
        doc.close()


def _coerce_page_number(page_number) -> int:
    """LLM-generated tool arguments arrive as whatever the model felt like emitting -- "12",
    12.0, or "page 12".

    @tool's own pydantic arg validation already coerces the easy cases ("12" -> 12) and
    rejects the hopeless ones before this function body runs (LangGraph's ToolNode turns that
    rejection into an error ToolMessage rather than killing the run). This stays as
    defence-in-depth for the paths that bypass that validation -- direct calls, and any future
    langchain version that relaxes coercion -- because the alternative is a TypeError escaping
    a @tool, which is the one thing this module must never do."""
    if isinstance(page_number, bool):
        raise ValueError(f"Invalid page number: {page_number!r}")
    if isinstance(page_number, int):
        return page_number
    if isinstance(page_number, float):
        return int(page_number)
    import re
    match = re.search(r"-?\d+", str(page_number))
    if not match:
        raise ValueError(f"Invalid page number: {page_number!r}")
    return int(match.group(0))


@tool
def analyze_document_visually(source_filename: str, page_number: int, question: str) -> str:
    """
    Visually inspects a specific page of an uploaded PDF using vision-capable AI. Use this
    when a question is about a table, chart, figure, or scanned content that the plain-text
    retrieval context doesn't answer cleanly -- e.g. "what does the segment revenue table on
    page 12 show" or "summarize the chart on page 4".

    CRITICAL INSTRUCTION: Do NOT use this tool in a loop to read every page of a document.
    Do NOT use this tool for general text summarization. ONLY use this tool when the user
    explicitly asks about a specific table or chart on a specific page.

    Args:
        source_filename: The exact filename of the previously uploaded PDF (as shown by the
            /documents endpoint).
        page_number: 1-indexed page number to inspect.
        question: What you want to know about that page.
    """
    # Everything is inside try/except: this function is invoked by LangGraph's ToolNode, and
    # anything that escapes it aborts the entire graph run (see rule 1 in the module
    # docstring). Errors are returned as text so the agent can fall back to the retrieved
    # text context instead of the request dying.
    try:
        page_number = _coerce_page_number(page_number)
    except Exception:
        return (
            f"Error: '{page_number}' is not a valid page number. Provide a 1-indexed integer "
            f"page number, e.g. 12."
        )

    if not source_filename or not str(source_filename).strip():
        return (
            "Error: no source_filename was provided. Pass the exact filename of an uploaded "
            "PDF (as listed by the /documents endpoint)."
        )

    try:
        from mongo_storage import load_pdf
        pdf_bytes = load_pdf(source_filename)
    except Exception as e:
        logger.error(f"Failed to load PDF '{source_filename}' from storage: {e}", exc_info=True)
        return (
            f"Error: could not load '{source_filename}' from document storage ({e}). The "
            f"document store may be unavailable -- answer from the retrieved text context "
            f"instead."
        )

    if not pdf_bytes:
        return (
            f"Error: original PDF for '{source_filename}' is not available for visual "
            f"inspection (it may have been uploaded before visual analysis was enabled, or "
            f"the filename doesn't match exactly)."
        )

    try:
        png_bytes = _render_pdf_page_to_png_bytes(pdf_bytes, page_number)
    except Exception as e:
        logger.error(f"Failed to render page {page_number} of {source_filename}: {e}")
        return f"Error rendering page {page_number} of '{source_filename}': {e}"

    if not png_bytes:
        return (
            f"Error: page {page_number} of '{source_filename}' rendered to an empty image and "
            f"cannot be analyzed visually."
        )

    prompt_text = (
        "You are assisting a financial analyst. Carefully read this document page, "
        "paying special attention to any tables, charts, or figures -- extract exact "
        "numbers where visible. Then answer the question.\n\n"
        f"Question: {question}"
    )

    logger.info(
        f"Analyzing page {page_number} of {source_filename} "
        f"({len(png_bytes) / 1024:.0f}KB PNG)..."
    )
    try:
        from llm_config import invoke_vision_with_fallback, content_to_text
        response_text, provider = invoke_vision_with_fallback(png_bytes, prompt_text)
    except Exception as e:
        # invoke_vision_with_fallback raises LLMChainError with an actionable message when the
        # whole vision chain is unusable (e.g. VISION_PROVIDERS=ollama with no Ollama running,
        # which is the single most common way this tool "breaks"). Surface that message rather
        # than a bare connection stack trace, and tell the agent explicitly that it should
        # keep going -- otherwise the model tends to retry the same tool call in a loop.
        logger.error(f"Vision analysis failed for page {page_number} of {source_filename}: {e}")
        return (
            f"Error: visual analysis of page {page_number} of '{source_filename}' is "
            f"unavailable right now. {e} "
            f"Do not retry this tool; answer from the retrieved text context instead, and say "
            f"clearly if the answer isn't available there."
        )

    # response.content can be a list of content blocks rather than a plain string (Anthropic
    # always returns blocks, Gemini sometimes does). invoke_vision_with_fallback already
    # normalises, but a @tool must return a str no matter what, so normalise defensively here
    # too -- returning a list from a tool breaks ToolMessage serialisation downstream.
    text = response_text if isinstance(response_text, str) else content_to_text(response_text)
    if not text or not text.strip():
        return (
            f"Error: the vision model ({provider}) returned an empty analysis for page "
            f"{page_number} of '{source_filename}'."
        )

    logger.info(f"Page {page_number} of {source_filename} analyzed successfully via {provider}.")
    return text
