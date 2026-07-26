"""
Ingestion Pipeline for Tech Company Earnings Reports & Product Releases.
This script is responsible for taking raw markdown documents, chunking them
intelligently to preserve context, embedding them, and storing them in a local ChromaDB.
"""

import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Ingest")

import io
from pypdf import PdfReader

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extracts text from a PDF file's raw bytes and converts it into pseudo-Markdown.

    Why convert to Markdown at all? The rest of the ingestion pipeline (process_and_ingest)
    is built around MarkdownHeaderTextSplitter, which relies on '#'-style headers to keep
    semantically related content together and to attach useful metadata to each chunk.
    PDFs (earnings call transcripts, 10-Ks) have no such structure once text is extracted,
    so we synthesize a '### Page N' header per page. This gives the splitter a natural,
    page-level boundary to chunk around instead of blindly splitting on character count,
    and it lets retrieved chunks reference which page they came from.
    """
    reader = PdfReader(io.BytesIO(file_bytes))

    pages_text = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            pages_text.append(f"### Page {i + 1}\n\n{text}")

    if not pages_text:
        logger.warning("No extractable text found in PDF (it may be a scanned/image-only document).")

    return "\n\n".join(pages_text)

# We use Langchain's text splitters.
# MarkdownHeaderTextSplitter allows us to chunk by headers, retaining the document structure in metadata.
# RecursiveCharacterTextSplitter ensures that chunks don't exceed our model's token limits.
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# Vector store configuration and the shared retriever singleton both come from retriever.py.
#
# Why import rather than redeclare: this module used to keep its own copy of CHROMA_DB_DIR
# ("./chroma_db") and its own get_embeddings_model(). Two copies of the same constant is two
# places to get the path wrong, and they *did* drift in effect -- both were relative, so the
# directory each one resolved to depended on the process's working directory. Retrieval and
# ingestion silently writing to and reading from different databases is the worst possible
# version of that bug. There is now exactly one definition, in retriever.py.
#
# Note the old `from langchain_community.vectorstores import Chroma` and
# `from langchain_community.embeddings import HuggingFaceEmbeddings` imports are gone. Both
# were deprecated compatibility shims scheduled for removal from langchain-community; when
# they are removed those lines become an ImportError at import time, and because api.py
# imports this module at module scope, that takes down the entire API rather than just
# ingestion. The Chroma import was never used at all, and embeddings are obtained from
# retriever.py (which uses the maintained langchain-huggingface package) so that ingestion
# and querying cannot possibly disagree about the embedding configuration.
from retriever import CHROMA_DB_DIR, COLLECTION_NAME, get_retriever_instance

def process_and_ingest(markdown_text: str, source_metadata: dict):
    """
    Implements a Hybrid Parent-Child Chunking Strategy.

    Why this approach?
    1. MarkdownHeaderTextSplitter (Parent): It breaks the document based on structural headers.
       This ensures that semantic sections (like 'Fiscal Quarter', 'Financial Results') are kept
       together, and the headers are appended to the metadata of each chunk. This allows the LLM
       and retriever to know *exactly* which section a chunk belongs to.
    2. RecursiveCharacterTextSplitter (Child): The structural chunks might still be too large
       for the BAAI/bge-small-en-v1.5's strict 512-token limit. We use a recursive character
       splitter with a 1200 character limit (roughly equating to 250-350 tokens, well within
       the 512 limit) to break down large sections while preserving overlap for context.

    Re-ingesting a filename that is already in the store *replaces* its chunks rather than
    adding a second copy -- see step 3.

    Returns (vectorstore, chunk_count).
    """

    # 1. Parent Chunking: Split by Markdown Headers
    # We explicitly define the headers we expect in our tech reports to enforce metadata inheritance.
    headers_to_split_on = [
        ("#", "Company"),
        ("##", "Fiscal_Quarter"),
        ("###", "Section")
    ]

    # The strip_headers=False ensures the actual headers aren't removed from the text,
    # which can sometimes help the embedding model understand the immediate context.
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )

    # This splits the document into larger structural blocks with header metadata.
    header_splits = markdown_splitter.split_text(markdown_text)

    # Inject source metadata into the splits
    for split in header_splits:
        split.metadata.update(source_metadata)

    # 2. Child Chunking: Ensure compatibility with BGE model context window
    # chunk_size=1200 characters is a safe threshold for a 512-token limit (avg 4 chars/token).
    # chunk_overlap=120 characters ensures we don't sever important sentences midway,
    # providing sliding window context for the embedding model.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=120,
        separators=["\n\n", "\n", ".", " ", ""] # Hierarchical splitting priorities
    )

    # Split the document objects (which already contain metadata from the markdown splitter)
    final_splits = text_splitter.split_documents(header_splits)

    logger.info(f"Split document into {len(final_splits)} child chunks.")

    # 3. Vectorization and Persistence
    # Instead of creating a new Chroma client (which could cause SQLite locks or out-of-sync
    # memory states when the API is running), we reuse the globally initialized retriever
    # instance from retriever.py.
    retriever_instance = get_retriever_instance()
    vectorstore = retriever_instance.vectorstore

    # Replace-on-re-ingest: drop any chunks already stored under this `source` first.
    #
    # Why: Chroma's add_documents always inserts with fresh ids, so uploading the same file
    # twice used to leave two complete copies of every chunk in the collection. Retrieval then
    # returned the same passage two or three times, crowding real context out of the top-k and
    # doubling the tokens sent to the synthesis model. It also put the two persistence layers
    # into disagreement: db.record_document() upserts on `filename`, so Mongo reported one
    # document with N chunks while Chroma actually held 2N. Deleting first makes re-ingestion
    # idempotent and keeps the two layers describing the same reality.
    source_filename = source_metadata.get("source")
    if source_filename:
        _delete_existing_chunks(vectorstore, source_filename)

    vectorstore.add_documents(documents=final_splits)

    logger.info(
        f"Successfully vectorized and added {len(final_splits)} chunks for "
        f"'{source_filename}' to '{COLLECTION_NAME}' at {CHROMA_DB_DIR}"
    )

    return vectorstore, len(final_splits)


def _delete_existing_chunks(vectorstore, source_filename: str) -> int:
    """
    Deletes every chunk previously stored under this `source` metadata value.

    Failures here are logged but not raised: a stale-chunk cleanup that fails degrades
    retrieval quality (duplicate context), whereas raising would abort an otherwise perfectly
    good ingestion and leave the user with no chunks at all for the file they just uploaded.
    The worse outcome is the one we avoid.
    """
    try:
        existing = vectorstore.get(where={"source": source_filename}, include=[])
        ids = existing.get("ids", [])
        if ids:
            vectorstore.delete(ids=ids)
            logger.info(
                f"Replacing existing content for '{source_filename}': removed {len(ids)} stale chunk(s)."
            )
        return len(ids)
    except Exception as e:
        logger.warning(
            f"Could not remove existing chunks for '{source_filename}' ({e}); the new chunks "
            f"will be added alongside them, which may produce duplicate retrieval results."
        )
        return 0

def ingest_directory(directory_path: str):
    """
    Crawls the specified directory for Markdown and PDF files and ingests them.
    """
    import glob
    md_files = glob.glob(os.path.join(directory_path, "**/*.md"), recursive=True)
    pdf_files = glob.glob(os.path.join(directory_path, "**/*.pdf"), recursive=True)

    if not md_files and not pdf_files:
        print(f"No Markdown or PDF files found in {directory_path}")
        return

    for file_path in md_files:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # We attach the source file name as metadata so we know where chunks came from
        source_metadata = {"source": os.path.basename(file_path)}
        print(f"Processing: {file_path}")
        process_and_ingest(content, source_metadata)

    for file_path in pdf_files:
        with open(file_path, "rb") as f:
            file_bytes = f.read()

        content = extract_text_from_pdf(file_bytes)
        if not content.strip():
            print(f"Skipping {file_path}: no extractable text found.")
            continue

        source_metadata = {"source": os.path.basename(file_path)}
        print(f"Processing: {file_path}")
        process_and_ingest(content, source_metadata)

if __name__ == "__main__":
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist. Creating it...")
        os.makedirs(data_dir)
    else:
        print(f"Starting ingestion for {data_dir} into {CHROMA_DB_DIR}...")
        ingest_directory(data_dir)
