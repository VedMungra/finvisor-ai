"""
Retrieval Mechanism for the RAG Chatbot.
This script sets up the vector store retriever and the Stage-2 Cross-Encoder reranker.
"""

import os
import threading
import huggingface_hub.utils
import logging
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
huggingface_hub.utils.logging.set_verbosity_error()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Retriever")

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from typing import Optional

# langchain-community is being sunset, and unlike HuggingFaceEmbeddings (which moved to the
# maintained langchain-huggingface package) HuggingFaceCrossEncoder has no replacement there
# yet. So it stays as the primary import -- but behind a guard, because when that package
# finally drops the module this becomes an ImportError at import time, and api.py imports this
# module at module scope: the whole API would fail to start over a reranker.
#
# The fallback is not a reimplementation so much as an inlining: HuggingFaceCrossEncoder is a
# ~15-line wrapper around sentence_transformers.CrossEncoder(model_name).predict(pairs), and
# sentence-transformers is already a direct dependency (the embedding model needs it). Same
# model, same scores, one less way for the process to fail to boot.
try:
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder
except ImportError:  # pragma: no cover - only hit once langchain-community drops the module
    from sentence_transformers import CrossEncoder as _SentenceTransformersCrossEncoder

    class HuggingFaceCrossEncoder:
        """Minimal stand-in for langchain_community's HuggingFaceCrossEncoder."""

        def __init__(self, model_name: str, **model_kwargs):
            self.model_name = model_name
            self.client = _SentenceTransformersCrossEncoder(model_name, **model_kwargs)

        def score(self, text_pairs):
            scores = self.client.predict(text_pairs)
            # Some rerankers emit (not_relevant, relevant) pairs rather than a single score;
            # take the relevant column, matching langchain_community's behaviour.
            if getattr(scores, "ndim", 1) > 1:
                scores = [row[1] for row in scores]
            return scores

    logging.getLogger("Retriever").warning(
        "langchain_community.cross_encoders is unavailable; using the built-in "
        "sentence-transformers cross-encoder wrapper instead."
    )

# The persistent directory where the vector database lives.
#
# Why this is resolved to an absolute path anchored to *this file* rather than left as the
# relative "./chroma_db" it used to be: a relative path is resolved against the process's
# current working directory, so `uvicorn api:app` launched from the repo root pointed at
# <repo>/chroma_db while the same command launched from backend/ pointed at
# <repo>/backend/chroma_db. Chroma creates a persist directory silently when it doesn't
# exist, so the failure mode was not an error -- it was a brand-new *empty* vector store.
# Retrieval then returned nothing and every document-scoped question answered "I don't know",
# with no log line anywhere explaining why. Anchoring to __file__ makes the location a
# property of the code, not of however the server happened to be started.
#
# The CHROMA_DB_DIR environment variable still overrides it, which is what a containerised
# or multi-instance deployment needs in order to point at a mounted volume.
_DEFAULT_CHROMA_DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
CHROMA_DB_DIR = os.path.abspath(os.path.expanduser(os.getenv("CHROMA_DB_DIR") or _DEFAULT_CHROMA_DB_DIR))

# The Chroma collection name. This is NOT configurable on purpose: the on-disk database
# already contains a collection under this exact name, and changing it would silently point
# the app at a new, empty collection in the same directory -- the same class of bug as the
# relative-path issue above.
COLLECTION_NAME = "tech_documents"

# Bi-Encoder for Stage 1 retrieval
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
# Cross-Encoder for Stage 2 reranking
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Process-wide model singletons.
#
# Why: loading bge-small-en-v1.5 plus the ms-marco cross-encoder takes roughly 45 seconds on
# a cold CPU. These models are stateless at inference time, so there is no reason for a second
# TwoStageRetriever (agent.py holds one, ingest.py needs one) to pay that cost again.
#
# Why a lock rather than a plain `if _x is None` check: FastAPI runs sync endpoint functions
# in a threadpool, so two requests arriving before the first one finishes initialising would
# each see `None` and each start their own 45-second model load -- doubling memory and CPU at
# exactly the moment the process is already under load. The lock makes the second caller wait
# for (and then reuse) the first caller's model.
_embeddings_model = None
_cross_encoder_model = None
_model_lock = threading.Lock()


def get_embeddings_model():
    """
    Returns the shared embedding model, loading it on first use.

    This must stay byte-for-byte identical to the configuration used during ingestion --
    a different model, or the same model without normalisation, produces vectors that are
    not comparable to the ones already persisted in Chroma. encode_kwargs
    {'normalize_embeddings': True} is what makes BGE's dot product equivalent to cosine
    similarity, which is the metric Chroma is configured for.
    """
    global _embeddings_model
    if _embeddings_model is None:
        with _model_lock:
            if _embeddings_model is None:
                logger.info(f"Loading embedding model '{EMBEDDING_MODEL_NAME}' (first use)...")
                _embeddings_model = HuggingFaceEmbeddings(
                    model_name=EMBEDDING_MODEL_NAME,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                logger.info("Embedding model ready.")
    return _embeddings_model


def get_cross_encoder_model():
    """Returns the shared Stage-2 cross-encoder, loading it on first use (see
    get_embeddings_model for why this is a locked singleton)."""
    global _cross_encoder_model
    if _cross_encoder_model is None:
        with _model_lock:
            if _cross_encoder_model is None:
                logger.info(f"Loading cross-encoder '{RERANKER_MODEL_NAME}' (first use)...")
                _cross_encoder_model = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL_NAME)
                logger.info("Cross-encoder ready.")
    return _cross_encoder_model


class SearchQuery(BaseModel):
    """Schema for extracting a search query."""
    search_query: str = Field(description="The core search query to run against the database.")

class TwoStageRetriever:
    """
    Custom Retriever that implements a robust two-stage pipeline.
    Stage 1: Retrieve top k_initial documents using LLM-extracted metadata filters (ChromaDB + bge-small).
    Stage 2: Rerank those documents using a precise cross-encoder (ms-marco) and return top k_final.
    """
    def __init__(self, llm=None, k_initial: int = 10, k_final: int = 3):
        self.k_initial = k_initial
        self.k_final = k_final

        # Load Stage 1: Vector Store (Bi-Encoder)
        self.embeddings = get_embeddings_model()
        self.vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=CHROMA_DB_DIR,
            embedding_function=self.embeddings
        )
        # `llm` is retained for backwards compatibility with existing call sites (agent.py
        # passes one) but is not used: the metadata-filter-extraction step it was meant for
        # was replaced by explicit `source_filename` filtering, which is exact rather than
        # probabilistic. It defaults to None so callers -- and the __main__ block below --
        # don't have to construct an LLM just to run a search.
        self.llm = llm

        # Load Stage 2: Cross-Encoder Reranker
        self.cross_encoder = get_cross_encoder_model()

    def _source_exists(self, source_filename: str) -> bool:
        """
        Reports whether any chunk in the collection carries this `source` metadata value.

        Why this is worth a second query: Chroma's similarity_search with a metadata filter
        returns an empty list both when the document isn't in the store at all and when it
        is but nothing matched well. Those are very different problems -- the first is an
        ingestion/upload failure, the second is a genuine "no relevant context" -- and
        collapsing them into one silent `[]` is what made the CHROMA_DB_DIR bug above so
        hard to diagnose. This is a cheap metadata-only lookup (limit=1, no documents or
        embeddings fetched), and it only runs on the empty-result path.
        """
        try:
            hit = self.vectorstore.get(where={"source": source_filename}, limit=1, include=[])
            return bool(hit.get("ids"))
        except Exception as e:
            # Never let a diagnostic lookup change the outcome of a retrieval.
            logger.warning(f"Could not check whether source '{source_filename}' exists: {e}")
            return False

    def invoke(self, query: str, source_filename: Optional[str] = None) -> list[Document]:
        """
        Runs the two-stage retrieval and returns at most k_final documents.

        This is called from inside the agent graph, which is called from FastAPI's /chat
        endpoint, so an exception here becomes a 500 for the user. A corrupt or missing HNSW
        index, a Chroma schema mismatch after a version upgrade, or a cross-encoder scoring
        failure are all recoverable from the user's point of view -- the agent falls back to
        web search when local context is empty -- so they are logged loudly and degraded
        into an empty (or un-reranked) result rather than propagated.
        """
        search_text = query
        filter_kwargs = {"source": source_filename} if source_filename else None

        logger.info(f"Retriever searching for: '{search_text}' with filter: {filter_kwargs}")

        # 1. Retrieve top-k_initial documents fast via cosine similarity
        try:
            docs = self.vectorstore.similarity_search(search_text, k=self.k_initial, filter=filter_kwargs)
        except Exception as e:
            logger.error(
                f"Stage-1 vector search failed against '{CHROMA_DB_DIR}' "
                f"(collection '{COLLECTION_NAME}'): {e}",
                exc_info=True,
            )
            return []

        if not docs:
            # Distinguish "that document was never ingested" from "that document has nothing
            # relevant to say about this question" -- see _source_exists.
            if source_filename and not self._source_exists(source_filename):
                logger.warning(
                    f"No chunks exist for source '{source_filename}' in collection "
                    f"'{COLLECTION_NAME}' at '{CHROMA_DB_DIR}'. The document was either never "
                    f"ingested, was ingested under a different filename, or the vector store "
                    f"was cleared."
                )
            else:
                logger.info(f"No relevant chunks matched query '{search_text}' (filter={filter_kwargs}).")
            return []

        # 2. Pair query with each document for the cross-encoder
        text_pairs = [[query, doc.page_content] for doc in docs]

        # 3. Score pairs
        try:
            scores = self.cross_encoder.score(text_pairs)
        except Exception as e:
            # Stage 1 already returned documents ordered by cosine similarity. That ordering
            # is worse than the reranked one but far better than nothing, so degrade to it
            # rather than dropping usable context on the floor.
            logger.error(
                f"Stage-2 cross-encoder scoring failed ({e}); falling back to un-reranked "
                f"Stage-1 order.",
                exc_info=True,
            )
            return docs[:self.k_final]

        # 4. Sort documents by their cross-encoder score (highest to lowest)
        doc_score_pairs = list(zip(docs, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        # 5. Return the top-k_final documents
        reranked_docs = [doc for doc, score in doc_score_pairs[:self.k_final]]
        return reranked_docs

def get_retriever(llm=None, k_initial: int = 10, k_final: int = 3):
    """Constructs a TwoStageRetriever. Cheap after the first call -- the expensive part
    (the two models) is shared process-wide, so this is effectively just a Chroma handle."""
    return TwoStageRetriever(llm=llm, k_initial=k_initial, k_final=k_final)


# Shared retriever singleton.
#
# Why it lives here rather than in agent.py: ingest.py needs the *same* vector store handle
# the query path uses, but importing it from agent.py created a circular import
# (agent -> retriever, ingest -> agent) that had to be worked around with a function-local
# import inside process_and_ingest. Owning the singleton in retriever.py -- the module that
# owns the vector store in the first place -- removes the cycle entirely, and lets both
# agent.py and ingest.py import it from one place.
#
# agent.get_retriever_instance() remains the public name api.py imports; agent.py is free to
# re-export this one.
_retriever_instance = None
_retriever_lock = threading.Lock()


def get_retriever_instance():
    """Returns the process-wide Stage-2 reranked retriever, initializing it on first call.

    Locked for the same reason the model loaders are: FastAPI dispatches sync endpoints on a
    threadpool, so concurrent first requests would otherwise race to build separate instances.
    """
    global _retriever_instance
    if _retriever_instance is None:
        with _retriever_lock:
            if _retriever_instance is None:
                _retriever_instance = get_retriever(k_initial=10, k_final=3)
    return _retriever_instance


if __name__ == "__main__":
    # Test the retriever locally.
    # Note the missing `llm` argument here used to raise TypeError immediately, because `llm`
    # was a required positional parameter -- this smoke test could never actually run. It is
    # optional now (the retriever never used it), so this works as intended.
    print("Initializing Two-Stage Retriever...")
    print(f"Using Chroma directory: {CHROMA_DB_DIR}")
    retriever = get_retriever(k_initial=5, k_final=2)

    test_query = "What were the financial results for Q1?"
    print(f"\nQuerying: '{test_query}'")

    results = retriever.invoke(test_query)

    if not results:
        print("\nNo results. Check that the collection above actually contains documents.")

    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"Content snippet: {doc.page_content[:200]}...")
