# Backend service: FastAPI + LangGraph agent.
#
# Multi-stage on purpose. The dependency set here (torch via sentence-transformers, chromadb,
# PyMuPDF) needs a C toolchain to build a couple of wheels, but nothing in that toolchain is
# needed once the wheels are installed. Building into a virtualenv in stage one and copying
# only that venv into a clean runtime image keeps ~300MB of build-essential out of the shipped
# image without giving up the ability to compile a wheel that has no prebuilt manylinux build.
#
# Python 3.11 (not 3.10) because that is what this project is developed and tested against
# locally, and several pinned floors in requirements.txt (langchain 1.x, chromadb 1.5.x) ship
# wheels that assume a modern interpreter. Matching dev and prod here removes an entire class
# of "works locally, fails in the container" bug reports.

# --------------------------------------------------------------------------------------
# Stage 1: build the virtualenv
# --------------------------------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copied and installed before the application source so that editing a .py file doesn't
# invalidate the (very expensive) dependency layer.
COPY requirements.txt .
RUN pip install -r requirements.txt

# --------------------------------------------------------------------------------------
# Stage 2: runtime
# --------------------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# libgomp1 is a runtime shared-library dependency of torch/onnxruntime (used by
# sentence-transformers and chromadb). It is NOT pulled in by the slim base image, and its
# absence shows up as an opaque "libgomp.so.1: cannot open shared object file" ImportError
# the first time the embedding model loads -- i.e. on the first real query, not at startup,
# which makes it needlessly hard to diagnose. Install it explicitly.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# PYTHONUNBUFFERED is not cosmetic here: without it, Python block-buffers stdout when it is a
# pipe (which it always is under Docker), so `docker logs` shows nothing at all until the
# buffer flushes. Every "the container is running but produces no output" report traces back
# to this. PYTHONDONTWRITEBYTECODE keeps .pyc files out of the read-only-ish app layer.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# Pre-download the embedding + reranker models at BUILD time rather than on first request.
# The two-stage retriever needs both BAAI/bge-small-en-v1.5 (bi-encoder) and
# cross-encoder/ms-marco-MiniLM-L-6-v2 (reranker); downloading them lazily costs ~45s on the
# first user query and, worse, makes the app hard-depend on outbound HuggingFace access at
# runtime. Baking them into the image makes the first query fast and lets the container run
# in a network-restricted environment.
RUN python -c "\
from langchain_huggingface import HuggingFaceEmbeddings; \
HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5', model_kwargs={'device':'cpu'}); \
from langchain_community.cross_encoders import HuggingFaceCrossEncoder; \
HuggingFaceCrossEncoder(model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'); \
print('models cached')"

COPY . .

# Run as an unprivileged user. Ownership of /app is needed because chroma_db/ is written at
# runtime and HF_HOME lives under /app/.cache.
RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

# Only chroma_db is a real runtime-persistent path now. The old uploaded_docs/ volume is gone:
# original PDFs and generated charts live in MongoDB GridFS (see mongo_storage.py), which is
# the whole point of that module -- declaring a volume for a directory the app never writes to
# just creates a confusing anonymous volume on every run.
VOLUME ["/app/chroma_db"]

EXPOSE 8000

# Hits the dependency-aware /health endpoint rather than just checking that the port is open,
# so an instance that is listening but has lost MongoDB is correctly reported as unhealthy.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
