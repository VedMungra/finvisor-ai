"""
Shared MongoDB connection.

Why this exists:
Both db.py (document metadata + feedback collections) and mongo_storage.py (GridFS file
storage) need a MongoDB connection to the same database. A single shared, lazily-created
client/database handle avoids opening two independent connections for the same database.

Retry-with-backoff on connect:
On flaky networks (campus/hostel Wi-Fi doing DNS interception or TLS inspection is a common
culprit), the very first connection attempt to Atlas can fail with a transient DNS timeout or
SSL handshake error even though the network recovers a few seconds later. Without retries,
that one bad attempt crashes the whole FastAPI startup (db.init_db() runs in an @app.on_event
("startup") handler, so an unhandled exception there takes the whole app down) -- forcing a
manual Ctrl+C and restart every time the network blips. Retrying a few times with a short
delay before giving up handles the common transient case automatically instead of making that
the user's job.

The backoff is capped rather than purely exponential. Uncapped doubling from a 3 second base
spends 3 + 6 + 12 = 21 seconds sleeping before giving up, on top of four 8-second server
selection timeouts -- a request or a startup can sit there for the better part of a minute
with nothing in the log explaining the delay. Capping each delay at RETRY_DELAY_CAP_SECONDS
bounds the worst case, and _worst_case_wait_seconds() below turns that bound into a number
that is logged up front, so anyone watching the terminal knows how long the retry loop can
possibly take before it fails.

Thread safety:
FastAPI runs sync endpoint functions in a threadpool, so get_db() can be called concurrently
by several requests before the connection exists. Without a lock, each of those threads sees
`_db is None` and builds its own MongoClient -- several connection pools to the same server,
all but one of them orphaned (and never closed) once the last writer wins the `_client`
global. The lock makes the first caller connect while the rest wait and reuse the result.
"""
import os
import time
import threading
import logging
from pymongo import MongoClient
from pymongo.errors import PyMongoError

logger = logging.getLogger("MongoClient")

MONGODB_URI = os.getenv("MONGODB_URI") or "mongodb://localhost:27017"
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME") or "finvisor"

MAX_CONNECT_RETRIES = 4
RETRY_DELAY_SECONDS = 3
# Ceiling on any single backoff delay, so the total is bounded and predictable.
RETRY_DELAY_CAP_SECONDS = 6
SERVER_SELECTION_TIMEOUT_MS = 8000

_client = None
_db = None
_connect_lock = threading.Lock()


def _retry_delay(attempt: int) -> float:
    """Exponential backoff for `attempt` (1-based), capped at RETRY_DELAY_CAP_SECONDS."""
    return min(RETRY_DELAY_SECONDS * (2 ** (attempt - 1)), RETRY_DELAY_CAP_SECONDS)


def _worst_case_wait_seconds() -> float:
    """Total time the retry loop can burn before raising: every server selection timeout plus
    every backoff sleep. Logged on the first failure so the delay is never a mystery."""
    timeouts = MAX_CONNECT_RETRIES * (SERVER_SELECTION_TIMEOUT_MS / 1000.0)
    sleeps = sum(_retry_delay(a) for a in range(1, MAX_CONNECT_RETRIES))
    return timeouts + sleeps


def _safe_uri(uri: str) -> str:
    """Strips any user:password@ prefix so a connection string never reaches the logs."""
    return uri.split("@")[-1] if "@" in uri else uri


def get_db():
    """Returns the shared MongoDB Database handle, connecting lazily on first use, with
    retry-with-backoff on transient connection failures (see module docstring)."""
    global _client, _db

    # Fast path: once connected, every subsequent call is a plain global read with no lock
    # contention. Assignment to `_db` below happens only after the connection is verified, so
    # a non-None `_db` seen here is always a fully usable handle.
    if _db is not None:
        return _db

    with _connect_lock:
        # Re-check inside the lock: another thread may have connected while this one waited.
        if _db is not None:
            return _db

        # Resolved at call time rather than import time so that a load_dotenv() running after
        # this module is imported is still honoured -- otherwise a perfectly correct .env can
        # be ignored purely because of module import order, and the app quietly talks to
        # localhost instead of the configured server.
        uri = os.getenv("MONGODB_URI") or MONGODB_URI
        db_name = os.getenv("MONGODB_DB_NAME") or MONGODB_DB_NAME

        last_error = None
        for attempt in range(1, MAX_CONNECT_RETRIES + 1):
            client = None
            try:
                client = MongoClient(uri, serverSelectionTimeoutMS=SERVER_SELECTION_TIMEOUT_MS)
                # Force a round trip now (a lightweight admin command) so connection problems
                # surface here -- where we can retry -- rather than on whatever query happens
                # to run first later.
                client.admin.command("ping")

                _client = client
                _db = _client[db_name]
                logger.info(f"Connected to MongoDB database '{db_name}' at {_safe_uri(uri)}")
                return _db

            except PyMongoError as e:
                last_error = e
                # Close the half-built client instead of leaking its connection pool and
                # background monitor threads on every failed attempt.
                if client is not None:
                    try:
                        client.close()
                    except Exception:
                        pass

                if attempt == 1:
                    logger.warning(
                        f"MongoDB at {_safe_uri(uri)} did not respond. Retrying up to "
                        f"{MAX_CONNECT_RETRIES} times; worst case this takes "
                        f"~{_worst_case_wait_seconds():.0f}s before failing."
                    )

                if attempt < MAX_CONNECT_RETRIES:
                    delay = _retry_delay(attempt)
                    logger.warning(
                        f"MongoDB connection attempt {attempt}/{MAX_CONNECT_RETRIES} failed: "
                        f"{e}. Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.warning(
                        f"MongoDB connection attempt {attempt}/{MAX_CONNECT_RETRIES} failed: {e}."
                    )

        logger.error(
            f"MongoDB connection to {_safe_uri(uri)} failed after {MAX_CONNECT_RETRIES} attempts."
        )
        if last_error is not None:
            raise last_error
        # Defensive: only reachable if MAX_CONNECT_RETRIES were ever configured <= 0, in which
        # case the loop never runs and there is no underlying error to re-raise.
        raise RuntimeError(
            f"Failed to connect to MongoDB (MAX_CONNECT_RETRIES={MAX_CONNECT_RETRIES})"
        )
