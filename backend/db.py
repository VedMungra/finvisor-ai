"""
Document metadata, user feedback, user accounts, portfolios, and chat history (MongoDB collections).

Why MongoDB instead of a separate SQL database:
This originally used SQLAlchemy (SQLite locally, Postgres in production) for this data,
alongside MongoDB GridFS for the actual file bytes (mongo_storage.py) -- two different
database technologies for two different kinds of data in the same app. That split has real
costs: two connection configs, two sets of drivers, two things that can be misconfigured
(the immediate trigger for this rewrite was exactly that -- a blank DATABASE_URL crashing
SQLAlchemy's URL parser). Consolidating both onto MongoDB removes the split entirely: one
connection, one technology, one .env variable (MONGODB_URI) to get right.

Document metadata (filename, chunk count, upload time) and feedback (question/answer/rating)
are still naturally row-shaped, but nothing about them actually requires SQL -- a Mongo
collection with a unique index on `filename` gives the same "one row per document, upsert on
re-ingest" guarantee a SQL unique constraint would.

Uses the shared MongoDB connection from mongo_client.py -- the same database mongo_storage.py
uses for GridFS file storage -- rather than opening a second connection.
"""
import datetime
import logging

from bson import ObjectId
from pymongo.errors import DuplicateKeyError, OperationFailure

logger = logging.getLogger("DB")

DOCUMENTS_COLLECTION = "documents"
FEEDBACK_COLLECTION = "feedback"
USERS_COLLECTION = "users"
PORTFOLIOS_COLLECTION = "portfolios"
CHAT_HISTORY_COLLECTION = "chat_history"


def _get_db():
    from mongo_client import get_db
    return get_db()


def init_db():
    """
    Ensures required indexes exist. Safe to call on every startup: create_index is
    idempotent (Mongo no-ops if an equivalent index already exists), unlike a raw CREATE
    TABLE which would need an IF NOT EXISTS guard.

    "Idempotent" is not the same as "cannot fail", though. Building a *unique* index is a
    validation of the data already in the collection: if the collection ever accumulated two
    rows with the same `filename` -- entirely possible for rows written before this index
    existed, or by an older build of the app -- Mongo refuses to create it and raises. This
    function runs from api.py's @app.on_event("startup") handler, where an unhandled
    exception takes down the whole API. Refusing to start the server over a duplicate row in
    a metadata collection is wildly out of proportion to the problem: every endpoint that
    matters (chat, ingest, retrieval) works fine without this index, and the duplicate is
    something a human needs to look at anyway. So the failure is caught, reported loudly
    enough to act on, and startup continues.
    """
    db = _get_db()

    try:
        db[DOCUMENTS_COLLECTION].create_index("filename", unique=True)
    except (DuplicateKeyError, OperationFailure) as e:
        duplicates = _find_duplicate_filenames(db)
        logger.error(
            f"Could not create the unique index on '{DOCUMENTS_COLLECTION}.filename': {e}. "
            f"This usually means the collection already contains rows sharing a filename"
            + (f" (duplicated: {', '.join(duplicates)})" if duplicates else "")
            + ". Startup continues without the index -- record_document() still upserts on "
              "filename, so no new duplicates are created, but the existing ones should be "
              "cleaned up manually before the index can be built."
        )

    try:
        db[FEEDBACK_COLLECTION].create_index("thread_id")
    except OperationFailure as e:
        # Non-unique index, so this can't fail on data content -- but it can fail on an
        # options conflict with an index of the same name. Again: a missing secondary index
        # is a performance issue, not a reason to refuse to serve requests.
        logger.error(
            f"Could not create the index on '{FEEDBACK_COLLECTION}.thread_id': {e}. "
            f"Startup continues; feedback lookups by thread will be slower."
        )

    # --- Auth & profile indexes ---
    try:
        db[USERS_COLLECTION].create_index("email", unique=True)
    except (DuplicateKeyError, OperationFailure) as e:
        logger.error(
            f"Could not create the unique index on '{USERS_COLLECTION}.email': {e}. "
            f"Startup continues; duplicate emails may cause registration errors."
        )

    try:
        db[PORTFOLIOS_COLLECTION].create_index("user_id", unique=True)
    except OperationFailure as e:
        logger.error(
            f"Could not create the index on '{PORTFOLIOS_COLLECTION}.user_id': {e}. "
            f"Startup continues."
        )

    try:
        db[CHAT_HISTORY_COLLECTION].create_index([("user_id", 1), ("context_key", 1)])
    except OperationFailure as e:
        logger.error(
            f"Could not create the index on '{CHAT_HISTORY_COLLECTION}': {e}. "
            f"Startup continues; chat history queries will be slower."
        )

    logger.info(
        f"MongoDB collections ready ('{DOCUMENTS_COLLECTION}', '{FEEDBACK_COLLECTION}', "
        f"'{USERS_COLLECTION}', '{PORTFOLIOS_COLLECTION}', '{CHAT_HISTORY_COLLECTION}')"
    )


def _find_duplicate_filenames(db, limit: int = 10):
    """Best-effort lookup of which filenames are duplicated, purely to make the log message
    above actionable. Returns [] if the aggregation itself fails -- a diagnostic must never
    become a second failure."""
    try:
        pipeline = [
            {"$group": {"_id": "$filename", "count": {"$sum": 1}}},
            {"$match": {"count": {"$gt": 1}}},
            {"$limit": limit},
        ]
        return [str(row["_id"]) for row in db[DOCUMENTS_COLLECTION].aggregate(pipeline)]
    except Exception:
        return []


def record_document(filename: str, content_type: str, chunk_count: int, has_original_file: bool):
    """Upserts a document's metadata row after successful ingestion. Re-ingesting a file with
    the same filename replaces its metadata rather than creating a duplicate entry, matching
    the unique index on `filename`."""
    db = _get_db()
    db[DOCUMENTS_COLLECTION].update_one(
        {"filename": filename},
        {
            "$set": {
                "filename": filename,
                "content_type": content_type,
                "chunk_count": chunk_count,
                "has_original_file": has_original_file,
                "uploaded_at": datetime.datetime.now(datetime.timezone.utc),
            }
        },
        upsert=True,
    )


def delete_document_metadata(filename: str) -> bool:
    """Deletes a document's metadata row. Returns True if a document was deleted."""
    db = _get_db()
    result = db[DOCUMENTS_COLLECTION].delete_one({"filename": filename})
    return result.deleted_count > 0


def record_feedback(thread_id: str, source_filename: str, question: str, answer: str,
                     rating: str, comment: str = None, user_id: str = None):
    db = _get_db()
    doc = {
        "thread_id": thread_id,
        "source_filename": source_filename,
        "question": question,
        "answer": answer,
        "rating": rating,
        "comment": comment,
        "created_at": datetime.datetime.now(datetime.timezone.utc),
    }
    if user_id:
        doc["user_id"] = user_id
    db[FEEDBACK_COLLECTION].insert_one(doc)


def get_all_documents():
    """
    Returns all document metadata rows, newest first, as plain dicts ready for a FastAPI
    JSON response. The {"_id": 0} projection drops Mongo's ObjectId field, which isn't
    JSON-serializable by default -- api.py's /documents/stats endpoint doesn't need it since
    `filename` is already the natural unique key for this collection.
    """
    db = _get_db()
    return list(db[DOCUMENTS_COLLECTION].find({}, {"_id": 0}).sort("uploaded_at", -1))


# ---------------------------------------------------------------------------
# User management
# ---------------------------------------------------------------------------

def create_user(username: str, email: str, hashed_password: str) -> dict:
    """Creates a new user document and returns it (with _id set by Mongo)."""
    db = _get_db()
    doc = {
        "username": username,
        "email": email,
        "hashed_password": hashed_password,
        "created_at": datetime.datetime.now(datetime.timezone.utc),
    }
    result = db[USERS_COLLECTION].insert_one(doc)
    doc["_id"] = result.inserted_id
    return doc


def get_user_by_email(email: str):
    """Returns the user document for the given email, or None."""
    db = _get_db()
    return db[USERS_COLLECTION].find_one({"email": email})


def get_user_by_id(user_id: str):
    """Returns the user document for the given ObjectId string, or None."""
    db = _get_db()
    try:
        return db[USERS_COLLECTION].find_one({"_id": ObjectId(user_id)})
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Portfolio (ticker watchlist)
# ---------------------------------------------------------------------------

def get_portfolio(user_id: str) -> list:
    """Returns the user's list of watched ticker symbols, or an empty list."""
    db = _get_db()
    doc = db[PORTFOLIOS_COLLECTION].find_one({"user_id": user_id})
    if doc:
        return doc.get("tickers", [])
    return []


def add_to_portfolio(user_id: str, ticker: str) -> list:
    """Adds a ticker to the user's watchlist (idempotent). Returns the updated list."""
    db = _get_db()
    ticker = ticker.strip().upper()
    db[PORTFOLIOS_COLLECTION].update_one(
        {"user_id": user_id},
        {"$addToSet": {"tickers": ticker}},
        upsert=True,
    )
    return get_portfolio(user_id)


def remove_from_portfolio(user_id: str, ticker: str) -> list:
    """Removes a ticker from the user's watchlist. Returns the updated list."""
    db = _get_db()
    ticker = ticker.strip().upper()
    db[PORTFOLIOS_COLLECTION].update_one(
        {"user_id": user_id},
        {"$pull": {"tickers": ticker}},
    )
    return get_portfolio(user_id)


# ---------------------------------------------------------------------------
# Chat history persistence
# ---------------------------------------------------------------------------

def save_chat_message(user_id: str, thread_id: str, context_key: str,
                      role: str, content: str, extra: dict = None):
    """Persists a single chat message (user or agent) to MongoDB."""
    db = _get_db()
    doc = {
        "user_id": user_id,
        "thread_id": thread_id,
        "context_key": context_key,
        "role": role,
        "content": content,
        "created_at": datetime.datetime.now(datetime.timezone.utc),
    }
    if extra:
        doc["extra"] = extra
    db[CHAT_HISTORY_COLLECTION].insert_one(doc)


def get_chat_history(user_id: str, context_key: str) -> list:
    """Returns all messages for a user's conversation in a specific context, oldest first."""
    db = _get_db()
    cursor = db[CHAT_HISTORY_COLLECTION].find(
        {"user_id": user_id, "context_key": context_key},
        {"_id": 0, "user_id": 0},
    ).sort("created_at", 1)
    return list(cursor)


def get_user_threads(user_id: str) -> list:
    """Returns a list of distinct conversation contexts for a user, with the latest message
    timestamp and thread_id for each."""
    db = _get_db()
    pipeline = [
        {"$match": {"user_id": user_id}},
        {"$sort": {"created_at": -1}},
        {"$group": {
            "_id": "$context_key",
            "thread_id": {"$first": "$thread_id"},
            "last_message_at": {"$first": "$created_at"},
            "message_count": {"$sum": 1},
        }},
        {"$sort": {"last_message_at": -1}},
    ]
    results = []
    for row in db[CHAT_HISTORY_COLLECTION].aggregate(pipeline):
        results.append({
            "context_key": row["_id"],
            "thread_id": row["thread_id"],
            "last_message_at": row["last_message_at"],
            "message_count": row["message_count"],
        })
    return results


def delete_chat_history(user_id: str, context_key: str) -> int:
    """Deletes all messages for a user's conversation in a specific context.
    Returns the number of deleted messages."""
    db = _get_db()
    result = db[CHAT_HISTORY_COLLECTION].delete_many(
        {"user_id": user_id, "context_key": context_key}
    )
    return result.deleted_count
