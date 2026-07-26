"""
MongoDB (GridFS) storage for original uploaded PDF bytes and generated chart images.

Why this replaces local-disk storage:
vision_tool.py needs the original PDF bytes to re-render a specific page as an image, and
agent.py's plot_stock_chart tool generates a PNG chart that api.py needs to hand back to the
frontend in the same request. Saving either of these to local disk (the first version of this
project did both -- PDFs under backend/uploaded_docs/, charts as loose PNG files in the
working directory) works for a single always-on local process, but breaks in three realistic
situations:
  1. Ephemeral filesystems: Render's free/starter web services wipe local disk on every
     restart or redeploy.
  2. Multiple instances: if this backend ever scales beyond one process, each instance only
     has the files that happened to be written to it.
  3. Dev-loop noise: uvicorn's --reload file watcher treats every new PNG written to the
     backend/ folder as a source-code change and triggers a reload -- which is exactly what
     was happening every time a chart got generated during local testing.

MongoDB gives a persistent, shared store instead, and storing charts in memory -> Mongo
(rather than memory -> disk -> Mongo) means the app never writes them to disk at all. GridFS
specifically (not a plain document in a normal collection) is used because MongoDB documents
cap out at 16MB -- GridFS transparently chunks larger files. Most single-file earnings
reports and chart PNGs here are well under that limit, but GridFS costs nothing extra and
removes the ceiling entirely, so there's no reason to special-case large files later.

PDFs are stored/looked up by filename (matching how ChromaDB's metadata and db.py's
DocumentRecord already key on filename); re-uploading a file with the same name replaces the
previous GridFS copy. Charts are stored/looked up by a unique chart_id (already
uuid-suffixed by the caller), so no replace-on-write logic is needed there.
"""
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Optional

import gridfs

logger = logging.getLogger("MongoStorage")

MAX_PUT_RETRIES = 3
PUT_RETRY_BASE_DELAY_SECONDS = 2.0

_fs = None
_fs_lock = threading.Lock()


def _get_fs():
    """Lazily creates the GridFS handle on top of the shared MongoDB connection
    (mongo_client.py) on first use. Locked because FastAPI dispatches sync endpoints on a
    threadpool -- concurrent first uploads would otherwise each trigger their own
    mongo_client.get_db() connect."""
    global _fs
    if _fs is None:
        with _fs_lock:
            if _fs is None:
                from mongo_client import get_db
                _fs = gridfs.GridFS(get_db())
    return _fs


def _save_bytes(filename: str, file_bytes: bytes, content_type: str, replace: bool = True) -> str:
    """
    Shared GridFS put logic used by both PDF and chart storage below. Always either returns
    the new file's id or raises -- never None.

    Write-then-delete, not delete-then-write:
    The replace path used to delete every existing copy of `filename` *before* attempting the
    put. If all the put retries then failed (Mongo restarting, disk full, network drop
    mid-upload), the previous good PDF was already gone -- a failed re-upload destroyed the
    working document it was meant to replace, and vision_tool.py lost the page images for a
    file the app still listed as ingested. Writing first means a failure leaves the old copy
    untouched: worst case the re-upload is rejected and the user retries, best case the old
    copy is deleted a fraction of a second later. The only cost is a brief window where two
    copies coexist, which _load_bytes handles by always reading the newest.

    Structure:
    The retry loop returns on success and raises on the last attempt, so falling out of the
    bottom should be impossible -- but "should be impossible" flow that returns an implicit
    None is exactly what bites later (a MAX_PUT_RETRIES of 0 would have made save_pdf return
    None and every caller treat that as a valid file id). The loop below leaves the failure
    as an explicit raise after it, so there is no path off the end of this function that
    doesn't produce a value or an exception.
    """
    fs = _get_fs()

    # Snapshot what is already stored under this name, but do not touch it yet.
    superseded_ids = []
    if replace:
        try:
            superseded_ids = [existing._id for existing in fs.find({"filename": filename})]
        except Exception as e:
            # Not fatal: worst case we leave an older copy behind, and _load_bytes still
            # returns the newest one.
            logger.warning(f"Could not list existing GridFS copies of '{filename}': {e}")

    last_error = None
    for attempt in range(1, MAX_PUT_RETRIES + 1):
        try:
            file_id = fs.put(
                file_bytes,
                filename=filename,
                content_type=content_type,
                uploaded_at=datetime.now(timezone.utc),
            )
        except Exception as e:
            last_error = e
            if attempt < MAX_PUT_RETRIES:
                delay = PUT_RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
                logger.warning(
                    f"GridFS put failed for '{filename}' (attempt {attempt}/{MAX_PUT_RETRIES}): "
                    f"{e}. Retrying in {delay}s..."
                )
                time.sleep(delay)
            continue

        logger.info(f"Stored '{filename}' in GridFS ({len(file_bytes)} bytes, id={file_id})")

        # The new copy is durable now, so the old ones can go. A failure here is cosmetic --
        # it leaves an orphaned older revision that _load_bytes will never return -- so it
        # must not turn a successful upload into an error for the caller.
        for old_id in superseded_ids:
            try:
                fs.delete(old_id)
            except Exception as e:
                logger.warning(
                    f"Stored the new copy of '{filename}' but could not delete superseded "
                    f"copy {old_id}: {e}"
                )

        return str(file_id)

    if last_error is not None:
        logger.error(
            f"Failed to store '{filename}' in GridFS after {MAX_PUT_RETRIES} attempts: {last_error}"
        )
        if replace and superseded_ids:
            logger.error(
                f"The previously stored copy of '{filename}' was left in place and is still "
                f"readable -- the failed upload did not destroy it."
            )
        raise last_error

    # Only reachable if MAX_PUT_RETRIES were configured <= 0 so the loop never ran at all.
    raise RuntimeError(
        f"Refusing to report success for '{filename}': MAX_PUT_RETRIES is "
        f"{MAX_PUT_RETRIES}, so no GridFS write was ever attempted."
    )


def _load_bytes(filename: str) -> Optional[bytes]:
    """
    Returns the bytes of the most recently stored file under `filename`, or None.

    Sorting by uploadDate descending rather than using fs.find_one() matters because
    _save_bytes writes the replacement before deleting the copy it supersedes: for the brief
    window where both exist, an unsorted find_one can return whichever Mongo yields first --
    which is typically the *older* one. Explicitly taking the newest makes reads deterministic
    regardless of that window, and also does the right thing for any duplicate copies left
    behind by an earlier failed cleanup.
    """
    fs = _get_fs()
    for grid_out in fs.find({"filename": filename}).sort("uploadDate", -1).limit(1):
        return grid_out.read()
    return None


def save_pdf(filename: str, file_bytes: bytes) -> str:
    """
    Stores (or replaces) the original PDF bytes in GridFS under `filename`.
    Returns the GridFS file id (as a string) of the stored file.
    """
    return _save_bytes(filename, file_bytes, content_type="application/pdf", replace=True)


def load_pdf(filename: str) -> Optional[bytes]:
    """Retrieves the original PDF bytes for a given filename, or None if not found."""
    return _load_bytes(filename)


def pdf_exists(filename: str) -> bool:
    fs = _get_fs()
    return fs.find_one({"filename": filename}) is not None


def delete_pdf(filename: str) -> bool:
    """Deletes all GridFS copies stored under this filename. Returns True if anything was
    deleted."""
    fs = _get_fs()
    deleted_any = False
    for existing in fs.find({"filename": filename}):
        fs.delete(existing._id)
        deleted_any = True
    return deleted_any


def save_chart(chart_id: str, png_bytes: bytes) -> str:
    """
    Stores a generated matplotlib chart (PNG bytes, built entirely in memory by
    agent.py's plot_stock_chart tool -- never written to local disk) under a unique
    chart_id. chart_id already includes a uuid from the caller, so collisions aren't a
    real concern; replace=False skips an unnecessary find-and-delete pass on every chart.
    """
    return _save_bytes(f"chart_{chart_id}.png", png_bytes, content_type="image/png", replace=False)


def load_chart(chart_id: str) -> Optional[bytes]:
    """Retrieves a previously generated chart's PNG bytes, or None if not found."""
    return _load_bytes(f"chart_{chart_id}.png")
