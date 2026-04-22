"""
logging_config.py
-----------------
HTTP access logging for the Streamlit / Tornado app.

Strategy
--------
Streamlit owns the Tornado IOLoop and aggressively configures Python's logging
system, which means any handler we attach to the standard ``logging`` hierarchy
can be silenced or swallowed before it reaches Railway's log collector.

This module therefore uses two complementary, Streamlit-proof mechanisms:

1. **stderr writes via sys.stderr.write()** – Railway captures everything
   written to file-descriptor 2 (stderr) regardless of the Python logging
   configuration.  We bypass the logging module entirely and write formatted
   JSON lines straight to stderr with flush=True.

2. **File mirror at /tmp/streamlit_access.log** – a persistent on-disk copy
   that survives across Streamlit reruns and can be tailed / inspected
   independently.

Both outputs are written by ``log_access()``, which you can call from any
Streamlit page or background thread.

The module also attempts to patch Tornado's ``RequestHandler.log_request``
so that real HTTP requests are captured automatically.  If Streamlit has
already replaced that method, the patch is a no-op and callers must invoke
``log_access()`` manually.

Log fields (JSON)
-----------------
timestamp   : ISO-8601 UTC time the request completed
method      : HTTP verb (GET, POST, …)
path        : Request URI / path
status      : HTTP response status code
duration_ms : Request duration in milliseconds (float, 2 dp)
client_ip   : Remote IP address of the caller
user_agent  : User-Agent header sent by the client

Usage
-----
Import once before any Streamlit rendering code runs:

    import logging_config  # noqa: F401  – activates HTTP access logging

To emit a manual log line from anywhere in the app:

    from logging_config import log_access
    log_access(method="GET", path="/", status=200, duration_ms=12.3,
               client_ip="1.2.3.4", user_agent="Mozilla/5.0 ...")
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Path for the on-disk access log mirror
# ---------------------------------------------------------------------------
_LOG_FILE = "/tmp/streamlit_access.log"

# ---------------------------------------------------------------------------
# Public helper – write one access-log record
# ---------------------------------------------------------------------------

def log_access(
    *,
    method: str = "-",
    path: str = "-",
    status: int = 0,
    duration_ms: float | None = None,
    client_ip: str = "-",
    user_agent: str = "",
) -> None:
    """Emit a single HTTP access-log record to stderr and /tmp/streamlit_access.log.

    Both outputs use flush=True so the line is visible in Railway logs
    immediately, without waiting for a buffer to fill.
    """
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "method": method,
        "path": path,
        "status": status,
        "duration_ms": duration_ms,
        "client_ip": client_ip,
        "user_agent": user_agent,
    }
    line = json.dumps(record) + "\n"

    # 1. Write directly to stderr – Railway captures fd-2 unconditionally.
    sys.stderr.write(line)
    sys.stderr.flush()

    # 2. Mirror to /tmp so the log survives across Streamlit reruns.
    try:
        with open(_LOG_FILE, "a", encoding="utf-8") as fh:
            fh.write(line)
    except OSError:
        pass  # /tmp not writable in some sandboxed environments – ignore


# ---------------------------------------------------------------------------
# Tornado patch – capture real HTTP requests automatically
# ---------------------------------------------------------------------------

def _make_log_request_handler():
    """Return a Tornado log_request replacement that calls log_access()."""

    def _log_request(handler) -> None:  # noqa: ANN001
        request = handler.request

        try:
            duration_ms = round(request.request_time() * 1000, 2)
        except Exception:
            duration_ms = None

        try:
            status = handler.get_status()
        except Exception:
            status = 0

        log_access(
            method=request.method or "-",
            path=request.uri or "-",
            status=status,
            duration_ms=duration_ms,
            client_ip=request.remote_ip or "-",
            user_agent=request.headers.get("User-Agent", ""),
        )

    return _log_request


def _patch_tornado() -> bool:
    """Attempt to patch Tornado's RequestHandler.log_request.

    Returns True if the patch was applied, False otherwise.
    """
    try:
        from tornado.web import RequestHandler  # type: ignore

        if getattr(RequestHandler, "_access_logging_patched", False):
            return True  # already patched by a previous import

        RequestHandler.log_request = _make_log_request_handler()  # type: ignore[method-assign]
        RequestHandler._access_logging_patched = True  # type: ignore[attr-defined]
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Module initialisation – runs once on first import
# ---------------------------------------------------------------------------

_tornado_patched = _patch_tornado()

# Emit a startup banner so Railway logs confirm the module loaded correctly.
_banner = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "event": "access_logging_initialised",
    "tornado_patched": _tornado_patched,
    "log_file": _LOG_FILE,
}
sys.stderr.write(json.dumps(_banner) + "\n")
sys.stderr.flush()
