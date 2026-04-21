"""
logging_config.py
-----------------
HTTP access logging for the Streamlit app.

Streamlit is built on Tornado. This module monkey-patches
Tornado's RequestHandler.log_request so that every HTTP request
is emitted as a structured JSON log line to stderr, where Railway
captures and indexes it automatically.

Log fields
----------
timestamp   : ISO-8601 UTC time the request completed
method      : HTTP verb (GET, POST, …)
path        : Request URI / path
status      : HTTP response status code
duration_ms : Request duration in milliseconds
client_ip   : Remote IP address of the caller
user_agent  : User-Agent header sent by the client

Usage
-----
Import this module once, before any Streamlit rendering code runs:

    import logging_config  # noqa: F401  – activates HTTP access logging
"""

import json
import logging
import sys
import time
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Root access logger – writes directly to stderr so Railway captures it
# ---------------------------------------------------------------------------
_access_logger = logging.getLogger("http.access")

if not _access_logger.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _access_logger.addHandler(_handler)
    _access_logger.setLevel(logging.INFO)
    _access_logger.propagate = False  # prevent Streamlit's logging from interfering

print("[logging_config] HTTP access logging initialised – writing to stderr", flush=True)


def _log_request(handler) -> None:
    """Replacement for Tornado's RequestHandler.log_request.

    Called by Tornado at the end of every HTTP request with the handler
    instance, which exposes the full request object and response status.
    """
    request = handler.request

    # Duration: Tornado stores request_time() in seconds as a float.
    try:
        duration_ms = round(request.request_time() * 1000, 2)
    except Exception:
        duration_ms = None

    # Status code: use handler.get_status() which is always set.
    try:
        status = handler.get_status()
    except Exception:
        status = 0

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "method": request.method,
        "path": request.uri,
        "status": status,
        "duration_ms": duration_ms,
        "client_ip": request.remote_ip,
        "user_agent": request.headers.get("User-Agent", ""),
    }

    print(json.dumps(record), file=sys.stderr, flush=True)


def setup_access_logging() -> None:
    """Patch Tornado's RequestHandler to emit structured access logs.

    Safe to call multiple times – the patch is applied only once.
    """
    try:
        from tornado.web import RequestHandler  # type: ignore

        if getattr(RequestHandler, "_access_logging_patched", False):
            return  # already patched

        RequestHandler.log_request = _log_request  # type: ignore[method-assign]
        RequestHandler._access_logging_patched = True  # type: ignore[attr-defined]

        logging.getLogger(__name__).info(
            "HTTP access logging enabled (Tornado RequestHandler patched)"
        )
    except ImportError:
        logging.getLogger(__name__).warning(
            "tornado not found – HTTP access logging not enabled"
        )


# Apply the patch immediately on import.
setup_access_logging()
