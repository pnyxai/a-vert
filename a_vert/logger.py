"""A-VERT logging utilities.

We use structlog for structured, machine-parseable & human-friendly logs but
we do NOT modify the global/root logging configuration to avoid impacting
other libraries that also rely on structlog.

Each A-VERT logger gets its own processor chain. Setting AVERT_LOG_LEVEL only
affects A-VERT loggers and will not increase verbosity of third-party loggers.

Example usage:
    from a_vert.logger import get_logger
    logger = get_logger(__name__)
    logger.debug("Candidate ranking", group=group_name, distance=0.42)
"""

from __future__ import annotations

import logging
import os
import sys
import structlog

_LOGGERS: dict[str, structlog.stdlib.BoundLogger] = {}


def _build_processors():
    """Return processors for isolated A-VERT loggers.

    Using KeyValueRenderer keeps console output compact while preserving
    structure. We include a timestamp & level for operational clarity.
    """
    # We omit timestamp & level here because the stdlib handler's formatter
    # will render them in the prefix. Keeping them out of the key=value body
    # avoids duplication.
    return [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.KeyValueRenderer(sort_keys=False, key_order=["event"]),
    ]


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get (or create) an isolated structlog logger for A-VERT.

    Honors AVERT_LOG_LEVEL without touching the root logger. A StreamHandler
    to stdout is attached ONLY to this named logger if no handlers exist.
    """
    if name in _LOGGERS:
        return _LOGGERS[name]

    level_name = os.getenv("AVERT_LOG_LEVEL", "WARNING").upper()
    level = getattr(logging, level_name, logging.WARNING)

    base_logger = logging.getLogger(name)
    base_logger.setLevel(level)
    # Prevent log records from bubbling up to root and being re-emitted (avoids duplicates)
    base_logger.propagate = False
    if not base_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        # Prefix style similar to previous behavior:
        # 2025-11-29 23:57:19 DEBUG    [a_vert.processing:223] message key=value...
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)-8s [%(name)s:%(lineno)d] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        base_logger.addHandler(handler)

    bound = structlog.wrap_logger(
        base_logger,
        processors=_build_processors(),
        context_class=dict,
        wrapper_class=structlog.stdlib.BoundLogger,
    )
    _LOGGERS[name] = bound
    return bound
