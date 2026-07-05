"""Dynamic time context so the model can resolve relative dates correctly."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)


def build_time_system_message(tz_name: str) -> str:
    try:
        tz = ZoneInfo(tz_name)
        label = tz_name
    except Exception:
        logger.warning("Unknown timezone %r; falling back to UTC", tz_name)
        tz = UTC
        label = "UTC"

    now = datetime.now(tz)
    return (
        "Dynamic time context:\n"
        f"- Current local time: {now.strftime('%A, %B %d, %Y %H:%M:%S')} ({label})\n"
        '- Interpret relative dates like "today/tomorrow/this week" using this timestamp.'
    )
