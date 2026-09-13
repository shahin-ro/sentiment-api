"""Lightweight text cleaning helpers."""

from __future__ import annotations

import re

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Normalize social/review-style text for bag-of-words models."""
    value = str(text).lower()
    value = _URL_RE.sub(" ", value)
    value = _MENTION_RE.sub(" ", value)
    value = _HASHTAG_RE.sub(r"\1", value)
    value = re.sub(r"[^a-z0-9'\s]", " ", value)
    value = _SPACE_RE.sub(" ", value).strip()
    return value
