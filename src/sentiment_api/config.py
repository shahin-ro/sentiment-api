"""Shared paths and defaults."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = ROOT / "data" / "samples" / "reviews.csv"
DEFAULT_MODEL_PATH = ROOT / "artifacts" / "sentiment_pipeline.joblib"
LABELS = ("negative", "neutral", "positive")
