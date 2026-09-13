"""Inference helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.pipeline import Pipeline

from sentiment_api.config import DEFAULT_MODEL_PATH
from sentiment_api.preprocess import clean_text
from sentiment_api.train import load_model


@dataclass(slots=True)
class Prediction:
    text: str
    cleaned_text: str
    label: str
    confidence: float
    probabilities: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def predict_text(text: str, model: Pipeline) -> Prediction:
    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("Text is empty after cleaning.")

    proba = model.predict_proba([cleaned])[0]
    classes = list(model.classes_)
    best_idx = int(np.argmax(proba))
    probabilities = {
        str(label): float(score) for label, score in zip(classes, proba, strict=True)
    }
    return Prediction(
        text=text,
        cleaned_text=cleaned,
        label=str(classes[best_idx]),
        confidence=float(proba[best_idx]),
        probabilities=probabilities,
    )


def predict_batch(texts: list[str], model: Pipeline) -> list[Prediction]:
    return [predict_text(text, model) for text in texts]


def get_or_load_model(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    *,
    cache: dict[str, Pipeline] | None = None,
) -> Pipeline:
    key = str(Path(model_path))
    store = cache if cache is not None else {}
    if key not in store:
        store[key] = load_model(model_path)
    return store[key]
