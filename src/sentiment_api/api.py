"""FastAPI application for sentiment prediction."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from sentiment_api import __version__
from sentiment_api.config import DEFAULT_MODEL_PATH
from sentiment_api.predict import predict_batch, predict_text
from sentiment_api.train import load_model

app = FastAPI(
    title="Sentiment API",
    description="Classify short English texts as negative, neutral, or positive.",
    version=__version__,
)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, examples=["I really loved this product."])


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)


class PredictResponse(BaseModel):
    text: str
    cleaned_text: str
    label: str
    confidence: float
    probabilities: dict[str, float]


@lru_cache(maxsize=1)
def _model():
    return load_model(DEFAULT_MODEL_PATH)


@app.get("/health")
def health() -> dict[str, str]:
    model_exists = Path(DEFAULT_MODEL_PATH).exists()
    return {
        "status": "ok" if model_exists else "model_missing",
        "version": __version__,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    if not Path(DEFAULT_MODEL_PATH).exists():
        raise HTTPException(status_code=503, detail="Model not trained yet.")
    try:
        result = predict_text(payload.text, _model())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PredictResponse(**result.to_dict())


@app.post("/predict/batch", response_model=list[PredictResponse])
def predict_many(payload: BatchPredictRequest) -> list[PredictResponse]:
    if not Path(DEFAULT_MODEL_PATH).exists():
        raise HTTPException(status_code=503, detail="Model not trained yet.")
    try:
        results = predict_batch(payload.texts, _model())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return [PredictResponse(**item.to_dict()) for item in results]
