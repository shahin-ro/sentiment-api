from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from sentiment_api.api import app
from sentiment_api.datasets import write_sample_csv
from sentiment_api.predict import predict_text
from sentiment_api.preprocess import clean_text
from sentiment_api.train import train_model


def test_clean_text_strips_urls_and_mentions() -> None:
    cleaned = clean_text("Wow!!! Visit https://example.com @shop #Sale")
    assert "https" not in cleaned
    assert "@shop" not in cleaned
    assert "sale" in cleaned


def test_train_and_predict(tmp_path: Path) -> None:
    data = write_sample_csv(tmp_path / "reviews.csv")
    model_path = tmp_path / "model.joblib"
    result = train_model(data, model_path=model_path, test_size=0.3)
    assert model_path.exists()
    assert 0.0 <= result.accuracy <= 1.0

    from sentiment_api.train import load_model

    model = load_model(model_path)
    pred = predict_text("This is amazing and I love it", model)
    assert pred.label in {"negative", "neutral", "positive"}
    assert 0.0 <= pred.confidence <= 1.0


def test_api_predict(tmp_path: Path, monkeypatch) -> None:
    data = write_sample_csv(tmp_path / "reviews.csv")
    model_path = tmp_path / "model.joblib"
    train_model(data, model_path=model_path, test_size=0.3)

    monkeypatch.setattr("sentiment_api.api.DEFAULT_MODEL_PATH", model_path)
    monkeypatch.setattr("sentiment_api.config.DEFAULT_MODEL_PATH", model_path)
    # Clear cached model loader
    from sentiment_api import api as api_module

    api_module._model.cache_clear()

    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200

    response = client.post("/predict", json={"text": "Absolutely fantastic product"})
    assert response.status_code == 200
    body = response.json()
    assert body["label"] in {"negative", "neutral", "positive"}
    assert "probabilities" in body
