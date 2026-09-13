"""Model training and persistence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sentiment_api.config import DEFAULT_MODEL_PATH
from sentiment_api.preprocess import clean_text


@dataclass(slots=True)
class TrainResult:
    model_path: Path
    n_rows: int
    accuracy: float
    macro_f1: float
    report: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["model_path"] = str(self.model_path)
        return payload


def load_dataset(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "text" not in frame.columns or "label" not in frame.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns.")
    frame = frame.dropna(subset=["text", "label"]).copy()
    frame["text"] = frame["text"].astype(str).map(clean_text)
    frame["label"] = frame["label"].astype(str).str.lower().str.strip()
    if frame.empty:
        raise ValueError("Dataset is empty after cleaning.")
    return frame


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=8000,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    C=2.0,
                    random_state=42,
                ),
            ),
        ]
    )


def train_model(
    data_path: str | Path,
    *,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
) -> TrainResult:
    frame = load_dataset(data_path)
    x_train, x_test, y_train, y_test = train_test_split(
        frame["text"],
        frame["label"],
        test_size=test_size,
        random_state=random_state,
        stratify=frame["label"],
    )

    # Evaluate on holdout, then refit on all rows for the served model.
    eval_pipeline = build_pipeline()
    eval_pipeline.fit(x_train, y_train)
    pred = eval_pipeline.predict(x_test)

    pipeline = build_pipeline()
    pipeline.fit(frame["text"], frame["label"])

    out = Path(model_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out)

    return TrainResult(
        model_path=out,
        n_rows=len(frame),
        accuracy=float(accuracy_score(y_test, pred)),
        macro_f1=float(f1_score(y_test, pred, average="macro")),
        report=classification_report(y_test, pred, output_dict=True),
    )


def load_model(model_path: str | Path = DEFAULT_MODEL_PATH) -> Pipeline:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Train one first with `sentiment-api train`."
        )
    return joblib.load(path)
