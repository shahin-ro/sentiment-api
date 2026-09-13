# Sentiment API

Train a small text sentiment classifier and serve it behind a FastAPI endpoint.

Labels:

- `negative`
- `neutral`
- `positive`

The default model is TF-IDF + Logistic Regression. It is intentionally lightweight so you can train, test, and deploy quickly.

## Setup

```bash
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

pip install -r requirements-dev.txt
pip install -e .
```

## Quick start

```bash
# 1) Sample data
sentiment-api make-sample

# 2) Train
sentiment-api train

# 3) Local prediction
sentiment-api predict "I really loved this product"

# 4) API server
sentiment-api serve
```

Open docs at `http://127.0.0.1:8000/docs`.

### Example request

```bash
curl -X POST http://127.0.0.1:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"text\": \"This was a terrible experience\"}"
```

## API

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Service + model status |
| POST | `/predict` | Single text |
| POST | `/predict/batch` | List of texts |

## Layout

```text
src/sentiment_api/
  preprocess.py
  train.py
  predict.py
  api.py
  cli.py
data/samples/
artifacts/          # saved model (gitignored)
tests/
Dockerfile
```

## Train on your own CSV

CSV needs two columns: `text`, `label`.

```bash
sentiment-api train --data path\to\reviews.csv --model artifacts\sentiment_pipeline.joblib
```

## Docker

```bash
# train locally first so artifacts/sentiment_pipeline.joblib exists
docker compose up --build
```

## Tests

```bash
pytest -q
```

## Stack

Python 3.10+, scikit-learn, FastAPI, Uvicorn, Typer, pandas.

## License

MIT — see [LICENSE](LICENSE).
