from __future__ import annotations

from pathlib import Path

from sentiment_api.datasets import write_sample_csv


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "data" / "samples" / "reviews.csv"
    path = write_sample_csv(out)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
