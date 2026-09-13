"""Sample review dataset for demos and tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

POSITIVE = [
    "This product is amazing and exceeded my expectations.",
    "Absolutely love it, would buy again.",
    "Great quality and fast shipping.",
    "Works perfectly, highly recommended.",
    "Fantastic experience from start to finish.",
    "Really happy with this purchase.",
    "Excellent customer support and solid build.",
    "Best one I have tried so far.",
    "I enjoy using it every day.",
    "Super comfortable and stylish.",
    "Outstanding value for the money.",
    "The design is beautiful and practical.",
    "Impressed by the performance and durability.",
    "Five stars, totally worth it.",
    "Smooth setup and delightful to use.",
    "Very reliable and easy to recommend.",
    "Loved the packaging and the product itself.",
    "Perfect gift, everyone liked it.",
    "Clean finish and strong materials.",
    "Surprisingly good, better than expected.",
    "I love this product so much.",
    "This product made me very happy.",
    "Wonderful item, great quality.",
    "I like this product a lot.",
    "Such a great product, totally satisfied.",
    "Happy with the product and the service.",
    "Brilliant product, works like a charm.",
    "I am so pleased with this purchase.",
    "Awesome product and fast delivery.",
    "Truly excellent product overall.",
    "I really loved this product.",
    "I love this product.",
    "Loved this product from day one.",
    "This product is wonderful.",
    "Great product, I am very satisfied.",
    "This product is excellent.",
    "I adore this product.",
    "Best product I bought this year.",
    "This product is fantastic.",
    "I am happy with this product.",
]

NEUTRAL = [
    "It is okay, nothing special.",
    "Average product for the price.",
    "Does what it says, but not impressive.",
    "Fine for occasional use.",
    "Neither good nor bad, just acceptable.",
    "Standard quality, mixed feelings.",
    "It arrived on time and looks normal.",
    "Mediocre overall, may keep it.",
    "Pretty decent, no strong opinion.",
    "Shipping was fine, product is average.",
    "It works, though I expected more.",
    "Usable, but not memorable.",
    "Ordinary experience overall.",
    "Meets basic needs and nothing more.",
    "Neutral about this purchase.",
    "Acceptable quality for short-term use.",
    "Not bad, not great either.",
    "Seems fine after a few days.",
    "Typical product in this category.",
    "I might keep it for now.",
    "This product is okay for now.",
    "The product is fine, nothing more.",
    "An average product with normal quality.",
    "This product is neither good nor bad.",
    "I feel neutral about this product.",
    "The product is acceptable.",
    "Just a regular product.",
    "This product is alright.",
    "The product seems ordinary.",
    "I have no strong feelings about this product.",
]

NEGATIVE = [
    "Terrible quality, broke after one day.",
    "Waste of money, very disappointed.",
    "Awful experience and rude support.",
    "Do not buy this, completely useless.",
    "Poor packaging and defective item.",
    "Horrible product, regret buying it.",
    "Stopped working immediately.",
    "Bad smell and cheap materials.",
    "Customer service ignored my emails.",
    "The worst purchase this year.",
    "Frustrating to use and unreliable.",
    "Low quality and overpriced.",
    "Arrived damaged and delayed.",
    "Totally disappointed with the result.",
    "Not worth the cost at all.",
    "Feels fragile and poorly made.",
    "I want a refund as soon as possible.",
    "Annoying issues from day one.",
    "Misleading description and bad fit.",
    "Would never recommend this to anyone.",
    "I hate this product.",
    "I hate this product so much.",
    "I really hate this item.",
    "I hate it and want my money back.",
    "This product is awful and I hate it.",
    "I dislike this product a lot.",
    "I do not like this product at all.",
    "This product is bad and disappointing.",
    "Terrible product, I hate using it.",
    "I hate everything about this product.",
    "This is a bad product.",
    "Very bad product, do not buy.",
    "I am angry about this product.",
    "This product ruined my day.",
    "Useless product, total garbage.",
    "I regret buying this product.",
    "Worst product I have ever owned.",
    "This product is trash.",
    "I cannot stand this product.",
    "Disgusting quality, hate it.",
]


def build_sample_frame() -> pd.DataFrame:
    rows = (
        [(text, "positive") for text in POSITIVE]
        + [(text, "neutral") for text in NEUTRAL]
        + [(text, "negative") for text in NEGATIVE]
    )
    return pd.DataFrame(rows, columns=["text", "label"])


def write_sample_csv(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = build_sample_frame()
    frame.to_csv(out, index=False)
    return out
