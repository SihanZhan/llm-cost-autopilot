"""Load the trained classifier and score new prompts.

Usage:
    python -m classifier.predict "Summarize this in one sentence: ..."
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib

from classifier.features import feature_vector
from classifier.tiers import tier_name

MODEL_FILE = Path(__file__).parent / "model.joblib"


@lru_cache(maxsize=1)
def _load() -> dict:
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"{MODEL_FILE.name} not found — run `python -m classifier.train` first"
        )
    return joblib.load(MODEL_FILE)


def predict_tier(prompt: str) -> int:
    """Return the predicted complexity tier (1, 2, or 3) for ``prompt``."""
    bundle = _load()
    features = [feature_vector(prompt)]
    return int(bundle["model"].predict(features)[0])


if __name__ == "__main__":
    import sys

    sample = " ".join(sys.argv[1:]) or (
        "Summarize this in one sentence: remote work adoption has plateaued "
        "since 2024, with most large employers settling on a hybrid policy."
    )
    tier = predict_tier(sample)
    print(f"tier {tier} ({tier_name(tier)})  <-  {sample}")
