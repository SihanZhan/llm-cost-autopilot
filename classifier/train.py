"""Phase 2: train the complexity classifier on the hand-labeled prompt set.

Loads classifier/data/labeled_prompts.jsonl, extracts features with
classifier.features.feature_vector, trains a logistic regression and a
random forest on an 80/20 stratified split, prints accuracy + a confusion
matrix for both, and saves whichever scores higher to classifier/model.joblib.

Usage:
    python -m classifier.train
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from classifier.features import FEATURE_NAMES, feature_vector
from classifier.tiers import TIER_NAMES, TIERS

ROOT = Path(__file__).parent
DATA_FILE = ROOT / "data" / "labeled_prompts.jsonl"
FEEDBACK_FILE = ROOT / "data" / "failure_feedback.jsonl"
MODEL_FILE = ROOT / "model.joblib"

TARGET_ACCURACY = 0.80


def load_dataset() -> tuple[list[str], list[int]]:
    """Load the hand-labeled set, plus any accumulated routing-failure
    feedback (see classifier.feedback) if that file exists yet."""
    prompts, tiers = [], []
    for path in (DATA_FILE, FEEDBACK_FILE):
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                prompts.append(row["prompt"])
                tiers.append(row["tier"])
    return prompts, tiers


def build_matrix(prompts: list[str]) -> np.ndarray:
    return np.array([feature_vector(p) for p in prompts])


def evaluate(name: str, model, X_test: np.ndarray, y_test: np.ndarray) -> float:
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\n{name} — held-out accuracy: {acc:.1%}")
    print(
        classification_report(
            y_test, preds,
            labels=list(TIERS),
            target_names=[TIER_NAMES[t] for t in TIERS],
            zero_division=0,
        )
    )
    print(f"confusion matrix (rows=true, cols=predicted), tier order {TIERS}")
    print(confusion_matrix(y_test, preds, labels=list(TIERS)))
    return acc


def main() -> int:
    prompts, tiers = load_dataset()
    feedback_note = f" (incl. {FEEDBACK_FILE.name})" if FEEDBACK_FILE.exists() else ""
    print(f"{len(prompts)} labeled prompts loaded{feedback_note}")

    X = build_matrix(prompts)
    y = np.array(tiers)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"train={len(X_train)}  held-out={len(X_test)}")

    log_reg = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000)),
    ])
    log_reg.fit(X_train, y_train)
    acc_lr = evaluate("Logistic regression", log_reg, X_test, y_test)

    rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    rf.fit(X_train, y_train)
    acc_rf = evaluate("Random forest", rf, X_test, y_test)

    if acc_lr >= acc_rf:
        best_name, best_model, best_acc = "logistic_regression", log_reg, acc_lr
    else:
        best_name, best_model, best_acc = "random_forest", rf, acc_rf

    print(f"\nSaving best model ({best_name}, {best_acc:.1%} held-out accuracy) -> {MODEL_FILE.name}")
    joblib.dump(
        {
            "model": best_model,
            "model_name": best_name,
            "feature_names": FEATURE_NAMES,
            "held_out_accuracy": best_acc,
        },
        MODEL_FILE,
    )

    if best_acc < TARGET_ACCURACY:
        print(f"\nWARNING: held-out accuracy {best_acc:.1%} is below the {TARGET_ACCURACY:.0%} target.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
