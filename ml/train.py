
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "synthetic" / "training_data.csv"
MODEL_DIR = ROOT / "ml" / "artifacts"
MODEL_FILE = MODEL_DIR / "glucose_spike_model.joblib"

FEATURES = [
    "age",
    "bmi",
    "hba1c",
    "current_glucose",
    "glucose_change_30m",
    "heart_rate",
    "sleep_hours",
    "steps",
]

TARGET = "glucose_spike_within_2h"


def train():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Training data missing. Run ml/generate_data.py first."
        )

    df = pd.read_csv(DATA_FILE)

    X = df[FEATURES]
    y = df[TARGET]

    if y.nunique() < 2:
        raise ValueError(
            "Training data must contain both target classes."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("Evaluation on synthetic holdout data")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")

    if y_test.nunique() == 2:
        print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.3f}")

    print(classification_report(
        y_test,
        predictions,
        zero_division=0,
    ))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {"model": model, "features": FEATURES},
        MODEL_FILE,
    )

    print(f"Model saved to: {MODEL_FILE}")


if __name__ == "__main__":
    train()
