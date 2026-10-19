
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = ROOT / "ml" / "artifacts" / "glucose_spike_model.joblib"


def predict_glucose_spike(features: dict) -> dict:
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "Model not found. Run python ml/train.py first."
        )

    bundle = joblib.load(MODEL_FILE)
    model = bundle["model"]
    feature_names = bundle["features"]

    missing = set(feature_names) - set(features)

    if missing:
        raise ValueError(
            f"Missing features: {', '.join(sorted(missing))}"
        )

    # Keep the feature order identical to training.
    input_data = pd.DataFrame(
        [{name: features[name] for name in feature_names}]
    )

    probability = float(
        model.predict_proba(input_data)[0][1]
    )

    return {
        "event": "glucose_spike_within_2h",
        "probability": round(probability, 4),
        "model_type": "synthetic_demo",
        "clinical_use": False,
    }


if __name__ == "__main__":
    sample = {
        "age": 45,
        "bmi": 28.4,
        "hba1c": 6.1,
        "current_glucose": 118,
        "glucose_change_30m": 12,
        "heart_rate": 78,
        "sleep_hours": 5.4,
        "steps": 1500,
    }

    print(predict_glucose_spike(sample))
