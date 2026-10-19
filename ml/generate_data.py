
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "synthetic" / "training_data.csv"


def generate_data(n_samples=5000, seed=42):
    rng = np.random.default_rng(seed)

    # Illustrative synthetic patient and sensor features.
    data = pd.DataFrame({
        "age": rng.integers(18, 80, n_samples),
        "bmi": rng.uniform(18, 38, n_samples),
        "hba1c": rng.uniform(4.5, 9.0, n_samples),
        "current_glucose": rng.uniform(75, 190, n_samples),
        "glucose_change_30m": rng.uniform(-15, 30, n_samples),
        "heart_rate": rng.integers(55, 115, n_samples),
        "sleep_hours": rng.uniform(3, 10, n_samples),
        "steps": rng.integers(0, 10000, n_samples),
    })

    # A synthetic future-glucose simulator creates labels.
    # These rules are illustrative, not medically validated.
    future_change = (
        0.9 * data["glucose_change_30m"]
        + 0.10 * (data["current_glucose"] - 110)
        + 1.2 * (7 - data["sleep_hours"])
        + 0.25 * (data["bmi"] - 25)
        + rng.normal(0, 12, n_samples)
    )

    future_peak = data["current_glucose"] + future_change

    # Demo definition: glucose rises at least 30 mg/dL
    # above its current value within the simulated horizon.
    data["glucose_spike_within_2h"] = (
        future_peak >= data["current_glucose"] + 30
    ).astype(int)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT, index=False)

    print(f"Generated {len(data)} synthetic examples.")
    print(f"Saved to: {OUTPUT}")
    print("\nLabel distribution:")
    print(data["glucose_spike_within_2h"].value_counts())


if __name__ == "__main__":
    generate_data()
