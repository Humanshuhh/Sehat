
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd


# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

# This file is located at backend/services/twin_service.py.
# parents[2] points to the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

PATIENTS_FILE = (
    PROJECT_ROOT / "data" / "synthetic" / "patients.csv"
)

WEARABLE_FILE = (
    PROJECT_ROOT / "data" / "synthetic" / "wearable_data.csv"
)

# Keep the existing name used by the simulator.
WEARABLE_CSV = WEARABLE_FILE

REQUIRED_WEARABLE_COLUMNS = [
    "patient_id",
    "timestamp",
    "glucose",
    "heart_rate",
    "sleep_hours",
    "steps",
]


# --------------------------------------------------
# TIMESTAMP HELPER
# --------------------------------------------------

def parse_timestamps(values):
    """
    Handle both time-only timestamps such as 08:15
    and full timestamps such as 2026-10-17 08:45:00.
    Invalid values become NaT.
    """
    try:
        return pd.to_datetime(
            values,
            format="mixed",
            errors="coerce",
        )
    except (TypeError, ValueError):
        # Compatibility fallback for older pandas versions.
        return pd.to_datetime(
            values,
            errors="coerce",
        )


# --------------------------------------------------
# LOAD AND VALIDATE DATA
# --------------------------------------------------

def load_patient_data():
    """Load patient profiles and wearable readings."""

    if not PATIENTS_FILE.exists():
        raise FileNotFoundError(
            f"Patient data not found: {PATIENTS_FILE}"
        )

    if not WEARABLE_FILE.exists():
        raise FileNotFoundError(
            f"Wearable data not found: {WEARABLE_FILE}"
        )

    patients = pd.read_csv(PATIENTS_FILE)
    wearable = pd.read_csv(WEARABLE_FILE)

    required_patient_columns = {
        "patient_id",
        "age",
        "bmi",
        "hba1c",
        "previous_diabetes",
    }

    if not required_patient_columns.issubset(patients.columns):
        raise ValueError(
            "patients.csv is missing required columns."
        )

    if not set(REQUIRED_WEARABLE_COLUMNS).issubset(
        wearable.columns
    ):
        raise ValueError(
            "wearable_data.csv is missing required columns. "
            f"Expected: {REQUIRED_WEARABLE_COLUMNS}"
        )

    # Normalize patient IDs.
    patients["patient_id"] = (
        patients["patient_id"].astype(str).str.strip()
    )

    wearable["patient_id"] = (
        wearable["patient_id"].astype(str).str.strip()
    )

    # Handle time-only and full datetime values.
    wearable["timestamp"] = parse_timestamps(
        wearable["timestamp"]
    )

    # Convert numeric columns safely.
    numeric_columns = [
        "glucose",
        "heart_rate",
        "sleep_hours",
        "steps",
    ]

    for column in numeric_columns:
        wearable[column] = pd.to_numeric(
            wearable[column],
            errors="coerce",
        )

    # Remove malformed rows rather than breaking every endpoint.
    wearable = wearable.dropna(
        subset=[
            "patient_id",
            "timestamp",
            "glucose",
            "heart_rate",
            "sleep_hours",
            "steps",
        ]
    ).copy()

    if wearable.empty:
        raise ValueError(
            "No valid wearable readings remain. "
            "Check wearable_data.csv."
        )

    return patients, wearable


# --------------------------------------------------
# PATIENT DIGITAL TWIN
# --------------------------------------------------

def get_patient_twin(patient_id: str) -> dict:
    """
    Combine a patient's profile with their latest wearable
    measurements to create a synthetic digital twin.
    """

    patients, wearable = load_patient_data()
    patient_id = str(patient_id).strip()

    patient_records = patients[
        patients["patient_id"] == patient_id
    ]

    if patient_records.empty:
        raise ValueError(
            f"Patient '{patient_id}' was not found."
        )

    if len(patient_records) > 1:
        raise ValueError(
            f"Multiple historical records found for '{patient_id}'."
        )

    patient_readings = wearable[
        wearable["patient_id"] == patient_id
    ].copy()

    if patient_readings.empty:
        raise ValueError(
            f"No wearable readings found for '{patient_id}'."
        )

    patient_readings = patient_readings.sort_values(
        "timestamp"
    )

    latest = patient_readings.iloc[-1]
    patient = patient_records.iloc[0]

    return {
        "patient_id": patient_id,
        "age": int(patient["age"]),
        "bmi": float(patient["bmi"]),
        "hba1c": float(patient["hba1c"]),
        "previous_diabetes": int(
            patient["previous_diabetes"]
        ),
        "current_glucose": float(latest["glucose"]),
        "heart_rate": int(latest["heart_rate"]),
        "sleep_hours": float(latest["sleep_hours"]),
        "steps": int(latest["steps"]),
        "last_updated": latest["timestamp"].isoformat(),
    }


# --------------------------------------------------
# GLUCOSE HISTORY
# --------------------------------------------------

def get_glucose_history(patient_id: str):
    """Return all valid glucose readings for a patient."""

    _, wearable = load_patient_data()
    patient_id = str(patient_id).strip()

    history = wearable[
        wearable["patient_id"] == patient_id
    ].copy()

    if history.empty:
        raise ValueError(
            f"Patient '{patient_id}' was not found in wearable data."
        )

    history = (
        history.sort_values("timestamp")
        .reset_index(drop=True)
    )

    return [
        {
            "timestamp": row["timestamp"].isoformat(),
            "glucose": float(row["glucose"]),
        }
        for _, row in history.iterrows()
    ]


# --------------------------------------------------
# GLUCOSE TREND ANALYSIS
# --------------------------------------------------

def get_glucose_trend(patient_id: str):
    """
    Analyze recorded glucose values.

    This is descriptive analysis of synthetic data,
    not a medical prediction or diagnostic tool.
    """

    _, wearable = load_patient_data()
    patient_id = str(patient_id).strip()

    history = wearable[
        wearable["patient_id"] == patient_id
    ].copy()

    if history.empty:
        raise ValueError(
            f"Patient '{patient_id}' was not found in wearable data."
        )

    history["glucose"] = pd.to_numeric(
        history["glucose"],
        errors="coerce",
    )

    history = (
        history.dropna(subset=["timestamp", "glucose"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if history.empty:
        raise ValueError(
            f"No valid glucose readings for patient {patient_id}."
        )

    if len(history) < 2:
        return {
            "patient_id": patient_id,
            "status": "insufficient_data",
            "message": "At least two valid readings are needed.",
            "readings_count": len(history),
            "direction": None,
            "first_timestamp": history.iloc[0][
                "timestamp"
            ].isoformat(),
            "last_timestamp": history.iloc[0][
                "timestamp"
            ].isoformat(),
            "first_glucose": float(
                history.iloc[0]["glucose"]
            ),
            "latest_glucose": float(
                history.iloc[0]["glucose"]
            ),
            "total_change": None,
            "elapsed_hours": None,
            "rate_per_hour": None,
            "earlier_average": None,
            "recent_average": None,
            "average_difference": None,
        }

    first = history.iloc[0]
    last = history.iloc[-1]

    elapsed_hours = (
        last["timestamp"] - first["timestamp"]
    ).total_seconds() / 3600

    total_change = float(
        last["glucose"] - first["glucose"]
    )

    # Simple demo classification, not a clinical threshold.
    if total_change > 1:
        direction = "rising"
    elif total_change < -1:
        direction = "falling"
    else:
        direction = "relatively stable"

    rate_per_hour = (
        total_change / elapsed_hours
        if elapsed_hours > 0
        else None
    )

    # Compare the earlier half of readings with the recent half.
    split_index = len(history) // 2

    earlier = history.iloc[:split_index]
    recent = history.iloc[split_index:]

    earlier_average = float(earlier["glucose"].mean())
    recent_average = float(recent["glucose"].mean())

    average_difference = (
        recent_average - earlier_average
    )

    return {
        "patient_id": patient_id,
        "status": "ok",
        "readings_count": len(history),
        "direction": direction,
        "first_timestamp": first["timestamp"].isoformat(),
        "last_timestamp": last["timestamp"].isoformat(),
        "first_glucose": round(float(first["glucose"]), 2),
        "latest_glucose": round(float(last["glucose"]), 2),
        "total_change": round(total_change, 2),
        "elapsed_hours": round(elapsed_hours, 2),
        "rate_per_hour": (
            round(rate_per_hour, 2)
            if rate_per_hour is not None
            else None
        ),
        "earlier_average": round(earlier_average, 2),
        "recent_average": round(recent_average, 2),
        "average_difference": round(
            average_difference, 2
        ),
    }


# --------------------------------------------------
# WEARABLE READING SIMULATOR
# --------------------------------------------------

def simulate_next_reading(patient_id: str) -> dict:
    """
    Generate one synthetic reading 15 simulated minutes
    after the patient's latest reading and append it to CSV.

    Demo only; this does not predict actual patient physiology.
    """

    if not WEARABLE_CSV.exists():
        raise FileNotFoundError(
            f"Wearable CSV not found: {WEARABLE_CSV}"
        )

    # Use the same CSV validation as the other functions.
    _, df = load_patient_data()
    patient_id = str(patient_id).strip()

    patient_rows = df[
        df["patient_id"] == patient_id
    ].copy()

    if patient_rows.empty:
        raise ValueError(
            f"No wearable readings found for patient {patient_id}."
        )

    patient_rows = patient_rows.sort_values("timestamp")
    latest = patient_rows.iloc[-1]

    next_timestamp = (
        latest["timestamp"] + timedelta(minutes=15)
    )

    # Deterministic generator for reproducible demo readings.
    seed_text = f"{patient_id}-{next_timestamp.isoformat()}"
    seed = sum(
        (index + 1) * ord(char)
        for index, char in enumerate(seed_text)
    ) % (2**32)

    rng = np.random.default_rng(seed)

    glucose = float(
        np.clip(
            float(latest["glucose"]) + rng.normal(0, 4),
            55,
            250,
        )
    )

    heart_rate = int(
        np.clip(
            round(
                float(latest["heart_rate"])
                + rng.normal(0, 3)
            ),
            45,
            150,
        )
    )

    sleep_hours = float(
        np.clip(float(latest["sleep_hours"]), 0, 24)
    )

    previous_steps = int(float(latest["steps"]))
    additional_steps = int(rng.integers(0, 151))
    steps = min(previous_steps + additional_steps, 100000)

    new_reading = {
        "patient_id": patient_id,
        "timestamp": next_timestamp.isoformat(sep=" "),
        "glucose": round(glucose, 1),
        "heart_rate": heart_rate,
        "sleep_hours": round(sleep_hours, 2),
        "steps": steps,
    }

    # IMPORTANT: ensure the existing file ends with a newline.
    # This prevents the new reading joining the previous CSV row.
    with WEARABLE_CSV.open("rb+") as csv_file:
        csv_file.seek(0, 2)
        file_size = csv_file.tell()

        if file_size > 0:
            csv_file.seek(-1, 2)
            last_byte = csv_file.read(1)

            if last_byte not in (b"\n", b"\r"):
                csv_file.seek(0, 2)
                csv_file.write(b"\n")

    # Append the new reading without rewriting old data.
    pd.DataFrame(
        [new_reading],
        columns=REQUIRED_WEARABLE_COLUMNS,
    ).to_csv(
        WEARABLE_CSV,
        mode="a",
        header=False,
        index=False,
        lineterminator="\n",
    )

    return {
        "message": "Synthetic reading generated and saved.",
        "patient_id": patient_id,
        "interval_minutes": 15,
        "reading": new_reading,
        "demo": True,
    }


# --------------------------------------------------
# OPTIONAL LOCAL TEST
# --------------------------------------------------

if __name__ == "__main__":
    try:
        print("PATIENT DIGITAL TWIN")
        print("-" * 30)
        print(get_patient_twin("P001"))

        print("\nGLUCOSE TREND")
        print("-" * 30)
        print(get_glucose_trend("P001"))

    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}")
