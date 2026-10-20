
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path

from backend.services.twin_service import  (
    get_patient_twin,
    load_patient_data,
    get_glucose_history,
    get_glucose_trend,
    simulate_next_reading,
)

from backend.services.alert_history import (
    initialize_alert_history,
    record_alerts,
    get_alert_history,
    get_alert_history_summary,
)

from ml.predict import predict_glucose_spike
from backend.services.anamoly import (
    detect_patient_anomalies,
)

app = FastAPI(
    title="Healthcare Digital Twin API",
    description=(
        "API for synthetic patient records, "
        "glucose trends, predictions, and simulated readings."
    ),
    version="1.1.0",
)


@app.get("/")
def home():
    return {
        "message": "Healthcare Digital Twin API is running",
        "status": "healthy",
    }


@app.get("/patients")
def list_patients():
    try:
        patients, _ = load_patient_data()

        required_columns = ["patient_id", "age", "bmi", "hba1c"]
        missing = [
            column for column in required_columns
            if column not in patients.columns
        ]

        if missing:
            raise HTTPException(
                status_code=500,
                detail=f"Missing patient columns: {missing}",
            )

        return {
            "count": len(patients),
            "patients": patients[required_columns].to_dict(
                orient="records"
            ),
        }

    except HTTPException:
        raise
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/patients/{patient_id}/twin")
def patient_twin(patient_id: str):
    try:
        return get_patient_twin(patient_id)

    except ValueError as exc:
        message = str(exc)
        status_code = 404 if "was not found" in message.lower() else 400
        raise HTTPException(
            status_code=status_code,
            detail=message,
        ) from exc

    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/patients/{patient_id}/history")
def patient_glucose_history(patient_id: str):
    try:
        return {
            "patient_id": patient_id,
            "history": get_glucose_history(patient_id),
        }

    except ValueError as exc:
        message = str(exc)
        status_code = 404 if "was not found" in message.lower() else 400
        raise HTTPException(
            status_code=status_code,
            detail=message,
        ) from exc

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail="Patient data files not found",
        ) from exc


@app.get("/patients/{patient_id}/trend")
def patient_glucose_trend(patient_id: str):
    try:
        return get_glucose_trend(patient_id)

    except ValueError as exc:
        message = str(exc)
        status_code = 404 if "was not found" in message.lower() else 400
        raise HTTPException(
            status_code=status_code,
            detail=message,
        ) from exc

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail="Patient data files not found",
        ) from exc


class PredictionInput(BaseModel):
    age: int = Field(ge=18, le=120)
    bmi: float = Field(gt=0, le=100)
    hba1c: float = Field(ge=0, le=30)
    current_glucose: float = Field(gt=0, le=1000)
    glucose_change_30m: float = Field(ge=-1000, le=1000)
    heart_rate: int = Field(ge=1, le=300)
    sleep_hours: float = Field(ge=0, le=24)
    steps: int = Field(ge=0, le=200000)


@app.post("/predict")
def predict_endpoint(payload: PredictionInput):
    try:
        return predict_glucose_spike(
            payload.model_dump()
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


@app.get("/patients/{patient_id}/forecast")
def forecast_patient_glucose_risk(patient_id: str):
    """Estimate synthetic-demo glucose spike risk using the latest reading."""
    import pandas as pd

    try:
        patients, wearable = load_patient_data()

        # Find the patient's demographic information.
        patient_rows = patients[
            patients["patient_id"].astype(str) == str(patient_id)
        ]

        if patient_rows.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Patient {patient_id} was not found",
            )

        # Get the patient's recorded wearable readings.
        readings = wearable[
            wearable["patient_id"].astype(str) == str(patient_id)
        ].copy()

        if readings.empty:
            raise HTTPException(
                status_code=404,
                detail="No wearable readings found for this patient",
            )

        required_columns = [
            "timestamp", "glucose", "heart_rate",
            "sleep_hours", "steps",
        ]
        missing = [
            column for column in required_columns
            if column not in readings.columns
        ]

        if missing:
            raise HTTPException(
                status_code=500,
                detail=f"Missing wearable columns: {missing}",
            )

        readings["timestamp"] = pd.to_datetime(
            readings["timestamp"], errors="coerce"
        )
        readings = (
            readings.dropna(subset=["timestamp", "glucose"])
            .sort_values("timestamp")
        )

        if readings.empty:
            raise HTTPException(
                status_code=400,
                detail="No valid timestamped glucose readings found",
            )

        latest = readings.iloc[-1]
        previous = readings.iloc[-2] if len(readings) >= 2 else None

        # Calculate change from the previous recorded reading.
        # This is not necessarily an exact 30-minute change.
        glucose_change = (
            float(latest["glucose"]) - float(previous["glucose"])
            if previous is not None else 0.0
        )

        patient = patient_rows.iloc[0]

        features = {
            "age": int(patient["age"]),
            "bmi": float(patient["bmi"]),
            "hba1c": float(patient["hba1c"]),
            "current_glucose": float(latest["glucose"]),
            "glucose_change_30m": glucose_change,
            "heart_rate": int(latest["heart_rate"]),
            "sleep_hours": float(latest["sleep_hours"]),
            "steps": int(latest["steps"]),
        }

        prediction = predict_glucose_spike(features)

        return {
            "patient_id": patient_id,
            "latest_reading_time": latest["timestamp"].isoformat(),
            "current_glucose": features["current_glucose"],
            "estimated_spike_probability": prediction["probability"],
            "predicted_event": prediction["event"],
            "forecast_horizon": "within 2 hours",
            "model_type": prediction["model_type"],
            "clinical_use": False,
            "demo_only": True,
            "note": (
                "Experimental prediction from synthetic/demo data; "
                "not a medical diagnosis or treatment recommendation."
            ),
        }

    except HTTPException:
        raise
    except (FileNotFoundError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


@app.post("/patients/{patient_id}/simulate")
def simulate_patient_reading(patient_id: str):
    try:
        return simulate_next_reading(patient_id)

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        message = str(exc)
        status_code = 404 if "was not found" in message.lower() else 400
        raise HTTPException(
            status_code=status_code,
            detail=message,
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not simulate reading: {exc}",
        ) from exc


@app.get("/patients/{patient_id}/alerts")
def patient_anomaly_alerts(patient_id: str):
    """Return rule-based alerts for synthetic wearable data."""

    try:
        return detect_patient_anomalies(patient_id)

    except ValueError as error:
        message = str(error)

        status_code = (
            404 if "not found" in message.lower()
            else 400
        )

        raise HTTPException(
            status_code=status_code,
            detail=message,
        ) from error

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error
@app.on_event("startup")
def setup_alert_history():
    initialize_alert_history()


@app.get("/patients/{patient_id}/alerts/history")
def patient_alert_history(patient_id: str, limit: int = 100):
    return {
        "patient_id": patient_id,
        "alerts": get_alert_history(patient_id, limit),
        "demo_only": True,
    }


@app.get("/patients/{patient_id}/alerts/summary")
def patient_alert_summary(patient_id: str):
    return get_alert_history_summary(patient_id)



# ---------- SMART GLUCOSE ALERTS ----------

CSV_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "synthetic"
    / "patients.csv"
)

# Configurable thresholds for this synthetic demo only.
HIGH_GLUCOSE = 180.0
LOW_GLUCOSE = 70.0
RAPID_RISE = 30.0
RAPID_RISE_WINDOW_MINUTES = 30


def detect_glucose_alerts(patient_df: pd.DataFrame) -> list[dict]:
    """Detect unusual glucose patterns in one patient's readings."""

    required = {"timestamp", "glucose"}
    if not required.issubset(patient_df.columns):
        raise ValueError("Patient data must contain timestamp and glucose.")

    df = patient_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["glucose"] = pd.to_numeric(df["glucose"], errors="coerce")

    df = (
        df.dropna(subset=["timestamp", "glucose"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    alerts = []

    for i, row in df.iterrows():
        glucose = float(row["glucose"])
        timestamp = row["timestamp"].isoformat()

        # Check the current reading.
        if glucose >= HIGH_GLUCOSE:
            alerts.append({
                "type": "high_glucose",
                "severity": "warning",
                "timestamp": timestamp,
                "glucose": glucose,
                "message": (
                    f"Glucose reading {glucose:.1f} is at or above "
                    f"the demo threshold of {HIGH_GLUCOSE:.0f}."
                ),
            })

        elif glucose < LOW_GLUCOSE:
            alerts.append({
                "type": "low_glucose",
                "severity": "warning",
                "timestamp": timestamp,
                "glucose": glucose,
                "message": (
                    f"Glucose reading {glucose:.1f} is below "
                    f"the demo threshold of {LOW_GLUCOSE:.0f}."
                ),
            })

        # Compare with the immediately previous reading.
        if i > 0:
            previous = df.iloc[i - 1]
            minutes = (
                row["timestamp"] - previous["timestamp"]
            ).total_seconds() / 60

            rise = glucose - float(previous["glucose"])

            if (
                0 < minutes <= RAPID_RISE_WINDOW_MINUTES
                and rise >= RAPID_RISE
            ):
                alerts.append({
                    "type": "rapid_rise",
                    "severity": "warning",
                    "timestamp": timestamp,
                    "glucose": glucose,
                    "change": round(rise, 1),
                    "minutes": round(minutes, 1),
                    "message": (
                        f"Glucose rose by {rise:.1f} within "
                        f"{minutes:.1f} minutes."
                    ),
                })

    # Show the newest alerts first.
    alerts.sort(key=lambda alert: alert["timestamp"], reverse=True)
    return alerts


@app.get("/patients/{patient_id}/alerts")
def get_patient_glucose_alerts(patient_id: str):
    """Return demo glucose alerts for a patient."""

    if not CSV_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Patient CSV not found: {CSV_PATH}",
        )

    try:
        df = pd.read_csv(CSV_PATH, dtype={"patient_id": str})
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read patient data: {exc}",
        )

    required = {"patient_id", "timestamp", "glucose"}
    if not required.issubset(df.columns):
        raise HTTPException(
            status_code=500,
            detail="CSV must contain patient_id, timestamp, and glucose.",
        )

    patient_df = df[df["patient_id"] == patient_id]

    if patient_df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No readings found for patient {patient_id}.",
        )

    try:
        alerts = detect_glucose_alerts(patient_df)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "patient_id": patient_id,
        "alert_count": len(alerts),
        "alerts": alerts,
        "demo_only": True,
    }
