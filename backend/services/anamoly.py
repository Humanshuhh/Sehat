import pandas as  pd 
from backend.services.twin_service import load_patient_data


# Demonstration thresholds only.
# These are NOT clinically validated alert limits.
DEMO_RULES = {
    "glucose_low": 70,
    "glucose_high": 180,
    "heart_rate_low": 50,
    "heart_rate_high": 120,
    "glucose_change": 30,
}


def detect_patient_anomalies(patient_id: str) -> dict:
    """
    Inspect the latest synthetic wearable reading and
    return explainable demo alerts.

    Does not modify CSV files or generate readings.
    """

    _, wearable = load_patient_data()
    patient_id = str(patient_id)

    readings = wearable[
        wearable["patient_id"].astype(str) == patient_id
    ].copy()

    if readings.empty:
        raise ValueError(
            f"No wearable readings found for patient {patient_id}"
        )

    readings["timestamp"] = pd.to_datetime(
    readings["timestamp"], errors="coerce"
)

    readings = (
        readings.dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if readings.empty:
        raise ValueError("No valid wearable timestamps found.")

    latest = readings.iloc[-1]
    alerts = []

    def add_alert(rule, message, value):
        alerts.append({
            "rule": rule,
            "message": message,
            "value": float(value),
            "timestamp": latest["timestamp"].isoformat(),
            "demo_only": True,
        })

    # Rule 1: Demonstration glucose thresholds
    glucose = float(latest["glucose"])

    if glucose < DEMO_RULES["glucose_low"]:
        add_alert(
            "glucose_below_demo_threshold",
            "Glucose is below the configured demo threshold.",
            glucose,
        )
    elif glucose > DEMO_RULES["glucose_high"]:
        add_alert(
            "glucose_above_demo_threshold",
            "Glucose is above the configured demo threshold.",
            glucose,
        )

    # Rule 2: Demonstration heart-rate thresholds
    heart_rate = float(latest["heart_rate"])

    if heart_rate < DEMO_RULES["heart_rate_low"]:
        add_alert(
            "heart_rate_below_demo_threshold",
            "Heart rate is below the configured demo threshold.",
            heart_rate,
        )
    elif heart_rate > DEMO_RULES["heart_rate_high"]:
        add_alert(
            "heart_rate_above_demo_threshold",
            "Heart rate is above the configured demo threshold.",
            heart_rate,
        )

    # Rule 3: Large change between consecutive glucose readings
    if len(readings) >= 2:
        previous = readings.iloc[-2]
        previous_glucose = float(previous["glucose"])
        change = glucose - previous_glucose

        if abs(change) >= DEMO_RULES["glucose_change"]:
            alerts.append({
                "rule": "large_glucose_change_demo",
                "message": (
                    "A large change between consecutive "
                    "synthetic glucose readings was detected."
                ),
                "value": round(change, 2),
                "previous_timestamp": (
                    previous["timestamp"].isoformat()
                ),
                "timestamp": latest["timestamp"].isoformat(),
                "demo_only": True,
            })

    return {
        "patient_id": patient_id,
        "status": "alert" if alerts else "no_demo_alerts",
        "alerts_count": len(alerts),
        "latest_timestamp": latest["timestamp"].isoformat(),
        "rules_used": DEMO_RULES,
        "alerts": alerts,
        "demo_only": True,
        "clinical_use": False,
    }
