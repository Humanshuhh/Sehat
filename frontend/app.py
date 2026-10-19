
import requests
import pandas as pd
import streamlit as st
import plotly.express as px
from streamlit_autorefresh import st_autorefresh

API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Healthcare Digital Twin",
    page_icon="🩺",
    layout="wide",
)

# Refresh dashboard every 5 seconds
st_autorefresh(
    interval=5000,
    key="live_monitor_refresh",
)

st.title("🩺 Healthcare Digital Twin")
st.caption("Interactive monitoring demo using synthetic patient data.")
st.warning(
    "Demo only — values are synthetic and must not be used "
    "for diagnosis or treatment decisions."
)


# ---------------- API HELPERS ----------------

def api_get(path):
    response = requests.get(
        f"{API_BASE_URL}{path}",
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def api_post(path):
    response = requests.post(
        f"{API_BASE_URL}{path}",
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def as_records(data):
    """Convert common API response formats into records."""
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ("history", "readings", "data", "records"):
            value = data.get(key)
            if isinstance(value, list):
                return value

        if data and all(isinstance(v, dict) for v in data.values()):
            return list(data.values())

    return []


def safe_default(
    data, key, fallback, minimum, maximum, cast=float
):
    """Read a numeric value safely and clamp it to widget limits."""
    try:
        value = cast(data.get(key, fallback))
    except (TypeError, ValueError, OverflowError):
        value = cast(fallback)

    return max(minimum, min(maximum, value))


def numeric_from_record(record, names):
    """Return the first available numeric field."""
    if record is None:
        return None

    for name in names:
        if name in record.index:
            value = pd.to_numeric(
                record[name],
                errors="coerce",
            )
            if pd.notna(value):
                return float(value)

    return None


def safe_number(value):
    """Convert a value to a finite float or return None."""
    try:
        number = float(value)
        if pd.notna(number) and number not in (
            float("inf"), float("-inf")
        ):
            return number
    except (TypeError, ValueError, OverflowError):
        pass

    return None


# ---------------- EXPLAINABLE AI HELPER ----------------

XAI_REFERENCE_VALUES = {
    "age": 45,
    "bmi": 28.4,
    "hba1c": 6.1,
    "current_glucose": 118.0,
    "glucose_change_30m": 12.0,
    "heart_rate": 78,
    "sleep_hours": 5.4,
    "steps": 1500,
}

XAI_FEATURE_LABELS = {
    "age": "Age",
    "bmi": "BMI",
    "hba1c": "HbA1c",
    "current_glucose": "Current glucose",
    "glucose_change_30m": "Glucose change (30 min)",
    "heart_rate": "Heart rate",
    "sleep_hours": "Sleep hours",
    "steps": "Steps",
}


def get_model_probability(payload):
    """Get the existing model's probability through FastAPI."""
    response = requests.post(
        f"{API_BASE_URL}/predict",
        json=payload,
        timeout=20,
    )
    response.raise_for_status()

    result = response.json()
    probability = float(result["probability"])

    if not 0 <= probability <= 1:
        raise ValueError("Model probability must be between 0 and 1.")

    return probability


def explain_prediction(payload):
    """
    One-feature-at-a-time sensitivity analysis.

    Each input is replaced with a fixed synthetic reference value.
    The difference shows how the model output changes under that
    specific change. It is not a causal explanation.
    """
    original_probability = get_model_probability(payload)
    explanations = []

    for feature, reference_value in XAI_REFERENCE_VALUES.items():
        if feature not in payload:
            continue

        modified_payload = payload.copy()
        modified_payload[feature] = reference_value

        # Skip features already equal to their reference value.
        if modified_payload[feature] == payload[feature]:
            difference = 0.0
        else:
            modified_probability = get_model_probability(
                modified_payload
            )

            difference = (
                original_probability - modified_probability
            ) * 100

        explanations.append({
            "Feature": XAI_FEATURE_LABELS.get(feature, feature),
            "Current value": payload[feature],
            "Reference value": reference_value,
            "Output difference (percentage points)": round(
                difference, 3
            ),
        })

    result_df = pd.DataFrame(explanations)

    if not result_df.empty:
        result_df["Absolute difference"] = result_df[
            "Output difference (percentage points)"
        ].abs()

        result_df = result_df.sort_values(
            "Absolute difference",
            ascending=False,
        ).drop(columns=["Absolute difference"])

    return original_probability, result_df


# ---------------- PATIENT SELECTOR ----------------

st.sidebar.header("Patient controls")

try:
    patients_response = api_get("/patients")

    # Support either a list or {"patients": [...]}
    if isinstance(patients_response, list):
        patient_records = patients_response
    else:
        patient_records = patients_response.get("patients", [])

    if not patient_records:
        st.error("No patients were returned by the API.")
        st.stop()

    patient_ids = [
        str(patient["patient_id"])
        for patient in patient_records
        if patient.get("patient_id") is not None
    ]

    if not patient_ids:
        st.error("The API returned no valid patient IDs.")
        st.stop()

    selected_patient = st.sidebar.selectbox(
        "Select patient",
        patient_ids,
    )

except requests.RequestException as exc:
    st.error(
        "Cannot connect to FastAPI. Start the backend first, "
        "then refresh this dashboard."
    )
    st.code(str(exc))
    st.stop()


# ---------------- PATIENT OVERVIEW ----------------

st.subheader(f"Patient overview — {selected_patient}")

patient_info = next(
    (
        patient
        for patient in patient_records
        if str(patient.get("patient_id")) == selected_patient
    ),
    {},
)

try:
    twin = api_get(
        f"/patients/{selected_patient}/twin"
    )

    if not isinstance(twin, dict):
        twin = {}

except requests.RequestException as exc:
    twin = {}
    st.error(f"Could not load the patient twin: {exc}")

overview = {**patient_info, **twin}

col1, col2, col3 = st.columns(3)

col1.metric("Age", overview.get("age", "—"))
col2.metric("BMI", overview.get("bmi", "—"))
col3.metric("HbA1c", overview.get("hba1c", "—"))

with st.expander("View raw Digital Twin response"):
    st.json(twin)


# ---------------- GLUCOSE HISTORY ----------------

st.divider()
st.subheader("📈 Glucose history")

history_df = pd.DataFrame()
timestamp_col = None
glucose_col = None

try:
    history_response = api_get(
        f"/patients/{selected_patient}/history"
    )

    records = as_records(history_response)

    if records:
        history_df = pd.DataFrame(records)

        timestamp_col = next(
            (
                col for col in (
                    "timestamp",
                    "time",
                    "datetime",
                    "recorded_at",
                )
                if col in history_df.columns
            ),
            None,
        )

        glucose_col = next(
            (
                col for col in (
                    "glucose",
                    "current_glucose",
                    "glucose_level",
                )
                if col in history_df.columns
            ),
            None,
        )

        if timestamp_col and glucose_col:
            history_df[timestamp_col] = pd.to_datetime(
                history_df[timestamp_col],
                errors="coerce",
                format="mixed",
            )

            history_df[glucose_col] = pd.to_numeric(
                history_df[glucose_col],
                errors="coerce",
            )

            history_df = (
                history_df.dropna(
                    subset=[timestamp_col, glucose_col]
                )
                .sort_values(timestamp_col)
                .reset_index(drop=True)
            )

            if not history_df.empty:
                latest_history = history_df.iloc[-1]

                previous = (
                    history_df.iloc[-2]
                    if len(history_df) > 1
                    else None
                )

                col1, col2 = st.columns(2)

                col1.metric(
                    "Latest recorded glucose",
                    f"{latest_history[glucose_col]:g}",
                )

                if previous is not None:
                    change = (
                        latest_history[glucose_col]
                        - previous[glucose_col]
                    )

                    col2.metric(
                        "Change from previous reading",
                        f"{change:+g}",
                    )
                else:
                    col2.metric(
                        "Change from previous reading",
                        "—",
                    )

                fig = px.line(
                    history_df,
                    x=timestamp_col,
                    y=glucose_col,
                    markers=True,
                    title="Recorded glucose readings over time",
                    labels={
                        timestamp_col: "Timestamp",
                        glucose_col: "Glucose (recorded units)",
                    },
                )

                fig.update_layout(
                    xaxis_title="Timestamp",
                    yaxis_title="Glucose",
                    hovermode="x unified",
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

                with st.expander("View glucose history table"):
                    st.dataframe(
                        history_df,
                        use_container_width=True,
                    )

            else:
                st.info("No valid glucose readings were found.")

        else:
            st.warning(
                "The history response has no recognizable "
                "timestamp or glucose column."
            )
            st.json(history_response)

    else:
        st.info("No glucose history records were returned.")

except requests.RequestException as exc:
    st.error(f"Could not load glucose history: {exc}")


# ---------------- LIVE PATIENT MONITORING ----------------

st.divider()
st.subheader("📡 Live Patient Monitoring")

st.caption(
    "Synthetic demo data • Auto-refresh enabled • "
    "Not for medical diagnosis"
)

latest = {}

if not history_df.empty:
    latest = history_df.iloc[-1].to_dict()


def get_latest_value(column, default=None):
    value = latest.get(column, default)

    if value is None or pd.isna(value):
        return default

    return value


# Prefer latest recorded values; fall back to the twin snapshot.
glucose = safe_number(
    get_latest_value("glucose", twin.get("current_glucose"))
)

heart_rate = safe_number(
    get_latest_value("heart_rate", twin.get("heart_rate"))
)

steps = safe_number(
    get_latest_value("steps", twin.get("steps"))
)

sleep_hours = safe_number(
    get_latest_value("sleep_hours", twin.get("sleep_hours"))
)

reading_time = twin.get(
    "last_updated",
    get_latest_value("timestamp", "Not available"),
)

metric1, metric2, metric3, metric4 = st.columns(4)

metric1.metric(
    "Glucose",
    f"{glucose:g}" if glucose is not None else "N/A",
)

metric2.metric(
    "Heart rate",
    f"{heart_rate:g}" if heart_rate is not None else "N/A",
)

metric3.metric(
    "Steps",
    f"{steps:,.0f}" if steps is not None else "N/A",
)

metric4.metric(
    "Sleep hours",
    f"{sleep_hours:g}" if sleep_hours is not None else "N/A",
)

st.caption(f"Latest recorded timestamp: {reading_time}")


# ---------------- GLUCOSE TREND SUMMARY ----------------

st.divider()
st.subheader("📊 Glucose trend analysis")

try:
    trend = api_get(
        f"/patients/{selected_patient}/trend"
    )

    if isinstance(trend, dict):
        preferred_keys = [
            "message",
            "trend",
            "direction",
            "average_glucose",
            "min_glucose",
            "max_glucose",
            "change",
            "readings_count",
            "latest_glucose",
            "total_change",
            "rate_per_hour",
            "recent_average",
            "earlier_average",
        ]

        summary = {
            key: trend[key]
            for key in preferred_keys
            if key in trend
        }

        st.json(summary if summary else trend)

    else:
        st.write(trend)

except requests.RequestException as exc:
    st.error(f"Could not load trend analysis: {exc}")


# ---------------- PREDICTIVE GLUCOSE RISK ----------------

st.subheader("🔮 Predictive Glucose Risk")

try:
    response = requests.get(
        f"{API_BASE_URL}/patients/{selected_patient}/forecast",
        timeout=10,
    )

    if response.status_code == 200:
        forecast = response.json()

        probability = forecast["estimated_spike_probability"]

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Estimated Spike Probability",
                f"{probability * 100:.1f}%"
            )

        with col2:
            st.metric(
                "Current Glucose",
                f"{forecast['current_glucose']:.1f}"
            )

        st.caption(
            "Prediction horizon: "
            + forecast.get("forecast_horizon", "within 2 hours")
        )

        st.info(
            "Experimental prediction using a synthetic/demo model. "
            "Not a medical diagnosis or treatment recommendation."
        )

        # This appears after a simulation causes the app to rerun.
        if st.session_state.get("simulation_notice"):
            st.success(
                "Forecast refreshed using the latest simulated reading."
            )

    else:
        st.warning(
            f"Forecast unavailable ({response.status_code}): "
            f"{response.text}"
        )

except requests.RequestException as exc:
    st.error(f"Could not connect to the prediction API: {exc}")


# ---------------- PATIENT ANOMALY ALERTS ----------------

st.divider()
st.subheader("🚨 Patient Anomaly Alerts")

try:
    alert_data = api_get(
        f"/patients/{selected_patient}/alerts"
    )

    alerts = (
        alert_data.get("alerts", [])
        if isinstance(alert_data, dict)
        else []
    )

    if alerts:
        st.warning(
            f"Detected {len(alerts)} demo alert(s)."
        )

        for alert in alerts:
            with st.container(border=True):
                st.markdown(
                    f"**{alert.get('message', 'Alert detected')}**"
                )

                st.write(
                    f"Rule: {alert.get('rule', 'N/A')}"
                )

                st.write(
                    f"Value: {alert.get('value', 'N/A')}"
                )

                st.caption(
                    "Timestamp: "
                    f"{alert.get('timestamp', alert_data.get('latest_timestamp', 'N/A'))}"
                )

    else:
        st.success(
            "No demo alerts detected for the latest reading."
        )

    st.caption(
        "Educational demo using synthetic data. Alerts are not "
        "clinically validated and must not be used for medical decisions."
    )

except requests.RequestException as exc:
    st.error(f"Could not load anomaly alerts: {exc}")


# ---------------- SIMULATE READING ----------------

st.divider()
st.subheader("⚙️ Simulate a wearable reading")

st.write(
    "Generate a synthetic reading through the FastAPI backend. "
    "This may append a row to your wearable CSV."
)

if st.button("Generate next reading", type="primary"):
    try:
        with st.spinner("Generating synthetic reading..."):
            result = api_post(
                f"/patients/{selected_patient}/simulate"
            )

        # Save the response before rerunning the app.
        st.session_state["simulation_notice"] = True
        st.session_state["last_simulation_result"] = result

        # Rerun all dashboard sections so they fetch updated API data.
        st.rerun()

    except requests.RequestException as exc:
        st.error(f"Simulation failed: {exc}")


# Show the simulation result after the rerun.
if st.session_state.get("last_simulation_result") is not None:
    st.success(
        "Synthetic reading generated. Dashboard and forecast refreshed."
    )

    st.json(st.session_state["last_simulation_result"])

    if st.button("Dismiss simulation result"):
        st.session_state.pop("last_simulation_result", None)
        st.session_state.pop("simulation_notice", None)
        st.rerun()


# ---------------- ALERT HISTORY ----------------

st.subheader("Alert History")
st.caption("Demo-only severity labels — not clinical triage.")

try:
    history_response = requests.get(
        f"{API_BASE_URL}/patients/{selected_patient}/alerts/history",
        params={"limit": 100},
        timeout=5,
    )

    if history_response.ok:
        history_data = history_response.json()
        alert_rows = history_data.get("alerts", [])

        if alert_rows:
            alert_history_df = pd.DataFrame(alert_rows)

            if "demo_severity" in alert_history_df.columns:
                severity_options = [
                    "All",
                    *sorted(
                        alert_history_df["demo_severity"]
                        .dropna()
                        .unique()
                    ),
                ]

                selected_severity = st.selectbox(
                    "Filter by demo severity",
                    severity_options,
                )

                if selected_severity != "All":
                    alert_history_df = alert_history_df[
                        alert_history_df["demo_severity"]
                        == selected_severity
                    ]

            st.dataframe(
                alert_history_df,
                use_container_width=True,
            )

        else:
            st.info(
                "No alerts have been recorded for this patient yet."
            )

    else:
        st.warning(
            "Could not load alert history: "
            f"{history_response.status_code}"
        )

except requests.RequestException as exc:
    st.warning(f"Alert history API is unavailable: {exc}")


# ---------------- AI GLUCOSE SPIKE PREDICTION ----------------

st.divider()
st.header("🤖 AI Glucose Spike Prediction")

st.warning(
    "Synthetic-data demonstration only. "
    "This model is not a medical diagnostic or treatment tool."
)

prediction_defaults = {**patient_info, **twin}

latest_reading = None
prediction_history = pd.DataFrame()
glucose_change_default = 0.0

glucose_change_note = (
    "No earlier reading was available to estimate a 30-minute change."
)

if (
    not history_df.empty
    and timestamp_col is not None
    and glucose_col is not None
):
    prediction_history = history_df.copy()
    latest_reading = prediction_history.iloc[-1]

    prediction_defaults["current_glucose"] = float(
        latest_reading[glucose_col]
    )

    # Find the closest reading at least 30 minutes earlier.
    latest_time = latest_reading[timestamp_col]

    earlier = prediction_history[
        prediction_history[timestamp_col]
        <= latest_time - pd.Timedelta(minutes=30)
    ]

    if not earlier.empty:
        earlier_reading = earlier.iloc[-1]

        glucose_change_default = float(
            latest_reading[glucose_col]
            - earlier_reading[glucose_col]
        )

        glucose_change_note = (
            "Estimated from the latest reading and the closest "
            "available reading at least 30 minutes earlier."
        )

    else:
        glucose_change_note = (
            "No reading at least 30 minutes earlier was found. "
            "The change defaults to 0."
        )


# Prefill wearable fields from the latest history record.
if latest_reading is not None:
    for target, candidates in {
        "heart_rate": ("heart_rate", "current_heart_rate"),
        "sleep_hours": ("sleep_hours", "sleep"),
        "steps": ("steps", "step_count"),
    }.items():

        value = numeric_from_record(
            latest_reading,
            candidates,
        )

        if value is not None:
            prediction_defaults[target] = value

st.caption(f"Selected patient: {selected_patient}")

if latest_reading is not None:
    st.caption(
        "Latest recorded glucose: "
        f"{float(latest_reading[glucose_col]):g} "
        f"(timestamp: {latest_reading[timestamp_col]})."
    )

else:
    st.info(
        "No usable glucose history was found. Available patient "
        "overview values will be used; missing fields use demo defaults."
    )


threshold = st.slider(
    "Alert threshold",
    min_value=0.50,
    max_value=0.95,
    value=0.70,
    step=0.05,
    format="%.2f",
)

st.caption(
    f"Current demo alert threshold: {threshold:.0%}"
)


with st.form("glucose_prediction_form"):
    st.caption(
        "Values are prefilled from available patient data. "
        "Review them before running the model."
    )

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=120,
            value=safe_default(
                prediction_defaults,
                "age",
                25,
                18,
                120,
                int,
            ),
        )

        bmi = st.number_input(
            "BMI",
            min_value=0.1,
            max_value=100.0,
            value=safe_default(
                prediction_defaults,
                "bmi",
                23.5,
                0.1,
                100.0,
                float,
            ),
        )

        hba1c = st.number_input(
            "HbA1c",
            min_value=0.0,
            max_value=30.0,
            value=safe_default(
                prediction_defaults,
                "hba1c",
                5.6,
                0.0,
                30.0,
                float,
            ),
        )

        current_glucose = st.number_input(
            "Current glucose",
            min_value=0.1,
            max_value=1000.0,
            value=safe_default(
                prediction_defaults,
                "current_glucose",
                110.0,
                0.1,
                1000.0,
                float,
            ),
        )

    with col2:
        glucose_change = st.number_input(
            "Glucose change over 30 minutes",
            min_value=-1000.0,
            max_value=1000.0,
            value=max(
                -1000.0,
                min(1000.0, float(glucose_change_default)),
            ),
            help=(
                "Prefilled from history when a suitable earlier "
                "reading exists."
            ),
        )

        prediction_heart_rate = st.number_input(
            "Heart rate",
            min_value=1,
            max_value=300,
            value=safe_default(
                prediction_defaults,
                "heart_rate",
                78,
                1,
                300,
                int,
            ),
        )

        prediction_sleep_hours = st.number_input(
            "Sleep hours",
            min_value=0.0,
            max_value=24.0,
            value=safe_default(
                prediction_defaults,
                "sleep_hours",
                7.0,
                0.0,
                24.0,
                float,
            ),
        )

        prediction_steps = st.number_input(
            "Steps",
            min_value=0,
            max_value=200000,
            value=safe_default(
                prediction_defaults,
                "steps",
                2500,
                0,
                200000,
                int,
            ),
        )

    st.caption(glucose_change_note)

    run_prediction = st.form_submit_button(
        "Run AI Demo Prediction",
        type="primary",
    )


if run_prediction:
    payload = {
        "age": age,
        "bmi": bmi,
        "hba1c": hba1c,
        "current_glucose": current_glucose,
        "glucose_change_30m": glucose_change,
        "heart_rate": prediction_heart_rate,
        "sleep_hours": prediction_sleep_hours,
        "steps": prediction_steps,
    }

    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=payload,
            timeout=20,
        )

        
        # ---------------- XAI: LOCAL SENSITIVITY ----------------

        st.divider()
        st.subheader("🧠 Explainable AI — Why did the output change?")

        st.write(
            "This analysis changes one input at a time to a fixed "
            "synthetic reference value and measures how the model "
            "output changes."
        )

        with st.spinner("Analysing model sensitivity..."):
            try:
                original_probability, explanation_df = (
                    explain_prediction(payload)
                )

                st.metric(
                    "Original model output",
                    f"{original_probability:.1%}",
                )

                if not explanation_df.empty:
                    chart_df = explanation_df.sort_values(
                        "Output difference (percentage points)"
                    )

                    fig = px.bar(
                        chart_df,
                        x="Output difference (percentage points)",
                        y="Feature",
                        orientation="h",
                        title=(
                            "Change in model output when one input "
                            "is replaced by its reference value"
                        ),
                        labels={
                            "Output difference (percentage points)": (
                                "Output difference (percentage points)"
                            ),
                            "Feature": "Input feature",
                        },
                    )

                    fig.add_vline(x=0, line_dash="dash")

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                    )

                    st.dataframe(
                        explanation_df,
                        use_container_width=True,
                    )

                    st.caption(
                        "Positive values mean the original input "
                        "produced a higher model output than the "
                        "reference-substituted input. Negative values "
                        "mean it produced a lower output. These are "
                        "local sensitivity comparisons, not causal "
                        "effects or clinical risk factors."
                    )

                st.info(
                    "The reference values are fixed synthetic examples. "
                    "This is not SHAP and does not prove that a feature "
                    "caused the prediction."
                )

            except (
                requests.RequestException,
                KeyError,
                TypeError,
                ValueError,
            ) as error:
                st.error(
                    f"Could not calculate the explanation: {error}"
                )


        response.raise_for_status()
        prediction = response.json()

        st.subheader("Prediction result")
        st.json(prediction)

        probability = prediction.get("probability")

        if probability is None:
            st.info(
                "The API responded, but no 'probability' field "
                "was returned. Check the JSON above against your "
                "model's response format."
            )

        else:
            probability = float(probability)

            if not 0.0 <= probability <= 1.0:
                st.error(
                    "The model probability must be between 0 and 1. "
                    "Check the prediction function's output."
                )

            else:
                st.metric(
                    "Model output",
                    f"{probability:.1%}",
                )

                st.progress(probability)

                if probability >= threshold:
                    st.warning(
                        f"Demo alert: model output {probability:.1%} "
                        f"meets the UI threshold of {threshold:.0%}."
                    )
                else:
                    st.info(
                        f"No demo alert triggered. Model output: "
                        f"{probability:.1%}; UI threshold: {threshold:.0%}."
                    )

                st.caption(
                    "This threshold is a UI setting, not a clinically "
                    "validated cutoff. The synthetic demo output is not "
                    "a real-world medical risk estimate."
                )

    except requests.HTTPError as error:
        st.error(f"Prediction API returned an error: {error}")

        if error.response is not None:
            st.code(error.response.text)

    except requests.RequestException as error:
        st.error(
            f"Could not connect to the prediction API: {error}"
        )

    except (TypeError, ValueError) as error:
        st.error(
            f"Could not interpret the prediction response: {error}"
        )
