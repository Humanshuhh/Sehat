# 🏥 Sehat — Healthcare Digital Twin

A synthetic-data-driven Healthcare Digital Twin prototype built with **Python, FastAPI, and Streamlit**. The project demonstrates patient data visualization, glucose trend monitoring, AI-based prediction, explainable AI (XAI), and rule-based smart alerts.

> **Disclaimer:** This project is for educational and demonstration purposes only. It is not a medical device and must not be used for diagnosis, treatment, or emergency medical decisions.

## ✨ Features

- **Patient Monitoring:** View synthetic patient readings and health-related metrics.
- **Glucose Trend Analysis:** Analyze recorded glucose readings over time.
- **AI Prediction Demo:** Submit patient inputs to an AI prediction endpoint.
- **Explainable AI (XAI):** Explore one-feature-at-a-time model sensitivity using reference values.
- **Smart Glucose Alerts:** Flag configured high, low, and rapid-rise glucose patterns.
- **Interactive Dashboard:** Explore data and results through a Streamlit interface.
- **REST API:** Access backend functionality through FastAPI endpoints and Swagger UI.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| FastAPI | Backend REST API |
| Uvicorn | ASGI development server |
| Streamlit | Interactive dashboard |
| Pandas | Data processing and analysis |
| Plotly | Interactive charts |
| Requests | Communication between frontend and backend |
| CSV | Synthetic patient data storage |

## 📁 Project Structure

```text
Twin/
├── backend/
│   ├── __init__.py
│   └── main.py
├── frontend/
│   └── app.py
├── data/
│   └── synthetic/
│       └── patients.csv
├── .gitignore
├── README.md
└── requirements.txt
```

> The structure above is the intended layout. Keep any additional modules already present in your project, and adjust the paths if your actual folder structure differs.

## 📋 Prerequisites

- Python 3.10 or a compatible newer version
- pip
- Git
- Visual Studio Code (recommended)

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Humanshuhh/Sehat.git
cd Sehat
```

If you already have the project locally, open its root folder in VS Code instead.

### 2. Create a virtual environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

If PowerShell blocks activation, use Command Prompt or run the Python commands using the environment's interpreter directly.

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install fastapi "uvicorn[standard]" streamlit pandas plotly requests
```

If your project already has a complete `requirements.txt`, install from it instead:

```bash
python -m pip install -r requirements.txt
```

### 4. Verify the data file

Ensure the synthetic dataset exists at:

```text
data/synthetic/patients.csv
```

The dataset currently uses these columns:

```text
patient_id, timestamp, glucose, heart_rate, sleep_hours, steps
```

The patient ID used for testing must exist in the dataset. For example, `P001` works only if that ID is present.

### 5. Start the FastAPI backend

Open a terminal in the project root and run:

```bash
python -m uvicorn backend.main:app --reload
```

The backend should be available at:

- API base URL: http://127.0.0.1:8000
- Interactive API documentation: http://127.0.0.1:8000/docs
- Alternative API documentation: http://127.0.0.1:8000/redoc

Keep this terminal running.

### 6. Start the Streamlit dashboard

Open a **second terminal** in the project root. Activate the same virtual environment if necessary, then run:

```bash
python -m streamlit run frontend/app.py
```

Open the dashboard at:

http://localhost:8501

Both the backend and frontend must be running for API-backed dashboard features to work.

## 🔌 API Endpoints

The following endpoints are part of the project's intended interface. Confirm the implemented routes in `/docs`, because endpoint availability depends on the current backend code.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/docs` | Interactive API documentation |
| GET | `/patients/{patient_id}/history` | Retrieve a patient's recorded history, if implemented |
| GET | `/patients/{patient_id}/alerts` | Retrieve rule-based glucose alerts |
| POST | `/predict` | Run the AI prediction demo |

### Example: Check glucose alerts

Once the backend is running, open:

```text
http://127.0.0.1:8000/patients/P001/alerts
```

The endpoint returns a JSON response containing the patient ID, alert count, alerts, and a demo-only indicator, provided the route and CSV are configured correctly.

Example response shape:

```json
{
  "patient_id": "P001",
  "alert_count": 0,
  "alerts": [],
  "demo_only": true
}
```

Actual results depend on the synthetic readings in the dataset.

## 🧠 Explainable AI

The current XAI prototype performs **one-feature-at-a-time sensitivity analysis**:

1. It obtains the model output for the original inputs.
2. It replaces one input feature at a time with a fixed reference value.
3. It compares the modified output with the original output.
4. It displays the differences in a chart and table.

This demonstrates how model outputs respond to individual input changes. It is **not SHAP, causal inference, or clinically validated medical reasoning**.

## 🚨 Smart Glucose Alerts

The alert prototype checks configured rules such as:

- Glucose readings at or above the configured high threshold.
- Glucose readings below the configured low threshold.
- A rapid glucose increase between consecutive readings within a configured time window.

The thresholds are configurable demo values. The rules are not a complete clinical policy, and absence of an alert does not establish that a patient is medically safe.

## 🧪 Testing

### Test the backend

1. Start FastAPI.
2. Open http://127.0.0.1:8000/docs.
3. Find the relevant endpoint.
4. Select **Try it out** and execute a request.
5. Inspect the response and any error messages.

### Test the dashboard

1. Start Streamlit in a second terminal.
2. Open http://localhost:8501.
3. Select a patient or enter a valid patient ID.
4. Review the available history and glucose trends.
5. Run the prediction demo and review its XAI explanation.
6. Check the Smart Glucose Alerts panel.

### Common issues

**`ModuleNotFoundError`**

Install the missing dependency in the active Python environment:

```bash
python -m pip install -r requirements.txt
```

**`Could not connect to FastAPI`**

Check that Uvicorn is running at `http://127.0.0.1:8000` and that `API_BASE_URL` in the frontend matches it.

**`404 Not Found`**

Check the endpoint path and confirm that the requested patient ID exists in the dataset.

**`FileNotFoundError` for the CSV**

Verify the dataset location and the path used by the backend. Prefer paths resolved relative to the Python file rather than the current working directory.

**`KeyError` in the dashboard**

Inspect the actual API JSON response and ensure the frontend reads keys that the backend returns.

## 🔒 Data and Privacy

- Use synthetic or properly de-identified data for demonstrations.
- Do not commit private patient records, API keys, credentials, or secrets.
- Keep environment files and virtual environments out of version control.
- Review the dataset and repository contents before publishing them.

## 🗺️ Roadmap

Potential future improvements:

- [ ] Patient selector integrated with all dashboard panels
- [ ] Improved glucose trend visualizations with alert markers
- [ ] Persistent prediction and alert history
- [ ] Automated tests for API endpoints and alert rules
- [ ] More robust input validation and error handling
- [ ] Model evaluation and documented limitations
- [ ] SHAP-based explanations, if compatible with the prediction model
- [ ] Database-backed patient history instead of CSV storage

## 🤝 Contributing

Contributions and suggestions are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Make focused changes.
4. Test the backend and dashboard.
5. Open a pull request describing the changes.

## 📄 License

No license is specified here. Add a `LICENSE` file before presenting the project as open source, and choose a license that fits your intended use.

## 👨‍💻 Author

**Sehat — Healthcare Digital Twin**

GitHub repository: https://github.com/Humanshuhh/Sehat

---

*Built as a learning project exploring healthcare data visualization, backend APIs, AI predictions, explainability, and rule-based monitoring.*
