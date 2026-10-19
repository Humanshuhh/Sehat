from pathlib import Path
from datetime import datetime, timezone
import sqlite3
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "synthetic" / "alert_history.db"


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_alert_history():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS alert_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                alert_key TEXT NOT NULL,
                rule TEXT NOT NULL,
                message TEXT NOT NULL,
                value TEXT,
                reading_timestamp TEXT NOT NULL,
                recorded_at TEXT NOT NULL,
                demo_severity TEXT NOT NULL,
                details_json TEXT,
                UNIQUE(patient_id, alert_key, reading_timestamp)
            )
        """)


def classify_demo_severity(rule):
    rule = str(rule).lower()

    if any(word in rule for word in ("high", "spike", "rapid_change")):
        return "High (demo)"
    if "low" in rule:
        return "Moderate (demo)"
    return "Info (demo)"


def record_alerts(patient_id, alerts):
    initialize_alert_history()
    inserted = 0

    with get_connection() as conn:
        for alert in alerts:
            rule = str(alert.get("rule") or alert.get("type") or "unknown_rule")
            message = str(
                alert.get("message") or alert.get("description") or rule
            )
            value = alert.get("value")
            timestamp = (
                alert.get("timestamp")
                or alert.get("reading_timestamp")
                or alert.get("observed_at")
                or "unknown-reading"
            )

            # Stable key prevents duplicates on automatic dashboard refresh.
            alert_key = f"{rule}:{message}"

            cursor = conn.execute("""
                INSERT OR IGNORE INTO alert_history (
                    patient_id, alert_key, rule, message, value,
                    reading_timestamp, recorded_at, demo_severity, details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                patient_id,
                alert_key,
                rule,
                message,
                None if value is None else str(value),
                str(timestamp),
                datetime.now(timezone.utc).isoformat(),
                classify_demo_severity(rule),
                json.dumps(alert, default=str),
            ))
            inserted += cursor.rowcount

    return inserted


def get_alert_history(patient_id, limit=100):
    initialize_alert_history()
    limit = max(1, min(int(limit), 500))

    with get_connection() as conn:
        rows = conn.execute("""
            SELECT id, patient_id, rule, message, value,
                   reading_timestamp, recorded_at, demo_severity
            FROM alert_history
            WHERE patient_id = ?
            ORDER BY id DESC
            LIMIT ?
        """, (patient_id, limit)).fetchall()

    return [dict(row) for row in rows]


def get_alert_history_summary(patient_id):
    initialize_alert_history()

    with get_connection() as conn:
        total = conn.execute("""
            SELECT COUNT(*) FROM alert_history WHERE patient_id = ?
        """, (patient_id,)).fetchone()[0]

        rows = conn.execute("""
            SELECT demo_severity, COUNT(*) AS count
            FROM alert_history
            WHERE patient_id = ?
            GROUP BY demo_severity
        """, (patient_id,)).fetchall()

    return {
        "patient_id": patient_id,
        "total_alerts": total,
        "by_demo_severity": {
            row["demo_severity"]: row["count"] for row in rows
        },
        "demo_only": True,
        "clinical_use": False,
    }