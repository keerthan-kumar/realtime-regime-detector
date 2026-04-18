import sqlite3
import time
import os
from typing import Optional

# ─── Config ───────────────────────────────────────────────────
DB_PATH = os.getenv("DB_PATH", "/data/events.db")

# ─── Connection ───────────────────────────────────────────────
def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.row_factory = sqlite3.Row
    return conn

# ─── One-time Initialization ──────────────────────────────────
def init_detections_table(conn: sqlite3.Connection):
    """
    Called ONCE at detector startup.
    Never called again — zero schema overhead on inference.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id           INTEGER,
            timestamp          TEXT    NOT NULL,
            value              REAL    NOT NULL,
            predicted_state    TEXT    NOT NULL,
            p_normal           REAL    NOT NULL,
            p_attack           REAL    NOT NULL,
            alert_level        TEXT    NOT NULL,
            rolling_stability  REAL,
            state_duration     INTEGER,
            transition_risk    REAL,
            z_score            REAL
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_detections_timestamp
        ON detections (timestamp DESC)
    """)
    conn.commit()
    print("[DB] Detections table ready.")

# ─── Reads ────────────────────────────────────────────────────
def fetch_latest_events(conn: sqlite3.Connection, limit: int = 200) -> list:
    """
    Fetches the last `limit` events ordered by timestamp DESC.
    Used by HMM for training and prediction window.
    """
    cursor = conn.execute("""
        SELECT id, timestamp, value, true_state
        FROM events
        ORDER BY timestamp DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    return list(reversed(rows))

def fetch_event_count(conn: sqlite3.Connection) -> int:
    """
    Returns total number of events collected so far.
    Returns 0 safely if events table doesn't exist yet.
    """
    try:
        cursor = conn.execute("SELECT COUNT(*) FROM events")
        return cursor.fetchone()[0]
    except sqlite3.OperationalError:
        return 0

def fetch_model_status(conn: sqlite3.Connection) -> Optional[dict]:
    """
    Reads the model_status row written by the generator.
    Returns None if table doesn't exist yet.
    """
    try:
        cursor = conn.execute("""
            SELECT phase, events_collected, warmup_required, last_updated
            FROM model_status
            WHERE id = 1
        """)
        row = cursor.fetchone()
        return dict(row) if row else None
    except sqlite3.OperationalError:
        return None

# ─── Writes ───────────────────────────────────────────────────
def insert_detection(
    conn: sqlite3.Connection,
    event_id: int,
    timestamp: str,
    value: float,
    predicted_state: str,
    p_normal: float,
    p_attack: float,
    alert_level: str,
    rolling_stability: float,
    state_duration: int,
    transition_risk: float,
    z_score: float
):
    """
    Pure insert with retry on lock.
    Table guaranteed to exist from init_detections_table() at startup.
    """
    for attempt in range(3):
        try:
            conn.execute("""
                INSERT INTO detections (
                    event_id, timestamp, value, predicted_state,
                    p_normal, p_attack, alert_level,
                    rolling_stability, state_duration,
                    transition_risk, z_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event_id, timestamp, value, predicted_state,
                p_normal, p_attack, alert_level,
                rolling_stability, state_duration,
                transition_risk, z_score
            ))
            conn.commit()
            return
        except sqlite3.OperationalError as e:
            if "locked" in str(e) and attempt < 2:
                time.sleep(0.1)
                continue
            print(f"[DB] insert_detection failed: {e}")

def fetch_detection_history(conn: sqlite3.Connection, limit: int = 50) -> list:
    """
    Returns last `limit` detection results for /history endpoint.
    """
    try:
        cursor = conn.execute("""
            SELECT * FROM detections
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        return [dict(row) for row in cursor.fetchall()]
    except sqlite3.OperationalError:
        return []

def fetch_validation_metrics(conn: sqlite3.Connection) -> Optional[dict]:
    """
    Computes real validation metrics by joining events (true_state)
    with detections (predicted_state) on event_id.
    """
    try:
        cursor = conn.execute("""
            SELECT
                e.true_state,
                d.predicted_state
            FROM detections d
            JOIN events e ON e.id = d.event_id
            WHERE e.true_state IS NOT NULL
            AND d.predicted_state IS NOT NULL
        """)
        rows = cursor.fetchall()

        if not rows or len(rows) < 10:
            return None

        tn = 0
        fp = 0
        fn = 0
        tp = 0

        for row in rows:
            true      = row[0].upper()
            predicted = row[1].upper()

            if true == "NORMAL" and predicted == "NORMAL":
                tn += 1
            elif true == "NORMAL" and predicted == "ATTACK":
                fp += 1
            elif true == "ATTACK" and predicted == "NORMAL":
                fn += 1
            elif true == "ATTACK" and predicted == "ATTACK":
                tp += 1

        total = tn + fp + fn + tp

        accuracy  = round((tp + tn) / total, 4) if total > 0 else 0.0
        precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
        recall    = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
        f1        = (
            round(2 * precision * recall / (precision + recall), 4)
            if (precision + recall) > 0 else 0.0
        )

        return {
            "total_evaluated": total,
            "accuracy":        accuracy,
            "precision":       precision,
            "recall":          recall,
            "f1_score":        f1,
            "confusion_matrix": {
                "true_normal_predicted_normal": tn,
                "true_normal_predicted_attack": fp,
                "true_attack_predicted_normal": fn,
                "true_attack_predicted_attack": tp
            }
        }

    except sqlite3.OperationalError:
        return None