import time
import threading
import numpy as np
import sqlite3
import os
from datetime import datetime
from typing import Optional

from db.database import (
    fetch_latest_events,
    fetch_event_count,
    fetch_model_status,
    insert_detection,
    get_connection
)
from engine.hmm_model import detector, WARMUP_POINTS, RETRAIN_EVERY

# ─── Config ───────────────────────────────────────────────────
POLL_INTERVAL  = 0.1
PREDICT_WINDOW = 30

# ─── Shared State ─────────────────────────────────────────────
_latest_result: Optional[dict] = None
_latest_result_lock = threading.Lock()

def get_latest_result() -> Optional[dict]:
    with _latest_result_lock:
        return _latest_result

def _set_latest_result(result: dict):
    global _latest_result
    with _latest_result_lock:
        _latest_result = result

# ─── Background Detection Loop ────────────────────────────────
def _detection_loop():
    """
    Runs forever in a background thread.
    Uses its OWN dedicated SQLite connection — never shares
    with the API thread. This eliminates all locking conflicts.
    """
    print("[DETECTOR] Background detection loop started.")

    # Own dedicated connection for this thread
    conn = get_connection()

    fitted_once       = False
    last_retrain_at   = 0
    last_processed_id = 0

    while True:
        try:
            # ── Wait for generator to start ───────────────────
            status = fetch_model_status(conn)
            if status is None:
                print("[DETECTOR] Waiting for generator to start...")
                time.sleep(2.0)
                continue

            event_count = fetch_event_count(conn)

            # ── Warmup phase ──────────────────────────────────
            if event_count < WARMUP_POINTS:
                pct = round((event_count / WARMUP_POINTS) * 100, 1)
                print(f"[DETECTOR] Warming up... {event_count}/{WARMUP_POINTS} ({pct}%)")
                time.sleep(POLL_INTERVAL)
                continue

            # ── Initial fit — runs exactly once ───────────────
            if not fitted_once and not detector.is_ready:
                rows = fetch_latest_events(conn, limit=WARMUP_POINTS)
                data = np.array([row["value"] for row in rows])
                detector.initial_fit(data)
                fitted_once     = True
                last_retrain_at = event_count
                time.sleep(1.0)
                continue

            # ── Wait for fit to complete ──────────────────────
            if not detector.is_ready:
                print("[DETECTOR] Waiting for initial fit to complete...")
                time.sleep(0.5)
                continue

            # ── Fetch prediction window ───────────────────────
            rows = fetch_latest_events(conn, limit=PREDICT_WINDOW)

            if not rows:
                time.sleep(POLL_INTERVAL)
                continue

            latest_id = rows[-1]["id"]

            # ── Skip if no new event ──────────────────────────
            if latest_id == last_processed_id:
                time.sleep(POLL_INTERVAL)
                continue

            last_processed_id = latest_id
            values = [row["value"] for row in rows]

            if len(values) < 2:
                time.sleep(POLL_INTERVAL)
                continue

            # ── Periodic background retrain ───────────────────
            if event_count - last_retrain_at >= RETRAIN_EVERY:
                retrain_rows = fetch_latest_events(
                    conn, limit=WARMUP_POINTS * 2
                )
                retrain_data = np.array(
                    [row["value"] for row in retrain_rows]
                )
                detector.retrain(retrain_data)
                last_retrain_at = event_count
                print(f"[DETECTOR] Retrain triggered at {event_count} events.")

            # ── Run inference ─────────────────────────────────
            result = detector.predict(values)

            if result is None:
                time.sleep(POLL_INTERVAL)
                continue

            result["timestamp"] = datetime.utcnow().isoformat()

            # ── Store result ──────────────────────────────────
            insert_detection(
                conn              = conn,
                event_id          = latest_id,
                timestamp         = result["timestamp"],
                value             = result["current_value"],
                predicted_state   = result["current_state"],
                p_normal          = result["state_probabilities"]["P(Normal)"],
                p_attack          = result["state_probabilities"]["P(Attack)"],
                alert_level       = result["alert_level"],
                rolling_stability = result["uncertainty_metrics"]["rolling_stability"],
                state_duration    = result["uncertainty_metrics"]["state_duration_ticks"],
                transition_risk   = result["uncertainty_metrics"]["transition_risk"],
                z_score           = result["uncertainty_metrics"]["observation_z_score"]
            )

            _set_latest_result(result)

            print(
                f"[DETECTOR] id={latest_id:<6} "
                f"state={result['current_state']:<8} "
                f"P(Attack)={result['state_probabilities']['P(Attack)']:.3f} "
                f"alert={result['alert_level']}"
            )

        except Exception as e:
            print(f"[DETECTOR] Loop error: {e}")
            time.sleep(1.0)

        time.sleep(POLL_INTERVAL)


# ─── Startup ──────────────────────────────────────────────────
def start_detection_loop(conn):
    """
    Called once at FastAPI startup.
    Note: conn parameter kept for API compatibility
    but detection loop uses its own dedicated connection.
    """
    t = threading.Thread(
        target=_detection_loop,
        daemon=True
    )
    t.start()
    print("[DETECTOR] Detection loop thread launched.")