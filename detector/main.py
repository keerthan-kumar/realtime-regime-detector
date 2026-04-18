from fastapi import FastAPI
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import sqlite3

from db.database import (
    get_connection,
    init_detections_table,
    fetch_detection_history,
    fetch_event_count,
    fetch_model_status,
    fetch_validation_metrics
)
from engine.detector import start_detection_loop, get_latest_result
from engine.hmm_model import detector, WARMUP_POINTS

# ─── Shared DB Connection ─────────────────────────────────────
conn: sqlite3.Connection = None

# ─── Lifespan ─────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global conn
    print("[API] Starting up...")
    conn = get_connection()
    init_detections_table(conn)
    start_detection_loop(conn)
    print("[API] Ready.")
    yield
    if conn:
        conn.close()
    print("[API] Shutdown complete.")


# ─── App ──────────────────────────────────────────────────────
app = FastAPI(
    title="Real-Time Regime Change Detector",
    description="Detects Normal vs Attack security regimes using a 2-state Gaussian HMM.",
    version="1.0.0",
    lifespan=lifespan
)


# ─── Endpoints ────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "realtime-regime-detector"
    }


@app.get("/status")
def status():
    event_count  = fetch_event_count(conn)
    model_status = fetch_model_status(conn)

    if model_status is None:
        return JSONResponse(
            status_code=503,
            content={
                "status":  "waiting_for_generator",
                "message": "Generator container has not started yet.",
                "hint":    "Run: docker-compose up and wait a few seconds."
            }
        )

    if event_count < WARMUP_POINTS:
        progress = detector.get_warmup_progress(event_count)
        return JSONResponse(
            status_code=503,
            content={
                "status":   "warming_up",
                "message":  "Collecting data for initial model training.",
                "progress": progress
            }
        )

    if not detector.is_ready:
        return JSONResponse(
            status_code=503,
            content={
                "status":  "training",
                "message": "Initial model training in progress.",
                "progress": {
                    "collected":   event_count,
                    "required":    WARMUP_POINTS,
                    "percent":     100.0,
                    "eta_seconds": 2.0
                }
            }
        )

    result = get_latest_result()
    if result is None:
        return JSONResponse(
            status_code=503,
            content={
                "status":  "initializing",
                "message": "First prediction not yet available. Retry in 1 second."
            }
        )

    return JSONResponse(
        status_code=200,
        content={
            "status":    "ok",
            "timestamp": result["timestamp"],
            "detection": {
                "current_state": result["current_state"],
                "alert_level":   result["alert_level"],
                "current_value": result["current_value"],
                "state_probabilities": result["state_probabilities"],
                "uncertainty_metrics": result["uncertainty_metrics"]
            },
            "model_meta": result["model_meta"]
        }
    )


@app.get("/history")
def history(limit: int = 50):
    limit = min(limit, 500)
    rows  = fetch_detection_history(conn, limit=limit)
    return JSONResponse(
        status_code=200,
        content={
            "status":  "ok",
            "count":   len(rows),
            "limit":   limit,
            "history": rows
        }
    )


@app.get("/metrics")
def metrics():
    if not detector.is_ready:
        return JSONResponse(
            status_code=503,
            content={
                "status":  "not_ready",
                "message": "Model not yet trained. Retry after warmup."
            }
        )

    result = fetch_validation_metrics(conn)

    if result is None:
        return JSONResponse(
            status_code=503,
            content={
                "status":  "insufficient_data",
                "message": "Not enough data to compute metrics yet. Retry in a few seconds."
            }
        )

    return JSONResponse(
        status_code=200,
        content={
            "status":  "ok",
            "metrics": result
        }
    )