import sqlite3
import time
import random
import numpy as np
from datetime import datetime
import os

# ─── Config ───────────────────────────────────────────────────
DB_PATH            = os.getenv("DB_PATH", "/data/events.db")
TICK_INTERVAL      = 0.1
WARMUP_POINTS      = 200
COMMIT_EVERY_N     = 10     # batch commits → reduces disk I/O

NORMAL_MEAN = 10.0
NORMAL_STD  = 3.0
ATTACK_MEAN = 20.0
ATTACK_STD  = 5.0

P_NORMAL_TO_ATTACK = 0.01
P_ATTACK_TO_NORMAL = 0.05

# ─── Database Setup ───────────────────────────────────────────
def init_db(conn: sqlite3.Connection):
    cursor = conn.cursor()

    # WAL mode — CRITICAL for concurrent Docker containers
    cursor.execute("PRAGMA journal_mode=WAL;")

    # Reduce fsync calls — safe for our use case
    cursor.execute("PRAGMA synchronous=NORMAL;")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    NOT NULL,
            value       REAL    NOT NULL,
            true_state  TEXT    NOT NULL
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_events_timestamp
        ON events (timestamp DESC)
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_status (
            id               INTEGER PRIMARY KEY CHECK (id = 1),
            phase            TEXT    NOT NULL DEFAULT 'warming_up',
            events_collected INTEGER NOT NULL DEFAULT 0,
            warmup_required  INTEGER NOT NULL DEFAULT 200,
            last_updated     TEXT
        )
    """)

    cursor.execute("""
        INSERT OR IGNORE INTO model_status
        (id, phase, events_collected, warmup_required, last_updated)
        VALUES (1, 'warming_up', 0, ?, ?)
    """, (WARMUP_POINTS, datetime.utcnow().isoformat()))

    conn.commit()
    print(f"[DB] WAL mode enabled. Initialized at {DB_PATH}")

# ─── Insert (no commit — batched) ─────────────────────────────
def insert_event(conn: sqlite3.Connection, timestamp: str, value: float, true_state: str):
    conn.execute(
        "INSERT INTO events (timestamp, value, true_state) VALUES (?, ?, ?)",
        (timestamp, value, true_state)
    )
    # No commit here — batched in main loop

def update_model_status(conn: sqlite3.Connection, phase: str, count: int):
    conn.execute("""
        UPDATE model_status
        SET phase=?, events_collected=?, last_updated=?
        WHERE id=1
    """, (phase, count, datetime.utcnow().isoformat()))
    # No commit here — batched in main loop

# ─── Stream Simulation ────────────────────────────────────────
def generate_stream(conn: sqlite3.Connection):
    current_state = "NORMAL"
    tick          = 0

    print("[GENERATOR] Starting live stream simulation...")
    print(f"[GENERATOR] Normal → mean={NORMAL_MEAN}, std={NORMAL_STD}")
    print(f"[GENERATOR] Attack → mean={ATTACK_MEAN}, std={ATTACK_STD}")
    print(f"[GENERATOR] Warmup: {WARMUP_POINTS} points → ~{int(WARMUP_POINTS * TICK_INTERVAL)}s")
    print(f"[GENERATOR] Committing every {COMMIT_EVERY_N} ticks")
    print("-" * 50)

    while True:
        tick += 1

        # ── State Transition ──────────────────────────────────
        if current_state == "NORMAL":
            if random.random() < P_NORMAL_TO_ATTACK:
                current_state = "ATTACK"
                print(f"\n[!!!] ATTACK STATE STARTED at tick {tick}\n")
        else:
            if random.random() < P_ATTACK_TO_NORMAL:
                current_state = "NORMAL"
                print(f"\n[   ] NORMAL STATE RESUMED at tick {tick}\n")

        # ── Generate Observation ──────────────────────────────
        if current_state == "NORMAL":
            value = np.random.normal(NORMAL_MEAN, NORMAL_STD)
        else:
            value = np.random.normal(ATTACK_MEAN, ATTACK_STD)

        value     = max(0.0, round(value, 4))
        timestamp = datetime.utcnow().isoformat()

        # ── Stage writes (no commit yet) ──────────────────────
        insert_event(conn, timestamp, value, current_state)

        if tick <= WARMUP_POINTS:
            update_model_status(conn, "warming_up", tick)
        elif tick == WARMUP_POINTS + 1:
            update_model_status(conn, "ready", tick)
            print("\n[GENERATOR] Warmup complete — detector can now train.\n")

        # ── Batch commit every N ticks ────────────────────────
        if tick % COMMIT_EVERY_N == 0:
            conn.commit()

        # ── Logging ──────────────────────────────────────────
        if tick <= WARMUP_POINTS and tick % 20 == 0:
            pct = round((tick / WARMUP_POINTS) * 100, 1)
            print(f"[WARMUP] {tick}/{WARMUP_POINTS} ({pct}%)")

        print(f"[Tick {tick:05d}] state={current_state:<8} value={value:>8.4f}  @ {timestamp}")

        time.sleep(TICK_INTERVAL)

# ─── Entry Point ──────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    try:
        init_db(conn)
        generate_stream(conn)
    except KeyboardInterrupt:
        # Flush any uncommitted batch before exit
        conn.commit()
        print("\n[GENERATOR] Final commit done. Shutting down cleanly.")
    finally:
        conn.close()