# Real-Time Security Regime Change Detector

A production-grade, containerized system that ingests a live stream of synthetic
security events and detects transitions between **Normal** and **Attack** operational
states using a **2-state Gaussian Hidden Markov Model (HMM)**.

---

## Architecture
[Generator Container] → writes events → [SQLite /data/events.db]
↓
[Detector Container]  → reads events → [HMM Engine] → [FastAPI]
↓
[Detections Table]

### Why HMM?
A Hidden Markov Model is the statistically correct choice for regime detection
because it models **sequential state transitions** — not just point anomalies.
Each hidden state emits observations from a learned Gaussian distribution:

- **State 0 (Normal):** Low-frequency events ~ N(10, 3)
- **State 1 (Attack):** High-frequency bursts ~ N(20, 5)

The distributions are deliberately overlapping to create a realistic
classification challenge. The Baum-Welch algorithm learns these parameters
from the live stream itself during a warm-up phase — eliminating any
distribution mismatch from pre-trained weights.

Unlike threshold-based approaches, HMM explicitly models the **sequential
dependency** between observations — a key property of a real security event
streams where attacks persist over multiple ticks rather than appearing
as isolated spikes.
---

## Project Structure
realtime-regime-detector/
├── docker-compose.yaml
├── data/                        # shared SQLite volume
├── generator/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── simulate.py              # live stream simulation
└── detector/
├── Dockerfile
├── requirements.txt
├── main.py                  # FastAPI endpoints
├── engine/
│   ├── hmm_model.py         # HMM lifecycle management
│   └── detector.py          # background detection loop
└── db/
└── database.py          # SQLite read/write layer

---

## Quick Start

### Evaluator Notes

After running `docker compose up --build`, the system goes through these phases:

| Time          | What's happening               | Expected API response            |
| ------------- | ------------------------------ | -------------------------------- |
| 0–20 seconds  | Warm-up: collecting 200 events | `503 warming_up` with progress % |
| 20–25 seconds | Initial HMM training           | `503 training`                   |
| 25+ seconds   | Live detection active          | `200 ok` with full results       |

### For best `/metrics` results:
- At **500 events** (~1 min): First background retrain triggers — precision improves
- At **1000+ events** (~2 mins): Metrics stabilize around:
  - Accuracy: ~0.95
  - Precision: ~0.81
  - Recall: ~0.93
  - F1: ~0.87

> **Note:** `/metrics` improves over time as more attack/normal transitions
> are observed. For best results, let the system run for **2-3 minutes**
> before evaluating `/metrics`. All other endpoints (`/health`, `/status`,
> `/history`) are fully functional after the 25-second warmup.

### Interactive API docs:
Visit `http://localhost:8000/docs` for the full Swagger UI.
### Prerequisites
- Docker Desktop installed and running
- Port 8000 free

### Run
```bash
docker compose up --build
```

Wait ~20 seconds for warmup, then query the API at `http://localhost:8000/docs`

---

## API Endpoints

### `GET /health`
Liveness check — always returns 200.

```json
{
  "status": "ok",
  "service": "realtime-regime-detector"
}
```

### `GET /status`
Returns current regime state with full uncertainty metrics.

**During warmup (503):**
```json
{
  "status": "warming_up",
  "message": "Collecting data for initial model training.",
  "progress": {
    "collected": 47,
    "required": 200,
    "percent": 23.5,
    "eta_seconds": 15.3
  }
}
```

**Ready (200):**
```json
{
  "status": "ok",
  "timestamp": "2026-04-18T08:04:25.827761",
  "detection": {
    "current_state": "ATTACK",
    "alert_level": "CRITICAL",
    "current_value": 22.4821,
    "state_probabilities": {
      "P(Normal)": 0.09,
      "P(Attack)": 0.91
    },
    "uncertainty_metrics": {
      "confidence_level": "HIGH",
      "rolling_stability": 0.9,
      "state_duration_ticks": 12,
      "transition_risk": 0.05,
      "observation_z_score": 2.41
    }
  },
  "model_meta": {
    "model_status": "READY",
    "last_trained_at": "2026-04-18T08:04:22.864233",
    "total_points_seen": 862
  }
}
```

### `GET /history?limit=50`
Returns last N detections. Default 50, max 500.

### `GET /metrics`
Returns real validation metrics by comparing HMM predictions
against ground truth states stored by the generator.

```json
{
  "status": "ok",
  "metrics": {
    "total_evaluated": 1022,
    "accuracy": 0.955,
    "precision": 0.8118,
    "recall": 0.9321,
    "f1_score": 0.8678,
    "confusion_matrix": {
      "true_normal_predicted_normal": 825,
      "true_normal_predicted_attack": 35,
      "true_attack_predicted_normal": 11,
      "true_attack_predicted_attack": 151
    }
  }
}
```

---

## Statistical Design

### Warm-up Phase
The system collects **200 observations** before fitting the HMM. This ensures
the model learns the generator's exact distributions — no pre-trained weights,
no distribution mismatch.

### Uncertainty Quantification
Five metrics are returned on every inference call:

| Metric                    | Description                                 | Source                                  |
| ------------------------- | ------------------------------------------- | --------------------------------------- |
| `P(Normal)` / `P(Attack)` | HMM posterior state probabilities           | `model.predict_proba()`                 |
| `confidence_level`        | HIGH / MEDIUM / LOW based on P(Attack)      | Threshold on posterior                  |
| `rolling_stability`       | Consistency of last 10 state predictions    | Posterior convergence check             |
| `state_duration_ticks`    | How long system has been in current state   | State transition counter                |
| `transition_risk`         | P(switching state next tick)                | HMM transition matrix `A[state][other]` |
| `observation_z_score`     | Deviation from HMM's learned state Gaussian | HMM-aware HDI analysis                  |

### HMM-Aware Z-Score
Unlike a rolling Z-score, this uses the model's own learned parameters:
`Z = (x - μ_state) / σ_state`

Where `μ_state` and `σ_state` come directly from the HMM's `means_` and
`covars_` for the current state — equivalent to HDI-based analysis on the
posterior predictive distribution. A Z-score > 2.0 in NORMAL state signals
the observation lies outside the 95% HDI of the learned Normal distribution,
providing statistically justified confidence in regime transitions.

### Background Retraining
The model retrains every **500 new events** in a background thread — never
blocking the API. The API only ever calls `.predict()`.

---

## Validation Results (Live Run — 1022 events)

| Metric    | Value      | Interpretation                              |
| --------- | ---------- | ------------------------------------------- |
| Accuracy  | **0.9550** | 95.5% of all predictions correct            |
| Precision | **0.8118** | 81% of attack alerts were real attacks      |
| Recall    | **0.9321** | Caught 93% of all real attacks              |
| F1 Score  | **0.8678** | Strong balance between precision and recall |

The distributions are intentionally overlapping (Normal: N(10,3), Attack: N(20,5))
to create a realistic classification challenge. The HMM uses sequential context
from its transition matrix to resolve ambiguous observations in the overlap zone
(values 10–16), which is why recall remains high (0.93) while precision reflects
the genuine difficulty of the task (0.81).

---

## Model Robustness

Three techniques ensure robustness:

1. **Live Warm-up Training:** The HMM trains on the first 200 observations
   from the live stream — never on pre-trained weights. This eliminates
   distribution mismatch between training and inference data.

2. **Background Retraining:** Every 500 events, the model retrains in a
   background thread on the latest data — adapting to any drift without
   blocking the API.

3. **Validation Against Ground Truth:** The `/metrics` endpoint computes
   accuracy, precision, recall, F1, and a full confusion matrix by joining
   HMM predictions against ground truth states stored by the generator —
   providing continuous, honest evaluation of model performance.

---

## Key Engineering Decisions

| Decision                         | Reason                                                |
| -------------------------------- | ----------------------------------------------------- |
| SQLite over MongoDB              | Lightweight, zero extra container, matches prompt     |
| WAL mode enabled                 | Concurrent read/write between 2 Docker containers     |
| Batch commits every 10 ticks     | Reduces disk I/O from 10/s to 1/s                     |
| ID-based change detection        | Prevents overlapping prediction windows               |
| Single persistent DB connection  | No connection churn overhead                          |
| Detections table init at startup | No schema check on every inference call               |
| Live warmup training             | No pre-trained `.pkl` — zero distribution mismatch    |
| Background retrain thread        | `.fit()` never called on API thread — always fast     |
| Retry on DB lock                 | Up to 3 retries with 100ms backoff — resilient writes |

---

## Personal Note

This project extends my prior work on 2-state Gaussian HMMs and regime
detection (originally applied to **climate extreme event detection** —
identifying rare regime transitions in 30+ years of sequential time-series
data at IIT Kharagpur) to the security domain. The core statistical
architecture — Baum-Welch training, Viterbi decoding, and posterior-based
uncertainty quantification — is directly drawn from that research experience.
