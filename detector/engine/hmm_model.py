import numpy as np
import threading
from hmmlearn import hmm
from datetime import datetime
from typing import Optional

# ─── Config ───────────────────────────────────────────────────
WARMUP_POINTS  = 200
RETRAIN_EVERY  = 500
PREDICT_WINDOW = 30

# ─── HMM Detector ─────────────────────────────────────────────
class HMMDetector:
    """
    Stateless regarding data storage.
    SQLite (via database.py) is the single source of truth.
    This class owns only the model lifecycle and inference logic.
    """

    def __init__(self):
        self.model: Optional[hmm.GaussianHMM] = None
        self.is_ready          = False
        self.is_training       = False
        self.total_points_seen = 0
        self.last_trained_at   = None
        self.last_state        = None
        self.state_duration    = 0

        # Resolved after first fit
        self._normal_state_idx = 0
        self._attack_state_idx = 1

        # Thread safety
        self._lock = threading.Lock()

    # ─── Training ─────────────────────────────────────────────
    def _build_model(self) -> hmm.GaussianHMM:
        return hmm.GaussianHMM(
            n_components=2,
            covariance_type="diag",
            n_iter=100,
            tol=1e-4,
            random_state=42
        )

    def _fit(self, data: np.ndarray):
        """
        Runs Baum-Welch on provided data snapshot.
        Always called in background thread — never blocks API.
        """
        try:
            model = self._build_model()
            X = data.reshape(-1, 1)
            model.fit(X)

            # Resolve Normal vs Attack by learned mean
            # Normal state always has the lower mean
            means = model.means_.flatten()
            normal_idx = int(np.argmin(means))
            attack_idx = int(np.argmax(means))

            with self._lock:
                self.model              = model
                self._normal_state_idx  = normal_idx
                self._attack_state_idx  = attack_idx
                self.is_ready           = True
                self.last_trained_at    = datetime.utcnow().isoformat()

            print(
                f"[HMM] Model trained. "
                f"Normal=state{normal_idx} (μ={means[normal_idx]:.2f}) | "
                f"Attack=state{attack_idx} (μ={means[attack_idx]:.2f})"
            )

        except Exception as e:
            print(f"[HMM] Training failed: {e}")
        finally:
            self.is_training = False

    def initial_fit(self, data: np.ndarray):
        """
        Called once after warmup completes.
        Receives data snapshot directly from SQLite fetch.
        """
        if len(data) < WARMUP_POINTS:
            print(f"[HMM] Not enough data: {len(data)}/{WARMUP_POINTS}")
            return

        print(f"[HMM] Starting initial fit on {len(data)} points...")
        self.is_training = True
        t = threading.Thread(
            target=self._fit,
            args=(data.copy(),),
            daemon=True
        )
        t.start()

    def retrain(self, data: np.ndarray):
        """
        Periodic background retrain.
        Receives fresh data snapshot from SQLite.
        """
        if self.is_training:
            return  # skip if already training
        print(f"[HMM] Background retrain triggered at {self.total_points_seen} points.")
        self.is_training = True
        t = threading.Thread(
            target=self._fit,
            args=(data.copy(),),
            daemon=True
        )
        t.start()

    # ─── Prediction ───────────────────────────────────────────
    def predict(self, recent_values: list) -> Optional[dict]:
        """
        Pure inference — no data storage, no .fit().
        Receives recent values fetched directly from SQLite.
        Returns full uncertainty metrics dict.
        """
        with self._lock:
            if not self.is_ready or self.model is None:
                return None
            if len(recent_values) < 2:
                return None

            try:
                window = np.array(
                    recent_values[-PREDICT_WINDOW:]
                ).reshape(-1, 1)

                # ── Core HMM Inference ────────────────────────
                state_sequence = self.model.predict(window)
                state_probs    = self.model.predict_proba(window)

                current_raw_state = int(state_sequence[-1])
                current_probs     = state_probs[-1]

                p_normal = float(current_probs[self._normal_state_idx])
                p_attack = float(current_probs[self._attack_state_idx])

                current_state = (
                    "ATTACK"
                    if current_raw_state == self._attack_state_idx
                    else "NORMAL"
                )

                # ── Metric 1: Rolling Stability ───────────────
                last_10        = state_sequence[-10:]
                rolling_stability = float(
                    np.mean(last_10 == current_raw_state)
                )

                # ── Metric 2: State Duration ──────────────────
                if current_state == self.last_state:
                    self.state_duration += 1
                else:
                    self.state_duration = 1
                    self.last_state     = current_state

                # ── Metric 3: Transition Risk ─────────────────
                # Read directly from HMM transition matrix
                other_idx = (
                    self._attack_state_idx
                    if current_state == "NORMAL"
                    else self._normal_state_idx
                )
                transition_risk = float(
                    self.model.transmat_[current_raw_state, other_idx]
                )

                # ── Metric 4: HMM-Aware Z-Score ───────────────
                # Uses model's learned Gaussian for current state
                # NOT a rolling statistic — proves HMM understanding
                current_value = float(recent_values[-1])
                state_mean    = float(
                    self.model.means_[current_raw_state][0]
                )
                state_std     = float(np.sqrt(
                    self.model.covars_[current_raw_state][0][0]
                ))
                z_score = (
                    round((current_value - state_mean) / state_std, 4)
                    if state_std > 0 else 0.0
                )

                # ── Metric 5: Confidence Level ────────────────
                if p_attack >= 0.90:
                    confidence_level = "HIGH"
                elif p_attack >= 0.70:
                    confidence_level = "MEDIUM"
                else:
                    confidence_level = "LOW"

                # ── Alert Level ───────────────────────────────
                if current_state == "ATTACK":
                    alert_level = "CRITICAL" if p_attack >= 0.90 else "WARNING"
                else:
                    alert_level = "NORMAL"

                self.total_points_seen += 1

                return {
                    "current_state": current_state,
                    "alert_level":   alert_level,
                    "current_value": round(current_value, 4),
                    "state_probabilities": {
                        "P(Normal)": round(p_normal, 4),
                        "P(Attack)": round(p_attack, 4)
                    },
                    "uncertainty_metrics": {
                        "confidence_level":     confidence_level,
                        "rolling_stability":    round(rolling_stability, 4),
                        "state_duration_ticks": self.state_duration,
                        "transition_risk":      round(transition_risk, 4),
                        "observation_z_score":  z_score
                    },
                    "model_meta": {
                        "model_status":      "READY",
                        "last_trained_at":   self.last_trained_at,
                        "total_points_seen": self.total_points_seen
                    }
                }

            except Exception as e:
                print(f"[HMM] Prediction failed: {e}")
                return None

    # ─── Warmup Progress ──────────────────────────────────────
    def get_warmup_progress(self, current_count: int) -> dict:
        percent = round((current_count / WARMUP_POINTS) * 100, 1)
        eta     = round((WARMUP_POINTS - current_count) * 0.1, 1)
        return {
            "collected":   current_count,
            "required":    WARMUP_POINTS,
            "percent":     min(percent, 100.0),
            "eta_seconds": max(0.0, eta)
        }


# ─── Singleton ────────────────────────────────────────────────
detector = HMMDetector()