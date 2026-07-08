"""
session.py
BCI Session Controller — Mentalab Explore (8ch, 500 Hz)
Pipeline:
    Mentalab (LSL) -> rASR -> covariance(C_t)
        -> [Global Riemannian EMA: R_t = Exp_R(alpha * Log_R(C_t)), alpha=0.005]
        -> [RPA Alignment: C_t_aligned = R_t^(-1/2) * C_t * R_t^(-1/2)]
        -> [MDM Prediction + Smooth Geodesic Adaptation]
        -> cursor
Centroid adaptation (Smooth Geodesic, Zanini 2018):
    - confidence > ADAPT_CONFIDENCE -> buffer n=BUFFER_SIZE
    - delta(mu_buf, mu_k) > ADAPT_THRESHOLD -> mu_k <- Exp_mu(alpha * Log_mu(mu_buf)), alpha=ALPHA_CENTROID
R updates unconditionally every trial — tracks global hardware covariance drift.
Adaptation buffers are seeded from calibration so the buffer Frechet mean is
well-formed from the first trial instead of building up from one or two samples.
Classes:
    0: UP    (up arrow)   -- imagine RIGHT FOOT moving upward
    1: DOWN  (down arrow) -- imagine TONGUE movement
    2: LEFT  (left arrow) -- imagine LEFT HAND clenching
    3: RIGHT (right arrow)-- imagine RIGHT HAND clenching
Usage:
    python session.py --mode calibrate   # first session
    python session.py --mode run         # real-time classification
    python session.py --mode test        # synthetic data simulation
    python session.py --mode recalibrate_r  # re-estimate R from rest
"""
from collections import deque
import argparse
import logging
import pickle
import queue
import threading
import time
from pathlib import Path
import numpy as np
#LSL
try:
    from pylsl import StreamInlet, resolve_byprop
    LSL_AVAILABLE = True
except ImportError:
    LSL_AVAILABLE = False
    logging.warning("pylsl not installed — only 'test' mode available")
# Cursor
try:
    import pyautogui
    pyautogui.FAILSAFE = True
    CURSOR_AVAILABLE = True
except ImportError:
    CURSOR_AVAILABLE = False
    logging.warning("pyautogui not installed — cursor movement disabled")
#Local modules
from ASR import asr_calibrate, asr_stream
from AMDM import (
    MDM, covariance, rpa_recenter,
    riemannian_mean, riemannian_ema, dist_riemann
)
# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
# Hardware constants
SRATE             = 500                 # Mentalab Explore sampling rate (Hz)
CHANNEL_SELECTION = [3, 4, 5, 6, 7]    # C3, Cz, C4, CP3, CP4 (0-indexed from LSL)
N_CHANNELS        = 5
WINDOW_SEC        = 2.0                 # MI trial window length (s)
N_SAMPLES         = int(WINDOW_SEC * SRATE)
REST_SEC          = 1.5
N_REST            = int(REST_SEC * SRATE)
CAL_SEC           = 120                 # rASR calibration duration (s)
ITI_SEC           = 2.5                 # inter-trial interval (s)
WARMUP_TRIALS     = 5                   # warmup trials discarded before calibration
TRIALS_PER_CLASS  = 30
CURSOR_STEP       = 80
CONFIDENCE_THRESH = 0.60
ALPHA_R          = 0.005
ALPHA_CENTROID   = 0.02
BUFFER_SIZE      = 50
ADAPT_THRESHOLD  = 0.05
ADAPT_CONFIDENCE = 0.70
RAW_DATA_PATH = Path("raw_session.npz")
MODEL_PATH    = Path("bci_model.pkl")
CLASS_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
CHANNEL_NAMES = ["Fz", "FC3", "FC4", "C3", "Cz", "C4", "CP3", "CP4"]
MI_CHANNELS   = [3, 4, 5]  # C3, Cz, C4
CLASS_INSTRUCTIONS = {
    0: "Imagine your RIGHT FOOT moving upward (like a kick)",
    1: "Imagine your TONGUE moving (like licking)",
    2: "Imagine clenching your LEFT HAND",
    3: "Imagine clenching your RIGHT HAND",
}
# Helpers
def print_cue(text: str, width: int = 50) -> None:
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)
def print_fixation() -> None:
    print("  +", end="\r", flush=True)
def estimate_R(rest_covs) -> np.ndarray:
    covs = np.array(rest_covs)
    norms = [np.linalg.norm(c, "fro") for c in covs]
    median_norm = np.median(norms)
    clean = [c for c, n in zip(covs, norms) if n < 3 * median_norm]
    logger.info(f"estimate_R: {len(clean)}/{len(covs)} windows after outlier rejection")
    return riemannian_mean(np.array(clean))
# predict_proba — Softmax over Riemannian distances
def mdm_predict_proba(mdm: MDM, cov: np.ndarray) -> dict:
    beta  = 2.0
    dists = {c: dist_riemann(cov, mdm.centroids_[c]) for c in mdm.classes_}
    exps  = {c: np.exp(-beta * d) for c, d in dists.items()}
    total = sum(exps.values())
    return {c: v / total for c, v in exps.items()}
MDM.predict_proba = lambda self, cov: mdm_predict_proba(self, cov)
# ASR stats
class ASRStats:
    def __init__(self):
        self._total     = 0
        self._artifacts = 0
    def update(self, artifact_log: list) -> None:
        for _, is_artifact in artifact_log:
            self._total += 1
            if is_artifact:
                self._artifacts += 1
    @property
    def artifact_pct(self) -> float:
        return 0.0 if self._total == 0 else 100.0 * self._artifacts / self._total
    def log(self) -> None:
        pct    = self.artifact_pct
        status = "OK" if pct < 30 else ("WARNING" if pct < 60 else "BAD")
        logger.info(f"[ASR] artifacts: {pct:.1f}% ({self._artifacts}/{self._total}) — {status}")
        if pct > 60:
            logger.warning("[ASR] Over 60% artifacts — check electrode contact!")
# Cursor
def move_cursor(label: int, confidence: float) -> bool:
    if confidence < CONFIDENCE_THRESH:
        logger.info(
            f"[CURSOR] {CLASS_NAMES[label]} — low confidence "
            f"({confidence*100:.0f}% < {CONFIDENCE_THRESH*100:.0f}%) — skipping"
        )
        return False
    if not CURSOR_AVAILABLE:
        logger.info(f"[CURSOR] -> {CLASS_NAMES[label]} ({confidence*100:.0f}%)")
        return True
    dx, dy = 0, 0
    if label == 0:   dy = -CURSOR_STEP
    elif label == 1: dy =  CURSOR_STEP
    elif label == 2: dx = -CURSOR_STEP
    elif label == 3: dx =  CURSOR_STEP
    try:
        x, y = pyautogui.position()
        pyautogui.moveTo(x + dx, y + dy, duration=0.1)
        logger.info(f"[CURSOR] {CLASS_NAMES[label]} ({confidence*100:.0f}%)")
        return True
    except pyautogui.FailSafeException:
        logger.warning("[CURSOR] Fail-safe triggered — cursor at screen corner")
        return False
# LSL Reader — Thread 1
class LSLReader(threading.Thread):
    def __init__(self, sample_queue: queue.Queue, window_size: int = N_SAMPLES):
        super().__init__(daemon=True)
        self.sample_queue = sample_queue
        self.window_size  = window_size
        self._stop_event  = threading.Event()
        self._buffer: list = []
    def run(self) -> None:
        if not LSL_AVAILABLE:
            logger.error("pylsl not available")
            return
        logger.info("Searching for LSL stream (type: ExG)...")
        streams = resolve_byprop("type", "ExG", timeout=10)
        if not streams:
            streams = resolve_byprop("name", "Explore", timeout=5)
        if not streams:
            logger.error("No LSL stream found.")
            return
        inlet = StreamInlet(streams[0])
        logger.info("Connected to LSL stream")
        while not self._stop_event.is_set():
            chunk, _ = inlet.pull_chunk(timeout=0.1, max_samples=32)
            if chunk:
                for sample in chunk:
                    self._buffer.append([sample[i] for i in CHANNEL_SELECTION])
                while len(self._buffer) >= self.window_size:
                    window = np.array(self._buffer[:self.window_size]).T
                    self._buffer = self._buffer[self.window_size:]
                    try:
                        self.sample_queue.put_nowait(window)
                    except queue.Full:
                        self.sample_queue.get_nowait()
                        self.sample_queue.put_nowait(window)
    def stop(self) -> None:
        self._stop_event.set()
# Preprocessing Worker — Thread 2
class PreprocessWorker(threading.Thread):
    """
    Applies rASR artifact removal to each EEG window and computes the
    sample covariance matrix C_t = (X - mean) (X - mean)^T / (T-1).
    Passes C_t to the classifier queue.
    """
    def __init__(self, raw_queue: queue.Queue, clean_queue: queue.Queue, asr_state: dict):
        super().__init__(daemon=True)
        self.raw_queue   = raw_queue
        self.clean_queue = clean_queue
        self.asr_state   = asr_state
        self._stop_event = threading.Event()
        self._asr_stats  = ASRStats()
        self._n_windows  = 0
    def run(self) -> None:
        logger.info("PreprocessWorker started")
        while not self._stop_event.is_set():
            try:
                window = self.raw_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                eeg_clean, artifact_log, self.asr_state = asr_stream(
                    window, self.asr_state, srate=SRATE
                )
                self._asr_stats.update(artifact_log)
                self._n_windows += 1
                if self._n_windows % 20 == 0:
                    self._asr_stats.log()
                cov = covariance(eeg_clean)
                self.clean_queue.put_nowait(cov)
            except Exception as e:
                logger.warning(f"PreprocessWorker error: {e}")
    def stop(self) -> None:
        self._stop_event.set()
        self._asr_stats.log()
# MDM Classifier Worker — Thread 3
#
#   C_t  (raw covariance from PreprocessWorker)
#    |
#    +-> STEP 1: R_t = riemannian_ema(R_{t-1}, C_t, alpha=ALPHA_R)
#    |           Unconditional, every trial.
#    |           Tracks global hardware covariance drift (electrode impedance,
#    |           perspiration, fatigue).
#    |
#    +-> STEP 2: C_t_aligned = rpa_recenter(C_t, R_t)
#    |           C_t_aligned = R_t^{-1/2} * C_t * R_t^{-1/2}  (Zanini 2018 eq.9)
#    |           Maps the data barycenter to the identity matrix I.
#    |           Affine invariance of AIM distance is preserved.
#    |
#    +-> STEP 3: proba = MDM.predict_proba(C_t_aligned)
#    |           Softmax(-beta * delta_AIM) over class centroids, beta=2.0.
#    |           k* = argmax proba
#    |
#    +-> STEP 4: move_cursor(k*, confidence)
#    |           Only moves if confidence > CONFIDENCE_THRESH.
#    |
#    +-> STEP 5: Smooth Geodesic Centroid Adaptation
#                If confidence > ADAPT_CONFIDENCE:
#                    buffer_k.append(C_t_aligned)   [FIFO n=BUFFER_SIZE, seeded from calibration]
#                    If len(buffer) >= BUFFER_SIZE:
#                        mu_buf = Karcher(buffer_k)
#                        If delta(mu_buf, mu_k) > ADAPT_THRESHOLD:
#                            mu_k <- Exp_mu(ALPHA_CENTROID * Log_mu(mu_buf))
#                            Smooth geodesic step — no discrete jumps.
#
class ClassifierWorker(threading.Thread):
    def __init__(
        self,
        clean_queue: queue.Queue,
        mdm:         MDM,
        R:           np.ndarray,
        cal_buffers: dict = None,
    ):
        super().__init__(daemon=True)
        self.clean_queue = clean_queue
        self.mdm         = mdm
        self.R           = R.copy()   # local copy — do not modify the model's R
        self.buffers = {cls: deque(maxlen=BUFFER_SIZE) for cls in range(4)}
        if cal_buffers:
            for cls, covs in cal_buffers.items():
                self.buffers[int(cls)].extend(covs)   # seed from calibration
        self._stop_event   = threading.Event()
        self._n_classified = 0
        self._n_moved      = 0
        self._n_adapted    = 0
    def run(self) -> None:
        logger.info(
            f"ClassifierWorker started | "
            f"alpha_R={ALPHA_R} alpha_centroid={ALPHA_CENTROID} "
            f"buffer={BUFFER_SIZE} threshold={ADAPT_THRESHOLD} "
            f"adapt_conf>{ADAPT_CONFIDENCE} cursor_conf>{CONFIDENCE_THRESH}"
        )
        while not self._stop_event.is_set():
            try:
                cov = self.clean_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self.R = riemannian_ema(self.R, cov, alpha=ALPHA_R)
                cov_aligned = rpa_recenter(cov, self.R)
                proba      = self.mdm.predict_proba(cov_aligned)
                label      = max(proba, key=proba.get)
                confidence = proba[label]
                moved = move_cursor(int(label), confidence)
                if moved:
                    self._n_moved += 1
                if confidence > ADAPT_CONFIDENCE:
                    self.buffers[int(label)].append(cov_aligned)
                    if len(self.buffers[int(label)]) >= BUFFER_SIZE:
                        buf_mean = riemannian_mean(list(self.buffers[int(label)]))
                        delta    = dist_riemann(self.mdm.centroids_[int(label)], buf_mean)
                        if delta > ADAPT_THRESHOLD:
                            self.mdm.update_centroid(
                                int(label), buf_mean, alpha=ALPHA_CENTROID
                            )
                            self._n_adapted += 1
                            logger.info(
                                f"[ADAPT] {CLASS_NAMES[int(label)]} "
                                f"delta={delta:.3f} -> geodesic update alpha={ALPHA_CENTROID}"
                            )
                self._n_classified += 1
                if self._n_classified % 10 == 0:
                    logger.info(
                        f"[MDM] classified={self._n_classified} "
                        f"moved={self._n_moved} "
                        f"adapted={self._n_adapted} "
                        f"ratio={self._n_moved/self._n_classified*100:.0f}%"
                    )
            except Exception as e:
                logger.warning(f"ClassifierWorker error: {e}")
    def stop(self) -> None:
        self._stop_event.set()
    def get_n_classified(self) -> int:
        return self._n_classified
# LSL data collection
def _collect_eeg_fresh(inlet, seconds: float) -> np.ndarray:
    """Flush LSL buffer then collect fresh samples."""
    inlet.flush()
    while True:
        chunk, _ = inlet.pull_chunk(timeout=2.0, max_samples=10)
        if chunk and len(chunk) > 0:
            break
    return _collect_eeg(inlet, seconds)
def _collect_eeg(inlet, seconds: float) -> np.ndarray:
    n_samples = int(seconds * SRATE)
    buffer: list = []
    while len(buffer) < n_samples:
        chunk, _ = inlet.pull_chunk(timeout=1.0, max_samples=128)
        if chunk:
            for sample in chunk:
                buffer.append([sample[i] for i in CHANNEL_SELECTION])
    return np.array(buffer[:n_samples]).T
# Calibration protocol
def run_calibration(inlet=None) -> dict:
    if inlet is None:
        return _calibrate_synthetic()
    return _calibrate_mentalab(inlet)
def _calibrate_synthetic() -> dict:
    """Synthetic calibration for offline testing without hardware."""
    logger.info("=== SYNTHETIC CALIBRATION (no rASR) ===")
    rng = np.random.default_rng(42)
    eeg_cal = rng.standard_normal((N_CHANNELS, SRATE * 60))
    eeg_cal *= np.random.default_rng(1).uniform(0.5, 2.0, (N_CHANNELS, 1))
    try:
        asr_state = asr_calibrate(eeg_cal, srate=SRATE)
    except RuntimeError:
        logger.warning("rASR calibration failed on synthetic data — skipping")
        asr_state = None
    scales = [1.0, 3.0, 6.0, 10.0]
    covs, labels = [], []
    for cls, scale in enumerate(scales):
        for _ in range(TRIALS_PER_CLASS):
            X = rng.standard_normal((N_CHANNELS, N_SAMPLES)) * scale
            if asr_state is not None:
                try:
                    X, _, asr_state = asr_stream(X, asr_state, srate=SRATE)
                except Exception:
                    pass
            covs.append(covariance(X))
            labels.append(cls)
    covs   = np.array(covs)
    labels = np.array(labels)
    R      = riemannian_mean(covs)
    covs_a = rpa_recenter(covs, R)
    mdm    = MDM()
    mdm.fit(covs_a, labels)
    # Seed adaptation buffers from the aligned calibration covariances
    cal_buffers = {
        int(c): list(covs_a[labels == c][-BUFFER_SIZE:])
        for c in np.unique(labels)
    }
    logger.info(f"MDM trained. sigmas: { {c: f'{s:.4f}' for c,s in mdm.sigmas_.items()} }")
    return {"mdm": mdm, "R": R, "asr_state": asr_state, "cal_buffers": cal_buffers}
def _calibrate_mentalab(inlet) -> dict:
    """Full calibration protocol with Mentalab hardware and rASR."""
    logger.info("=== rASR CALIBRATION ===")
    print_cue("Sit still for 2 minutes. Do not move.")
    eeg_cal   = _collect_eeg(inlet, seconds=CAL_SEC)
    asr_state = asr_calibrate(eeg_cal, srate=SRATE)
    asr_stats = ASRStats()
    logger.info("rASR calibrated")
    raw_data:   list = []
    covs_all:   list = []
    labels_all: list = []
    for cls, name in CLASS_NAMES.items():
        print_cue(f"Class {cls}: {name}")
        print(f"  {CLASS_INSTRUCTIONS[cls]}")
        print(f"  Press Enter when ready...")
        input()
        print(f"\n  Warmup ({WARMUP_TRIALS} trials — discarded)...")
        for _ in range(WARMUP_TRIALS):
            print_fixation()
            time.sleep(ITI_SEC)
            print(f"  >> IMAGINE: {name} <<", flush=True)
            _collect_eeg_fresh(inlet, seconds=WINDOW_SEC)
            time.sleep(0.3)
        print(f"\n  Calibration ({TRIALS_PER_CLASS} trials)...")
        for trial in range(TRIALS_PER_CLASS):
            print_fixation()
            time.sleep(ITI_SEC)
            print(f"  Trial {trial+1}/{TRIALS_PER_CLASS} — {name}", flush=True)
            eeg_trial = _collect_eeg_fresh(inlet, seconds=WINDOW_SEC)
            raw_data.append(eeg_trial)
            clean, artifact_log, asr_state = asr_stream(eeg_trial, asr_state, srate=SRATE)
            asr_stats.update(artifact_log)
            covs_all.append(covariance(clean))
            time.sleep(0.1)
        asr_stats.log()
        labels_all.extend([cls] * TRIALS_PER_CLASS)
        logger.info(f"Class {name} done ({TRIALS_PER_CLASS} trials)")
    covs   = np.array(covs_all)
    labels = np.array(labels_all)
    R          = riemannian_mean(covs)
    covs_algnd = rpa_recenter(covs, R)
    mdm        = MDM()
    mdm.fit(covs_algnd, labels)
    # Seed adaptation buffers from the aligned calibration covariances, per class,
    # so the buffer Frechet mean is well-formed from the first online trial.
    cal_buffers = {
        int(cls): list(covs_algnd[labels == cls][-BUFFER_SIZE:])
        for cls in CLASS_NAMES
    }
    logger.info(f"MDM trained. sigmas: { {c: f'{s:.4f}' for c,s in mdm.sigmas_.items()} }")
    if raw_data:
        np.savez(RAW_DATA_PATH, eeg=np.array(raw_data), labels=labels)
        logger.info(f"Raw data saved -> {RAW_DATA_PATH}")
    return {"mdm": mdm, "R": R, "asr_state": asr_state, "cal_buffers": cal_buffers}
# Main session loop
def run_session(model: dict) -> None:
    mdm       = model["mdm"]
    R         = model["R"]
    asr_state = model["asr_state"]
    raw_queue   = queue.Queue(maxsize=50)
    clean_queue = queue.Queue(maxsize=50)
    lsl_reader = LSLReader(raw_queue)
    preprocess = PreprocessWorker(raw_queue, clean_queue, asr_state)
    classifier = ClassifierWorker(clean_queue, mdm, R, model.get("cal_buffers"))
    logger.info(f"Electrode mapping: {dict(enumerate(CHANNEL_NAMES))}")
    logger.info(f"MI channels: {[(i, CHANNEL_NAMES[i]) for i in MI_CHANNELS]}")
    logger.info("Starting pipeline...")
    lsl_reader.start()
    preprocess.start()
    classifier.start()
    logger.info(f"Pipeline running. Confidence threshold: {CONFIDENCE_THRESH*100:.0f}%")
    logger.info("Ctrl+C to stop.")
    try:
        while True:
            time.sleep(5)
            logger.info(
                f"Status: raw_q={raw_queue.qsize()} "
                f"clean_q={clean_queue.qsize()} "
                f"classified={classifier.get_n_classified()}"
            )
    except KeyboardInterrupt:
        logger.info("Stopping pipeline...")
    finally:
        lsl_reader.stop()
        preprocess.stop()
        classifier.stop()
        lsl_reader.join(timeout=2)
        preprocess.join(timeout=2)
        classifier.join(timeout=2)
        logger.info("Pipeline stopped.")
# Test mode
def run_test() -> None:
    """
    Offline simulation using synthetic data. Runs the full pipeline
    (EMA R update -> RPA -> MDM) to verify end-to-end correctness.
    """
    pyautogui.moveTo(960, 540)
    logger.info("=== TEST MODE ===")
    model = run_calibration(inlet=None)
    mdm   = model["mdm"]
    R     = model["R"]
    rng    = np.random.default_rng(0)
    scales = {0: 1.0, 1: 3.0, 2: 6.0, 3: 10.0}
    correct, n_moved, n_trials = 0, 0, 40
    logger.info(f"Simulating {n_trials} trials (threshold={CONFIDENCE_THRESH*100:.0f}%)...")
    R_test = R.copy()
    for i in range(n_trials):
        true_label = i % 4
        X          = rng.standard_normal((N_CHANNELS, N_SAMPLES)) * scales[true_label]
        cov        = covariance(X)
        # Mirror the full run pipeline
        R_test      = riemannian_ema(R_test, cov, alpha=ALPHA_R)
        cov_aligned = rpa_recenter(cov, R_test)
        proba      = mdm.predict_proba(cov_aligned)
        label      = max(proba, key=proba.get)
        confidence = proba[label]
        if label == true_label:
            correct += 1
        moved = move_cursor(int(label), confidence)
        if moved:
            n_moved += 1
        time.sleep(0.05)
    acc = correct / n_trials
    logger.info(f"\nAcc: {acc*100:.1f}% | Cursor moved: {n_moved}/{n_trials}")
# CLI
def main() -> None:
    parser = argparse.ArgumentParser(description="BCI Session Controller")
    parser.add_argument(
        "--mode",
        choices=["calibrate", "run", "test", "recalibrate_r"],
        default="test",
    )
    parser.add_argument("--model", default=str(MODEL_PATH))
    args = parser.parse_args()
    if args.mode == "test":
        run_test()
        return
    if not LSL_AVAILABLE:
        logger.error("pylsl not available. Install: pip install pylsl")
        return
    logger.info("Searching for Mentalab LSL stream (type: ExG)...")
    streams = resolve_byprop("type", "ExG", timeout=15)
    if not streams:
        logger.warning("ExG stream not found — trying name 'Explore'...")
        streams = resolve_byprop("name", "Explore", timeout=5)
    if not streams:
        logger.error("No LSL stream found.")
        return
    inlet = StreamInlet(streams[0])
    logger.info("Connected to Mentalab.")
    if args.mode == "calibrate":
        model = run_calibration(inlet=inlet)
        with open(args.model, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"Model saved -> {args.model}")
    elif args.mode == "recalibrate_r":
        if not Path(args.model).exists():
            logger.error(f"Model not found: {args.model}. Run --mode calibrate first.")
            return
        with open(args.model, "rb") as f:
            model = pickle.load(f)
        print_cue("Sit still for 30 seconds — recalibrating R")
        rest_eeg  = _collect_eeg_fresh(inlet, seconds=30)
        rest_covs = []
        for start in range(0, rest_eeg.shape[1] - N_REST, N_REST):
            window = rest_eeg[:, start:start + N_REST]
            clean, _, _ = asr_stream(window, model["asr_state"], srate=SRATE)
            rest_covs.append(covariance(clean))
        model["R"] = estimate_R(rest_covs)
        logger.info(f"New R eigenvalues: {np.linalg.eigvalsh(model['R'])}")
        with open(args.model, "wb") as f:
            pickle.dump(model, f)
        logger.info("R updated -> bci_model.pkl")
    elif args.mode == "run":
        if not Path(args.model).exists():
            logger.error(f"Model not found: {args.model}. Run --mode calibrate first.")
            return
        with open(args.model, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {args.model}")
        run_session(model)
if __name__ == "__main__":
    main()
