# BCI Session Controller

Real-time motor-imagery BCI pipeline for the **Mentalab Explore** (8-channel, 500 Hz), using Riemannian geometry (MDM classifier with RPA alignment) to drive cursor movement from EEG.

## Pipeline

```
Mentalab (LSL) -> rASR -> covariance(C_t)
    -> Global Riemannian EMA:  R_t = Exp_R(alpha * Log_R(C_t)),  alpha=0.005
    -> RPA Alignment:          C_t_aligned = R_t^(-1/2) * C_t * R_t^(-1/2)
    -> MDM Prediction + Smooth Geodesic Adaptation
    -> cursor
```

**Centroid adaptation** (Smooth Geodesic, Zanini 2018):
- `confidence > ADAPT_CONFIDENCE` → buffer sample (size `BUFFER_SIZE`)
- `delta(mu_buf, mu_k) > ADAPT_THRESHOLD` → `mu_k <- Exp_mu(alpha * Log_mu(mu_buf))`, `alpha=ALPHA_CENTROID`

`R` updates unconditionally every trial to track global hardware covariance drift (electrode impedance, perspiration, fatigue). Adaptation buffers are seeded from calibration data so the buffer Fréchet mean is well-formed from the first online trial.

## Classes

| Label | Direction | Imagined movement |
|---|---|---|
| 0 | UP | Right foot moving upward |
| 1 | DOWN | Tongue movement |
| 2 | LEFT | Left hand clenching |
| 3 | RIGHT | Right hand clenching |

## Requirements

```
pip install -r requirements.txt
```

`pylsl` is required for live hardware modes (`calibrate`, `run`, `recalibrate_r`). Without it, only `test` mode works. `pyautogui` is required for actual cursor movement; without it, predicted moves are just logged.

This project also depends on two local modules that must sit alongside `session.py`:
- `ASR.py` — `asr_calibrate`, `asr_stream`
- `AMDM.py` — `MDM`, `covariance`, `rpa_recenter`, `riemannian_mean`, `riemannian_ema`, `dist_riemann`

## Usage

```bash
python session.py --mode calibrate      # first session: rASR calibration + MDM training
python session.py --mode run            # real-time classification + cursor control
python session.py --mode test           # offline simulation with synthetic data
python session.py --mode recalibrate_r  # re-estimate global covariance R from rest data
```

Calibrated models are saved to `bci_model.pkl`; raw calibration EEG to `raw_session.npz`.

## Hardware setup

- Device: Mentalab Explore, 8 channels, 500 Hz, streamed via LSL (type `ExG`, or name `Explore`)
- Channel layout: `Fz, FC3, FC4, C3, Cz, C4, CP3, CP4`
- Channels used for classification: `C3, Cz, C4, CP3, CP4` (indices `[3, 4, 5, 6, 7]`)

## Key parameters

| Constant | Value | Meaning |
|---|---|---|
| `CONFIDENCE_THRESH` | 0.60 | Minimum confidence to move the cursor |
| `ALPHA_R` | 0.005 | EMA rate for global covariance `R` |
| `ALPHA_CENTROID` | 0.02 | Geodesic step size for centroid adaptation |
| `BUFFER_SIZE` | 50 | Samples buffered per class before checking drift |
| `ADAPT_THRESHOLD` | 0.05 | Riemannian distance that triggers centroid adaptation |
| `ADAPT_CONFIDENCE` | 0.70 | Minimum confidence to buffer a sample for adaptation |


