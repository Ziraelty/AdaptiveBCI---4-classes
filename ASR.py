
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm, logm, fractional_matrix_power
import mne
import matplotlib.gridspec as gridspec
 
 
 
def temporal_embedding(eeg, n_delays=4, delay_step=6):

    n_ch, n_samples = eeg.shape
    embedded = np.zeros((n_ch * n_delays, n_samples))
 
    for i in range(n_delays):
        delay = i * delay_step
        block = slice(i * n_ch, (i + 1) * n_ch)
 
        if delay == 0:
            embedded[block, :] = eeg
        else:
            embedded[block, :delay]  = np.tile(eeg[:, [0]], (1, delay))
            embedded[block,  delay:] = eeg[:, :-delay]
 
    return embedded
 
 
 
def riemannian_mean(covs, n_iter=50, tol=1e-7):

    M = np.mean(covs, axis=0)
 
    for it in range(n_iter):
        M_half     = fractional_matrix_power(M, 0.5)
        M_neg_half = fractional_matrix_power(M, -0.5)
        S          = np.zeros_like(M)
 
        for C in covs:
            inner = M_neg_half @ C @ M_neg_half
            inner = (inner + inner.T) / 2
            S    += np.real(logm(inner))
 
        S /= len(covs)

        if np.linalg.norm(S, 'fro') < tol:
            print(f"  Riemannian mean zbiega w {it+1} iteracjach.")
            break
 
        M = M_half @ expm(S) @ M_half
        M = np.real((M + M.T) / 2)
 
    return M
 
 
def geodesic(M, C, alpha):
    M_half     = fractional_matrix_power(M, 0.5)
    M_neg_half = fractional_matrix_power(M, -0.5)
    inner      = M_neg_half @ C @ M_neg_half
    inner      = (inner + inner.T) / 2
    G          = M_half @ fractional_matrix_power(inner, alpha) @ M_half
    return np.real(G)
 

 
def asr_calibrate(eeg_cal, srate, cutoff=20, n_delays=4, delay_step=6,
                  win_sec=1.0, overlap=0.5, clean_percentile=75):
    n_ch_raw   = eeg_cal.shape[0]
    win_len    = int(win_sec * srate)
    hop_len    = int(win_len * (1 - overlap))
    n_samples  = eeg_cal.shape[1]
 
    windows_raw    = []
    rms_per_window = []  
 
    for start in range(0, n_samples - win_len + 1, hop_len):
        X = eeg_cal[:, start:start + win_len]
        rms_ch = np.sqrt(np.mean(X ** 2, axis=1))  
        windows_raw.append((start, X))
        rms_per_window.append(rms_ch)
 
    rms_array = np.array(rms_per_window)           
    rms_thresh = np.percentile(rms_array, clean_percentile, axis=0)  
    clean_mask = np.all(rms_array <= rms_thresh, axis=1)
 
    n_total = len(windows_raw)
    n_clean = clean_mask.sum()
    print(f"Okien kalibracyjnych: {n_total} → {n_clean} po filtracji RMS "
          f"({n_clean/n_total*100:.1f}%)")
 
    if n_clean < 10:
        raise RuntimeError
        
 
    covs   = []
    n_emb  = n_ch_raw * n_delays
 
    for i, (_, X) in enumerate(windows_raw):
        if not clean_mask[i]:
            continue
 
        X_emb = temporal_embedding(X, n_delays, delay_step)
        X_emb = X_emb - X_emb.mean(axis=1, keepdims=True)
        C     = (X_emb @ X_emb.T) / X_emb.shape[1]
        C     = (C + C.T) / 2
        eps   = 1e-6 * np.trace(C) / n_emb
        C    += eps * np.eye(n_emb)
        covs.append(C)
 
    print("Obliczanie średniej Riemannowskiej...")
    M     = riemannian_mean(np.array(covs))
    W     = np.real(fractional_matrix_power(M, -0.5))   
    W_inv = np.real(fractional_matrix_power(M,  0.5))   
 
    all_eigs = []
    for C in covs:
        C_w     = W @ C @ W.T
        C_w     = (C_w + C_w.T) / 2
        eigs, _ = np.linalg.eigh(C_w)
        all_eigs.append(eigs)
 
    all_eigs   = np.array(all_eigs)
    rms_eigs   = np.sqrt(np.mean(all_eigs ** 2, axis=0))
    thresholds = cutoff * rms_eigs
 
    print(f"Kalibracja gotowa. embedded={n_emb} ch, "
          f"cutoff={cutoff}, "
          f"próg min/max={thresholds.min():.4f}/{thresholds.max():.4f}")
 
    return {
        'M'          : M,
        'W'          : W,
        'W_inv'      : W_inv,      
        'thresholds' : thresholds,
        'cutoff'     : cutoff,
        'n_ch'       : n_emb,      
        'n_delays'   : n_delays,
        'delay_step' : delay_step,
        'n_ch_raw'   : n_ch_raw,
        'win_len'    : win_len,     
        'hop_len'    : hop_len,
    }
 
 
def asr_process_window(X, state, alpha=0.001):

    W          = state['W']
    M          = state['M']
    W_inv      = state['W_inv']     
    thresholds = state['thresholds']
 
    X          = X - X.mean(axis=1, keepdims=True)
    C          = (X @ X.T) / X.shape[1]
    C          = (C + C.T) / 2
    eps        = 1e-6 * np.trace(C) / C.shape[0] 

    C_w        = W @ C @ W.T
    C_w        = (C_w + C_w.T) / 2
    eigs, vecs = np.linalg.eigh(C_w)
 
    is_artifact = np.any(eigs > thresholds)
 
    if not is_artifact:
        state['W']     = np.real(fractional_matrix_power(state['M'], -0.5))
        state['W_inv'] = np.real(fractional_matrix_power(state['M'],  0.5))  
        return X, False, state
 
    clean_idx = np.where(eigs <= thresholds)[0]
 
    if len(clean_idx) == 0:
        return np.zeros_like(X), True, state
 
    V_clean = vecs[:, clean_idx]
    
    B       = np.real(W_inv @ V_clean @ V_clean.T @ W)
    X_clean = B @ X
 
    return X_clean, True, state
 
 
def asr_stream(eeg, state, srate):
    n_ch_raw   = state['n_ch_raw']
    n_delays   = state['n_delays']
    delay_step = state['delay_step']
    win_len    = state['win_len']    
    hop_len    = state['hop_len'] 
    n_ch, n_samples = eeg.shape
 
    win_len  = int(1.0 * srate)
    hop_len  = int(win_len * 0.5)
 
    buf          = np.tile(eeg[:, [0]], (1, win_len))
    eeg_clean    = np.zeros_like(eeg, dtype=float)
    weights      = np.zeros(n_samples)
    hann         = np.hanning(win_len)
    artifact_log = []
    step         = 0
    eeg_clean[:, :hop_len] = eeg[:, :hop_len]
    weights[:hop_len]      = 1.0
 
    for start in range(0, n_samples - hop_len + 1, hop_len):
        new_samples = eeg[:, start:start + hop_len]
        buf         = np.concatenate([buf[:, hop_len:], new_samples], axis=1)
        step       += 1
 

        if step < 2:
            continue
 
        buf_emb = temporal_embedding(buf, n_delays, delay_step)
 
        X_out, is_artifact, state = asr_process_window(buf_emb, state)
        recovered = X_out[:n_ch_raw, :]                 
 
        end = min(start + win_len, n_samples)
        seg_len = end - start
        eeg_clean[:, start:end] += recovered[:, :seg_len] * hann[:seg_len]
        weights[start:end]      += hann[:seg_len]
 
        artifact_log.append((start, is_artifact))
 
    weights   = np.maximum(weights, 1e-6)
    eeg_clean /= weights
 
    return eeg_clean, artifact_log, state
 
 
