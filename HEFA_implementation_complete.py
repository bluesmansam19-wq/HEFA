"""
HEFA — Harmonic-Envelope Fractal Analysis
Complete implementation — Version 1.0

Author  : Prof. MOMO Thierry Stéphane
Affil.  : SACEM Luxembourg / BBDA Burkina Faso / Primitive Blue
Date    : 2026-04-30
License : CC-BY 4.0
DOI     : (to be assigned)

Reference
---------
MOMO Thierry Stéphane (2026). HEFA — Harmonic-Envelope Fractal Analysis:
A protocol for robust fractal analysis of compressed music audio. Version 1.0.
SACEM Luxembourg / BBDA Burkina Faso / Primitive Blue.

Description
-----------
HEFA computes four fractal measures (Hurst R/S, Hurst DFA, Higuchi D,
spectral exponent α) on the RMS envelope of the harmonic component of
an audio signal, extracted via HPSS. This approach bypasses MP3
compression artifacts that degrade sample-level fractal estimation.

Validated on Binkontin OCORA 1999 and Buur OCORA 2007 (Lobi music,
Burkina Faso). Higuchi D and Hurst DFA show < 1.5 % deviation between
320 kbps and 128 kbps versions of the same recordings.

Dependencies
------------
    python  >= 3.8
    numpy   >= 1.20
    scipy   >= 1.7
    librosa >= 0.10

Usage
-----
    # Single window
    result = hefa_window("my_audio.mp3", start=0, duration=240)

    # 6 sliding windows (recommended for temporal evolution)
    results = hefa_sliding("my_audio.mp3", n_windows=6, window_sec=240)
"""

import numpy as np
import librosa
import librosa.effects
import librosa.feature


# ─────────────────────────────────────────────
#  STANDARD PARAMETERS  (do not modify without
#  documenting the change in your publication)
# ─────────────────────────────────────────────
SR            = 22050   # target sampling rate (Hz)
HPSS_MARGIN   = 3       # HPSS separation margin
FRAME_LENGTH  = 2048    # RMS frame length (samples)
HOP_LENGTH    = 512     # RMS hop length (samples)
FS_ENV        = SR / HOP_LENGTH   # envelope framerate ≈ 43.07 Hz
ALPHA_BAND    = (0.05, 5.0)       # spectral exponent frequency band (Hz)
HIGUCHI_KMAX  = 20      # Higuchi k_max


# ═══════════════════════════════════════════
#  FRACTAL MEASURES
# ═══════════════════════════════════════════

def hurst_rs(ts):
    """
    Hurst exponent via Rescaled Range (R/S) analysis.

    Parameters
    ----------
    ts : array-like
        Time series (1-D).

    Returns
    -------
    float or None
        Hurst exponent H. Returns None if fewer than 2 valid scales.

    Interpretation
    --------------
    H > 0.5 : long-memory persistence
    H = 0.5 : white noise (no memory)
    H < 0.5 : anti-persistence

    Reference: Hurst (1951); Mandelbrot & Van Ness (1968).
    """
    ts = np.asarray(ts, dtype=float)
    N = len(ts)
    ns, rs = [], []
    for n in [N // 16, N // 8, N // 4, N // 2]:
        if n < 16:
            continue
        nb = N // n
        rb = []
        for b in range(nb):
            blk = ts[b * n:(b + 1) * n]
            mb = np.mean(blk)
            dev = np.cumsum(blk - mb)
            R = np.max(dev) - np.min(dev)
            S = np.std(blk)
            if S > 0:
                rb.append(R / S)
        if rb:
            ns.append(n)
            rs.append(np.mean(rb))
    if len(ns) < 2:
        return None
    h, _ = np.polyfit(np.log(ns), np.log(rs), 1)
    return float(h)


def dfa(ts):
    """
    Hurst exponent via Detrended Fluctuation Analysis (DFA).

    More robust to non-stationarities than R/S. Same interpretation ranges.

    Parameters
    ----------
    ts : array-like
        Time series (1-D).

    Returns
    -------
    float or None
        DFA exponent. Returns None if fewer than 4 valid scales.

    Reference: Peng et al. (1994).
    """
    ts = np.asarray(ts, dtype=float) - np.mean(ts)
    N = len(ts)
    yi = np.cumsum(ts)
    scales = np.unique(np.logspace(1, np.log10(N // 4), 25).astype(int))
    F, valid = [], []
    for s in scales:
        if s < 4 or s > N // 2:
            continue
        ns = N // s
        if ns < 2:
            continue
        flu = []
        for j in range(ns):
            seg = yi[j * s:(j + 1) * s]
            x = np.arange(s)
            p = np.polyfit(x, seg, 1)
            flu.append(np.sqrt(np.mean((seg - np.polyval(p, x)) ** 2)))
        F.append(np.mean(flu))
        valid.append(s)
    if len(valid) < 4:
        return None
    h, _ = np.polyfit(np.log10(valid), np.log10(F), 1)
    return float(h)


def higuchi(ts, k_max=HIGUCHI_KMAX):
    """
    Higuchi fractal dimension D.

    Parameters
    ----------
    ts    : array-like  Time series (1-D).
    k_max : int         Maximum k value (default 20, HEFA standard).

    Returns
    -------
    float or None
        Higuchi dimension D ∈ [1, 2]. Returns None if fewer than 4 valid k.

    Interpretation
    --------------
    D = 1.0 : smooth, deterministic curve
    D = 1.5 : random walk (neutral memory)
    D → 2.0 : white noise (maximal complexity)

    Reference: Higuchi (1988).
    """
    ts = np.asarray(ts, dtype=float)
    N = len(ts)
    L, ks = [], []
    for k in range(1, k_max + 1):
        Lk = []
        for m in range(k):
            ids = np.arange(m, N, k)
            if len(ids) < 2:
                continue
            sub = ts[ids]
            denom = ((N - 1) // k) * k
            if denom == 0:
                continue
            length = np.sum(np.abs(np.diff(sub))) * (N - 1) / (denom * k)
            Lk.append(length)
        if Lk:
            L.append(np.mean(Lk))
            ks.append(k)
    if len(ks) < 4:
        return None
    log_k = np.log(1.0 / np.array(ks))
    log_L = np.log(L)
    d, _ = np.polyfit(log_k, log_L, 1)
    return float(d)


def alpha_spectral(ts, fs=FS_ENV, band=ALPHA_BAND):
    """
    Spectral exponent α_env of the RMS envelope.

    NOTE: α_env is computed on the harmonic-component RMS envelope
    (low-frequency band 0.05–5 Hz). It is distinct from α_signal
    (estimated on the direct audio signal, typically ≈ −2.30 ± 0.07
    on musical signals). Always use the notation α_env in publications
    to avoid confusion.

    Parameters
    ----------
    ts   : array-like  RMS envelope time series.
    fs   : float       Envelope framerate (default FS_ENV ≈ 43.07 Hz).
    band : tuple       Frequency band (Hz) for PSD estimation.

    Returns
    -------
    float or None
        Spectral exponent α_env. Returns None if fewer than 5 valid bins.

    Interpretation
    --------------
    α = 0 : white noise (spectrally flat)
    α = 1 : pink noise 1/f (self-organized complex systems)
    α = 2 : Brownian noise (integrated white noise)

    Reference: Voss & Clarke (1975).
    """
    ts = np.asarray(ts, dtype=float)
    fft = np.abs(np.fft.rfft(ts)) ** 2
    freqs = np.fft.rfftfreq(len(ts), 1 / fs)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if mask.sum() < 5:
        return None
    log_f = np.log10(freqs[mask])
    log_p = np.log10(fft[mask] + 1e-15)
    coef, _ = np.polyfit(log_f, log_p, 1)
    return float(-coef)


# ═══════════════════════════════════════════
#  CORE PIPELINE
# ═══════════════════════════════════════════

def hefa_window(audio_path, start=0, duration=None, label="window"):
    """
    Complete HEFA pipeline on a single time window.

    Steps
    -----
    1. Decode audio (sr=22050 Hz, mono)
    2. HPSS — keep harmonic component (margin=3)
    3. Compute RMS envelope (frame=2048, hop=512, fs_env≈43 Hz)
    4. Compute four fractal measures on the envelope

    Parameters
    ----------
    audio_path : str    Path to audio file (MP3, WAV, FLAC, …).
    start      : float  Window start in seconds (default 0).
    duration   : float  Window duration in seconds (None = full file).
    label      : str    Label for this window in the output dict.

    Returns
    -------
    dict with keys:
        label, start, duration,
        H_RS, H_DFA, D_Higuchi, alpha_env
    """
    # 1. Decode
    y, _ = librosa.load(audio_path, sr=SR, mono=True,
                        offset=start, duration=duration)

    # 2. HPSS — keep harmonic component
    y_h, _ = librosa.effects.hpss(y, margin=HPSS_MARGIN)
    del y

    # 3. RMS envelope on harmonic component
    rms = librosa.feature.rms(
        y=y_h,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH
    )[0]
    del y_h

    # 4. Fractal measures
    return {
        "label"     : label,
        "start_sec" : start,
        "duration_sec": duration,
        "H_RS"      : hurst_rs(rms),
        "H_DFA"     : dfa(rms),
        "D_Higuchi" : higuchi(rms, k_max=HIGUCHI_KMAX),
        "alpha_env" : alpha_spectral(rms, fs=FS_ENV, band=ALPHA_BAND),
    }


def hefa_sliding(audio_path, n_windows=6, window_sec=240, verbose=True):
    """
    HEFA pipeline over n equidistant sliding windows.

    Recommended for studying temporal evolution of a performance.
    With n=6, the Spearman correlation with position requires |ρ| ≥ 0.829
    for p < 0.05, providing a meaningful hypothesis-testing threshold.

    Parameters
    ----------
    audio_path : str    Path to audio file.
    n_windows  : int    Number of windows (default 6).
    window_sec : float  Window duration in seconds (default 240).
    verbose    : bool   Print progress (default True).

    Returns
    -------
    list of dicts (one per window), each with keys:
        label, start_sec, duration_sec,
        H_RS, H_DFA, D_Higuchi, alpha_env

    Notes
    -----
    If total_duration < n_windows × window_sec, window_sec is
    automatically reduced to total_duration / n_windows.
    Windows are non-overlapping by default (step = total_dur − window_sec
    distributed evenly across n_windows − 1 gaps).
    """
    # Get total duration without loading the full file
    y_full, _ = librosa.load(audio_path, sr=SR, mono=True)
    total_dur = len(y_full) / SR
    del y_full

    if total_dur < n_windows * window_sec:
        window_sec = total_dur / n_windows
        if verbose:
            print(f"[HEFA] Short corpus — window adjusted to {window_sec:.1f} s")

    step = (total_dur - window_sec) / max(n_windows - 1, 1)
    results = []

    for i in range(n_windows):
        start = i * step
        label = f"F{i + 1}"
        if verbose:
            pct = int(100 * start / total_dur)
            print(f"[HEFA] {label} — start={start:.0f}s ({pct}%) ...")
        result = hefa_window(audio_path, start=start,
                             duration=window_sec, label=label)
        results.append(result)
        if verbose:
            print(f"       H_RS={result['H_RS']:.3f}  "
                  f"H_DFA={result['H_DFA']:.3f}  "
                  f"D={result['D_Higuchi']:.3f}  "
                  f"α_env={result['alpha_env']:.3f}")

    return results


# ═══════════════════════════════════════════
#  OPTIONAL: BITRATE ROBUSTNESS CHECK
# ═══════════════════════════════════════════

def hefa_compare_bitrates(path_320k, path_128k, n_windows=6, window_sec=240):
    """
    Compare HEFA results between a 320 kbps and a 128 kbps version
    of the same recording. Reports mean and max relative deviation (%).

    Use this to validate HEFA robustness for a new corpus before
    running the full analysis.

    Parameters
    ----------
    path_320k  : str  Path to 320 kbps audio file.
    path_128k  : str  Path to 128 kbps audio file.
    n_windows  : int  Number of sliding windows (default 6).
    window_sec : float Window duration in seconds (default 240).

    Returns
    -------
    dict  Per-metric mean and max relative deviation (%).
    """
    print("[HEFA] Running 320 kbps ...")
    r320 = hefa_sliding(path_320k, n_windows, window_sec, verbose=False)
    print("[HEFA] Running 128 kbps ...")
    r128 = hefa_sliding(path_128k, n_windows, window_sec, verbose=False)

    metrics = ["H_RS", "H_DFA", "D_Higuchi", "alpha_env"]
    report = {}

    print("\n── Bitrate robustness report ──────────────────")
    print(f"{'Metric':<14} {'Δ mean %':>10} {'Δ max %':>10}")
    print("─" * 36)

    for m in metrics:
        v320 = np.array([w[m] for w in r320 if w[m] is not None])
        v128 = np.array([w[m] for w in r128 if w[m] is not None])
        if len(v320) == 0 or len(v128) == 0:
            continue
        n = min(len(v320), len(v128))
        delta = np.abs((v128[:n] - v320[:n]) / (v320[:n] + 1e-15)) * 100
        mean_d, max_d = float(np.mean(delta)), float(np.max(delta))
        report[m] = {"delta_mean_pct": mean_d, "delta_max_pct": max_d}
        print(f"{m:<14} {mean_d:>10.2f} {max_d:>10.2f}")

    print("─" * 36)
    return report


# ═══════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print(__doc__)
        print("\nUsage:")
        print("  python HEFA_implementation_complete.py <audio_file> [n_windows] [window_sec]")
        print("\nExample:")
        print("  python HEFA_implementation_complete.py corpus.mp3 6 240")
        sys.exit(0)

    audio_file = sys.argv[1]
    n_win  = int(sys.argv[2])   if len(sys.argv) > 2 else 6
    w_sec  = float(sys.argv[3]) if len(sys.argv) > 3 else 240.0

    print(f"\n[HEFA v1.0] Analysing: {audio_file}")
    print(f"            Windows  : {n_win} × {w_sec:.0f} s")
    print(f"            Params   : sr={SR}, hpss_margin={HPSS_MARGIN}, "
          f"frame={FRAME_LENGTH}, hop={HOP_LENGTH}, kmax={HIGUCHI_KMAX}\n")

    results = hefa_sliding(audio_file, n_windows=n_win,
                           window_sec=w_sec, verbose=True)

    print("\n── Summary ────────────────────────────────────────────")
    print(f"{'Window':<8} {'H_RS':>8} {'H_DFA':>8} {'D_Higuchi':>12} {'alpha_env':>12}")
    print("─" * 52)
    for r in results:
        def fmt(v): return f"{v:.3f}" if v is not None else "  N/A"
        print(f"{r['label']:<8} {fmt(r['H_RS']):>8} {fmt(r['H_DFA']):>8} "
              f"{fmt(r['D_Higuchi']):>12} {fmt(r['alpha_env']):>12}")

    out_path = audio_file.replace(".mp3", "").replace(".wav", "") + "_HEFA.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[HEFA] Results saved → {out_path}")
