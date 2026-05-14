# HEFA Quick-Start Guide

**HEFA — Harmonic-Envelope Fractal Analysis · Version 1.0**

---

## What HEFA does

HEFA computes four fractal measures on the **RMS envelope of the harmonic component** of an audio file, bypassing MP3 compression artifacts that degrade sample-level fractal estimation. It is designed for structural complexity analysis of recorded musical performances, particularly when the corpus includes heterogeneous bitrate sources (archives, YouTube recordings, field recordings).

**Measures computed**: Hurst R/S · Hurst DFA · Higuchi D · Spectral exponent α_env

---

## 1. Install dependencies

```bash
pip install numpy scipy librosa
```

Tested with: Python ≥ 3.8 · numpy ≥ 1.20 · scipy ≥ 1.7 · librosa ≥ 0.10

---

## 2. Run from the command line

```bash
# 6 windows of 240 s (recommended)
python HEFA_implementation_complete.py my_corpus.mp3

# Custom: 4 windows of 180 s
python HEFA_implementation_complete.py my_corpus.mp3 4 180
```

Results are printed to the console and saved as `my_corpus_HEFA.json`.

---

## 3. Use as a Python module

### Single window

```python
from HEFA_implementation_complete import hefa_window

result = hefa_window("my_corpus.mp3", start=0, duration=240)
print(result)
# {'label': 'window', 'start_sec': 0, 'duration_sec': 240,
#  'H_RS': 0.81, 'H_DFA': 0.89, 'D_Higuchi': 1.62, 'alpha_env': 1.23}
```

### 6 sliding windows (recommended for temporal evolution)

```python
from HEFA_implementation_complete import hefa_sliding

results = hefa_sliding("my_corpus.mp3", n_windows=6, window_sec=240)
for r in results:
    print(f"{r['label']}: H={r['H_DFA']:.3f}  D={r['D_Higuchi']:.3f}  α={r['alpha_env']:.3f}")
```

### Validate bitrate robustness before full analysis

```python
from HEFA_implementation_complete import hefa_compare_bitrates

report = hefa_compare_bitrates("corpus_320k.mp3", "corpus_128k.mp3")
# Prints mean and max % deviation per metric across all windows
```

---

## 4. Interpreting the results

| Measure | Range | Low value means | High value means |
|---|---|---|---|
| Hurst R/S / DFA | [0, 1] | Anti-persistent (< 0.5) or random (= 0.5) | Persistent long memory (> 0.5) |
| Higuchi D | [1, 2] | Smooth / regular | Complex / irregular |
| α_env | [0, ∞] | Spectrally flat (white noise) | Strong low-frequency structure |

**Typical values for complex musical performances**: H ∈ [0.65, 1.0], D ∈ [1.5, 1.75], α_env ∈ [0.5, 2.0]

> **Note**: α_env is computed on the harmonic-component RMS envelope (0.05–5 Hz band). It is distinct from α_signal (estimated directly on the audio waveform, typically ≈ −2.30 on musical signals). Always use the notation **α_env** in publications.

---

## 5. Standard parameters (HEFA v1.0)

| Parameter | Value | Do not change without documentation |
|---|---|---|
| Sampling rate | 22,050 Hz | |
| HPSS margin | 3 | |
| RMS frame length | 2,048 samples (≈ 93 ms) | |
| RMS hop length | 512 samples (≈ 23 ms) | |
| Envelope framerate | ≈ 43.07 Hz | |
| α_env frequency band | 0.05–5 Hz | |
| Higuchi k_max | 20 | |

---

## 6. What HEFA is **not** suitable for

| Task | Why | Alternative |
|---|---|---|
| Onset density measurement | Requires percussive component | Distinct Ku/k pipeline |
| Attack / transient analysis | Requires direct audio signal | Direct signal pipeline |
| Corpora < 60 s | Statistics too limited | — |
| Bitrates < 128 kbps | Not validated | — |

---

## 7. Reproducibility checklist

Before publishing results obtained with HEFA, confirm that you have documented:

- [ ] Unique identifier for each audio file (MD5 hash, DOI, or stable URL)
- [ ] Bitrate and sample rate of each source file
- [ ] Python version and exact library versions (`pip freeze > requirements.txt`)
- [ ] HEFA protocol version used (cite **"HEFA v1.0"**)
- [ ] Source code (this file or repository URL)
- [ ] All parameters (especially if any deviate from the v1.0 standard)
- [ ] Sliding window configuration (n, duration, distribution)
- [ ] Raw measurement data (JSON or CSV)
- [ ] Any special cases (non-standard duration, anomalies, exclusions)

---

## 8. Citation

> MOMO Thierry Stéphane (2026). *HEFA — Harmonic-Envelope Fractal Analysis: A protocol for robust fractal analysis of compressed music audio*. Version 1.0. SACEM Luxembourg / BBDA Burkina Faso / Primitive Blue. DOI: (to be assigned)

---

## 9. Files in this package

| File | Description |
|---|---|
| `HEFA_v1_0_EN.md` | Full protocol specification |
| `HEFA_implementation_complete.py` | **This script** — complete implementation |
| `HEFA_validation_data.json` | Raw validation data (Binkontin + Buur P4, OCORA) |
| `HEFA_quickstart_EN.md` | This guide |
