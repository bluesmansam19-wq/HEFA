# HEFA — Harmonic-Envelope Fractal Analysis

**Version 1.0 · CC-BY 4.0 · May 2026**

A validated protocol and Python implementation for robust fractal analysis of compressed music audio.

---

## The problem

Standard fractal estimators (Hurst, Higuchi, spectral exponent) applied directly to MP3 audio at 128 kbps produce unreliable results — deviations of 10–20% from uncompressed reference values. Most ethnomusicological archives, YouTube recordings, and streaming sources exist only at these bitrates.

## The solution

HEFA displaces fractal analysis from the raw audio signal to the **RMS envelope of the harmonic component** extracted via Harmonic-Percussive Source Separation (HPSS). The result:

| Metric | Mean deviation (320k vs 128k) |
|---|---|
| Higuchi D | **< 0.15 %** |
| Hurst DFA | **< 1.5 %** |
| Hurst R/S | < 2.7 % |
| α_env | < 4.3 % |

Validated on two independent corpora of Lobi music from Burkina Faso (OCORA 1999 and 2007).

---

## Quick start

```bash
pip install numpy scipy librosa
python HEFA_implementation_complete.py my_audio.mp3
```

Results are printed to console and saved as `my_audio_HEFA.json`.

See [`HEFA_quickstart_EN.md`](HEFA_quickstart_EN.md) for full usage instructions.

---

## Repository contents

| File | Description |
|---|---|
| `HEFA_implementation_complete.py` | Complete Python implementation (450 lines) |
| `HEFA_v1_0_EN_FINAL.md` | Full protocol specification |
| `HEFA_quickstart_EN.md` | Quick-start guide |
| `HEFA_validation_data.json` | Raw validation data (Binkontin + Buur P4, with MD5 hashes) |
| `requirements.txt` | Exact dependency versions |
| `paper.md` | JOSS paper |
| `paper.bib` | Bibliography |

---

## Four fractal measures computed

| Measure | Symbol | Interpretation |
|---|---|---|
| Hurst R/S | H_RS | Long-memory persistence |
| Hurst DFA | H_DFA | Persistence (robust to non-stationarity) |
| Higuchi dimension | D | Structural complexity [1–2] |
| Spectral exponent | α_env | Frequency scaling of envelope |

---

## Citation

```bibtex
@software{momo2026hefa,
  author  = {MOMO Thierry Stéphane},
  title   = {{HEFA}: Harmonic-Envelope Fractal Analysis v1.0},
  year    = {2026},
  url     = {https://github.com/[your-username]/HEFA},
  license = {CC-BY-4.0}
}
```

---

## Author

**Prof. MOMO Thierry Stéphane**  
Independent Researcher  
SACEM Luxembourg / BBDA Burkina Faso / Primitive Blue  
bluesmansam19@gmail.com

## License

[Creative Commons Attribution 4.0 International (CC-BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
