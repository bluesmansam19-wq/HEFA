---
title: 'HEFA: A Python package for harmonic-envelope fractal analysis of compressed music audio'
tags:
  - Python
  - music analysis
  - fractal analysis
  - ethnomusicology
  - signal processing
  - Higuchi dimension
  - Hurst exponent
authors:
  - name: MOMO Thierry Stéphane
    orcid: 0009-0005-8023-6930
    affiliation: "1, 2, 3"
affiliations:
  - name: SACEM Luxembourg
    index: 1
  - name: BBDA Burkina Faso
    index: 2
  - name: Primitive Blue
    index: 3
date: 14 May 2026
bibliography: paper.bib
---

# Summary

HEFA (Harmonic-Envelope Fractal Analysis) is a Python package that computes four fractal complexity measures on compressed music audio files: the Hurst exponent via Rescaled Range analysis [@hurst1951], the Hurst exponent via Detrended Fluctuation Analysis [@peng1994], the Higuchi fractal dimension [@higuchi1988], and the spectral exponent α. Unlike existing tools that apply these estimators directly to the audio waveform, HEFA operates on the RMS envelope of the harmonic component isolated via Harmonic-Percussive Source Separation (HPSS) [@fitzgerald2010], implemented using the librosa library [@mcfee2015]. This design choice makes HEFA robust to the compression artifacts that degrade fractal estimates in MP3 audio below 320 kbps — the dominant format in ethnomusicological archives, YouTube recordings, and streaming platforms.

# Statement of need

Fractal analysis of music audio has produced significant insights into the self-similar structure of musical signals [@voss1975; @su2012]. However, existing implementations assume access to high-quality, uncompressed audio. When applied to MP3 files at 128 kbps — the typical bitrate of archival recordings and streaming sources — standard estimators exhibit deviations of 10–20% from uncompressed reference values, rendering the results unreliable for comparative analysis.

This limitation is especially acute in ethnomusicology, where the primary sources for acoustic analysis are historical field recordings and commercial archives that exist only in compressed formats [@panteli2017]. No existing tool addresses this gap specifically.

HEFA addresses it by operating on the **RMS envelope of the harmonic component** rather than on the raw waveform. The harmonic component, isolated by HPSS (margin = 3), is insensitive to the high-frequency artifacts introduced by MP3 compression. Its RMS envelope performs implicit downsampling to musically meaningful timescales (≈ 0.2–20 seconds at the default framerate of 43 Hz), eliminating micro-temporal noise while preserving structural dynamics relevant to musical organization.

Validation on two independent corpora of Lobi music from Burkina Faso (OCORA recordings, 1999 and 2007) demonstrates that HEFA produces mean relative deviations below 0.15% for Higuchi D and below 1.5% for Hurst DFA between 320 kbps and 128 kbps versions of the same recordings. These results qualify HEFA as suitable for comparative fractal analysis of archival-quality compressed audio.

HEFA is intended for use by: (1) ethnomusicologists working with compressed archival recordings; (2) music information retrieval researchers studying structural complexity in world music corpora; (3) computational musicologists seeking a validated, reproducible fractal analysis pipeline with a clearly defined parameter standard.

# Implementation

HEFA implements four fractal measures:

**Hurst R/S** (`hurst_rs`): Rescaled Range analysis over four sub-window scales (N/16, N/8, N/4, N/2). Returns the slope of log(R/S) vs log(N).

**Hurst DFA** (`dfa`): Detrended Fluctuation Analysis over 25 logarithmically spaced scales. Linear detrending within each segment. Returns the slope of log(F) vs log(s).

**Higuchi D** (`higuchi`): Higuchi fractal dimension with k_max = 20 (HEFA standard). Returns the slope of log(L) vs log(1/k).

**Spectral exponent α_env** (`alpha_spectral`): Power spectral density slope in the 0.05–5 Hz frequency band of the RMS envelope, estimated via FFT.

The core pipeline function `hefa_window()` takes an audio file path and optional start/duration parameters, and returns a dictionary of the four measures. The convenience function `hefa_sliding()` applies the pipeline over n equidistant windows and is the recommended entry point for temporal evolution studies. A bitrate comparison function `hefa_compare_bitrates()` facilitates reproducibility validation for new corpora.

All parameters are fixed to the HEFA v1.0 standard values (sr = 22050 Hz, HPSS margin = 3, frame = 2048, hop = 512, k_max = 20, α band = 0.05–5 Hz) and documented in the protocol specification (`HEFA_v1_0_EN_FINAL.md`). Deviations from these parameters must be documented explicitly in any publication using HEFA, to ensure reproducibility.

# Validation

Validation was performed on two corpora:

- **Corpus 1** (Binkontin, OCORA Burkina Faso 1999, 32′47″, 6 windows × 240 s): Higuchi D deviation < 0.14% mean, < 0.25% max; Hurst DFA deviation < 0.12% mean, < 0.29% max across 6 windows.

- **Corpus 2** (Buur P4, OCORA Burkina Faso 2007, 3′16″, single window): Higuchi D deviation = 0.11%; Hurst DFA deviation = 1.46%.

Spearman rank-order correlations with window position are preserved between bitrates (α_env: ρ = −1.0 at 320k, ρ = −0.9 at 128k; Higuchi D: ρ = +0.5 at both). Raw validation data are provided in `HEFA_validation_data.json` with MD5 hashes for audio file traceability.

# Acknowledgements

The Lobi music corpora used for validation are from the OCORA Burkina Faso series (1999 and 2007). The author thanks the musicians of the Hien Bihoulèté ensemble for their art.

# References
