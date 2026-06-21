# DAMP — Dynamic Affective Micro-Preference Paradigm

**English** · [Русский](README.ru.md) · [日本語](README.ja.md)

> **Project status — methodological template; no IRB approval, no data collection.**
> This repository is an open **methodological and computational framework**. It has
> **not** received ethics-committee / IRB approval, and no human-subject data are
> collected, stored, or distributed here. Every manuscript figure is **simulated**.
> The experiment program (`main.py`) is a ready-to-adapt **template**: to run it with
> human participants you must first obtain approval from your own ethics board and
> complete the consent form's `[placeholders]`. The preprint is available on OSF
> (see **Citation** below for the DOI) and is not duplicated in this repository.

---

## Overview

DAMP records and analyzes the *dynamics* of forced-choice decisions between briefly
presented emotional images. Instead of the final choice alone, it captures how the
decision unfolds — motor competition (mouse trajectory) and choice stability — as
process-level indices. For research and exploration only; no clinical or diagnostic
use is implied.

## Experimental Design

**Stimuli.** Images from the OASIS database (Kurdi et al., 2017), categorized by
valence (**Positive / Negative / Neutral**), then preprocessed: background removal,
grayscale, Gaussian blur, and rendering in a single flat color. **Color is assigned
independently of valence** (crossed across categories) so hue and luminance cannot
proxy for affective category. The processed set is manually curated — by rater(s) other than the
stimulus author, ideally blind to valence — keeping an image only if the object
stays distinguishable and occupies ≳ 40% of the canvas.

**Trial flow.** (1) audio warning → (2) fixation cross (2.0–3.5 s) → (3) two stimuli
shown sequentially (timing below) → (4) 3.0 s response window on a blank screen →
(5) confirmation + inter-trial interval (1.0–2.5 s). No masking is used.

**Timing** (jittered per trial to prevent anticipation):

| Parameter | Range |
|---|---|
| Stimulus exposure (each) | 18–30 ms (3–5 frames @ 165 Hz) |
| Inter-stimulus interval | 80–150 ms |

**Trial types & composition:**

| Type | Count | |
|---|---|---|
| Positive–Neutral | 17 | neutral-containing |
| Negative–Neutral | 17 | neutral-containing |
| Positive–Negative | 17 | both-valenced |
| SWAP (mirror re-presentation) | 10 | stability check |
| **Total** | **61** | |

> **Note on the labels.** "both-valenced" vs "neutral-containing" defines the DCI contrast by *stimulus composition*. Whether the both-valenced condition actually produces greater trajectory deviation (DCI > 0) is the open empirical question below — not an assumption (preprint §2.2 / §4.2).

Missed trials (no response within 3.0 s) are repeated at the end and excluded from
primary analysis.

## Calibration

7 trials clicking on a square shown randomly left or right; the median RT becomes the
individual baseline (`base_rt`) used for RT normalization.

## Recording

- Mouse trajectory at **100 Hz** (every 10 ms) throughout each trial
- Reaction time (from response-window onset to click)
- Color category per stimulus (descriptive bias analysis)

## Behavioral Indices

**Primary indices:**

| Index | Formula | Interpretation |
|---|---|---|
| **DCI** — Decision Conflict Index | `(AUC_high − AUC_low) / AUC_low` | Relative trajectory deviation, both-valenced vs neutral-containing. Whether DCI > 0 holds is an **open empirical question**, not established here (see preprint §2.2 / §4.2). For group inference the difference contrast `AUC_high − AUC_low` is more robust than this ratio. |
| **RSI** — Representational Stability Index | `N_same / N_swap` | Share of SWAP trials in which the original choice is maintained despite spatial inversion. Above chance ⇒ preference stability (assess via confidence interval). |

AUC = integral of perpendicular deviation from the ideal straight-line path
(trapezoidal). In the DCI, `AUC_high` is the both-valenced (Positive–Negative) condition and `AUC_low` the neutral-containing pairs; "high"/"low" name stimulus composition, not an assumed amount of conflict.

**Exploratory index:**

| Index | Definition | Note |
|---|---|---|
| **FLIP** | Number of times the trajectory crosses the central vertical axis (x = 0), averaged across trials | Recorded but **excluded from the primary analytical framework** — low theoretical specificity, sensitive to motor noise (preprint §4.1). |

## Data Output

Each session writes to `data/[Session_ID]/`:

| File | Contents |
|---|---|
| `results.csv` | Trial-level behavioral data |
| `trajectories.csv` | Mouse coordinates at 100 Hz |
| `indices.csv` | Session-level DCI, RSI, FLIP |
| `scientific_summary.txt` | Human-readable index report |
| `indices.png` | Decision-dynamics profile |
| `auc_profile.png` | AUC per trial |
| `rt_dynamics.png` | RT across the session |
| `rt_by_color.png` | Mean RT by stimulus color |
| `swap_consistency.png` | SWAP choice stability |
| `emotion_choice_frequency.png` | Choice frequency by valence |
| `color_preference.png` | Descriptive color-bias analysis |
| `anomaly_trials/`, `normal_trials/` | Per-trial trajectory plots |

## Computational Model & Simulations

The preprint reports **no human data**; all model figures are simulated from the
Section 2.2 generative model. Two competing, falsifiable models bracket the predicted
**sign of DCI**:

- **Model 1 — minimal drift-diffusion** (`dampsimulation.py`): predicts **DCI ≤ 0**
  (the largest valence gap, Positive–Negative, resolves fastest → straightest path).
  Robust to the noise parameter σ.
- **Model 2 — leaky competing accumulator** (`damp_lca_model.py`; Usher & McClelland,
  2001): predicts **DCI > 0** once the salience-capture weight γ > 0.

The sign of DCI is therefore an empirical question for future data, not a result of
the simulation.

| Script | Purpose |
|---|---|
| `main.py` | Runs the experiment (PsychoPy): consent, calibration, 61 trials |
| `analyze_data.py` | Computes indices + per-session figures for every folder in `data/` |
| `dampsimulation.py` | Model 1 (minimal DDM) |
| `damp_lca_model.py` | Model 2 (LCA + salience-weighted competition); γ sweep |
| `dampmakefigure.py` | Figures 2–5 |
| `damp_model_comparison_figure.py` | Figure 6 (DDM vs LCA + emergent high-conflict trajectory) |
| `damp_recovery.py` | Figure 7 (parameter recovery of Model 1) |
| `damp_power_analysis.py` | Bootstrap power analysis |
| `damp_extra_checks.py` | σ-robustness of the DCI sign, AUC bimodality, power by effect size |
| `Database_2026/script.py` | Stimulus generation (background removal, grayscale, blur, flat color ⊥ valence) |
| `Color_images.py` | Labels each stimulus with its dominant color |

Figures live in `figures/` (Figure 2–7) and are **all simulated**.


## Setup

**Python 3.13.** Monitor with refresh rate **≥ 165 Hz** (frame-level timing), a
standard mouse, and a fixed viewing distance (≈ 1 m).

```bash
# Experiment, analysis, and simulations:
pip install psychopy numpy pandas matplotlib seaborn sounddevice

# Stimulus (re)generation only (Database_2026/script.py, Color_images.py):
pip install pillow scipy rembg
```

## Usage

1. Place the raw OASIS images in `Database_2026/images/`, grouped by valence.
2. Run `Database_2026/script.py` to standardize them (background removal,
   grayscale, blur, flat color). Output is written to
   `Database_2026/processed_stimuli/`.
3. Manually curate `processed_stimuli/`: keep an image only if the object is
   still recognizable and occupies ≳ 40% of the canvas; discard the rest. This
   pass should be done by rater(s) other than the stimulus author, ideally blind
   to valence. Move the kept images into
   `Database_2026/final_standardized_stimuli/` with `Positive/`, `Negative/`,
   `Neutral/` subfolders (≥ 50 images each).
4. Run `main.py` — enter participant ID, age, gender; complete consent and
   calibration; then the main task (~40 min).
5. Run `analyze_data.py` to generate indices and figures for all sessions in `data/`.

**Pilot mode** (`use_fixed_stimuli = True`): identical stimuli and timing across
participants via `pilot_template.json` (auto-generated if absent).
**Random mode** (`= False`): full per-session randomization.

## Citation

> Mayer, A. (2026). *The DAMP (Dynamic Affective Micro-Preference) Paradigm:
> A Methodological and Computational Framework for Tracing Affective Choice Dynamics.*
> Preprint, OSF. DOI: https://doi.org/10.17605/OSF.IO/3N8JW

## Disclaimer

For research purposes only. The indices characterize decision dynamics and do not
constitute clinical assessment or diagnosis.
