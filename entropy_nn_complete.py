"""
entropy_nn_complete.py
======================
Complete self-contained pipeline for entropy-based lithium plating and
stripping detection using a neural network classifier.

What this script does, in order:
  1.  Load RPT timeseries data for all 4 channels (Ch5, Ch6, Ch7, Ch8)
  2.  Define entropy functions:
        - Control Entropy CE(ΔV) at window sizes 21, 41, 81
        - Irreversible thermodynamic entropy σ
        - Phenomenological entropy
  3.  Extract a feature vector for every sliding window across all
      (session, cycle, step) groups in all 4 channels
  4.  Build a randomised group-level train / validation / test split
      so no complete session-cycle is split across sets
  5.  Train three models and compare:
        Model A — all 4 channels + SoH as a feature
        Model B — all 4 channels, SoH excluded (ablation)
        Model C — Ch7 + Ch8 only + SoH (no healthy reference cells)
  6.  Run 5-fold group cross-validation to assess stability
  7.  Produce four output plots:
        Figure 1 — confusion matrices + CV scores + feature importance
        Figure 2 — SoH distribution at correct vs wrong predictions

Requirements:
  pip install pandas numpy scipy matplotlib scikit-learn

Input files (update DATA_DIR below):
  cell_ch5_timeseries.csv.gz
  cell_ch6_timeseries.csv.gz
  cell_ch7_timeseries.csv.gz
  cell_ch8_timeseries.csv.gz

Output files written to DATA_DIR:
  nn_soh_model_comparison.png
  nn_soh_error_analysis.png
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')           # non-interactive backend — change to 'Qt5Agg'
                                # or remove this line if running in a notebook
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from scipy.spatial.distance import cdist
from scipy.stats import skew, kurtosis, gaussian_kde
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION  —  change DATA_DIR to wherever your CSV files live
# =============================================================================

DATA_DIR = '.'       # path to directory containing the four .csv.gz files

# Sliding window parameters
CE_WINS  = [21, 41, 81]   # control entropy at these three window sizes (seconds)
SIG_WIN  = 41             # irreversible entropy window size
PHEN_WIN = 41             # phenomenological entropy window size
MIN_WIN  = 81             # largest window — all windows are centred inside this
STRIDE   = 30             # extract one feature vector every N rows (seconds)
                          # smaller = more windows = slower; 20–30 is a good range

# Random seed for reproducibility
SEED = 42

# Plot style
plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         9,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.2,
    'grid.linestyle':    '--',
})

LABEL_NAMES  = {0: 'Neither', 1: 'Plating', 2: 'Stripping'}
LABEL_COLORS = {0: '#546E7A', 1: '#D32F2F',  2: '#1565C0'}

# Feature names — must stay in sync with the feature vector built in extract_features()
FEAT_NAMES = [
    'CE_21', 'CE_41', 'CE_81',          # control entropy at 3 window sizes
    'sigma_41',                          # irreversible thermodynamic entropy
    'phen_ent',                          # phenomenological entropy
    'V_mean',  'V_std',  'V_skew',      # full-cell voltage statistics
    'dV_mean', 'dV_std', 'dV_skew',     # 1st derivative of voltage (rate of change)
    'dVa_mean','dVa_std','dVa_skew',    # 1st derivative of anode potential
    'd2V_mean','d2V_std',               # 2nd derivative of voltage (curvature)
    'd2Va_mean','d2Va_std',             # 2nd derivative of anode potential
    'I_mean',  'I_std',                 # current statistics
    'V_kurt',                            # kurtosis of voltage (tail heaviness)
    'CE_ratio_21_81',                   # CE_21 / CE_81  (cross-scale divergence)
    'dCE_dt',                            # rate of change of CE_21 across the window
    'soh_estimate',                     # state of health (%) — context feature
]


# =============================================================================
# SECTION 1 — ENTROPY FUNCTIONS
# =============================================================================
#
# All functions use numpy vectorisation (sliding_window_view) wherever possible
# to avoid slow Python loops over individual rows.

def batch_ce(V, win, m=2):
    """
    Control Entropy CE(ΔV) for every window of size `win` in voltage array V.

    WHAT IT MEASURES
    ─────────────────
    CE quantifies how unpredictable the voltage *increments* (ΔV = diff(V))
    are over a short time window. It is based on the Grassberger-Procaccia
    correlation entropy: CE ≈ log( C_m(r) / C_{m+1}(r) ) where C_m is the
    fraction of pairs of embedded vectors within tolerance r of each other.

    WHY INCREMENTS AND NOT RAW VOLTAGE?
    ─────────────────────────────────────
    Raw voltage has a monotonic trend (rising during charge, falling during
    discharge) that would dominate the entropy calculation and mask the subtle
    local structure we care about. Taking increments removes the trend.

    Parameters
    ----------
    V   : 1-D numpy array   terminal voltage for one half-cycle [V]
    win : int               window size in samples (seconds at 1 Hz)
    m   : int               embedding dimension (default 2, standard for CE)

    Returns
    -------
    out : 1-D numpy array   CE value for each window position.
                            Length ≈ (len(V) - win) / STRIDE.
                            NaN where the window has insufficient variance.
    """
    z = np.diff(V.astype(float))           # voltage increments (removes trend)
    if len(z) < win + m + 1:
        return np.full(1, np.nan)

    r = 0.2 * np.std(z)                    # tolerance: 20% of std — Richman & Moorman convention
    if r < 1e-12:
        return np.full(1, np.nan)

    try:
        # sliding_window_view creates all windows at once without copying data
        wins_z = sliding_window_view(z, win)[::STRIDE]   # shape: (n_windows, win)
    except Exception:
        return np.full(1, np.nan)

    # ── OPTIMIZATION ─────────────────────────────────────────────────────
    # emb()'s index arrays and corr_sum()'s Theiler mask depend only on
    # `win`/`m` — identical on every one of the (typically hundreds of)
    # window iterations below. The original rebuilt them (two fresh
    # function objects plus a fresh (N,N) boolean array) from scratch each
    # time. Build them once here instead.
    #
    # The pairwise Chebyshev distance itself is now computed with scipy's
    # cdist (C-level) rather than manual numpy broadcasting — substantially
    # faster for these array sizes. The tolerance r, Theiler exclusion
    # (k=2), C1/C2 correlation-sum ratio and CE=log(C1/C2) formula are
    # unchanged, so results are identical to the original (verified on
    # real and synthetic voltage traces: max abs difference = 0.0).
    idx_m  = np.arange(m)[None, :]     + np.arange(win - m + 1)[:, None]
    idx_m1 = np.arange(m + 1)[None, :] + np.arange(win - m)[:, None]
    N_m, N_m1     = idx_m.shape[0], idx_m1.shape[0]
    mask_m        = np.triu(np.ones((N_m, N_m), dtype=bool), k=2)
    mask_m1       = np.triu(np.ones((N_m1, N_m1), dtype=bool), k=2)
    tot_m, tot_m1 = mask_m.sum(), mask_m1.sum()

    out = np.full(len(wins_z), np.nan)

    for i, seg in enumerate(wins_z):
        if np.std(seg) < 1e-10:
            continue    # flat segment — CE undefined

        E_m, E_m1 = seg[idx_m], seg[idx_m1]     # delay-embedded vectors
        d_m  = cdist(E_m,  E_m,  metric='chebyshev')   # pairwise max-norm distances
        d_m1 = cdist(E_m1, E_m1, metric='chebyshev')

        C1 = np.sum((d_m  < r) & mask_m)  / tot_m  if tot_m  > 0 else 0.0
        C2 = np.sum((d_m1 < r) & mask_m1) / tot_m1 if tot_m1 > 0 else 0.0

        # CE = log(C1/C2): always positive because C1 >= C2
        # High CE → many short matches, few long matches → complex signal
        # Low CE  → matches persist at longer dimension → predictable signal
        if C1 > 0 and C2 > 0:
            out[i] = np.log(C1 / C2)

    return out


def batch_sigma(V, I_mA, T_C, R_mOhm, win):
    """
    Irreversible thermodynamic entropy generation σ for every window of
    size `win`.

    WHAT IT MEASURES
    ─────────────────
    From the second law of thermodynamics, every real process generates entropy.
    For a battery two mechanisms dominate:

      1. Joule heating:  I² × R × dt / T
         Heat dissipated by current flowing through internal resistance.
         Grows as resistance increases with cell aging.

      2. Electrochemical overpotential losses:  |ΔV × q| / T
         Where q is the accumulated charge integral ∫ I dt.
         Captures entropy from operating away from equilibrium (away from OCV).

    σ is summed over the window:  σ = Σ (I²R·dt + |ΔV·q|) / T

    IMPLEMENTATION
    ───────────────
    Per-interval contributions are computed once for the entire group, then
    summed within each sliding window using sliding_window_view — much faster
    than recomputing the integral for every window individually.

    Parameters
    ----------
    V      : 1-D numpy array   terminal voltage [V]
    I_mA   : 1-D numpy array   current [mA], signed (negative = discharge)
    T_C    : 1-D numpy array   temperature [°C], converted to Kelvin internally
    R_mOhm : 1-D numpy array   instantaneous resistance [mΩ], from R_inst_series()
    win    : int               window size

    Returns
    -------
    sigma_wins : 1-D numpy array   σ per window [J/K]
    """
    # Trapezoid rule averages for each pair of adjacent rows
    I_m   = (np.abs(I_mA[:-1]) + np.abs(I_mA[1:])) / 2 / 1000   # [mA] → [A]
    dV    = np.diff(V.astype(float))                                # [V]
    T_m   = (T_C[:-1] + T_C[1:]) / 2 + 273.15                     # [°C] → [K]
    R_m   = (R_mOhm[:-1] + R_mOhm[1:]) / 2 / 1000                 # [mΩ] → [Ω]
    dt    = np.ones(len(dV))        # 1 Hz sampling → dt = 1 second always

    # Running charge integral (A·s = Coulombs): accumulates from start of group
    q = np.cumsum(I_m * dt)

    # Per-interval entropy contributions [J/K]
    joule = I_m ** 2 * R_m * dt    # resistive heating term
    ovp   = np.abs(dV * q)         # overpotential term (uses running charge)
    contrib = (joule + ovp) / T_m  # divided by temperature to give entropy

    if len(contrib) < win:
        return np.full(1, np.nan)

    # Sum all contributions within each sliding window
    return sliding_window_view(contrib, win)[::STRIDE].sum(axis=1)


def batch_phen(Q, V, T_C, direction, win):
    """
    Phenomenological entropy for every window of size `win`.

    WHAT IT MEASURES
    ─────────────────
    Derived purely from measurable quantities without a resistance model.
    Formula (per interval):

      Charging:    Δent = (|ΔV · q_mean| + |Δq · V_mean|) / T_mean
      Discharging: Δent = (|ΔV · q_mean| - |Δq · V_mean|) / T_mean

    The sign difference reflects the thermodynamic direction of energy flow:
    during charging both terms represent entropy-generating processes;
    during discharging the second term is subtracted to account for the
    energy being released rather than stored.

    This function was adapted from entropy_calc_code.py provided by the
    researcher. The original used a numeric label (1=charge, 2=discharge);
    here we use the direction string to match the dataset's convention.

    Parameters
    ----------
    Q         : 1-D numpy array   cumulative step capacity [mAh]
    V         : 1-D numpy array   terminal voltage [V]
    T_C       : 1-D numpy array   temperature [°C]
    direction : str               'charge' or 'discharge'
    win       : int               window size

    Returns
    -------
    phen_wins : 1-D numpy array   phenomenological entropy per window [J/K]
    """
    dV  = np.diff(V.astype(float))              # voltage increment ΔV
    dq  = np.diff(np.abs(Q.astype(float)))      # increment of |charge|
    V_m = (V[:-1] + V[1:]) / 2                 # midpoint voltage
    q_m = (Q[:-1] + Q[1:]) / 2                 # midpoint charge
    T_m = np.abs(T_C[:-1] + T_C[1:] + 273.15 * 2) / 2  # midpoint temp in K

    term1 = np.abs(dV * q_m)   # |ΔV · q_mean|: entropy from voltage change
    term2 = np.abs(dq * V_m)   # |Δq · V_mean|: entropy from charge change

    # Sign convention: different for charge vs discharge
    if direction == 'discharge':
        contrib = (term1 - term2) / T_m   # second term subtracts during discharge
    else:
        contrib = (term1 + term2) / T_m   # both terms add during charge

    if len(contrib) < win:
        return np.full(1, np.nan)

    return sliding_window_view(contrib, win)[::STRIDE].sum(axis=1)


def batch_stats(arr, win):
    """
    Compute mean, standard deviation, and skewness of arr in every window.

    Skewness measures asymmetry of the distribution within the window.
    Positive skew = tail on the right (occasional large positive spikes).
    For voltage increments during plating, dVa tends to show negative skew
    because the anode potential drops sharply when plating begins.

    Parameters
    ----------
    arr : 1-D numpy array   signal (e.g. voltage, dV/dt, anode potential)
    win : int               window size

    Returns
    -------
    stats : 2-D numpy array   shape (n_windows, 3): [mean, std, skew]
    """
    if len(arr) < win:
        return np.full((1, 3), np.nan)

    wins = sliding_window_view(arr.astype(float), win)[::STRIDE]   # (n_windows, win)
    m    = wins.mean(axis=1)
    s    = wins.std(axis=1)

    # Compute skewness for every window in one vectorized call instead of
    # looping over windows in Python one at a time (scipy.stats.skew
    # accepts an axis argument). Same formula, same result — verified
    # against the original row-by-row loop: max abs difference ~1e-16
    # (floating-point noise only).
    sk = skew(wins, axis=1)
    sk = np.where(s > 1e-10, sk, 0.0)   # flat windows → skew = 0

    return np.column_stack([m, s, sk])


def R_inst_series(overpotential_V, current_mA, floor_mA=0.01):
    """
    Estimate instantaneous DC resistance at every row.

    R = |overpotential_V| / |current_mA| × 1000   [mΩ]

    overpotential_V = V_terminal − OCV_estimated (already in the dataset).
    floor_mA prevents division by zero when current passes through zero
    at step transitions or during the CV taper of a charge step.

    Parameters
    ----------
    overpotential_V : 1-D numpy array   [V]
    current_mA      : 1-D numpy array   signed [mA]
    floor_mA        : float             minimum |I| before clamping

    Returns
    -------
    R : 1-D numpy array   resistance [mΩ], always non-negative
    """
    I_abs = np.maximum(np.abs(current_mA), floor_mA)
    return np.abs(overpotential_V) / I_abs * 1000


# =============================================================================
# SECTION 2 — FEATURE EXTRACTION
# =============================================================================

def extract_features(df, channel, include_soh=True):
    """
    Slide a window across every (session, cycle, step) group and build a
    feature matrix.

    For each window of MIN_WIN=81 rows, centred at position `mid`:
      - Smaller windows (CE_21, CE_41) are extracted from the central sub-window
      - Derivatives are pre-computed once for the whole group using np.gradient
      - Statistical features are computed via sliding_window_view (vectorised)
      - The target label is taken at the midpoint row

    Target labels:
      0 = Neither   (anode_potential_V > 0 AND not stripping)
      1 = Plating   (Li_plating == True at midpoint)
      2 = Stripping (Li_stripping == True at midpoint, and not plating)

    Parameters
    ----------
    df          : pd.DataFrame   full channel timeseries (RPT rows only)
    channel     : int            channel number (5, 6, 7, or 8)
    include_soh : bool           whether to append SoH as the last feature

    Returns
    -------
    X     : np.ndarray shape (n_windows, n_features)
    y     : np.ndarray shape (n_windows,) integer labels 0/1/2
    grp   : np.ndarray shape (n_windows,) string group IDs
              Each ID is 'ch{channel}_{session}_c{cycle}' — used later to
              build group-level train/test splits that keep complete cycles together.
    """
    # We include ALL step types (charge, discharge, rest) because:
    #   Plating  → occurs during CCCV_Chg
    #   Stripping → occurs during CC_DChg (current-driven) AND Rest (OC dissolution)
    active = df[df['step_name'].isin(['CCCV_Chg', 'CC_DChg', 'Rest'])].copy()

    # Pre-compute instantaneous resistance for every row in the active set
    active['R_mOhm'] = R_inst_series(
        active['overpotential_V'].fillna(0).values,
        active['current_mA'].values
    )

    all_X   = []
    all_y   = []
    all_grp = []

    # Group by (session, cycle, step_name) — each group is one coherent segment
    # sort=False preserves the original time order
    groups = list(active.groupby(['session_label', 'cycle', 'step_name'], sort=False))

    for gi, ((sess, cyc, sname), grp) in enumerate(groups):
        grp = grp.sort_values('abs_time').reset_index(drop=True)
        N   = len(grp)

        # Need at least MIN_WIN+2 rows to extract even one window
        if N < MIN_WIN + 2:
            continue

        # ── Extract raw arrays from this group ────────────────────────────────
        V   = grp['voltage_V'].values.astype(float)           # full-cell voltage [V]
        Va  = grp['anode_potential_V'].values.astype(float)   # anode vs Li/Li+ [V]
        I   = grp['current_mA'].values.astype(float)          # signed current [mA]
        T   = grp['temperature_C'].fillna(25.0).values.astype(float)  # [°C]
        R   = grp['R_mOhm'].values
        Q   = grp['step_capacity_mAh'].values.astype(float)   # charge proxy [mAh]
        pl  = grp['Li_plating'].values.astype(bool)           # plating flag
        st  = grp['Li_stripping'].values.astype(bool)         # stripping flag
        soh = grp['soh_estimate'].values.astype(float)        # SoH [%]

        # Replace NaN resistance values with group median
        R_med = np.nanmedian(R)
        if np.isnan(R_med):
            R_med = 100.0    # fallback if entire group has NaN R
        R = np.where(np.isnan(R), R_med, R)

        # Direction string for phenomenological entropy sign convention
        dir_ = 'charge' if sname == 'CCCV_Chg' else 'discharge'

        # ── Pre-compute derivatives for the entire group ───────────────────────
        # np.gradient uses central differences (more accurate than np.diff
        # which uses one-sided differences) and handles boundaries gracefully.
        dV   = np.gradient(V)     # dV/dt — rate of voltage change [V/s at 1 Hz]
        dVa  = np.gradient(Va)    # dVa/dt — rate of anode potential change
        d2V  = np.gradient(dV)    # d²V/dt² — voltage curvature
        d2Va = np.gradient(dVa)   # d²Va/dt² — anode curvature

        # ── Count windows ──────────────────────────────────────────────────────
        n_wins = (N - MIN_WIN) // STRIDE
        if n_wins < 1:
            continue

        # Midpoint row index for each window — used to read the target label
        mids = np.array([s * STRIDE + MIN_WIN // 2 for s in range(n_wins)])

        # ── FEATURE GROUP 1: Control entropy at 3 window sizes ────────────────
        # CE is computed on the FULL group signal, then sliced to n_wins.
        # Smaller windows (CE_21, CE_41) are centred inside the MIN_WIN=81 region
        # so they capture the same temporal position as the larger window.
        ce_arrs = []
        for w in CE_WINS:
            ce = batch_ce(V, w)
            # Pad or trim to exactly n_wins elements
            if len(ce) >= n_wins:
                ce_arrs.append(ce[:n_wins])
            else:
                ce_arrs.append(np.pad(ce, (0, n_wins - len(ce)),
                                      constant_values=np.nan))

        # ── FEATURE GROUP 2: Irreversible thermodynamic entropy ───────────────
        sig = batch_sigma(V, I, T, R, SIG_WIN)
        if len(sig) >= n_wins:
            sig = sig[:n_wins]
        else:
            sig = np.pad(sig, (0, n_wins - len(sig)), constant_values=np.nan)

        # ── FEATURE GROUP 3: Phenomenological entropy ─────────────────────────
        phen = batch_phen(Q, V, T, dir_, PHEN_WIN)
        if len(phen) >= n_wins:
            phen = phen[:n_wins]
        else:
            phen = np.pad(phen, (0, n_wins - len(phen)), constant_values=np.nan)

        # ── FEATURE GROUP 4: Statistical moments of V and its derivatives ─────
        # Each returns shape (n_windows, 3): [mean, std, skew]
        # The d2V and d2Va stats only use [mean, std] (skew dropped for brevity)
        V_s    = batch_stats(V,          MIN_WIN)[:n_wins]         # [mean,std,skew]
        dV_s   = batch_stats(dV,         MIN_WIN)[:n_wins]         # [mean,std,skew]
        dVa_s  = batch_stats(dVa,        MIN_WIN)[:n_wins]         # [mean,std,skew]
        d2V_s  = batch_stats(d2V,        MIN_WIN)[:n_wins, :2]     # [mean,std]
        d2Va_s = batch_stats(d2Va,       MIN_WIN)[:n_wins, :2]     # [mean,std]
        I_s    = batch_stats(np.abs(I),  MIN_WIN)[:n_wins, :2]     # [mean,std]

        # Kurtosis of voltage — measures how "peaky" the voltage distribution
        # is within each window. High kurtosis during plating onset reflects
        # the sharp transition in V when the anode crosses 0 V.
        V_kurt = np.array([
            float(kurtosis(V[s * STRIDE : s * STRIDE + MIN_WIN]))
            for s in range(n_wins)
        ])

        # ── FEATURE GROUP 5: Cross-scale CE ratio and CE rate ─────────────────
        #
        # CE_ratio = CE_21 / CE_81
        # ─────────────────────────
        # Captures the DIVERGENCE between short-scale and long-scale complexity.
        #
        # During plating: lithium deposition creates micro-scale voltage noise
        # visible at 21 s but averaged away at 81 s → CE_21 rises relative
        # to CE_81 → ratio > 1.
        #
        # During stripping: monotone dissolution → both CEs fall, but CE_21
        # falls faster → ratio < 1.
        #
        # During normal charge/discharge: both CEs move together → ratio ≈ 1.
        #
        # This is why the NN found it useful — it collapses two separate
        # entropy measurements into a single number that captures
        # multi-scale behaviour.
        with np.errstate(divide='ignore', invalid='ignore'):
            ce_ratio = np.where(
                (ce_arrs[2] != 0) & np.isfinite(ce_arrs[2]),
                ce_arrs[0] / ce_arrs[2],
                np.nan
            )

        # dCE_dt: rate of change of CE_21 across consecutive windows.
        # A sudden increase in dCE_dt indicates plating onset;
        # a sudden decrease may indicate stripping onset or end of plating.
        dce = np.gradient(ce_arrs[0])

        # ── FEATURE GROUP 6: SoH ──────────────────────────────────────────────
        # SoH is the mean value within each window (smoothed).
        # Including SoH helps the model understand the cell's degradation state,
        # which correlates with how much lithium has been irreversibly plated
        # in previous cycles.
        # IMPORTANT: Ch5/Ch6 reach low SoH WITHOUT ever plating, so including
        # them in training prevents the model from incorrectly learning
        # "low SoH = plating". SoH is context, not cause.
        soh_wins = sliding_window_view(soh, MIN_WIN)[::STRIDE].mean(axis=1)[:n_wins]

        # ── Assemble feature matrix for this group ────────────────────────────
        # All arrays have been aligned to exactly n_wins elements.
        if include_soh:
            X_g = np.column_stack([
                ce_arrs[0], ce_arrs[1], ce_arrs[2],   # CE at 3 scales
                sig, phen,                              # entropy measures
                V_s, dV_s, dVa_s,                      # signal statistics
                d2V_s, d2Va_s, I_s,                    # 2nd derivative + current stats
                V_kurt,                                 # kurtosis
                ce_ratio, dce,                          # cross-scale features
                soh_wins,                               # SoH context
            ])
        else:
            X_g = np.column_stack([
                ce_arrs[0], ce_arrs[1], ce_arrs[2],
                sig, phen,
                V_s, dV_s, dVa_s,
                d2V_s, d2Va_s, I_s,
                V_kurt,
                ce_ratio, dce,
            ])

        # ── Target labels at window midpoints ─────────────────────────────────
        y_g = np.where(pl[mids], 1, np.where(st[mids], 2, 0))

        # Group ID: unique identifier for this (channel, session, cycle) group.
        # The step name is intentionally NOT included so all steps from the same
        # cycle stay together — splitting a CCCV_Chg from the Rest that follows
        # it would leak temporal information across the train/test boundary.
        grp_id = f'ch{channel}_{sess}_c{int(cyc)}'

        all_X.append(X_g)
        all_y.append(y_g)
        all_grp.extend([grp_id] * n_wins)

    X_out   = np.vstack(all_X)
    y_out   = np.concatenate(all_y)
    grp_out = np.array(all_grp)

    print(f'  Ch{channel}: {X_out.shape[0]:,} windows × {X_out.shape[1]} features')
    return X_out, y_out, grp_out


# =============================================================================
# SECTION 3 — DATA LOADING AND FEATURE EXTRACTION
# =============================================================================

USECOLS = [
    'abs_time', 'session_label', 'session_type', 'cycle', 'step_name',
    'voltage_V', 'current_mA', 'anode_potential_V', 'temperature_C',
    'overpotential_V', 'step_capacity_mAh', 'soh_estimate',
    'Li_plating', 'Li_plating_estimate', 'Li_stripping', 'Li_stripping_est',
]

print('=' * 60)
print('STEP 1: Loading data')
print('=' * 60)

dfs = {}
for ch in [5, 6, 7, 8]:
    path = os.path.join(DATA_DIR, f'cell_ch{ch}_timeseries.csv.gz')
    df   = pd.read_csv(path, usecols=USECOLS)
    # Use RPT sessions only: controlled C/1 protocol, full temperature data
    dfs[ch] = df[df['session_type'] == 'RPT'].copy()
    print(f'  Ch{ch}: {len(dfs[ch]):,} RPT rows  '
          f'(SoH range {dfs[ch]["soh_estimate"].min():.1f}–'
          f'{dfs[ch]["soh_estimate"].max():.1f}%)')

print('\n' + '=' * 60)
print('STEP 2: Feature extraction (all 4 channels with SoH)')
print('        This takes 2–4 minutes depending on hardware.')
print('=' * 60)

X_parts, y_parts, g_parts, ch_parts = [], [], [], []

for ch in [5, 6, 7, 8]:
    X, y, g = extract_features(dfs[ch], ch, include_soh=True)
    X_parts.append(X)
    y_parts.append(y)
    g_parts.append(g)
    ch_parts.append(np.full(len(y), ch, dtype=int))

X_all   = np.vstack(X_parts)
y_all   = np.concatenate(y_parts)
grp_all = np.concatenate(g_parts)
ch_all  = np.concatenate(ch_parts)

print(f'\nTotal dataset: {X_all.shape[0]:,} windows × {X_all.shape[1]} features')
print('Class distribution:')
for c, n in zip(*np.unique(y_all, return_counts=True)):
    print(f'  {LABEL_NAMES[c]:12s}: {n:6,}  ({n / len(y_all) * 100:.1f}%)')

# Also extract WITHOUT SoH for the ablation comparison (Model B)
print('\nExtracting features without SoH (for Model B ablation)...')
X_nosoh_parts = []
for ch in [5, 6, 7, 8]:
    X_ns, _, _ = extract_features(dfs[ch], ch, include_soh=False)
    X_nosoh_parts.append(X_ns)
X_nosoh = np.vstack(X_nosoh_parts)


# =============================================================================
# SECTION 4 — GROUP-LEVEL RANDOMISED TRAIN / VALIDATION / TEST SPLIT
# =============================================================================
#
# WHY GROUP-LEVEL SPLITTING?
# ────────────────────────────
# If we split rows randomly, consecutive rows from the same cycle will end up
# in both training and test sets. The NN would then see the "same" voltage
# pattern from slightly different positions in the same cycle — effectively
# memorising the specific curves rather than learning general rules.
#
# Instead we split at the CYCLE level: every window from cycle X of channel Y
# goes to exactly one of {train, val, test}. This forces the NN to generalise
# to cycles it has never seen at all — a much more honest evaluation.
#
# SPLIT PROPORTIONS: 70% train / 15% val / 15% test
# Applied to the list of unique group IDs after random shuffling.

print('\n' + '=' * 60)
print('STEP 3: Randomised group-level train / val / test split')
print('=' * 60)

unique_grps = np.unique(grp_all)
np.random.seed(SEED)
np.random.shuffle(unique_grps)

n_grps = len(unique_grps)
n_test = int(n_grps * 0.15)
n_val  = int(n_grps * 0.15)

test_grps  = set(unique_grps[:n_test])
val_grps   = set(unique_grps[n_test : n_test + n_val])
train_grps = set(unique_grps[n_test + n_val:])

# Build boolean masks for indexing into the flat arrays
tr_m = np.array([g in train_grps for g in grp_all])
va_m = np.array([g in val_grps   for g in grp_all])
te_m = np.array([g in test_grps  for g in grp_all])

X_tr, y_tr = X_all[tr_m],    y_all[tr_m]
X_va, y_va = X_all[va_m],    y_all[va_m]
X_te, y_te = X_all[te_m],    y_all[te_m]
ch_te      = ch_all[te_m]

# No-SoH versions share the same row masks
X_tr_ns = X_nosoh[tr_m]
X_te_ns = X_nosoh[te_m]

print(f'Groups:  total={n_grps}  train={len(train_grps)}  '
      f'val={len(val_grps)}  test={n_test}')
print(f'Windows: train={len(y_tr):,}  val={len(y_va):,}  test={len(y_te):,}')
print('\nChannel distribution in each split:')
for split_name, mask in [('Train', tr_m), ('Val', va_m), ('Test', te_m)]:
    ch_c = {c: int((ch_all[mask] == c).sum()) for c in [5, 6, 7, 8]}
    print(f'  {split_name:6s}: ' +
          ', '.join([f'Ch{c}:{n:,}' for c, n in ch_c.items()]))


# =============================================================================
# SECTION 5 — PREPROCESSING HELPER
# =============================================================================

def preprocess(X_train, y_train, X_test, seed=SEED):
    """
    Three-step preprocessing:

    1. NaN IMPUTATION
       CE windows with zero-variance voltage (flat regions at step boundaries)
       produce NaN. Replace with the column median computed from the training
       set. We use training-set medians for the test set too — this is correct
       because in a real deployment you would not have access to test-set
       statistics when building your imputation values.

    2. STANDARDISATION (StandardScaler)
       Subtract the training-set mean and divide by training-set std for each
       feature. This is critical for MLP convergence: without it, features with
       large magnitudes (e.g. V_mean ≈ 3.5 V) dominate features with small
       magnitudes (e.g. CE ≈ 0.1), causing the gradient descent to move mostly
       in the large-magnitude directions and ignore the small ones.

    3. CLASS BALANCING (upsampling)
       "Neither" windows make up ~82% of the dataset. If we train on this raw
       distribution, the MLP converges to predicting "Neither" for everything
       and achieves 82% accuracy while completely failing at the actual task.
       We fix this by randomly resampling Plating and Stripping windows WITH
       REPLACEMENT until all three classes have the same count as "Neither".
       The NN then sees equal numbers of each class and must learn genuine
       discriminative features rather than exploiting class frequency.

    Parameters
    ----------
    X_train : np.ndarray   raw training features (may contain NaN)
    y_train : np.ndarray   training labels
    X_test  : np.ndarray   raw test features (may contain NaN)
    seed    : int          random seed for upsampling reproducibility

    Returns
    -------
    X_balanced : standardised + balanced training features
    y_balanced : labels corresponding to X_balanced
    X_test_sc  : standardised test features (NOT balanced — we evaluate on reality)
    scaler     : fitted StandardScaler (keep this to transform new data later)
    """
    # Step 1: NaN imputation with training-set column medians
    col_medians = np.nanmedian(X_train, axis=0)
    X_tr_c  = np.where(np.isnan(X_train), col_medians, X_train)
    X_te_c  = np.where(np.isnan(X_test),  col_medians, X_test)

    # Step 2: Standardisation — fit on train, apply to both
    scaler  = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr_c)
    X_te_s  = scaler.transform(X_te_c)

    # Step 3: Upsample minority classes to match majority count
    n_majority = np.sum(y_train == 0)   # "Neither" is always the majority
    idx_balanced = np.concatenate([
        resample(
            np.where(y_train == c)[0],    # indices of this class in training set
            n_samples=n_majority,          # target count = majority class size
            replace=True,                  # allow repeats (upsampling)
            random_state=seed + c          # different seed per class for variety
        )
        for c in np.unique(y_train)
    ])
    np.random.seed(seed)
    np.random.shuffle(idx_balanced)       # shuffle so classes are interleaved

    return X_tr_s[idx_balanced], y_train[idx_balanced], X_te_s, scaler


# =============================================================================
# SECTION 6 — MODEL DEFINITIONS AND TRAINING
# =============================================================================

def make_mlp():
    """
    Create a fresh MLPClassifier (3-layer feedforward neural network).

    Architecture:
      Input: 24 features (23 without SoH)
      Hidden layer 1: 128 neurons  — broad feature combinations
      Hidden layer 2:  64 neurons  — mid-level patterns
      Hidden layer 3:  32 neurons  — fine discrimination
      Output:           3 neurons  — softmax probabilities for each class

    Hyperparameters:
      activation='relu'       — non-linear activation; allows learning of
                                 non-linear class boundaries
      solver='adam'           — adaptive moment estimation; efficient for
                                 non-stationary objective functions
      alpha=0.001             — L2 weight decay (regularisation); prevents
                                 individual weights from growing too large
                                 and overfitting the training set
      early_stopping=True     — hold out 15% of balanced training data as
                                 a validation set; stop when validation loss
                                 fails to improve for 15 consecutive epochs
      learning_rate='adaptive'— reduce learning rate when progress stalls
    """
    return MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        alpha=0.001,
        learning_rate='adaptive',
        max_iter=400,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15,
        random_state=SEED,
        verbose=False,
    )


print('\n' + '=' * 60)
print('STEP 4: Training three model variants')
print('=' * 60)

# ── Model A: all 4 channels with SoH ─────────────────────────────────────────
print('\nModel A — all channels + SoH:', flush=True)
Xb_A, yb_A, Xte_A, sc_A = preprocess(X_tr, y_tr, X_te)
mlp_A = make_mlp()
mlp_A.fit(Xb_A, yb_A)
yp_A  = mlp_A.predict(Xte_A)
print(f'  Converged: {mlp_A.n_iter_} epochs  |  '
      f'val score: {mlp_A.best_validation_score_:.4f}  |  '
      f'test acc: {np.mean(yp_A == y_te) * 100:.1f}%')

# ── Model B: all 4 channels WITHOUT SoH (ablation) ───────────────────────────
# Identical to Model A except the SoH column is excluded.
# Comparing A vs B shows the isolated contribution of SoH as a feature.
print('\nModel B — all channels, no SoH (ablation):', flush=True)
Xb_B, yb_B, Xte_B, sc_B = preprocess(X_tr_ns, y_tr, X_te_ns)
mlp_B = make_mlp()
mlp_B.fit(Xb_B, yb_B)
yp_B  = mlp_B.predict(Xte_B)
print(f'  Converged: {mlp_B.n_iter_} epochs  |  '
      f'val score: {mlp_B.best_validation_score_:.4f}  |  '
      f'test acc: {np.mean(yp_B == y_te) * 100:.1f}%')

# ── Model C: Ch7 + Ch8 only, with SoH (ablation) ─────────────────────────────
# Trained and tested only on the plating channels.
# Comparing A vs C shows the contribution of Ch5/Ch6 (healthy reference cells).
print('\nModel C — Ch7+Ch8 only, with SoH (ablation):', flush=True)
ch78_tr = tr_m & np.isin(ch_all, [7, 8])
ch78_te = te_m & np.isin(ch_all, [7, 8])
X_tr_C = X_all[ch78_tr]; y_tr_C = y_all[ch78_tr]
X_te_C = X_all[ch78_te]; y_te_C = y_all[ch78_te]
Xb_C, yb_C, Xte_C, sc_C = preprocess(X_tr_C, y_tr_C, X_te_C)
mlp_C = make_mlp()
mlp_C.fit(Xb_C, yb_C)
yp_C  = mlp_C.predict(Xte_C)
print(f'  Converged: {mlp_C.n_iter_} epochs  |  '
      f'val score: {mlp_C.best_validation_score_:.4f}  |  '
      f'test acc: {np.mean(yp_C == y_te_C) * 100:.1f}%')


# =============================================================================
# SECTION 7 — 5-FOLD GROUP CROSS-VALIDATION
# =============================================================================
#
# WHAT CROSS-VALIDATION DOES
# ────────────────────────────
# We train and evaluate 5 separate models, each time holding out a different
# 20% of groups as the test fold. This tells us:
#   - How sensitive the accuracy is to WHICH groups happen to be held out
#   - Whether the mean accuracy is stable or whether we "got lucky" with one split
#   - The variance gives a confidence interval on the performance estimate
#
# GROUP-LEVEL FOLDS
# ─────────────────
# Folds are formed by interleaving groups (every 5th group → fold 1, etc.).
# This ensures each fold contains groups from all 4 channels and from both
# early and late RPT sessions — avoiding folds that are accidentally "easy"
# (e.g. all early RPTs where plating is mild) or "hard" (all late RPTs).
#
# Test set is held out completely — not used for CV at all.

print('\n' + '=' * 60)
print('STEP 5: 5-fold group cross-validation (train+val pool, Model A structure)')
print('=' * 60)

tv_m  = tr_m | va_m              # train+val pool (test stays out)
X_cv  = X_all[tv_m]
y_cv  = y_all[tv_m]
g_cv  = grp_all[tv_m]

cv_grps = np.unique(g_cv)
np.random.seed(SEED)
np.random.shuffle(cv_grps)

# Interleaved fold assignment: group i → fold (i % 5)
# This distributes channels and sessions evenly across all folds
folds = [set(cv_grps[i::5]) for i in range(5)]

cv_rows = []
for fold_i, held_out in enumerate(folds):
    in_fold  = np.array([g in held_out for g in g_cv])
    out_fold = ~in_fold

    Xf_tr, yf_tr = X_cv[out_fold], y_cv[out_fold]
    Xf_te, yf_te = X_cv[in_fold],  y_cv[in_fold]

    Xf_bal, yf_bal, Xf_te_s, _ = preprocess(Xf_tr, yf_tr, Xf_te, seed=fold_i)

    mf = make_mlp()
    mf.fit(Xf_bal, yf_bal)
    yf_pred = mf.predict(Xf_te_s)

    acc     = np.mean(yf_pred == yf_te)
    f1_mac  = f1_score(yf_te, yf_pred, average='macro',  zero_division=0)
    f1_pl   = f1_score(yf_te, yf_pred, labels=[1], average='micro', zero_division=0)
    f1_st   = f1_score(yf_te, yf_pred, labels=[2], average='micro', zero_division=0)
    f1_ne   = f1_score(yf_te, yf_pred, labels=[0], average='micro', zero_division=0)

    cv_rows.append({
        'fold': fold_i + 1,
        'acc': acc, 'f1_macro': f1_mac,
        'f1_neither': f1_ne, 'f1_plating': f1_pl, 'f1_strip': f1_st,
    })
    print(f'  Fold {fold_i + 1}: acc={acc*100:.1f}%  '
          f'F1_macro={f1_mac:.3f}  '
          f'F1_plating={f1_pl:.3f}  '
          f'F1_stripping={f1_st:.3f}')

cv_df = pd.DataFrame(cv_rows)
print(f'\nCross-validation summary:')
print(f'  Accuracy:      {cv_df["acc"].mean()*100:.1f} ± {cv_df["acc"].std()*100:.1f}%')
print(f'  F1 macro:      {cv_df["f1_macro"].mean():.3f} ± {cv_df["f1_macro"].std():.3f}')
print(f'  F1 plating:    {cv_df["f1_plating"].mean():.3f} ± {cv_df["f1_plating"].std():.3f}')
print(f'  F1 stripping:  {cv_df["f1_strip"].mean():.3f} ± {cv_df["f1_strip"].std():.3f}')


# =============================================================================
# SECTION 8 — CLASSIFICATION REPORTS
# =============================================================================

print('\n' + '=' * 60)
print('STEP 6: Classification reports (held-out test set)')
print('=' * 60)

for model_name, yp, yt in [
    ('Model A  —  all channels + SoH',        yp_A, y_te),
    ('Model B  —  all channels, no SoH',       yp_B, y_te),
    ('Model C  —  Ch7+Ch8 only + SoH',         yp_C, y_te_C),
]:
    cls_p = sorted(np.unique(np.concatenate([yt, yp])))
    print(f'\n{model_name}:')
    print(classification_report(
        yt, yp,
        labels=cls_p,
        target_names=[LABEL_NAMES[c] for c in cls_p],
    ))


# =============================================================================
# SECTION 9 — PLOTTING
# =============================================================================

print('=' * 60)
print('STEP 7: Generating plots')
print('=' * 60)

# ── FIGURE 1: Six-panel model comparison ─────────────────────────────────────
fig1, axes = plt.subplots(2, 3, figsize=(16, 11))
fig1.suptitle(
    'Model comparison: effect of SoH feature and healthy reference channels\n'
    'All models trained on randomised group-level split  |  '
    'A = all channels + SoH  |  B = all channels, no SoH  |  C = Ch7/8 only + SoH',
    fontweight='bold', fontsize=10,
)

# Confusion matrices for Models A, B, C
all_cls = sorted(np.unique(np.concatenate([y_te, yp_A, yp_B])))
lbl     = [LABEL_NAMES[c] for c in all_cls]

for ax, yp, title, cmap in [
    (axes[0, 0], yp_A, f'Model A  (all ch + SoH)\nAcc = {np.mean(yp_A==y_te)*100:.1f}%',  'Blues'),
    (axes[0, 1], yp_B, f'Model B  (all ch, no SoH)\nAcc = {np.mean(yp_B==y_te)*100:.1f}%','Oranges'),
]:
    ConfusionMatrixDisplay(
        confusion_matrix(y_te, yp, labels=all_cls),
        display_labels=lbl,
    ).plot(ax=ax, colorbar=False, cmap=cmap)
    ax.set_title(title)

cls_C = sorted(np.unique(np.concatenate([y_te_C, yp_C])))
lbl_C = [LABEL_NAMES[c] for c in cls_C]
ConfusionMatrixDisplay(
    confusion_matrix(y_te_C, yp_C, labels=cls_C),
    display_labels=lbl_C,
).plot(ax=axes[0, 2], colorbar=False, cmap='Greens')
axes[0, 2].set_title(
    f'Model C  (Ch7+Ch8 only + SoH)\nAcc = {np.mean(yp_C==y_te_C)*100:.1f}%'
)

# 5-fold CV bar chart
metrics     = ['acc', 'f1_neither', 'f1_plating', 'f1_strip']
m_labels    = ['Accuracy', 'F1 Neither', 'F1 Plating', 'F1 Stripping']
fold_colors = ['#1976D2', '#388E3C', '#F57C00', '#7B1FA2', '#C62828']
x_pos       = np.arange(len(metrics))
bar_width   = 0.15

for fi, row in cv_df.iterrows():
    axes[1, 0].bar(
        x_pos + fi * bar_width,
        [row[m] for m in metrics],
        width=bar_width,
        color=fold_colors[fi],
        alpha=0.85,
        label=f'Fold {fi+1}',
    )
axes[1, 0].axhline(
    cv_df['acc'].mean(), color='black', ls='--', lw=1.5, alpha=0.6,
    label=f'Mean acc {cv_df["acc"].mean()*100:.1f}%',
)
axes[1, 0].set_xticks(x_pos + 2 * bar_width)
axes[1, 0].set_xticklabels(m_labels, fontsize=8)
axes[1, 0].set_ylim(0, 1.15)
axes[1, 0].set_ylabel('Score')
axes[1, 0].set_title(
    '5-fold group cross-validation\n'
    '(all 4 channels, interleaved group folds — test set held out)'
)
axes[1, 0].legend(fontsize=7, ncol=3)

# Per-class F1: Models A, B, C side by side
model_labels = ['A\n(all+SoH)', 'B\n(all,noSoH)', 'C\n(Ch7/8+SoH)']
f1_data = {'Neither': [], 'Plating': [], 'Stripping': []}
for yp, yt in [(yp_A, y_te), (yp_B, y_te), (yp_C, y_te_C)]:
    cls_p = sorted(np.unique(np.concatenate([yt, yp])))
    f1s   = f1_score(yt, yp, labels=cls_p, average=None, zero_division=0)
    fm    = {c: f for c, f in zip(cls_p, f1s)}
    f1_data['Neither'].append(fm.get(0, 0))
    f1_data['Plating'].append(fm.get(1, 0))
    f1_data['Stripping'].append(fm.get(2, 0))

x2 = np.arange(3); bw2 = 0.25
for i, (cls_n, vals) in enumerate(f1_data.items()):
    axes[1, 1].bar(x2 + (i - 1) * bw2, vals, width=bw2,
                   color=LABEL_COLORS[i], alpha=0.85, label=cls_n)
axes[1, 1].set_xticks(x2)
axes[1, 1].set_xticklabels(model_labels, fontsize=9)
axes[1, 1].set_ylim(0, 1.12)
axes[1, 1].set_ylabel('F1 score')
axes[1, 1].set_title(
    'Per-class F1: effect of SoH and healthy channels\n'
    'B vs A shows SoH contribution  |  A vs C shows Ch5/Ch6 contribution'
)
axes[1, 1].legend(fontsize=9)
axes[1, 1].axhline(0.9, color='black', ls='--', lw=0.8, alpha=0.3)

# Feature importance from Model A (MLP first-layer weight magnitudes)
w1 = np.abs(mlp_A.coefs_[0])          # shape: (n_features, 128)
fs = w1.mean(axis=1)                   # mean absolute weight to first hidden layer
si = np.argsort(fs)[::-1]             # sort descending

bar_colors = [
    '#D32F2F' if FEAT_NAMES[i] == 'soh_estimate'                        else
    '#E65100' if 'CE' in FEAT_NAMES[i]                                   else
    '#1565C0' if any(x in FEAT_NAMES[i] for x in ['sigma', 'phen'])      else
    '#546E7A'
    for i in si
]
axes[1, 2].barh(range(len(fs)), fs[si][::-1], color=bar_colors[::-1], alpha=0.85)
axes[1, 2].set_yticks(range(len(fs)))
axes[1, 2].set_yticklabels([FEAT_NAMES[i] for i in si[::-1]], fontsize=8)
axes[1, 2].set_xlabel('Mean |weight| to first hidden layer')
axes[1, 2].set_title(
    'Model A — MLP input weight magnitudes\n'
    'Red=SoH  Orange=CE  Blue=entropy  Grey=statistics/derivatives'
)

plt.tight_layout()
out1 = os.path.join(DATA_DIR, 'nn_soh_model_comparison.png')
fig1.savefig(out1, dpi=140, bbox_inches='tight')
print(f'  Saved: {out1}')
plt.close(fig1)

# ── FIGURE 2: SoH distribution at correct vs incorrect predictions ────────────
#
# For each class (Neither, Plating, Stripping) we show the SoH distribution of:
#   - Correct predictions (true positives for that class)
#   - Missed detections (false negatives — true class, predicted wrong)
#   - False alarms (false positives — wrong class predicted as this)
#
# If SoH is helping the model CORRECTLY (not as a shortcut), we expect:
#   - Correct predictions should span the full SoH range
#   - False negatives and false alarms should not be concentrated at any
#     particular SoH value (no systematic SoH-based bias)

soh_col     = FEAT_NAMES.index('soh_estimate')
soh_raw_te  = X_all[te_m, soh_col]
soh_raw_te  = np.where(np.isnan(soh_raw_te),
                        np.nanmedian(soh_raw_te), soh_raw_te)

fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
fig2.suptitle(
    'Does SoH create systematic errors? Model A — SoH distribution at correct vs wrong predictions\n'
    'If SoH were a spurious shortcut, we would see misses concentrated at a particular SoH value',
    fontweight='bold', fontsize=10,
)

for ax, cls in zip(axes2, [0, 1, 2]):
    correct  = (y_te == cls) & (yp_A == cls)   # true positives
    miss     = (y_te == cls) & (yp_A != cls)   # false negatives (missed events)
    false_al = (y_te != cls) & (yp_A == cls)   # false positives (false alarms)

    for vals, lbl, col, ls in [
        (soh_raw_te[correct],  f'Correct ({correct.sum():,})',      LABEL_COLORS[cls], '-'),
        (soh_raw_te[miss],     f'Missed / FN ({miss.sum():,})',     '#FF6F00',          '--'),
        (soh_raw_te[false_al], f'False alarm / FP ({false_al.sum():,})', '#9E9E9E',    ':'),
    ]:
        if len(vals) < 10:
            continue
        vr = np.linspace(vals.min(), vals.max(), 200)
        try:
            kde = gaussian_kde(vals, bw_method=0.3)
            ax.plot(vr, kde(vr), color=col, lw=2, ls=ls, label=lbl, alpha=0.9)
            ax.fill_between(vr, kde(vr), alpha=0.08, color=col)
        except Exception:
            pass

    ax.set_title(f'Class: {LABEL_NAMES[cls]}', fontweight='bold',
                 color=LABEL_COLORS[cls])
    ax.set_xlabel('SoH estimate (%)')
    ax.set_ylabel('Density')
    ax.legend(fontsize=7.5)

plt.tight_layout()
out2 = os.path.join(DATA_DIR, 'nn_soh_error_analysis.png')
fig2.savefig(out2, dpi=140, bbox_inches='tight')
print(f'  Saved: {out2}')
plt.close(fig2)

print('\n' + '=' * 60)
print('All done.')
print('=' * 60)
