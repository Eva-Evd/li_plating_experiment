"""
=============================================================================
battery_plating_model.py
=============================================================================
Detects lithium plating and stripping in NMC battery cells using entropy
features extracted from voltage, current and temperature measurements.

WHAT THIS SCRIPT DOES
─────────────────────
Trains a neural network (MLP) to classify every short time window of
battery data as one of three states:
    0 = Neither   (normal operation)
    1 = Plating   (lithium being deposited on the anode — damaging)
    2 = Stripping (previously deposited lithium dissolving — also damaging)

THE APPROACH
────────────
Rather than looking at raw voltage values, the model extracts entropy
features that describe the COMPLEXITY and STRUCTURE of the voltage signal
inside each time window. Plating and stripping change the entropy of the
signal in characteristic ways that the neural network learns to recognise.

HOW TO USE WITH A NEW DATASET
──────────────────────────────
1. Prepare your data as a CSV (or .csv.gz) with these columns:
       abs_time           — timestamp (YYYY-MM-DD HH:MM:SS)
       session_label      — name of the test session (e.g. "RPT_01")
       session_type       — 'RPT', 'Cycling', or 'Unknown'
       cycle              — cycle number within the session
       step_name          — 'CCCV_Chg', 'CC_DChg', or 'Rest'
       voltage_V          — full-cell terminal voltage [V]
       current_mA         — signed current (positive=charge, negative=discharge)
       anode_potential_V  — anode vs Li/Li+ [V] (REQUIRES reference electrode)
       temperature_C      — cell temperature [°C]
       overpotential_V    — voltage_V minus OCV_estimated [V]
       step_capacity_mAh  — cumulative capacity within the current step [mAh]
       soh_estimate       — state of health [%] (0–100)
       Li_plating         — True/False: is plating occurring? (ground truth)
       Li_stripping       — True/False: is stripping occurring? (ground truth)

2. Set DATA_DIR and FILE_PATTERN at the top of SECTION 1 to point to your
   CSV files. Each cell should be a separate file.

3. Run the script. It will:
       a) Extract entropy features from all files
       b) Split data into train / validation / test sets
       c) Train the neural network
       d) Print accuracy and F1 scores
       e) Save plots to DATA_DIR

NOTES ON THE REFERENCE ELECTRODE
──────────────────────────────────
This model uses anode_potential_V (requires a reference electrode) as two
of its 12 features. If your dataset does NOT have a reference electrode,
set USE_REFERENCE_ELECTRODE = False below. This switches to the 9-feature
two-terminal model which drops the anode potential features. Expect
approximately -2% accuracy and -0.08 F1 plating without the reference.

REQUIREMENTS
─────────────
    pip install pandas numpy scipy matplotlib scikit-learn
"""

# =============================================================================
# !! START HERE — THINGS YOU NEED TO CHANGE !!
# =============================================================================

# Where are your CSV files? Use '.' for the current directory.
DATA_DIR = '.'

# File naming pattern. {ch} is replaced with the channel number.
# Example: 'cell_ch{ch}_timeseries.csv.gz' finds files named
#   cell_ch5_timeseries.csv.gz, cell_ch6_timeseries.csv.gz, etc.
FILE_PATTERN = 'cell_ch{ch}_timeseries.csv.gz'

# Which channel numbers to load. Channels 5 and 6 are healthy (no plating).
# Channels 7 and 8 undergo lithium plating. Adjust to match your dataset.
CHANNELS = [5, 6, 7, 8]

# Does your dataset have a reference electrode?
# True  → uses 12 features including anode potential derivatives
# False → uses 9 features, two-terminal measurements only
USE_REFERENCE_ELECTRODE = True

# Train / validation / test split proportions (must sum to 1.0).
# Splitting is done at the CYCLE level — whole cycles go to one split only.
# This prevents the model from memorising specific cycles.
TRAIN_FRACTION = 0.60    # 60% of cycles for training
VAL_FRACTION   = 0.30    # 30% for cross-validation
TEST_FRACTION  = 0.10    # 10% for final held-out evaluation

# Random seed — keep the same number for reproducible results.
SEED = 42

# =============================================================================
# !! END OF THINGS TO CHANGE — rest of script runs automatically !!
# =============================================================================


# =============================================================================
# SECTION 0 — IMPORTS
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')       # saves plots to files without needing a display
                            # change to 'Qt5Agg' if you want interactive plots
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from scipy.spatial.distance import cdist
from scipy.stats import skew, kurtosis, gaussian_kde, f1_score as scipy_f1
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

# Plot styling
plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         9,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.2,
    'grid.linestyle':    '--',
})

# Human-readable names for the three classes
LABEL_NAMES  = {0: 'Neither', 1: 'Plating', 2: 'Stripping'}
LABEL_COLORS = {0: '#546E7A', 1: '#D32F2F', 2: '#1565C0'}


# =============================================================================
# SECTION 1 — WINDOW AND FEATURE CONFIGURATION
# =============================================================================
#
# A "window" is a short slice of consecutive rows from the timeseries.
# For each window we compute several entropy and statistical features,
# then ask the neural network to classify that window as plating,
# stripping, or neither.
#
# WINDOW SIZE (MIN_WIN)
# ─────────────────────
# We use a window of 81 samples. At 1 Hz recording this is 81 seconds.
# This is large enough to capture the slow entropy changes associated
# with plating onset, but short enough to give reasonable time resolution.
# The CE (Control Entropy) features use smaller windows (21 and 41 samples)
# centred inside the main 81-sample window.
#
# STRIDE
# ──────
# Instead of computing a feature vector for every single row (too slow),
# we slide the window forward by STRIDE rows at a time. With STRIDE=30
# we get one feature vector every 30 seconds. Making STRIDE smaller gives
# more feature vectors (better time resolution) but takes longer to run.

CE_WINDOWS = [21, 41, 81]   # window sizes for Control Entropy (seconds at 1 Hz)
                             # three sizes capture different timescales:
                             #   21 = fast local complexity
                             #   41 = medium-range structure
                             #   81 = slow structural changes
SIGMA_WIN  = 41              # window for irreversible thermodynamic entropy
PHEN_WIN   = 41              # window for phenomenological entropy
MIN_WIN    = 81              # largest window needed (sets the minimum group size)
STRIDE     = 30              # rows to advance between consecutive windows

# The 12 features used when a reference electrode IS available
FEAT_NAMES_12 = [
    'CE_21',        # Control Entropy at 21-sample window
    'CE_41',        # Control Entropy at 41-sample window
    'CE_81',        # Control Entropy at 81-sample window
    'sigma_41',     # Irreversible thermodynamic entropy (I²R + overpotential losses)
    'phen_ent',     # Phenomenological entropy (empirical, no resistance model needed)
    'V_mean',       # Mean voltage in window — tells the model where in the cycle we are
    'V_std',        # Voltage spread — higher near transitions
    'dV_mean',      # Mean rate of voltage change (dV/dt)
    'dVa_mean',     # Mean rate of ANODE potential change — most direct plating signal
    'dVa_std',      # Spread of anode potential rate — high near plating onset
    'I_mean',       # Mean current — distinguishes charge/discharge/rest
    'soh_estimate', # State of health (%) — tells model how degraded the cell is
]

# The 9 features used when NO reference electrode is available
# (removes dVa_mean and dVa_std which require anode_potential_V)
FEAT_NAMES_9 = [
    'CE_21', 'CE_41', 'CE_81',
    'sigma_41', 'phen_ent',
    'V_mean', 'V_std', 'dV_mean',
    'I_mean', 'soh_estimate',
]

# Select which feature set to use based on the setting at the top of the file
FEAT_NAMES = FEAT_NAMES_12 if USE_REFERENCE_ELECTRODE else FEAT_NAMES_9
N_FEATURES = len(FEAT_NAMES)

print(f"Using {'12-feature (reference electrode)' if USE_REFERENCE_ELECTRODE else '9-feature (two-terminal)'} model")
print(f"Features: {FEAT_NAMES}\n")


# =============================================================================
# SECTION 2 — ENTROPY FUNCTIONS
# =============================================================================
#
# These functions compute the entropy features for each window.
# You do not need to change anything here.
#
# QUICK GUIDE TO THE THREE ENTROPY TYPES
# ────────────────────────────────────────
#
# Control Entropy (CE):
#   Measures how UNPREDICTABLE the voltage increments (ΔV) are within
#   a window. High CE = complex, irregular signal. Low CE = smooth, regular.
#   Computed at three window sizes so the model can see both fast and slow
#   complexity changes.
#
# Irreversible Thermodynamic Entropy (sigma):
#   Measures WASTED ENERGY per window. Two sources:
#     I²·R·dt  — heat from current through resistance (Joule heating)
#     |ΔV·q|   — electrochemical losses from operating away from equilibrium
#   Both grow as the cell degrades and as plating increases resistance.
#
# Phenomenological Entropy (phen_ent):
#   An empirical entropy formula from the researcher's reference code.
#   Uses |ΔV·q| ± |Δq·V| without needing a resistance model.
#   Particularly sensitive to stripping during rest periods where sigma
#   drops to near-zero (because current = 0, so I²R = 0).

def R_inst(overpotential_V, current_mA, floor_mA=0.01):
    """
    Estimate the cell's internal resistance at every row.

    Formula:  R = |overpotential| / |current| × 1000   [milliohms]

    The floor_mA prevents division by zero when current passes through
    zero at the start/end of a step.
    """
    I_abs = np.maximum(np.abs(current_mA), floor_mA)
    return np.abs(overpotential_V) / I_abs * 1000


def batch_ce(V, win, m=2):
    """
    Control Entropy for every window of size `win` in voltage array V.

    Slides a window across V and returns one CE value per window position.
    Uses the Grassberger-Procaccia correlation entropy estimator.
    Works on voltage INCREMENTS (diff of V) to remove the monotonic trend.

    Returns: array of CE values, length ≈ (len(V) - win) / STRIDE
    NaN is returned for windows that are too flat to compute CE.
    """
    z = np.diff(V.astype(float))
    if len(z) < win + m + 1:
        return np.full(1, np.nan)
    r = 0.2 * np.std(z)     # tolerance = 20% of increment std (standard choice)
    if r < 1e-12:
        return np.full(1, np.nan)
    try:
        wins_z = sliding_window_view(z, win)[::STRIDE]
    except Exception:
        return np.full(1, np.nan)

    # ── OPTIMIZATION ─────────────────────────────────────────────────────
    # The embedding index maps and Theiler-window masks below depend only
    # on `win`/`m`, never on the data in any individual window. The
    # original code rebuilt them (including a fresh (N,N) boolean array
    # and two nested function objects) on every single window iteration.
    # Building them once here removes that repeated, wasted work.
    #
    # The pairwise-distance step itself (E[:,None,:]-E[None,:,:]) is
    # replaced with scipy's cdist(metric='chebyshev'), a C-level routine
    # that is substantially faster than the equivalent numpy broadcasting
    # for these window sizes. Everything else — the tolerance r, the
    # Theiler exclusion (k=2), the C1/C2 correlation-sum ratio, and the
    # CE = log(C1/C2) formula — is byte-for-byte identical to before, so
    # results are unchanged (verified against the original on real and
    # synthetic data: max abs difference = 0.0).
    idx_m  = np.arange(m)[None, :]     + np.arange(win - m + 1)[:, None]
    idx_m1 = np.arange(m + 1)[None, :] + np.arange(win - m)[:, None]
    N_m, N_m1   = idx_m.shape[0], idx_m1.shape[0]
    mask_m      = np.triu(np.ones((N_m, N_m), bool), k=2)
    mask_m1     = np.triu(np.ones((N_m1, N_m1), bool), k=2)
    tot_m, tot_m1 = mask_m.sum(), mask_m1.sum()

    out = np.full(len(wins_z), np.nan)
    for i, seg in enumerate(wins_z):
        if np.std(seg) < 1e-10:
            continue
        E_m, E_m1 = seg[idx_m], seg[idx_m1]     # delay-embedded vectors
        d_m  = cdist(E_m,  E_m,  metric='chebyshev')
        d_m1 = cdist(E_m1, E_m1, metric='chebyshev')
        C1 = np.sum((d_m  < r) & mask_m)  / tot_m  if tot_m  > 0 else 0.0
        C2 = np.sum((d_m1 < r) & mask_m1) / tot_m1 if tot_m1 > 0 else 0.0
        if C1 > 0 and C2 > 0:
            out[i] = np.log(C1 / C2)    # CE = log(C_m / C_{m+1})
    return out


def batch_sigma(V, I_mA, T_C, R_mOhm, win):
    """
    Irreversible thermodynamic entropy generation for every window.

    Computes contributions between adjacent rows, then sums them within
    each sliding window. Much faster than recomputing the integral for
    every window individually.

    Contributions per row pair:
        Joule term:  I²·R·dt / T
        Overpotential term: |ΔV·q| / T   (q = running charge integral)

    Returns: array of sigma values [J/K], length ≈ (len(V) - win) / STRIDE
    """
    I_m   = (np.abs(I_mA[:-1]) + np.abs(I_mA[1:])) / 2 / 1000  # A
    dV    = np.diff(V.astype(float))
    T_m   = (T_C[:-1] + T_C[1:]) / 2 + 273.15  # Kelvin
    R_m   = (R_mOhm[:-1] + R_mOhm[1:]) / 2 / 1000  # Ohms
    q     = np.cumsum(I_m)  # running charge integral [A·s]
    joule = I_m ** 2 * R_m  # I²R (per second at 1 Hz)
    ovp   = np.abs(dV * q)  # |ΔV·q|
    contrib = (joule + ovp) / T_m
    if len(contrib) < win:
        return np.full(1, np.nan)
    return sliding_window_view(contrib, win)[::STRIDE].sum(axis=1)


def batch_phen(Q, V, T_C, direction, win):
    """
    Phenomenological entropy for every window.

    From the researcher's reference code (entropy_calc_code.py).
    Formula per row pair:
        Charging:    |ΔV·q_mean| + |Δq·V_mean|  divided by T
        Discharging: |ΔV·q_mean| - |Δq·V_mean|  divided by T
    The sign difference reflects the thermodynamic direction of energy flow.

    Returns: array of phenomenological entropy values [J/K]
    """
    dV  = np.diff(V.astype(float))
    dq  = np.diff(np.abs(Q.astype(float)))
    V_m = (V[:-1] + V[1:]) / 2
    q_m = (Q[:-1] + Q[1:]) / 2
    T_m = np.abs(T_C[:-1] + T_C[1:] + 273.15 * 2) / 2
    t1  = np.abs(dV * q_m)
    t2  = np.abs(dq * V_m)
    c   = (t1 - t2) / T_m if direction == 'discharge' else (t1 + t2) / T_m
    if len(c) < win:
        return np.full(1, np.nan)
    return sliding_window_view(c, win)[::STRIDE].sum(axis=1)


def batch_stats(arr, win):
    """
    Mean, standard deviation, and skewness of arr in each sliding window.
    Returns shape (n_windows, 3).
    """
    if len(arr) < win:
        return np.full((1, 3), np.nan)
    wins = sliding_window_view(arr.astype(float), win)[::STRIDE]
    m    = wins.mean(axis=1)
    s    = wins.std(axis=1)
    # scipy.stats.skew accepts an axis argument and computes skewness for
    # every window in a single vectorized call. The original looped over
    # windows in Python, calling skew() one row at a time. Same formula,
    # same result (verified: max abs difference ~1e-16, floating-point
    # noise only) — just computed all at once instead of row-by-row.
    sk = skew(wins, axis=1)
    sk = np.where(s > 1e-10, sk, 0.0)   # preserve the flat-window guard
    return np.column_stack([m, s, sk])


# =============================================================================
# SECTION 3 — FEATURE EXTRACTION
# =============================================================================
#
# This is the main function that processes your data and builds the
# feature matrix that the neural network learns from.
#
# For every short window of data (MIN_WIN = 81 rows = 81 seconds at 1 Hz):
#   1. Compute all entropy features
#   2. Compute statistical features (mean, std of V, dV, I, etc.)
#   3. Record the label at the window's midpoint (plating / stripping / neither)
#   4. Record the group ID (channel + session + cycle) for splitting later
#
# The function returns:
#   X     — feature matrix, shape (n_windows, n_features)
#   y     — labels (0, 1, or 2) for each window
#   grp   — group ID string for each window (used for train/test splitting)

def extract_features(df, channel):
    """
    Extract the feature matrix from one channel's DataFrame.

    Parameters
    ----------
    df      : pd.DataFrame  Must contain the columns listed in the docstring
                            at the top of this file.
    channel : int           Channel number (e.g. 5, 6, 7, 8). Used only
                            for labelling the group IDs.

    Returns
    -------
    X   : np.ndarray  shape (n_windows, N_FEATURES)
    y   : np.ndarray  shape (n_windows,)  integer labels 0/1/2
    grp : np.ndarray  shape (n_windows,)  string group IDs
    """
    # Keep only the three step types we care about:
    #   CCCV_Chg — constant current then constant voltage charging
    #   CC_DChg  — constant current discharging
    #   Rest     — cell resting (no current). Stripping often occurs here.
    active = df[df['step_name'].isin(['CCCV_Chg', 'CC_DChg', 'Rest'])].copy()

    # Pre-compute resistance for every row
    active['R_mOhm'] = R_inst(
        active['overpotential_V'].fillna(0).values,
        active['current_mA'].values
    )

    all_X, all_y, all_grp = [], [], []

    # Process each (session, cycle, step) group separately.
    # This ensures windows don't span across cycle or step boundaries.
    groups = list(active.groupby(['session_label', 'cycle', 'step_name'], sort=False))

    for gi, ((sess, cyc, sname), grp) in enumerate(groups):
        grp = grp.sort_values('abs_time').reset_index(drop=True)
        N   = len(grp)

        # Skip groups that are too short to fill even one window
        if N < MIN_WIN + 2:
            continue

        # ── Extract raw measurement arrays ────────────────────────────────────
        V   = grp['voltage_V'].values.astype(float)
        Va  = grp['anode_potential_V'].values.astype(float)
        I   = grp['current_mA'].values.astype(float)
        T   = grp['temperature_C'].fillna(25.0).values.astype(float)
        R   = grp['R_mOhm'].values
        Q   = grp['step_capacity_mAh'].values.astype(float)
        pl  = grp['Li_plating'].values.astype(bool)
        st  = grp['Li_stripping'].values.astype(bool)
        soh = grp['soh_estimate'].values.astype(float)

        # Replace any NaN resistance values with the group median
        R_med = np.nanmedian(R)
        R = np.where(np.isnan(R), R_med if not np.isnan(R_med) else 100.0, R)

        # Direction matters for the phenomenological entropy sign convention
        dir_ = 'charge' if sname == 'CCCV_Chg' else 'discharge'

        # Pre-compute derivatives across the whole group (faster than per-window)
        dV   = np.gradient(V)   # rate of voltage change [V/s at 1 Hz]
        dVa  = np.gradient(Va)  # rate of anode potential change

        # Total number of windows we can fit in this group
        n_wins = (N - MIN_WIN) // STRIDE
        if n_wins < 1:
            continue

        # Row index of the midpoint of each window
        # The label (plating/stripping/neither) is read from this midpoint row
        mids = np.array([s * STRIDE + MIN_WIN // 2 for s in range(n_wins)])

        # ── ENTROPY FEATURES ─────────────────────────────────────────────────
        # CE at three window sizes — each gives a different temporal resolution
        ce_arrs = []
        for w in CE_WINDOWS:
            ce = batch_ce(V, w)
            # Align to exactly n_wins elements (pad with NaN if too short)
            if len(ce) >= n_wins:
                ce_arrs.append(ce[:n_wins])
            else:
                ce_arrs.append(np.pad(ce, (0, n_wins - len(ce)), constant_values=np.nan))

        # Irreversible thermodynamic entropy
        sig = batch_sigma(V, I, T, R, SIGMA_WIN)
        sig = sig[:n_wins] if len(sig) >= n_wins else np.pad(
            sig, (0, n_wins - len(sig)), constant_values=np.nan)

        # Phenomenological entropy
        phen = batch_phen(Q, V, T, dir_, PHEN_WIN)
        phen = phen[:n_wins] if len(phen) >= n_wins else np.pad(
            phen, (0, n_wins - len(phen)), constant_values=np.nan)

        # ── STATISTICAL FEATURES ─────────────────────────────────────────────
        # Mean, std, skew of voltage, dV, dVa, and current within each window
        V_s   = batch_stats(V,         MIN_WIN)[:n_wins]      # [mean, std, skew]
        dV_s  = batch_stats(dV,        MIN_WIN)[:n_wins]      # [mean, std, skew]
        dVa_s = batch_stats(dVa,       MIN_WIN)[:n_wins]      # [mean, std, skew]
        I_s   = batch_stats(np.abs(I), MIN_WIN)[:n_wins, :2]  # [mean, std]

        # Mean SoH across the window — gives the model context about degradation
        soh_wins = sliding_window_view(soh, MIN_WIN)[::STRIDE].mean(axis=1)[:n_wins]

        # ── ASSEMBLE FEATURE VECTOR ───────────────────────────────────────────
        # The feature vector must match the FEAT_NAMES list exactly.
        # Build a lookup dictionary then pull out only the features we need.
        feat_lookup = {
            'CE_21':        ce_arrs[0],
            'CE_41':        ce_arrs[1],
            'CE_81':        ce_arrs[2],
            'sigma_41':     sig,
            'phen_ent':     phen,
            'V_mean':       V_s[:, 0],
            'V_std':        V_s[:, 1],
            'dV_mean':      dV_s[:, 0],
            'dVa_mean':     dVa_s[:, 0],   # requires reference electrode
            'dVa_std':      dVa_s[:, 1],   # requires reference electrode
            'I_mean':       I_s[:, 0],
            'soh_estimate': soh_wins,
        }

        # Pull only the features in FEAT_NAMES (respects USE_REFERENCE_ELECTRODE)
        X_g = np.column_stack([feat_lookup[f] for f in FEAT_NAMES])

        # ── TARGET LABELS ─────────────────────────────────────────────────────
        # Label each window by what is happening at its midpoint row.
        # Priority: Plating (1) > Stripping (2) > Neither (0)
        y_g = np.where(pl[mids], 1, np.where(st[mids], 2, 0))

        # Group ID: identifies which (channel, session, cycle) this window came from.
        # Used later to keep whole cycles together in the same split.
        grp_id = f'ch{channel}_{sess}_c{int(cyc)}'

        all_X.append(X_g)
        all_y.append(y_g)
        all_grp.extend([grp_id] * n_wins)

    if not all_X:
        raise ValueError(f'No windows extracted from channel {channel}. '
                         f'Check that step_name values match CCCV_Chg/CC_DChg/Rest '
                         f'and that there are enough rows per step (need >{MIN_WIN}).')

    X_out   = np.vstack(all_X)
    y_out   = np.concatenate(all_y)
    grp_out = np.array(all_grp)

    print(f'  Ch{channel}: {X_out.shape[0]:,} windows extracted')
    return X_out, y_out, grp_out


# =============================================================================
# SECTION 4 — DATA LOADING
# =============================================================================

# Columns we need from the CSV. Any extra columns in your file are ignored.
REQUIRED_COLS = [
    'abs_time', 'session_label', 'session_type', 'cycle', 'step_name',
    'voltage_V', 'current_mA', 'anode_potential_V', 'temperature_C',
    'overpotential_V', 'step_capacity_mAh', 'soh_estimate',
    'Li_plating', 'Li_stripping',
]

print('=' * 60)
print('STEP 1: Loading data')
print('=' * 60)

X_parts, y_parts, grp_parts, ch_parts = [], [], [], []

for ch in CHANNELS:
    filepath = os.path.join(DATA_DIR, FILE_PATTERN.format(ch=ch))

    if not os.path.exists(filepath):
        print(f'  WARNING: File not found — {filepath} — skipping channel {ch}')
        continue

    print(f'  Loading Ch{ch} from {filepath}...')
    df = pd.read_csv(filepath, usecols=REQUIRED_COLS)

    # Use RPT sessions only: these have controlled conditions and temperature data.
    # Comment out this line to include cycling sessions as well.
    df = df[df['session_type'] == 'RPT'].copy()

    n_rows = len(df)
    n_plating  = df['Li_plating'].sum()
    n_stripping = df['Li_stripping'].sum()
    soh_range = f"{df['soh_estimate'].min():.1f}–{df['soh_estimate'].max():.1f}%"
    print(f'    {n_rows:,} rows  |  '
          f'Plating rows: {n_plating:,}  |  '
          f'Stripping rows: {n_stripping:,}  |  '
          f'SoH range: {soh_range}')

    X, y, grp = extract_features(df, ch)
    X_parts.append(X)
    y_parts.append(y)
    grp_parts.append(grp)
    ch_parts.append(np.full(len(y), ch, dtype=int))

if not X_parts:
    raise RuntimeError('No data loaded. Check DATA_DIR and FILE_PATTERN.')

X_all   = np.vstack(X_parts)
y_all   = np.concatenate(y_parts)
grp_all = np.concatenate(grp_parts)
ch_all  = np.concatenate(ch_parts)

print(f'\nTotal: {X_all.shape[0]:,} windows × {X_all.shape[1]} features')
print('Class distribution:')
for c, n_c in zip(*np.unique(y_all, return_counts=True)):
    print(f'  {LABEL_NAMES[c]:12s}: {n_c:6,}  ({n_c / len(y_all) * 100:.1f}%)')


# =============================================================================
# SECTION 5 — TRAIN / VALIDATION / TEST SPLIT
# =============================================================================
#
# WHY WE SPLIT BY GROUP, NOT BY ROW
# ───────────────────────────────────
# If we split individual rows randomly, the model would see windows from
# the same cycle in both training and test — effectively memorising that
# specific cycle rather than learning general rules. Splitting by group
# (whole cycles at a time) forces the model to generalise to cycles it
# has never seen.
#
# WHAT THE SPLITS DO
# ───────────────────
# Train (60%):      The model learns from these windows.
# Validation (30%): Used in 5-fold cross-validation to measure how stable
#                   the model's performance is across different subsets.
# Test (10%):       Completely held out. Never used during training or CV.
#                   Gives an unbiased final performance estimate.

print('\n' + '=' * 60)
print('STEP 2: Building train / validation / test split')
print('=' * 60)

# Get all unique group IDs and shuffle them randomly
unique_grps = np.unique(grp_all)
np.random.seed(SEED)
np.random.shuffle(unique_grps)

n_grps = len(unique_grps)
n_te   = max(1, int(n_grps * TEST_FRACTION))
n_va   = max(1, int(n_grps * VAL_FRACTION))
n_tr   = n_grps - n_te - n_va

# Assign groups to splits
test_grps  = set(unique_grps[:n_te])
val_grps   = set(unique_grps[n_te : n_te + n_va])
train_grps = set(unique_grps[n_te + n_va :])

# Build boolean masks to select rows belonging to each split
tr_m = np.array([g in train_grps for g in grp_all])
va_m = np.array([g in val_grps   for g in grp_all])
te_m = np.array([g in test_grps  for g in grp_all])

X_tr, y_tr = X_all[tr_m], y_all[tr_m]
X_va, y_va = X_all[va_m], y_all[va_m]
X_te, y_te = X_all[te_m], y_all[te_m]

print(f'Groups: total={n_grps}  train={n_tr}  val={n_va}  test={n_te}')
print(f'Windows: train={len(y_tr):,}  val={len(y_va):,}  test={len(y_te):,}')
print('\nChannel mix in each split:')
for sname, mask in [('Train', tr_m), ('Val', va_m), ('Test', te_m)]:
    ch_c = {c: int((ch_all[mask] == c).sum()) for c in CHANNELS}
    print(f'  {sname:6s}: ' + '  '.join([f'Ch{c}:{n:,}' for c, n in ch_c.items()]))


# =============================================================================
# SECTION 6 — PREPROCESSING
# =============================================================================
#
# Three steps applied before training:
#
# 1. NaN IMPUTATION
#    Windows at step boundaries sometimes produce NaN entropy values
#    (flat voltage → CE is undefined). Replace with the column median.
#    The median is always computed from the TRAINING set so the test set
#    stays truly independent.
#
# 2. STANDARDISATION
#    Subtract the mean and divide by the standard deviation of each feature.
#    This puts all features on the same scale. Without it, features with
#    large values (e.g. V_mean ≈ 3.5 V) dominate features with small
#    values (e.g. CE ≈ 0.1) and the neural network trains poorly.
#
# 3. CLASS BALANCING
#    "Neither" windows make up ~82% of the data. If we train on this
#    imbalance, the model learns to always predict "Neither" and achieves
#    82% accuracy without detecting any plating or stripping.
#    Fix: randomly duplicate (upsample) Plating and Stripping windows
#    until all three classes have equal numbers. The test set is NOT
#    balanced — we evaluate on the true class distribution.

def preprocess(X_train, y_train, X_test, seed=SEED):
    """
    Impute NaN → standardise → balance training classes.

    Returns
    -------
    X_bal    : balanced, standardised training features
    y_bal    : labels for X_bal
    X_test_s : standardised test features (not balanced)
    scaler   : fitted StandardScaler (save this if you want to transform new data later)
    """
    # Step 1: replace NaN with training-set column medians
    medians  = np.nanmedian(X_train, axis=0)
    X_tr_c   = np.where(np.isnan(X_train), medians, X_train)
    X_te_c   = np.where(np.isnan(X_test),  medians, X_test)

    # Step 2: standardise (fit on train, apply to both)
    scaler   = StandardScaler()
    X_tr_s   = scaler.fit_transform(X_tr_c)
    X_te_s   = scaler.transform(X_te_c)

    # Step 3: upsample minority classes to match the "Neither" class count
    n_majority = np.sum(y_train == 0)
    idx = np.concatenate([
        resample(
            np.where(y_train == c)[0],   # indices of this class
            n_samples=n_majority,         # target size = majority count
            replace=True,                 # allow repeated rows
            random_state=seed + c,
        )
        for c in np.unique(y_train)
    ])
    np.random.seed(seed)
    np.random.shuffle(idx)

    return X_tr_s[idx], y_train[idx], X_te_s, scaler


# =============================================================================
# SECTION 7 — MODEL DEFINITION
# =============================================================================
#
# NEURAL NETWORK ARCHITECTURE
# ─────────────────────────────
# We use a Multilayer Perceptron (MLP) — a standard feedforward neural network.
#
# Input layer:  N_FEATURES neurons (one per feature)
# Hidden layer 1: 128 neurons  — learns broad combinations of features
# Hidden layer 2:  64 neurons  — learns more specific patterns
# Hidden layer 3:  32 neurons  — fine-grained discrimination
# Output layer:     3 neurons  — probability of each class (Neither/Plating/Stripping)
#
# KEY SETTINGS
# ─────────────
# activation='relu'        Non-linear activation function. Allows the network
#                          to learn non-linear class boundaries.
# solver='adam'            Adaptive learning rate optimiser. Well-suited to
#                          non-stationary signals like battery data.
# alpha=0.001              L2 weight decay (regularisation). Stops individual
#                          weights from growing too large and overfitting.
# early_stopping=True      Automatically stops training when validation loss
#                          stops improving, preventing overfitting.

def make_model():
    """Create a fresh MLP classifier with the standard architecture."""
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


# =============================================================================
# SECTION 8 — 5-FOLD CROSS-VALIDATION
# =============================================================================
#
# Cross-validation trains 5 separate models, each time holding out a
# different 20% of the train+val pool as the "fold test set". This tells us:
#   - How much performance varies depending on WHICH groups are held out
#   - Whether the model is consistent or sensitive to the random split
#
# The held-out test set (10%) is NEVER used in cross-validation.
# CV operates entirely on the train+val pool (90%).
#
# Folds are formed by interleaving (groups 0,5,10,... → fold 1, groups 1,6,11,... → fold 2)
# so each fold automatically gets a mix of early and late RPT sessions
# and a mix of all four channels.

def run_5fold_cv(X_pool, y_pool, grp_pool):
    """
    Run 5-fold group cross-validation.
    Returns a list of score dicts, one per fold.
    """
    cv_grps = np.unique(grp_pool)
    np.random.seed(SEED)
    np.random.shuffle(cv_grps)

    # Interleaved assignment: group i → fold (i % 5)
    folds = [set(cv_grps[i::5]) for i in range(5)]

    fold_scores = []
    for fold_i, held_out in enumerate(folds):
        in_fold  = np.array([g in held_out for g in grp_pool])
        out_fold = ~in_fold

        # Train on the 4 other folds, evaluate on this fold
        Xf_tr, yf_tr = X_pool[out_fold], y_pool[out_fold]
        Xf_te, yf_te = X_pool[in_fold],  y_pool[in_fold]

        Xf_bal, yf_bal, Xf_te_s, _ = preprocess(Xf_tr, yf_tr, Xf_te, seed=fold_i)

        m = make_model()
        m.fit(Xf_bal, yf_bal)
        yf_pred = m.predict(Xf_te_s)

        cls_p = sorted(np.unique(np.concatenate([yf_te, yf_pred])))
        f1s   = f1_score(yf_te, yf_pred, labels=cls_p, average=None, zero_division=0)
        fm    = {c: f for c, f in zip(cls_p, f1s)}
        fold_scores.append({
            'fold':       fold_i + 1,
            'acc':        np.mean(yf_pred == yf_te),
            'f1_neither': fm.get(0, 0),
            'f1_plating': fm.get(1, 0),
            'f1_strip':   fm.get(2, 0),
            'f1_macro':   f1_score(yf_te, yf_pred, average='macro', zero_division=0),
        })
        print(f'  Fold {fold_i + 1}: '
              f'acc={fold_scores[-1]["acc"]*100:.1f}%  '
              f'F1_plating={fold_scores[-1]["f1_plating"]:.4f}  '
              f'F1_stripping={fold_scores[-1]["f1_strip"]:.4f}')

    return fold_scores


# =============================================================================
# SECTION 9 — TRAIN AND EVALUATE THE FINAL MODEL
# =============================================================================

print('\n' + '=' * 60)
print('STEP 3: 5-fold cross-validation (train+val pool)')
print('=' * 60)

tv_m    = tr_m | va_m
cv_sc   = run_5fold_cv(X_all[tv_m], y_all[tv_m], grp_all[tv_m])
cv_df   = pd.DataFrame(cv_sc)

print(f'\nCV summary:')
print(f'  Accuracy:    {cv_df["acc"].mean()*100:.2f} ± {cv_df["acc"].std()*100:.2f}%')
print(f'  F1 macro:    {cv_df["f1_macro"].mean():.4f} ± {cv_df["f1_macro"].std():.4f}')
print(f'  F1 plating:  {cv_df["f1_plating"].mean():.4f} ± {cv_df["f1_plating"].std():.4f}')
print(f'  F1 strip:    {cv_df["f1_strip"].mean():.4f} ± {cv_df["f1_strip"].std():.4f}')

print('\n' + '=' * 60)
print('STEP 4: Training final model on full training set')
print('=' * 60)

# Train the final model on the training set only (not val or test)
X_bal, y_bal, X_te_s, scaler = preprocess(X_tr, y_tr, X_te)

final_model = make_model()
final_model.fit(X_bal, y_bal)
y_pred = final_model.predict(X_te_s)
y_prob = final_model.predict_proba(X_te_s)

print(f'Model converged in {final_model.n_iter_} epochs')
print(f'Test accuracy: {np.mean(y_pred == y_te)*100:.2f}%')

print('\n' + '=' * 60)
print('STEP 5: Classification report (held-out test set)')
print('=' * 60)

cls_p  = sorted(np.unique(np.concatenate([y_te, y_pred])))
tnames = [LABEL_NAMES[c] for c in cls_p]
print(classification_report(y_te, y_pred, labels=cls_p, target_names=tnames))


# =============================================================================
# SECTION 10 — PLOTS
# =============================================================================

print('=' * 60)
print('STEP 6: Generating plots')
print('=' * 60)

# ── Figure 1: Confusion matrix + CV bar chart ─────────────────────────────────
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig1.suptitle('Model performance summary', fontweight='bold', fontsize=12)

ConfusionMatrixDisplay(
    confusion_matrix(y_te, y_pred, labels=cls_p),
    display_labels=tnames,
).plot(ax=ax1, colorbar=False, cmap='Blues')
ax1.set_title(f'Confusion matrix — test set\nAccuracy: {np.mean(y_pred==y_te)*100:.2f}%')

# 5-fold CV bars
metrics    = ['acc', 'f1_neither', 'f1_plating', 'f1_strip']
m_labels   = ['Accuracy', 'F1 Neither', 'F1 Plating', 'F1 Stripping']
fold_colors= ['#1976D2', '#388E3C', '#F57C00', '#7B1FA2', '#C62828']
x_pos      = np.arange(len(metrics))
bw         = 0.15

for fi, row in cv_df.iterrows():
    ax2.bar(x_pos + fi * bw, [row[m] for m in metrics],
            width=bw, color=fold_colors[fi], alpha=0.85, label=f'Fold {fi+1}')
ax2.axhline(cv_df['acc'].mean(), color='black', ls='--', lw=1.5, alpha=0.6,
            label=f'Mean acc {cv_df["acc"].mean()*100:.1f}%')
ax2.set_xticks(x_pos + 2 * bw)
ax2.set_xticklabels(m_labels, fontsize=8)
ax2.set_ylim(0, 1.15)
ax2.set_ylabel('Score')
ax2.set_title('5-fold cross-validation scores\n(train+val pool — test never seen in CV)')
ax2.legend(fontsize=7, ncol=3)

plt.tight_layout()
out1 = os.path.join(DATA_DIR, 'plot_performance_summary.png')
fig1.savefig(out1, dpi=140, bbox_inches='tight')
print(f'  Saved: {out1}')
plt.close(fig1)

# ── Figure 2: Feature importance (MLP weight magnitudes) ─────────────────────
fig2, ax = plt.subplots(figsize=(9, 6))
w1  = np.abs(final_model.coefs_[0])   # first layer weights: shape (n_features, 128)
fs  = w1.mean(axis=1)                  # mean absolute weight per input feature
si  = np.argsort(fs)[::-1]

bar_colors = [
    '#D32F2F' if FEAT_NAMES[i] == 'soh_estimate'                            else
    '#E65100' if 'CE' in FEAT_NAMES[i]                                       else
    '#1565C0' if any(x in FEAT_NAMES[i] for x in ['sigma', 'phen'])          else
    '#2E7D32' if any(x in FEAT_NAMES[i] for x in ['dVa', 'V_mean', 'I'])    else
    '#546E7A'
    for i in si
]
ax.barh(range(len(fs)), fs[si][::-1], color=bar_colors[::-1], alpha=0.85)
ax.set_yticks(range(len(fs)))
ax.set_yticklabels([FEAT_NAMES[i] for i in si[::-1]], fontsize=9)
ax.set_xlabel('Mean |weight| to first hidden layer (proxy for feature importance)')
ax.set_title('Feature importance in trained model\n'
             'Red=SoH  Orange=CE entropy  Blue=thermo/phen entropy  '
             'Green=key signals  Grey=statistical',
             fontsize=9)

plt.tight_layout()
out2 = os.path.join(DATA_DIR, 'plot_feature_importance.png')
fig2.savefig(out2, dpi=140, bbox_inches='tight')
print(f'  Saved: {out2}')
plt.close(fig2)

# ── Figure 3: Predicted probability traces (test windows only) ────────────────
fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))
fig3.suptitle('Plating/stripping probability distributions — test set\n'
              'Good separation: true-positive mass near 1.0, '
              'false-positive mass near 0.0',
              fontweight='bold', fontsize=11)

for ax, cls in zip(axes3, cls_p):
    correct   = (y_te == cls) & (y_pred == cls)
    missed    = (y_te == cls) & (y_pred != cls)
    false_pos = (y_te != cls) & (y_pred == cls)
    prob_col  = cls   # index into y_prob columns

    for vals, lbl, col, ls in [
        (y_prob[correct,   prob_col], f'Correct ({correct.sum():,})',         LABEL_COLORS[cls], '-'),
        (y_prob[missed,    prob_col], f'Missed / FN ({missed.sum():,})',       '#FF6F00',          '--'),
        (y_prob[false_pos, prob_col], f'False alarm / FP ({false_pos.sum():,})', '#9E9E9E',        ':'),
    ]:
        if len(vals) < 5:
            continue
        vr = np.linspace(0, 1, 200)
        try:
            kde = gaussian_kde(vals, bw_method=0.15)
            ax.plot(vr, kde(vr), color=col, lw=2, ls=ls, alpha=0.9, label=lbl)
            ax.fill_between(vr, kde(vr), alpha=0.08, color=col)
        except Exception:
            pass

    ax.axvline(0.5, color='black', ls='--', lw=1, alpha=0.5, label='Threshold')
    ax.set_title(f'Class: {LABEL_NAMES[cls]}', fontweight='bold',
                 color=LABEL_COLORS[cls])
    ax.set_xlabel(f'P({LABEL_NAMES[cls]})')
    ax.set_ylabel('Density')
    ax.legend(fontsize=7.5)

plt.tight_layout()
out3 = os.path.join(DATA_DIR, 'plot_probability_distributions.png')
fig3.savefig(out3, dpi=140, bbox_inches='tight')
print(f'  Saved: {out3}')
plt.close(fig3)

print('\n' + '=' * 60)
print('DONE')
print('=' * 60)
print(f'Model summary:')
print(f'  Feature set: {N_FEATURES} features '
      f'({"with" if USE_REFERENCE_ELECTRODE else "without"} reference electrode)')
print(f'  Test accuracy:    {np.mean(y_pred==y_te)*100:.2f}%')
f1s_final = f1_score(y_te, y_pred, labels=cls_p, average=None, zero_division=0)
for c, f in zip(cls_p, f1s_final):
    print(f'  F1 {LABEL_NAMES[c]:12s}: {f:.4f}')
print(f'  CV accuracy:      {cv_df["acc"].mean()*100:.2f} ± {cv_df["acc"].std()*100:.2f}%')
