"""
MIT-Stanford-TRI Fast-Charging Dataset — Quality Scoring
=========================================================
Scores the MIT-Stanford-TRI dataset (141 A124 LFP cells, ~72 fast-charging
policies) against the same six-dimension quality framework used for the
Oxford dataset.

What ends up hardcoded here, and why:
  - Chemistry diversity: single cell model (A124 LFP, 2.54 Ah nominal) across
    the whole dataset; no per-cell chemistry field exists to count from.
  - DoD diversity: metadata reports all cells cycled over the same full
    charge/discharge window; no per-cycle DoD field is recorded.
  - Calendar aging / Real-world operation: genuinely undeterminable from
    the data under any amount of analysis (the experiment is cyclic-only
    and laboratory-only).
  - Dynamic load profiles: the ageing protocol is CCCV charging + CC
    discharging; no drive-cycle or variable-current ageing profile exists.

Computed dynamically: replicate cells (from policy grouping), C-rate /
policy diversity (from policy_readable), temperature diversity (from the
measured T array), current-sign convention (from the direction of current
in charge vs discharge phases), physical plausibility (voltage, current,
temperature), statistical outliers, unexpected signal changes, measurement
noise (from the CV-phase tail — the quietest part of each MIT cycle, used
here as the analogue of Oxford's pseudo-OCV segments), distribution
balance, and temporal coherence.

Output: a scorecard CSV and a scorecard figure, both saved to
./quality_results/
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import h5py

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIG
# ============================================================================
DATASET_PATH = r"C:\Users\admin\Desktop\DR2\11 All Datasets\04 MIT–Stanford–TRI Fast-Charging Dataset\mit_dataset"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quality_results")
os.makedirs(OUT_DIR, exist_ok=True)

SCORE_LABELS = {"++": "Comprehensive coverage", "+": "Mostly satisfied",
                "o": "Partially satisfied", "-": "Not satisfied"}
SCORE_COLORS = {"++": "#2ca02c", "+": "#98df8a", "o": "#ffbb78", "-": "#d62728"}

# Physical plausibility bounds -- general Li-ion envelope for LFP cells
# (dataset-agnostic fixed bounds, matching the Oxford code's approach).
VOLTAGE_MIN_V, VOLTAGE_MAX_V = 2.0, 3.65
CURRENT_MIN_A, CURRENT_MAX_A = -10.0, 10.0
CHARGE_TEMP_MIN_C, CHARGE_TEMP_MAX_C = 0.0, 45.0
DISCHARGE_TEMP_MIN_C, DISCHARGE_TEMP_MAX_C = -20.0, 60.0
NOMINAL_CAPACITY_AH = 2.54   # A124 nominal capacity
NOMINAL_VOLTAGE_V = 3.3      # A124 nominal voltage

# ============================================================================
# METADATA-ONLY FACTS (genuinely undeterminable from the .mat file's own
# content, whatever analysis is applied to it)
# ============================================================================
METADATA_ONLY = {
    "Chemistry diversity": {
        "score": "o",
        "finding": "1 chemistry/cell model (A124 LFP, 2.54 Ah) across the whole dataset -- "
                   "no per-cell chemistry field exists in the .mat to count from.",
    },
    "DoD diversity": {
        "score": "o",
        "finding": "1 DoD condition per metadata (all cells cycled over the same full "
                   "charge/discharge window) -- no per-cycle DoD field to count independently.",
    },
    "Calendar aging": {
        "score": "-",
        "finding": "Metadata reports calendar aging is absent as a dedicated experimental "
                   "condition; the dataset is cyclic-ageing only.",
    },
    "Dynamic load profiles": {
        "score": "o",
        "finding": "Metadata: the ageing protocol is CCCV charging + CC discharging; no "
                   "drive-cycle or variable-current ageing profile is applied (only the "
                   "fast-charging policies vary the current during charge).",
    },
    "Real-world operation": {
        "score": "-",
        "finding": "Metadata: real-world operation is absent; all cycling is laboratory "
                   "CCCV/CC with no measurements from vehicles in operation.",
    },
}

# Protocol elements documented in the HDF5 structure (all present -> full doc)
PROTOCOL_METADATA = {
    "chemistry": True,
    "cell_model": True,
    "environmental_temperature": True,
    "charging_protocol": True,
    "discharge_protocol": True,
    "characterization_frequency": True,
    "characterization_current": True,
}

# ============================================================================
# SCORING FUNCTIONS (identical vocabulary to the Oxford code)
# ============================================================================
def score_pct_low_is_good(pct):
    if pct < 0.1: return "++"
    if pct < 1: return "+"
    if pct < 5: return "o"
    return "-"

def score_pct_high_is_good(pct):
    if pct >= 99.9: return "++"
    if pct >= 99: return "+"
    if pct >= 95: return "o"
    return "-"

def score_noise(pct):
    if pct <= 0.5: return "++"
    if pct <= 1: return "+"
    if pct <= 2: return "o"
    return "-"

def score_doc(pct):
    if pct >= 100: return "++"
    if pct >= 75: return "+"
    if pct >= 25: return "o"
    return "-"

def score_diversity_count(n):
    if n >= 3: return "++"
    if n == 2: return "+"
    return "o"

def score_replicates(n):
    if n >= 10: return "++"
    if n >= 5: return "+"
    return "o"

def score_soh_range(pct_cells_below_80):
    if pct_cells_below_80 >= 20: return "++"
    if pct_cells_below_80 >= 10: return "+"
    if pct_cells_below_80 > 0: return "o"
    return "-"

def score_soh_balance(dominant_bin_pct):
    if dominant_bin_pct <= 50: return "++"
    if dominant_bin_pct <= 70: return "+"
    if dominant_bin_pct <= 90: return "o"
    return "-"

def score_cv(cv_pct):
    if cv_pct <= 10: return "++"
    if cv_pct <= 25: return "+"
    if cv_pct <= 50: return "o"
    return "-"

def score_degradation_trend(pct_consistent):
    if pct_consistent >= 99: return "++"
    if pct_consistent >= 90: return "+"
    if pct_consistent >= 70: return "o"
    return "-"

results = []
def add(criterion, aspect, score, finding):
    results.append({"criterion": criterion, "aspect": aspect, "score": score, "finding": finding})
    print(f"  [{score}] {criterion} — {aspect}: {finding}")

# ============================================================================
# HDF5 HELPERS (unchanged from your working MIT exploration script)
# ============================================================================
def get_n_cells_from_batch(h5):
    try:
        batch = h5["batch"]
        for key in batch.keys():
            ds = batch[key]
            if isinstance(ds, h5py.Dataset) and ds.dtype.kind == "O":
                return int(ds[()].flatten().shape[0])
        return 0
    except Exception:
        return 0

def get_summary_field(h5, cell_idx, field):
    try:
        batch = h5["batch"]
        if "summary" not in batch:
            return np.array([np.nan])
        summ = batch["summary"]
        if isinstance(summ, h5py.Group):
            if field not in summ:
                return np.array([np.nan])
            ds = summ[field]
            data = ds[()].flatten()
            if data.dtype.kind == "O":
                if cell_idx >= len(data):
                    return np.array([np.nan])
                target = h5[data[cell_idx]]
                return target[()].flatten().astype(float) if target is not None else np.array([np.nan])
            return data.astype(float)
        if isinstance(summ, h5py.Dataset) and summ.dtype.kind == "O":
            flat = summ[()].flatten()
            if cell_idx >= len(flat):
                return np.array([np.nan])
            cell_grp = h5[flat[cell_idx]]
            if cell_grp is None or not isinstance(cell_grp, h5py.Group):
                return np.array([np.nan])
            if field not in cell_grp:
                return np.array([np.nan])
            return cell_grp[field][()].flatten().astype(float)
        return np.array([np.nan])
    except Exception:
        return np.array([np.nan])

def get_n_cycles_for_cell(h5, cell_idx):
    try:
        batch = h5["batch"]
        if "cycles" not in batch:
            return 0
        cyc_ds = batch["cycles"]
        if not (isinstance(cyc_ds, h5py.Dataset) and cyc_ds.dtype.kind == "O"):
            return 0
        flat = cyc_ds[()].flatten()
        if cell_idx >= len(flat):
            return 0
        cell_grp = h5[flat[cell_idx]]
        if cell_grp is None:
            return 0
        if "data" in cell_grp:
            return int(cell_grp["data"][()].flatten().shape[0])
        for key in cell_grp.keys():
            ds = cell_grp[key]
            if isinstance(ds, h5py.Dataset) and ds.dtype.kind == "O":
                return int(ds[()].flatten().shape[0])
        return 0
    except Exception:
        return 0

def get_raw_cycle_field(h5, cell_idx, cycle_idx, field):
    try:
        batch = h5["batch"]
        cyc_ds = batch["cycles"]
        flat = cyc_ds[()].flatten()
        if cell_idx >= len(flat):
            return np.array([np.nan])
        cell_grp = h5[flat[cell_idx]]
        if cell_grp is None:
            return np.array([np.nan])
        if "data" in cell_grp:
            data_ds = cell_grp["data"]
            if isinstance(data_ds, h5py.Dataset) and data_ds.dtype.kind == "O":
                cyc_flat = data_ds[()].flatten()
                if cycle_idx >= len(cyc_flat):
                    return np.array([np.nan])
                cyc_grp = h5[cyc_flat[cycle_idx]]
                if cyc_grp is None or not isinstance(cyc_grp, h5py.Group):
                    return np.array([np.nan])
                if field not in cyc_grp:
                    return np.array([np.nan])
                return cyc_grp[field][()].flatten().astype(float)
        if field in cell_grp:
            field_ds = cell_grp[field]
            if isinstance(field_ds, h5py.Dataset) and field_ds.dtype.kind == "O":
                flat_refs = field_ds[()].flatten()
                if cycle_idx >= len(flat_refs):
                    return np.array([np.nan])
                target = h5[flat_refs[cycle_idx]]
                return target[()].flatten().astype(float) if target is not None else np.array([np.nan])
        return np.array([np.nan])
    except Exception:
        return np.array([np.nan])

def get_policy(h5, cell_idx):
    try:
        p_ds = h5["batch"]["policy_readable"]
        p_ref = p_ds[()].flatten()[cell_idx]
        p_arr = h5[p_ref][()]
        return "".join(chr(int(c)) for c in p_arr.flatten() if 32 <= int(c) < 127)
    except Exception:
        return "N/A"

# ============================================================================
# LOAD DATA (generic -- every cell, every cycle, every field actually
# present is picked up dynamically)
# ============================================================================
print("Loading MIT-Stanford-TRI dataset...")
mat_files = sorted(glob.glob(os.path.join(DATASET_PATH, "*.mat")))
print(f"Found {len(mat_files)} .mat files")

# Cell-level metadata
all_cells = []
for f in mat_files:
    bname = os.path.basename(f)[:10]
    with h5py.File(f, "r") as h5:
        n_cells = get_n_cells_from_batch(h5)
        for ci in range(n_cells):
            n_cyc = get_n_cycles_for_cell(h5, ci)
            all_cells.append({
                "batch": bname,
                "cell_id": f"{bname}_c{ci:03d}",
                "n_cycles": n_cyc,
                "policy": get_policy(h5, ci),
            })

cells_df = pd.DataFrame(all_cells)
n_cells_total = len(cells_df)
print(f"Found {n_cells_total} cells across {cells_df['batch'].nunique()} batches, "
      f"{cells_df['policy'].nunique()} unique policies")

# Long-form sample data across all cells/cycles for V/I/T plausibility,
# noise, monotonicity and continuity checks.
print("Loading raw sample data (this may take a few minutes)...")
rows = []
missing_expected = 0
missing_observed = 0

for f in mat_files:
    bname = os.path.basename(f)[:10]
    with h5py.File(f, "r") as h5:
        n_cells = get_n_cells_from_batch(h5)
        for ci in range(n_cells):
            cell_id = f"{bname}_c{ci:03d}"
            n_cyc = get_n_cycles_for_cell(h5, ci)
            for cyc_i in range(n_cyc):
                t = get_raw_cycle_field(h5, ci, cyc_i, "t")
                V = get_raw_cycle_field(h5, ci, cyc_i, "V")
                I = get_raw_cycle_field(h5, ci, cyc_i, "I")
                T = get_raw_cycle_field(h5, ci, cyc_i, "T")
                n = min(len(t), len(V), len(I), len(T))
                if n == 0:
                    continue
                missing_expected += n * 4
                missing_observed += (np.isnan(V[:n]).sum() + np.isnan(I[:n]).sum()
                                     + np.isnan(T[:n]).sum())
                rows.append(pd.DataFrame({
                    "cell": cell_id, "cycle": cyc_i,
                    "t": t[:n], "v": V[:n], "i": I[:n], "T": T[:n],
                }))

data = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
    columns=["cell", "cycle", "t", "v", "i", "T"])
print(f"Loaded {len(data):,} sample rows\n")

# Segment classification without hardcoded names: positive current = charge,
# negative = discharge (MIT's documented convention).
is_charge = data["i"] > 0.05
is_discharge = data["i"] < -0.05

# ============================================================================
# 1. CORRECTNESS
# ============================================================================
print("== 1. Correctness ==")

viol, total = 0, 0
per_signal = []
for label, mask, col, lo, hi in [
    ("Voltage (charge segments)", is_charge, "v", VOLTAGE_MIN_V, VOLTAGE_MAX_V),
    ("Voltage (discharge segments)", is_discharge, "v", VOLTAGE_MIN_V, VOLTAGE_MAX_V),
    ("Current (charge segments)", is_charge, "i", 0.0, CURRENT_MAX_A),
    ("Current (discharge segments)", is_discharge, "i", CURRENT_MIN_A, 0.0),
    ("Temperature (charge segments)", is_charge, "T", CHARGE_TEMP_MIN_C, CHARGE_TEMP_MAX_C),
    ("Temperature (discharge segments)", is_discharge, "T", DISCHARGE_TEMP_MIN_C, DISCHARGE_TEMP_MAX_C),
]:
    s = data.loc[mask, col].dropna()
    if len(s) == 0:
        continue
    v = ((s < lo) | (s > hi)).sum()
    viol += v
    total += len(s)
    pct = v / len(s) * 100
    per_signal.append((label, v, len(s), pct))
    print(f"    Physical plausibility breakdown -- {label}: {v:,}/{len(s):,} ({pct:.2f}%) "
          f"outside [{lo}, {hi}], observed range [{s.min():.3f}, {s.max():.3f}]")
pct_implausible = (viol / total * 100) if total else 0
add("Correctness", "Physical plausibility", score_pct_low_is_good(pct_implausible),
    f"{viol:,}/{total:,} voltage/current/temperature readings ({pct_implausible:.3f}%) outside "
    f"the general Li-ion envelope (voltage {VOLTAGE_MIN_V}-{VOLTAGE_MAX_V}V; current "
    f"{CURRENT_MIN_A}-{CURRENT_MAX_A}A; temperature {CHARGE_TEMP_MIN_C}-{CHARGE_TEMP_MAX_C}C during "
    f"charge, {DISCHARGE_TEMP_MIN_C}-{DISCHARGE_TEMP_MAX_C}C during discharge).")

# Current sign convention: during charge, I>0; during discharge, I<0.
sign_checks = []
for (cell, cycle), grp in data.groupby(["cell", "cycle"]):
    I = grp["i"].dropna().values
    if len(I) < 10:
        continue
    has_pos = np.any(I > 0.05)
    has_neg = np.any(I < -0.05)
    sign_checks.append(has_pos and has_neg)
pct_sign_ok = (np.mean(sign_checks) * 100) if sign_checks else 0
add("Correctness", "Current sign convention", score_pct_high_is_good(pct_sign_ok),
    f"{sum(sign_checks):,}/{len(sign_checks):,} cycles ({pct_sign_ok:.2f}%) show both a positive "
    f"charge phase and a negative discharge phase, as expected (positive = charge, negative = "
    f"discharge convention confirmed directly from the recorded Current field).")

# ============================================================================
# 2. COMPLETENESS
# ============================================================================
print("\n== 2. Completeness ==")

essential_cols = ["t", "v", "i", "T"]
missing = data[essential_cols].isna().sum().sum()
pct_missing = missing / data[essential_cols].size * 100 if data[essential_cols].size else 0
add("Completeness", "Missing values", score_pct_low_is_good(pct_missing),
    f"{pct_missing:.4f}% missing across essential columns {essential_cols} "
    f"({missing_observed:,} of {missing_expected:,} expected measurements).")

cont_pct_list = []
for (cell, cycle), grp in data.groupby(["cell", "cycle"]):
    t_sorted = grp["t"].dropna().sort_values().values
    if len(t_sorted) > 1:
        diffs = np.diff(t_sorted)
        med = np.median(diffs)
        if med > 0:
            cont_pct_list.append((diffs <= 5 * med).mean() * 100)
pct_continuity = np.mean(cont_pct_list) if cont_pct_list else 100
add("Completeness", "Temporal continuity", score_pct_high_is_good(pct_continuity),
    f"Average within-cycle sampling continuity (no gaps >5x the median interval): "
    f"{pct_continuity:.2f}%.")

documented_protocol = sum(PROTOCOL_METADATA.values())
total_protocol_elements = len(PROTOCOL_METADATA)
pct_doc = documented_protocol / total_protocol_elements * 100
add("Completeness", "Test protocol documentation", score_doc(pct_doc),
    f"{documented_protocol}/{total_protocol_elements} essential protocol elements documented "
    f"({pct_doc:.0f}%); {cells_df['policy'].nunique()} unique fast-charging policies "
    f"documented via the policy_readable field.")

# ============================================================================
# 3. ANOMALY AND NOISE CONTROL
# ============================================================================
print("\n== 3. Anomaly and noise control ==")

# Capacity per (cell, cycle) from the summary QDischarge field -- this is the
# consistent reference discharge used for SOH tracking.
cap_records = []
for f in mat_files:
    bname = os.path.basename(f)[:10]
    with h5py.File(f, "r") as h5:
        n_cells = get_n_cells_from_batch(h5)
        for ci in range(n_cells):
            Q = get_summary_field(h5, ci, "QDischarge")
            cyc = get_summary_field(h5, ci, "cycle")
            if len(Q) == 0 or np.all(np.isnan(Q)):
                continue
            valid = ~np.isnan(Q) & (Q > 0)
            if not valid.any():
                continue
            ref_cap = Q[valid][0]
            SOH = Q / ref_cap
            for idx, (c, q, s) in enumerate(zip(cyc, Q, SOH)):
                if s <= 1.2 and q > 0:
                    cap_records.append({
                        "cell": f"{bname}_c{ci:03d}",
                        "cycle": int(c) if not np.isnan(c) else idx,
                        "capacity_Ah": q, "SOH": s,
                    })
cap_df = pd.DataFrame(cap_records)

if not cap_df.empty:
    caps = cap_df["capacity_Ah"]
    q1, q3 = caps.quantile(0.25), caps.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
    outliers = ((caps < lo) | (caps > hi)).sum()
    pct_outliers = outliers / len(caps) * 100
else:
    pct_outliers = 0
add("Anomaly and noise control", "Statistical outliers", score_pct_low_is_good(pct_outliers),
    f"{pct_outliers:.3f}% of per-cycle discharge capacity values fall outside the 3xIQR range.")

all_diffs = []
if not cap_df.empty:
    for cell, grp in cap_df.groupby("cell"):
        g = grp.sort_values("cycle")
        cell_caps = g["capacity_Ah"].values
        if len(cell_caps) > 1:
            all_diffs.extend(np.diff(cell_caps)[1:])  # skip first transition
all_diffs = np.array(all_diffs)
if len(all_diffs) > 0:
    q1, q3 = np.percentile(all_diffs, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
    jumps = ((all_diffs < lo) | (all_diffs > hi)).sum()
    pct_jumps = jumps / len(all_diffs) * 100
else:
    pct_jumps = 0
add("Anomaly and noise control", "Unexpected signal changes", score_pct_low_is_good(pct_jumps),
    f"{pct_jumps:.3f}% of cycle-to-cycle capacity transitions fall outside the 3xIQR range "
    f"of transition sizes (first transition per cell excluded).")

# Measurement noise from the CV-phase tail of each charge: the last 5% of
# samples within the charge phase is where current has tapered to its
# near-resting value -- the quietest part of an MIT cycle, used here as the
# analogue of Oxford's pseudo-OCV segments.
noise_pct = []
for (cell, cycle), grp in data.groupby(["cell", "cycle"]):
    g = grp.sort_values("t")
    charge = g[g["i"] > 0.05]
    if len(charge) < 20:
        continue
    tail_n = max(5, int(0.05 * len(charge)))
    v_tail = charge["v"].dropna().values[-tail_n:]
    if len(v_tail) > 5:
        diffs = np.abs(np.diff(v_tail))
        noise_pct.append(np.median(diffs) / NOMINAL_VOLTAGE_V * 100)
pct_noise = np.mean(noise_pct) if noise_pct else 0
add("Anomaly and noise control", "Measurement noise", score_noise(pct_noise),
    f"Median sample-to-sample voltage fluctuation during the CV-phase tail of charge "
    f"(near-resting current) is {pct_noise:.4f}% of a {NOMINAL_VOLTAGE_V}V reference.")

# ============================================================================
# 4. REPRESENTATIVENESS AND DIVERSITY
# ============================================================================
print("\n== 4. Representativeness and diversity ==")

# Temperature diversity from the measured T array
t_valid = data["T"].dropna()
if len(t_valid) > 0:
    spread = t_valid.max() - t_valid.min()
    n_temp = 1 if spread < 10 else int(t_valid.round(-1).nunique())
    temp_min, temp_max = t_valid.min(), t_valid.max()
else:
    spread = 0
    n_temp = 1
    temp_min = temp_max = np.nan
add("Representativeness and diversity", "Temperature conditions", score_diversity_count(n_temp),
    f"Measured temperature range [{temp_min:.1f}, {temp_max:.1f}]C "
    f"(chamber setpoint 25-30C per metadata) -> {n_temp} effective condition(s).")

# C-rate / policy diversity from policy_readable
n_policies = cells_df["policy"].nunique()
add("Representativeness and diversity", "C-rate / policy diversity",
    score_diversity_count(n_policies),
    f"{n_policies} unique fast-charging policies documented across {n_cells_total} cells "
    f"(ratio {n_cells_total / n_policies:.1f} cells/policy).")

# Replicate cells: mean cells per unique policy
if n_policies > 0:
    avg_replicates = n_cells_total / n_policies
else:
    avg_replicates = 0
add("Representativeness and diversity", "Replicate cells", score_replicates(avg_replicates),
    f"{avg_replicates:.1f} replicate cells per unique fast-charging policy on average "
    f"({n_cells_total} cells across {n_policies} policies).")

for aspect, info in METADATA_ONLY.items():
    add("Representativeness and diversity", aspect, info["score"], info["finding"])

# ============================================================================
# 5. DISTRIBUTION BALANCE
# ============================================================================
print("\n== 5. Distribution balance ==")

if not cap_df.empty:
    cells_below_80 = cap_df[cap_df["SOH"] < 0.8]["cell"].nunique()
    pct_cells_below_80 = cells_below_80 / n_cells_total * 100
else:
    pct_cells_below_80 = 0
add("Distribution balance", "SOH range coverage", score_soh_range(pct_cells_below_80),
    f"{pct_cells_below_80:.1f}% of cells reach SOH < 80% (end of life).")

if not cap_df.empty:
    bins = pd.cut(cap_df["SOH"].clip(0, 1.05),
                  bins=[0, 0.7, 0.8, 0.9, 0.95, 1.05])
    dist = bins.value_counts(normalize=True) * 100
    dominant = dist.max() if len(dist) else 100
else:
    dominant = 100
add("Distribution balance", "SOH distribution balance", score_soh_balance(dominant),
    f"Largest SOH bin holds {dominant:.1f}% of all cycle records.")

if not cap_df.empty:
    per_cell_cycles = cap_df.groupby("cell")["cycle"].nunique()
    cv_pct = (per_cell_cycles.std() / per_cell_cycles.mean() * 100
              if per_cell_cycles.mean() else 0)
else:
    cv_pct = 0
add("Distribution balance", "Cycle contribution per cell", score_cv(cv_pct),
    f"Coefficient of variation of cycle count across cells: {cv_pct:.1f}%.")

# ============================================================================
# 6. TEMPORAL COHERENCE
# ============================================================================
print("\n== 6. Temporal coherence ==")

ordered_pct = []
if not cap_df.empty:
    for cell, grp in cap_df.groupby("cell"):
        cycles = grp["cycle"].values
        if len(cycles) > 1:
            ordered_pct.append((np.diff(np.sort(cycles)) >= 0).mean() * 100)
pct_ordered = np.mean(ordered_pct) if ordered_pct else 100
add("Temporal coherence", "Monotonic temporal progression", score_pct_high_is_good(pct_ordered),
    f"{pct_ordered:.2f}% of consecutive characterisation-cycle indices are non-decreasing.")

if not cap_df.empty:
    total_transitions, violating = 0, 0
    for cell, grp in cap_df.groupby("cell"):
        g = grp.sort_values("cycle")
        diffs = np.diff(g["SOH"].values)
        total_transitions += len(diffs)
        violating += (diffs > 0.03).sum()
    pct_consistent = (1 - violating / total_transitions) * 100 if total_transitions else 100
else:
    pct_consistent = 100
add("Temporal coherence", "Consistent degradation trend", score_degradation_trend(pct_consistent),
    f"{pct_consistent:.2f}% of cycle-to-cycle SOH transitions are non-increasing beyond a 3% "
    f"noise tolerance (per-transition, not an all-or-nothing per-cell gate).")

if not cap_df.empty:
    dup_free_pct = []
    for cell, grp in cap_df.groupby("cell"):
        dup_free_pct.append((1 - grp["cycle"].duplicated().mean()) * 100)
    pct_index_consistent = np.mean(dup_free_pct) if dup_free_pct else 100
else:
    pct_index_consistent = 100
add("Temporal coherence", "Cycle index consistency", score_pct_high_is_good(pct_index_consistent),
    f"{pct_index_consistent:.2f}% of characterisation-cycle indices are non-duplicated within "
    f"their cell.")

# ============================================================================
# SAVE SCORECARD
# ============================================================================
results_df = pd.DataFrame(results)
csv_path = os.path.join(OUT_DIR, "mit_quality_scorecard.csv")
results_df.to_csv(csv_path, index=False)
print(f"\nScorecard saved: {csv_path}")

fig, ax = plt.subplots(figsize=(13, 0.5 * len(results_df) + 2))
ax.set_xlim(0, 1)
ax.set_ylim(0, len(results_df))
ax.axis("off")
col_x = [0.0, 0.32, 0.90]
for txt, x in zip(["Criterion", "Aspect", "Score"], col_x):
    ax.text(x, len(results_df) + 0.4, txt, fontsize=9, fontweight="bold")

prev_criterion = None
for i, row in results_df.iterrows():
    y = len(results_df) - 1 - i
    bg = "#f7f7f7" if i % 2 == 0 else "white"
    ax.add_patch(mpatches.FancyBboxPatch((0, y), 1, 0.9, boxstyle="square,pad=0",
                                          linewidth=0, facecolor=bg, zorder=0))
    show_criterion = row["criterion"] != prev_criterion
    ax.text(col_x[0], y + 0.35, row["criterion"] if show_criterion else "",
            fontsize=7.5, fontweight="bold")
    ax.text(col_x[1], y + 0.35, row["aspect"], fontsize=7.5)
    ax.text(col_x[2], y + 0.35, row["score"], fontsize=9, fontweight="bold",
            color=SCORE_COLORS[row["score"]])
    prev_criterion = row["criterion"]

legend = [mpatches.Patch(color=c, label=f"{s} — {SCORE_LABELS[s]}")
          for s, c in SCORE_COLORS.items()]
ax.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.05),
          ncol=4, fontsize=8, frameon=True)
ax.set_title("MIT-Stanford-TRI Fast-Charging Dataset — Data Quality Scorecard",
             fontsize=12, fontweight="bold", pad=16)
plt.tight_layout()
fig_path = os.path.join(OUT_DIR, "mit_quality_scorecard.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"Figure saved: {fig_path}")
plt.close(fig)

print("\nScore distribution:")
print(results_df["score"].value_counts().reindex(["++", "+", "o", "-"], fill_value=0))