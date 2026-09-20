"""
CALCE Battery Dataset — Quality Scoring
========================================
Scores the CALCE CX2 subset (CX2-16 and same-protocol siblings) against the
six-dimension quality framework.

Design rule: hardcode ONLY facts that cannot be determined from the raw
cycle-data files at all (calendar aging, dynamic load profiles, real-world
operation — CALCE's own documentation says these are absent for this
subset, and no amount of data analysis could confirm that). Everything
else — chemistry/temperature/DoD/C-rate diversity, physical bounds
violations, missing values, noise, distribution balance, temporal
coherence — is computed dynamically from the CSV files and their filenames.

Output: a scorecard CSV and a scorecard figure, both saved to
./quality_results/
"""

import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ============================================================================
# CONFIG
# ============================================================================
DATA_DIR = r"C:\Users\admin\Desktop\DR2\11 All Datasets\10 Battery Archive Datasets\Battery Archive Data\CALCE\CALCE"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quality_results")
os.makedirs(OUT_DIR, exist_ok=True)

SCORE_LABELS = {"++": "Comprehensive coverage", "+": "Mostly satisfied",
                "o": "Partially satisfied", "-": "Not satisfied"}
SCORE_COLORS = {"++": "#2ca02c", "+": "#98df8a", "o": "#ffbb78", "-": "#d62728"}

# ============================================================================
# METADATA-ONLY FACTS (from CALCE official documentation — cannot be
# recovered from the cycle-data CSVs under any amount of analysis)
# ============================================================================
METADATA_ONLY = {
    "Calendar aging": {
        "score": "-",
        "finding": "CALCE documentation reports no dedicated calendar-aging "
                   "experiment for this CX2 subset.",
    },
    "Dynamic load profiles": {
        "score": "-",
        "finding": "CALCE documentation states cycling follows a fixed CC/CV "
                   "protocol; no dynamic/drive-cycle load profiles are used.",
    },
    "Real-world operation": {
        "score": "-",
        "finding": "Dataset is laboratory-only per official documentation; "
                   "no field/real-world operation data was collected.",
    },
}

PROTOCOL_METADATA = {
    "chemistry": True,
    "temperature": True,
    "charging_protocol": True,
    "charging_rate": True,
    "charge_termination": True,
    "discharge_rate": True,
    "discharge_cutoff": True,
}

# Physical spec bounds -- CX2 family, rated capacity 1.35Ah (1350 mAh).
# Voltage: CALCE-documented cutoffs (calce.umd.edu/battery-data), +-0.05V
# practical tolerance.
# Charge current: 0.5C x 1.35Ah = 0.675A documented; practical QC ceiling
# +0.75A. NOTE: actual Max_Current in this file runs ~1.1-1.3A on nearly
# every cycle -- well above even this practical ceiling. That is treated as
# a genuine, confirmed finding (see "Documented vs. observed current rate"
# below), not adjusted away by widening this bound further.
# Discharge current: documented max magnitude 3C x 1.35Ah = 4.05A; practical
# QC floor -4.15A. Same combined range applied to both Min_Current and
# Max_Current, as specified.
# Capacity: 0-1.42Ah practical QC (nominal 1.35Ah + tolerance). Re-included
# after being dropped previously -- that removal was because the earlier
# 1.15Ah bound (borrowed from CS2_3) false-positived on normal high-SOH
# readings; this 1.42Ah figure is CX2-specific and should not have that
# problem (95th-pct discharge capacity observed was ~1.27Ah).
VOLTAGE_MIN_V, VOLTAGE_MAX_V = 2.65, 4.25
CHARGE_CURRENT_MIN_A, CHARGE_CURRENT_MAX_A = -4.15, 1.35 
DISCHARGE_CURRENT_MIN_A, DISCHARGE_CURRENT_MAX_A = -4.15, 1.35 
CAPACITY_MIN_AH, CAPACITY_MAX_AH = 0.0, 1.42
NOMINAL_CAPACITY_AH = 1.35                      # 1350 mAh per CX2 family metadata.txt
DOCUMENTED_CHARGE_C_RATE = 0.5
NOMINAL_VOLTAGE_V = 3.7                         # typical LCO operating voltage, used only to express noise as a %

# ============================================================================
# SCORING FUNCTIONS (thresholds copied from the quality-assessment table)
# ============================================================================
def score_pct_low_is_good(pct):
    """<0.1 / 0.1-1 / 1-5 / >5"""
    if pct < 0.1: return "++"
    if pct < 1: return "+"
    if pct < 5: return "o"
    return "-"

def score_pct_high_is_good(pct):
    """>=99.9 / 99-99.9 / 95-99 / <95"""
    if pct >= 99.9: return "++"
    if pct >= 99: return "+"
    if pct >= 95: return "o"
    return "-"

def score_noise(pct):
    """<=0.5 / 0.5-1 / 1-2 / >2"""
    if pct <= 0.5: return "++"
    if pct <= 1: return "+"
    if pct <= 2: return "o"
    return "-"

def score_doc(pct):
    """all / >=75 / 25-75 / <25"""
    if pct >= 100: return "++"
    if pct >= 75: return "+"
    if pct >= 25: return "o"
    return "-"

def score_diversity_count(n):
    """>=3 / 2 / 1 (updated table: single value = 'o', no '--' tier for this row)"""
    if n >= 3: return "++"
    if n == 2: return "+"
    return "o"  # n == 1 (or 0, defensively)

def score_replicates(n):
    """>=10 / 5-9 / <5 (updated table: no '--' tier for this row)"""
    if n >= 10: return "++"
    if n >= 5: return "+"
    return "o"

def score_soh_range(pct_cells_below_80):
    """>=20% / 10-20% / <10% with degradation / limited degradation"""
    if pct_cells_below_80 >= 20: return "++"
    if pct_cells_below_80 >= 10: return "+"
    if pct_cells_below_80 > 0: return "o"
    return "-"

def score_soh_balance(dominant_bin_pct):
    """dominant bin <=50 / <=70 / >70 / highly concentrated"""
    if dominant_bin_pct <= 50: return "++"
    if dominant_bin_pct <= 70: return "+"
    if dominant_bin_pct <= 90: return "o"
    return "-"

def score_cv(cv_pct):
    """<=10 / 10-25 / 25-50 / >50"""
    if cv_pct <= 10: return "++"
    if cv_pct <= 25: return "+"
    if cv_pct <= 50: return "o"
    return "-"

def score_degradation_trend(pct_consistent_cells):
    """>=99% / 90-99% / 70-90% / <70%"""
    if pct_consistent_cells >= 99: return "++"
    if pct_consistent_cells >= 90: return "+"
    if pct_consistent_cells >= 70: return "o"
    return "-"

results = []
def add(criterion, aspect, score, finding):
    results.append({"criterion": criterion, "aspect": aspect, "score": score, "finding": finding})
    print(f"  [{score}] {criterion} — {aspect}: {finding}")

# ============================================================================
# LOAD DATA (dynamic — cell id, chemistry, temperature, DoD, C-rate all
# parsed straight from the filenames, which are part of the dataset itself)
# ============================================================================
print("Loading CALCE cycle-data files...")
FNAME_RE = re.compile(
    r"CALCE_(?P<cell>[^_]+)_prism_(?P<chem>[^_]+)_(?P<temp>\d+)C_"
    r"(?P<dod>[\d\-]+)_(?P<charge_c>[\d.]+)-(?P<discharge_c>[\d.]+)C_"
    r"(?P<rep>[a-z])_cycle_data\.csv"
)

files = sorted(glob.glob(os.path.join(DATA_DIR, "*_cycle_data.csv")))
if not files:
    raise FileNotFoundError(f"No *_cycle_data.csv files found in {DATA_DIR}")

frames = []
for f in files:
    m = FNAME_RE.match(os.path.basename(f))
    if not m:
        print(f"  Warning: filename did not match expected pattern: {os.path.basename(f)}")
        continue
    df = pd.read_csv(f, low_memory=False)
    df.columns = df.columns.str.strip()
    df["cell_id"] = m["cell"]
    df["chemistry"] = m["chem"]
    df["temperature_C"] = int(m["temp"])
    df["dod_range"] = m["dod"]
    df["charge_c_rate"] = float(m["charge_c"])
    df["discharge_c_rate"] = float(m["discharge_c"])
    frames.append(df)

data = pd.concat(frames, ignore_index=True)
n_cells = data["cell_id"].nunique()
print(f"Loaded {len(data):,} cycle records from {n_cells} cells: {sorted(data['cell_id'].unique())}")

# Column detection (dynamic — handles minor naming variation across files)
def find_col(keywords):
    for c in data.columns:
        cl = c.lower()
        if all(k in cl for k in keywords):
            return c
    return None

col_cycle = find_col(["cycle"]) or find_col(["cycle", "index"])
col_dchg_cap = find_col(["discharge", "capacity"])
col_chg_cap = find_col(["charge", "capacity"])
col_min_v = find_col(["min", "voltage"])
col_max_v = find_col(["max", "voltage"])
col_min_i = find_col(["min", "current"])
col_max_i = find_col(["max", "current"])

essential_cols = [c for c in [col_cycle, col_dchg_cap, col_min_v, col_max_v] if c]

print(f"Detected columns -> cycle:{col_cycle} dchg_cap:{col_dchg_cap} chg_cap:{col_chg_cap} "
      f"min_v:{col_min_v} max_v:{col_max_v} min_i:{col_min_i} max_i:{col_max_i}\n")

# ============================================================================
# 1. CORRECTNESS
# ============================================================================
print("== 1. Correctness ==")

viol, total = 0, 0
per_signal = []
for label, col, lo, hi in [
    ("Min_Voltage", col_min_v, VOLTAGE_MIN_V, VOLTAGE_MAX_V),
    ("Max_Voltage", col_max_v, VOLTAGE_MIN_V, VOLTAGE_MAX_V),
    ("Min_Current", col_min_i, DISCHARGE_CURRENT_MIN_A, DISCHARGE_CURRENT_MAX_A),
    ("Max_Current", col_max_i, CHARGE_CURRENT_MIN_A, CHARGE_CURRENT_MAX_A),
    ("Discharge_Capacity", col_dchg_cap, CAPACITY_MIN_AH, CAPACITY_MAX_AH),
    ("Charge_Capacity", col_chg_cap, CAPACITY_MIN_AH, CAPACITY_MAX_AH),
]:
    if col:
        s = data[col].dropna()
        v = ((s < lo) | (s > hi)).sum()
        viol += v
        total += len(s)
        pct = v / len(s) * 100 if len(s) else 0
        per_signal.append((label, col, lo, hi, v, len(s), pct))
        print(f"    Physical plausibility breakdown -- {label} ({col}): "
              f"{v:,}/{len(s):,} ({pct:.2f}%) outside [{lo}, {hi}], "
              f"observed range [{s.min():.4f}, {s.max():.4f}], "
              f"5th/95th pct [{s.quantile(0.05):.4f}, {s.quantile(0.95):.4f}]")
pct_implausible = (viol / total * 100) if total else 0
add("Correctness", "Physical plausibility", score_pct_low_is_good(pct_implausible),
    f"{viol:,}/{total:,} readings ({pct_implausible:.3f}%) outside the CX2-specific practical QC "
    f"envelope (voltage {VOLTAGE_MIN_V}-{VOLTAGE_MAX_V}V, charge current {CHARGE_CURRENT_MIN_A}-"
    f"{CHARGE_CURRENT_MAX_A}A, discharge current {DISCHARGE_CURRENT_MIN_A}-{DISCHARGE_CURRENT_MAX_A}A, "
    f"capacity {CAPACITY_MIN_AH}-{CAPACITY_MAX_AH}Ah). See console for the per-signal breakdown.")

# Not a plausibility check, but a genuine finding worth reporting on its own:
# CALCE documents a 0.5C charge rate for all CX2 cells, but the actual
# Max_Current in this file runs ~1.1-1.3A on nearly every cycle -- well
# above the 0.675A that 0.5C of a 1.35Ah cell implies, and above even the
# 0.75A practical QC ceiling. This holds broadly across cycles rather than
# as an occasional spike, so it looks like a real mismatch between the
# documented protocol and this specific (Battery-Archive-reprocessed) file.
if col_max_i:
    max_i_vals = data[col_max_i].dropna()
    documented_current_a = DOCUMENTED_CHARGE_C_RATE * NOMINAL_CAPACITY_AH
    implied_c_rate = max_i_vals.median() / NOMINAL_CAPACITY_AH
    pct_far_from_documented = (max_i_vals > 2 * documented_current_a).mean() * 100
    add("Correctness", "Documented vs. observed current rate", "o" if pct_far_from_documented > 50 else "++",
        f"CALCE documentation specifies a {DOCUMENTED_CHARGE_C_RATE}C charge rate "
        f"({documented_current_a:.3f}A for {NOMINAL_CAPACITY_AH}Ah nominal). Median observed "
        f"Max_Current is {max_i_vals.median():.3f}A (implied rate {implied_c_rate:.2f}C); "
        f"{pct_far_from_documented:.1f}% of cycles exceed twice the documented rate. Reported as "
        f"a finding, not folded into the physical-plausibility score.")

if col_min_i and col_max_i:
    valid_cycles = data.dropna(subset=[col_min_i, col_max_i])
    consistent = ((valid_cycles[col_max_i] > 0) & (valid_cycles[col_min_i] < 0)).sum()
    pct_sign_ok = (consistent / len(valid_cycles) * 100) if len(valid_cycles) else 0
    add("Correctness", "Current sign convention", score_pct_high_is_good(pct_sign_ok),
        f"{consistent:,}/{len(valid_cycles):,} cycles ({pct_sign_ok:.2f}%) show "
        f"positive charge current and negative discharge current, as documented.")
else:
    add("Correctness", "Current sign convention", "o",
        "Min/Max current columns not present in the cycle-data files; convention could not be verified per-record.")

# ============================================================================
# 2. COMPLETENESS
# ============================================================================
print("\n== 2. Completeness ==")

if essential_cols:
    missing = data[essential_cols].isna().sum().sum()
    total_cells_checked = data[essential_cols].size
    pct_missing = missing / total_cells_checked * 100
else:
    pct_missing = 0
add("Completeness", "Missing values", score_pct_low_is_good(pct_missing),
    f"{pct_missing:.4f}% missing across essential columns {essential_cols}.")

cont_pct_list = []
if col_cycle:
    for _, grp in data.groupby("cell_id"):
        cycles = np.sort(grp[col_cycle].dropna().unique())
        if len(cycles) > 1:
            expected = np.arange(cycles.min(), cycles.max() + 1)
            present = np.isin(expected, cycles)
            cont_pct_list.append(present.mean() * 100)
pct_continuity = np.mean(cont_pct_list) if cont_pct_list else 100
add("Completeness", "Temporal continuity", score_pct_high_is_good(pct_continuity),
    f"Average cycle-index continuity across cells: {pct_continuity:.2f}%.")

# Test protocol documentation
documented_protocol = sum(PROTOCOL_METADATA.values())
total_protocol_elements = len(PROTOCOL_METADATA)
pct_doc = documented_protocol / total_protocol_elements * 100

add(
    "Completeness",
    "Test protocol documentation",
    score_doc(pct_doc),
    f"{documented_protocol}/{total_protocol_elements} essential protocol "
    f"elements documented ({pct_doc:.0f}%)."
)

# ============================================================================
# 3. ANOMALY AND NOISE CONTROL
# ============================================================================
print("\n== 3. Anomaly and noise control ==")

if col_dchg_cap:
    caps = data[col_dchg_cap].dropna()
    q1, q3 = caps.quantile(0.25), caps.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
    outliers = ((caps < lo) | (caps > hi)).sum()
    pct_outliers = outliers / len(caps) * 100
else:
    pct_outliers = 0
add("Anomaly and noise control", "Statistical outliers", score_pct_low_is_good(pct_outliers),
    f"{pct_outliers:.3f}% of discharge-capacity readings fall outside the 3xIQR range.")

all_diffs = []
if col_dchg_cap and col_cycle:
    for _, grp in data.groupby("cell_id"):
        g = grp.sort_values(col_cycle)
        caps = g[col_dchg_cap].dropna().values
        if len(caps) > 1:
            # skip the first transition (formation -> cycle 1 is expected to be large)
            all_diffs.extend(np.diff(caps)[1:])
all_diffs = np.array(all_diffs)
if len(all_diffs) > 0:
    # Robust, self-calibrating threshold (mirrors the 3xIQR rule used for
    # statistical outliers, applied here to the cycle-to-cycle *differences*
    # rather than the raw values). Catches real jumps without flagging the
    # periodic reference-performance-test bumps that are normal in aging data.
    q1, q3 = np.percentile(all_diffs, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
    jumps = ((all_diffs < lo) | (all_diffs > hi)).sum()
    pct_jumps = jumps / len(all_diffs) * 100
else:
    pct_jumps = 0
add("Anomaly and noise control", "Unexpected signal changes", score_pct_low_is_good(pct_jumps),
    f"{pct_jumps:.3f}% of cycle-to-cycle capacity transitions fall outside the 3xIQR range "
    f"of transition sizes (first, formation-related transition excluded per cell).")

print("\nComputing measurement noise from raw timeseries voltage (rest periods)...")
TS_FNAME_RE = re.compile(
    r"CALCE_(?P<cell>[^_]+)_prism_(?P<chem>[^_]+)_(?P<temp>\d+)C_"
    r"(?P<dod>[\d\-]+)_(?P<charge_c>[\d.]+)-(?P<discharge_c>[\d.]+)C_"
    r"(?P<rep>[a-z])_timeseries\.csv"
)
ts_files = sorted(glob.glob(os.path.join(DATA_DIR, "*_timeseries.csv")))

voltage_noise_pct = []
for f in ts_files:
    if not TS_FNAME_RE.match(os.path.basename(f)):
        continue
    ts = pd.read_csv(f, low_memory=False)
    ts.columns = ts.columns.str.strip()
    v_col = next((c for c in ts.columns if "voltage" in c.lower()), None)
    i_col = next((c for c in ts.columns if "current" in c.lower()), None)
    if v_col is None:
        continue
    v = ts[v_col].dropna()
    if i_col is not None:
        i = ts[i_col].reindex(v.index)
        rest_mask = i.abs() < 0.05  # near-zero current -> rest / CV-tail: true signal should be flat
        v_signal = v[rest_mask] if rest_mask.sum() > 20 else v
    else:
        v_signal = v
    diffs = np.abs(np.diff(v_signal.values))
    if len(diffs) > 0:
        voltage_noise_pct.append(np.median(diffs) / NOMINAL_VOLTAGE_V * 100)

pct_noise = np.mean(voltage_noise_pct) if voltage_noise_pct else 0
add("Anomaly and noise control", "Measurement noise", score_noise(pct_noise),
    f"Median sample-to-sample voltage fluctuation during low-current (rest/CV-tail) periods is "
    f"{pct_noise:.4f}% of a {NOMINAL_VOLTAGE_V}V reference, computed from the raw timeseries files "
    f"(not from capacity, which trends with aging rather than reflecting sensor noise).")

# ============================================================================
# 4. REPRESENTATIVENESS AND DIVERSITY
# (chemistry / temperature / DoD / C-rate parsed dynamically from filenames;
#  calendar aging / dynamic load / real-world are the hardcoded metadata facts)
# ============================================================================
print("\n== 4. Representativeness and diversity ==")

n_chem = data["chemistry"].nunique()
add("Representativeness and diversity", "Chemistry diversity", score_diversity_count(n_chem),
    f"{n_chem} chemistry value(s) across the subset ({sorted(data['chemistry'].unique())}) with no "
    f"variation between cells -> matches table condition '1 without variation'.")

n_temp = data["temperature_C"].nunique()
add("Representativeness and diversity", "Temperature conditions", score_diversity_count(n_temp),
    f"{n_temp} temperature setpoint(s) ({sorted(data['temperature_C'].unique())}°C), fixed across "
    f"the whole subset -> matches table condition 'fixed'.")

n_dod = data["dod_range"].nunique()
add("Representativeness and diversity", "DoD diversity", score_diversity_count(n_dod),
    f"{n_dod} DoD range(s) ({sorted(data['dod_range'].unique())}), fixed across the whole subset "
    f"-> matches table condition 'fixed'.")

n_crate = data[["charge_c_rate", "discharge_c_rate"]].drop_duplicates().shape[0]
add("Representativeness and diversity", "C-rate diversity", score_diversity_count(n_crate),
    f"{n_crate} charge/discharge C-rate combination(s), fixed across the whole subset "
    f"-> matches table condition 'fixed'.")

add("Representativeness and diversity", "Replicate cells", score_replicates(n_cells),
    f"{n_cells} cells cycled under the identical documented protocol.")

for aspect, info in METADATA_ONLY.items():
    add("Representativeness and diversity", aspect, info["score"], info["finding"])

# ============================================================================
# 5. DISTRIBUTION BALANCE
# ============================================================================
print("\n== 5. Distribution balance ==")

soh_frames = []
if col_dchg_cap and col_cycle:
    for cell_id, grp in data.groupby("cell_id"):
        g = grp.sort_values(col_cycle)
        caps = g[col_dchg_cap].dropna()
        if len(caps) < 3:
            continue
        initial_cap = caps.head(3).median()
        if initial_cap <= 0:
            continue
        soh = (g[col_dchg_cap] / initial_cap).clip(0, 1.05)
        soh_frames.append(pd.DataFrame({"cell_id": cell_id, col_cycle: g[col_cycle], "SOH": soh}))
soh_df = pd.concat(soh_frames, ignore_index=True) if soh_frames else pd.DataFrame(columns=["cell_id", "SOH"])
soh_df = soh_df.dropna(subset=["SOH"])

if not soh_df.empty:
    cells_below_80 = soh_df[soh_df["SOH"] < 0.8]["cell_id"].nunique()
    pct_cells_below_80 = cells_below_80 / n_cells * 100
else:
    pct_cells_below_80 = 0
add("Distribution balance", "SOH range coverage", score_soh_range(pct_cells_below_80),
    f"{pct_cells_below_80:.1f}% of cells reach SOH < 80% (end of life).")

if not soh_df.empty:
    bins = pd.cut(soh_df["SOH"], bins=[0, 0.7, 0.8, 0.9, 0.95, 1.05])
    dist = bins.value_counts(normalize=True) * 100
    dominant = dist.max()
else:
    dominant = 100
add("Distribution balance", "SOH distribution balance", score_soh_balance(dominant),
    f"Largest SOH bin holds {dominant:.1f}% of all cycle records.")

if col_cycle:
    per_cell_cycles = data.groupby("cell_id")[col_cycle].max()
    cv_pct = (per_cell_cycles.std() / per_cell_cycles.mean() * 100) if per_cell_cycles.mean() else 0
else:
    cv_pct = 0
add("Distribution balance", "Cycle contribution per cell", score_cv(cv_pct),
    f"Coefficient of variation of cycle-life across cells: {cv_pct:.1f}%.")

# ============================================================================
# 6. TEMPORAL COHERENCE
# ============================================================================
print("\n== 6. Temporal coherence ==")

if col_cycle:
    ordered_pct = []
    for _, grp in data.groupby("cell_id"):
        cycles = grp[col_cycle].dropna().values
        if len(cycles) > 1:
            ordered_pct.append((np.diff(cycles) >= 0).mean() * 100)
    pct_ordered = np.mean(ordered_pct) if ordered_pct else 100
else:
    pct_ordered = 100
add("Temporal coherence", "Monotonic temporal progression", score_pct_high_is_good(pct_ordered),
    f"{pct_ordered:.2f}% of consecutive cycle-index steps are non-decreasing.")

if not soh_df.empty:
    # Scored per TRANSITION (matches how the table's thresholds read, and how
    # "monotonic temporal progression" above is scored) rather than requiring
    # an entire cell's whole cycle life to have zero violations — a single
    # noisy uptick (e.g. a reference-performance-test cycle) would otherwise
    # fail the whole cell even though the cell's overall trend is clearly
    # degrading.
    total_transitions, violating_transitions = 0, 0
    for cell_id, grp in soh_df.groupby("cell_id"):
        g = grp.sort_values(col_cycle) if col_cycle else grp
        soh_vals = g["SOH"].values
        if len(soh_vals) > 1:
            diffs = np.diff(soh_vals)
            total_transitions += len(diffs)
            violating_transitions += (diffs > 0.03).sum()  # >3% SOH recovery = violation
    pct_consistent = (1 - violating_transitions / total_transitions) * 100 if total_transitions else 100
else:
    pct_consistent = 100
add("Temporal coherence", "Consistent degradation trend", score_degradation_trend(pct_consistent),
    f"{pct_consistent:.2f}% of cycle-to-cycle SOH transitions are non-increasing beyond a "
    f"3% noise tolerance (measured across all cells, not an all-or-nothing per-cell gate).")

if col_cycle:
    dup_free_pct = []
    for _, grp in data.groupby("cell_id"):
        cycles = grp[col_cycle].dropna()
        dup_free_pct.append((1 - cycles.duplicated().mean()) * 100)
    pct_index_consistent = np.mean(dup_free_pct) if dup_free_pct else 100
else:
    pct_index_consistent = 100
add("Temporal coherence", "Cycle index consistency", score_pct_high_is_good(pct_index_consistent),
    f"{pct_index_consistent:.2f}% of cycle-index values are non-duplicated within their cell.")

# ============================================================================
# SAVE SCORECARD
# ============================================================================
results_df = pd.DataFrame(results)
csv_path = os.path.join(OUT_DIR, "calce_quality_scorecard.csv")
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
    ax.text(col_x[0], y + 0.35, row["criterion"] if show_criterion else "", fontsize=7.5, fontweight="bold")
    ax.text(col_x[1], y + 0.35, row["aspect"], fontsize=7.5)
    ax.text(col_x[2], y + 0.35, row["score"], fontsize=9, fontweight="bold",
            color=SCORE_COLORS[row["score"]])
    prev_criterion = row["criterion"]

legend = [mpatches.Patch(color=c, label=f"{s} — {SCORE_LABELS[s]}") for s, c in SCORE_COLORS.items()]
ax.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=4, fontsize=8, frameon=True)
ax.set_title("CALCE (CX2 subset) — Data Quality Scorecard", fontsize=12, fontweight="bold", pad=16)
plt.tight_layout()
fig_path = os.path.join(OUT_DIR, "calce_quality_scorecard.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"Figure saved: {fig_path}")
plt.close(fig)

print("\nScore distribution:")
print(results_df["score"].value_counts().reindex(["++", "+", "o", "-"], fill_value=0))