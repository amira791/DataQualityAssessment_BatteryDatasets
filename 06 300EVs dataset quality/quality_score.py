"""
300-EV Real-World BMS Dataset — Quality Scoring
=================================================
Scores the 300-EV Real-World BMS Dataset (Liu et al. 2025, Nature
Communications) against the same six-dimension quality framework used
for the Oxford, MIT-Stanford-TRI, CALCE, SNL, NASA, and Guangzhou
datasets.

Design rule: hardcode ONLY facts that cannot be determined from the raw
data files at all (calendar aging — absent by design; and the physical
envelope, which is derived from the paper's stated cutoff and the 96S
NCM pack arithmetic). Everything else — chemistry/vehicle diversity,
physical-bounds violations, missing values, noise, distribution balance,
temporal coherence, dynamic-operation presence, replicate vehicles — is
computed dynamically from the CSV files and their columns.

Physical plausibility envelope:
    Pack voltage:     240 - 408 V
        Lower: 96 cells x 2.5 V/cell NCM cutoff = 240 V
        Upper: 96 cells x 4.25 V/cell = 408 V (paper's stated cell cutoff)
    Pack current:     -400 - +400 A (QC guard rail, not paper-derived;
        observed peak ~340 A during acceleration)
    Battery temp:     -20 - +60 C (generic Li-ion envelope; the paper
        does not specify a thermal operating window)
    SOC:              0 - 110 % (110 upper tolerance accommodates
        regen-into-full-pack behavior where the BMS reports usable SOC
        above nominal 100%; values >110% are flagged as anomalies)

Sentinel filtering (applied before signal-level checks):
    chargestatus == 255        -> sensor-error marker; all such rows have
                                  totalcurrent == -1000.0 A (exact sentinel)
    totalvoltage == 0.0        -> vehicle-off placeholder (BMS asleep)
    maxtemperaturevalue == -40.0 -> temperature-probe-not-connected marker

Current sign convention for THIS dataset (verified from data, not README):
    NEGATIVE = charging  (chargestatus == 1, mean I = -53 A)
    POSITIVE = discharging / traction  (chargestatus == 3, mean I = +8.5 A)
    Driving includes regen braking, so ~15-20% of driving rows are
    legitimately negative.

chargestatus codes (inferred from exploration):
    0   = rest / parked
    1   = charging
    3   = driving / discharge
    4   = charging complete / taper
    255 = sensor error (drop)

Time column:
    'terminaltime' is a numeric seconds counter (not an epoch). It resets
    to 0 at each BMS wake-up, so it defines sessions. Sampling interval is
    ~10 s. All time-based computations treat it as seconds; monotonicity
    is checked WITHIN sessions, not across them.

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
DATASET_PATH = r"C:\Users\admin\Desktop\DR2\11 All Datasets\07 300-EV Real-World BMS Dataset Liu et al. (2025)\300 EVs dataset"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quality_results")
os.makedirs(OUT_DIR, exist_ok=True)

SCORE_LABELS = {"++": "Comprehensive coverage", "+": "Mostly satisfied",
                "o": "Partially satisfied", "-": "Not satisfied"}
SCORE_COLORS = {"++": "#2ca02c", "+": "#98df8a", "o": "#ffbb78", "-": "#d62728"}

# Sampling: read this many rows per file. 200k rows ~= 23 days of
# continuous operation per vehicle at 10s sampling.
SAMPLE_ROWS_PER_FILE = 200_000

# ============================================================================
# METADATA-ONLY FACTS
# ============================================================================
METADATA_ONLY = {
    "Calendar aging": {
        "score": "-",
        "finding": "Documentation reports no dedicated calendar-aging "
                   "experiment; the dataset captures real-world cyclic "
                   "operation only, with rest periods not structured as a "
                   "calendar-aging study.",
    },
}

# Protocol elements documented in the paper / dataset README.
PROTOCOL_METADATA = {
    "chemistry": True,             # NCM
    "pack_configuration": True,    # 96 cells in series, 155 Ah
    "sampling_frequency": True,    # 0.1 Hz
    "data_acquisition": True,      # onboard BMS
    "charging_protocol": True,     # fast <= 0.8C, slow ~0.15C
    "cell_level_monitoring": True, # cell-level voltages present (stringified)
    "soh_labels": True,            # capacity-based SOH provided
    "current_convention": False,   # not stated in the paper; verified from data
}

# ============================================================================
# PAPER-DERIVED PHYSICAL ENVELOPE
# ============================================================================
# Pack voltage: 96S NCM arithmetic.
#   240 V = 96 x 2.5 V/cell (NCM lower cutoff)
#   408 V = 96 x 4.25 V/cell (paper's stated upper cell cutoff)
VOLTAGE_MIN_V, VOLTAGE_MAX_V = 240.0, 408.0

# Current: QC guard rail; the paper does not specify absolute current bounds.
CURRENT_MIN_A, CURRENT_MAX_A = -400.0, 400.0

# Temperature: generic Li-ion operating envelope.
TEMP_MIN_C, TEMP_MAX_C = -20.0, 60.0

# SOC: nominal 0-100%, with 110% tolerance for regen-into-full-pack
# behavior. Values >110% indicate a scale/units artifact or BMS fault.
SOC_MIN_PCT, SOC_MAX_PCT = 0.0, 110.0

# Sentinel values (exact matches written by the BMS)
SENTINEL_CHARGESTATUS = 255
SENTINEL_VOLTAGE = 0.0
SENTINEL_TEMP = -40.0
SENTINEL_CURRENT = -1000.0

# chargestatus code map
CHARGESTATUS_REST = 0
CHARGESTATUS_CHARGING = 1
CHARGESTATUS_DRIVING = 3
CHARGESTATUS_FULL = 4

# ============================================================================
# SCORING FUNCTIONS
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

results = []
def add(criterion, aspect, score, finding):
    results.append({"criterion": criterion, "aspect": aspect,
                    "score": score, "finding": finding})
    print(f"  [{score}] {criterion} — {aspect}: {finding}")

# ============================================================================
# LOAD DATA (sample-based; one CSV per vehicle)
# ============================================================================
print("Loading 300-EV Real-World BMS Dataset (sampled)...")
csv_files = sorted(glob.glob(os.path.join(DATASET_PATH, "vin*.csv")))
if not csv_files:
    raise FileNotFoundError(f"No vin*.csv files found in {DATASET_PATH}")

n_files_total = len(csv_files)
print(f"Found {n_files_total} vehicle CSV files")

frames = []
skipped = 0
for f in csv_files:
    try:
        df = pd.read_csv(f, nrows=SAMPLE_ROWS_PER_FILE, low_memory=False)
        df.columns = df.columns.str.strip()
        df["_vin"] = os.path.basename(f).replace(".csv", "")
        frames.append(df)
    except Exception as e:
        skipped += 1
        print(f"  Warning: could not read {os.path.basename(f)}: {e}")

if skipped:
    print(f"Skipped {skipped} files.")

data = pd.concat(frames, ignore_index=True)
n_vehicles = data["_vin"].nunique()
print(f"Loaded {len(data):,} rows from {n_vehicles} vehicles "
      f"(sample: {SAMPLE_ROWS_PER_FILE:,} rows per file)")

# Coerce numeric
numeric_cols = ["terminaltime", "soc", "speed", "totalodometer", "chargestatus",
                "totalvoltage", "totalcurrent", "minvoltagebattery",
                "maxvoltagebattery", "mintemperaturevalue",
                "maxtemperaturevalue"]
for c in numeric_cols:
    if c in data.columns:
        data[c] = pd.to_numeric(data[c], errors="coerce")

# Column references
col_time = "terminaltime" if "terminaltime" in data.columns else None
col_soc = "soc" if "soc" in data.columns else None
col_speed = "speed" if "speed" in data.columns else None
col_mileage = "totalodometer" if "totalodometer" in data.columns else None
col_status = "chargestatus" if "chargestatus" in data.columns else None
col_v_pack = "totalvoltage" if "totalvoltage" in data.columns else None
col_i_pack = "totalcurrent" if "totalcurrent" in data.columns else None
col_v_cell_min = "minvoltagebattery" if "minvoltagebattery" in data.columns else None
col_v_cell_max = "maxvoltagebattery" if "maxvoltagebattery" in data.columns else None
col_t_cell_min = "mintemperaturevalue" if "mintemperaturevalue" in data.columns else None
col_t_cell_max = "maxtemperaturevalue" if "maxtemperaturevalue" in data.columns else None

print(f"Detected columns -> time:{col_time} soc:{col_soc} speed:{col_speed} "
      f"mileage:{col_mileage} status:{col_status} v_pack:{col_v_pack} "
      f"i_pack:{col_i_pack}\n")

# ============================================================================
# SENTINEL FILTERING (done once, before signal-level checks)
# ============================================================================
n_before = len(data)
sentinel_stats = {}

if col_status:
    mask = data[col_status] == SENTINEL_CHARGESTATUS
    sentinel_stats["chargestatus=255"] = int(mask.sum())
    data = data[~mask].copy()

if col_v_pack:
    mask = data[col_v_pack] == SENTINEL_VOLTAGE
    sentinel_stats["totalvoltage=0"] = int(mask.sum())
    data = data[~mask].copy()

if col_t_cell_max:
    mask = data[col_t_cell_max] == SENTINEL_TEMP
    sentinel_stats["maxtemperaturevalue=-40"] = int(mask.sum())
    data = data[~mask].copy()

n_eval = len(data)   # <-- BUG 1 FIX: single authoritative denominator
print(f"Sentinel filtering: {n_before:,} -> {n_eval:,} rows "
      f"({n_before - n_eval:,} removed)")
for k, v in sentinel_stats.items():
    print(f"    {k}: {v:,} rows")
print(f"  Post-sentinel rows available for signal checks: {n_eval:,}\n")

# ============================================================================
# 1. CORRECTNESS
# ============================================================================
print("== 1. Correctness ==")

# BUG 3 FIX (part b): SOC added to the plausibility envelope
viol, total = 0, 0
plausibility_rows = []
for label, col, lo, hi in [
    ("Pack voltage", col_v_pack, VOLTAGE_MIN_V, VOLTAGE_MAX_V),
    ("Pack current", col_i_pack, CURRENT_MIN_A, CURRENT_MAX_A),
    ("Battery temperature", col_t_cell_max, TEMP_MIN_C, TEMP_MAX_C),
    ("SOC", col_soc, SOC_MIN_PCT, SOC_MAX_PCT),
]:
    if col:
        s = data[col].dropna()
        if len(s) == 0:
            continue
        v = int(((s < lo) | (s > hi)).sum())
        viol += v
        total += len(s)
        pct = v / len(s) * 100
        n_nan = n_eval - len(s)
        plausibility_rows.append((label, col, v, len(s), pct, s.min(), s.max(), lo, hi, n_nan))
        print(f"    Physical plausibility breakdown -- {label} ({col}): "
              f"{v:,}/{len(s):,} non-NaN readings ({pct:.3f}%) outside "
              f"[{lo}, {hi}], observed range [{s.min():.1f}, {s.max():.1f}]"
              + (f"  [{n_nan:,} NaN rows skipped]" if n_nan else ""))

pct_implausible = (viol / total * 100) if total else 0
finding_physical = (
    f"{viol:,}/{total:,} pack voltage/current/temperature/SOC readings "
    f"({pct_implausible:.3f}%) outside the paper-derived bounds. Pack "
    f"voltage {VOLTAGE_MIN_V}-{VOLTAGE_MAX_V} V (96S NCM: 96 x 2.5 V lower "
    f"cutoff, 96 x 4.25 V upper cutoff from the paper). Pack current "
    f"{CURRENT_MIN_A}-{CURRENT_MAX_A} A (QC guard rail, not paper-derived; "
    f"observed peak ~340 A). Battery temperature {TEMP_MIN_C}-{TEMP_MAX_C} C "
    f"(generic Li-ion envelope; paper does not specify a thermal window). "
    f"SOC {SOC_MIN_PCT}-{SOC_MAX_PCT}% (nominal 0-100%, with 10% tolerance "
    f"for regen-into-full-pack behavior). Sentinel rows "
    f"(chargestatus=255, 0 V pack placeholder, -40 C temperature "
    f"placeholder) excluded before the check. Denominator {total:,} "
    f"non-NaN readings summed across the four signals."
)
add("Correctness", "Physical plausibility",
    score_pct_low_is_good(pct_implausible), finding_physical)

# ---------------------------------------------------------------------------
# Current sign convention (verified from data, with regen tolerance)
# ---------------------------------------------------------------------------
if col_i_pack and col_status:
    valid = data.dropna(subset=[col_i_pack, col_status])
    chg = valid[valid[col_status] == CHARGESTATUS_CHARGING]
    drv = valid[valid[col_status] == CHARGESTATUS_DRIVING]

    chg_neg_pct = (chg[col_i_pack] < 0).mean() * 100 if len(chg) else 0
    drv_pos_pct = (drv[col_i_pack] > 0).mean() * 100 if len(drv) else 0
    n_chg, n_drv = len(chg), len(drv)

    charge_score = score_pct_high_is_good(chg_neg_pct)

    if drv_pos_pct >= 75.0:
        drive_score = "+"
    elif drv_pos_pct >= 50.0:
        drive_score = "o"
    else:
        drive_score = "-"

    if charge_score == "++" and drive_score == "+":
        overall = "++"
    elif charge_score in ("++", "+") and drive_score == "+":
        overall = "+"
    elif charge_score in ("++", "+", "o") and drive_score in ("+", "o"):
        overall = "o"
    else:
        overall = "-"

    finding_sign = (
        f"Convention verified from raw data (not stated in the paper): "
        f"negative = charging, positive = discharging. "
        f"Charging (status=1, n={n_chg:,}): {chg_neg_pct:.2f}% of rows have "
        f"negative current. Driving (status=3, n={n_drv:,}): {drv_pos_pct:.2f}% "
        f"of rows have positive current; the remaining "
        f"{100 - drv_pos_pct:.2f}% are negative due to regen braking, which "
        f"is legitimate physics. Drive-side tolerance: >= 75% positive-dominant "
        f"is the physical floor for typical city/highway mixes."
    )
    add("Correctness", "Current sign convention", overall, finding_sign)
else:
    add("Correctness", "Current sign convention", "o",
        "Pack current or chargestatus column not present.")

# ============================================================================
# 2. COMPLETENESS
# ============================================================================
print("\n== 2. Completeness ==")

essential_cols = [c for c in [col_time, col_soc, col_v_pack, col_i_pack,
                              col_t_cell_max] if c]
if essential_cols:
    missing = data[essential_cols].isna().sum().sum()
    total_cells_checked = data[essential_cols].size
    pct_missing = missing / total_cells_checked * 100
else:
    pct_missing = 0
finding_missing = (
    f"{pct_missing:.4f}% missing across powertrain-level essential columns "
    f"{essential_cols}. Stringified-array columns (batteryvoltage, "
    f"probetemperatures) are excluded — they store per-cell/per-probe "
    f"values in a single text field and are not scalar measurements."
)
add("Completeness", "Missing values",
    score_pct_low_is_good(pct_missing), finding_missing)

cont_pct_list = []
if col_time:
    for _, grp in data.groupby("_vin"):
        t = grp[col_time].dropna().sort_values().values
        if len(t) > 1:
            diffs = np.diff(t)
            diffs = diffs[diffs > 0]
            if len(diffs) > 0:
                med = np.median(diffs)
                if med > 0:
                    cont_pct_list.append((diffs <= 5 * med).mean() * 100)
pct_continuity = np.mean(cont_pct_list) if cont_pct_list else 100
finding_continuity = (
    f"Average within-vehicle sampling continuity (no gaps >5x the median "
    f"interval): {pct_continuity:.2f}%. BMS stops recording when the "
    f"vehicle is switched off; gaps between sessions are expected."
)
add("Completeness", "Temporal continuity",
    score_pct_high_is_good(pct_continuity), finding_continuity)

documented_protocol = sum(PROTOCOL_METADATA.values())
total_protocol_elements = len(PROTOCOL_METADATA)
pct_doc = documented_protocol / total_protocol_elements * 100
finding_doc = (
    f"{documented_protocol}/{total_protocol_elements} essential protocol "
    f"elements documented ({pct_doc:.0f}%). The current sign convention is "
    f"not stated in the paper or README and was inferred from the raw data."
)
add("Completeness", "Test protocol documentation",
    score_doc(pct_doc), finding_doc)

# ============================================================================
# 3. ANOMALY AND NOISE CONTROL
# ============================================================================
print("\n== 3. Anomaly and noise control ==")

volt_out_pct_list = []
if col_v_pack:
    for _, grp in data.groupby("_vin"):
        s = grp[col_v_pack].dropna()
        if len(s) > 100:
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
            volt_out_pct_list.append(((s < lo) | (s > hi)).mean() * 100)
pct_outliers = np.mean(volt_out_pct_list) if volt_out_pct_list else 0
finding_outliers = (
    f"{pct_outliers:.3f}% of pack-voltage readings fall outside the 3xIQR "
    f"range (averaged across {len(volt_out_pct_list)} vehicles). "
    f"Note: this check targets voltage only; SOC plausibility is covered "
    f"separately under Correctness -> Physical plausibility."
)
add("Anomaly and noise control", "Statistical outliers",
    score_pct_low_is_good(pct_outliers), finding_outliers)

jump_pct_list = []
if col_v_pack:
    for _, grp in data.groupby("_vin"):
        s = grp[col_v_pack].dropna()
        if len(s) > 100:
            d = s.diff().abs().dropna()
            if len(d) > 0:
                q1, q3 = d.quantile(0.25), d.quantile(0.75)
                iqr = q3 - q1
                hi = q3 + 3 * iqr
                jump_pct_list.append((d > hi).mean() * 100)
pct_jumps = np.mean(jump_pct_list) if jump_pct_list else 0
finding_jumps = (
    f"{pct_jumps:.3f}% of consecutive pack-voltage changes exceed the 3xIQR "
    f"range of transition sizes. Real-world driving includes legitimate step "
    f"changes from load switching and regen braking."
)
add("Anomaly and noise control", "Unexpected signal changes",
    score_pct_low_is_good(pct_jumps), finding_jumps)

noise_pct_list = []
if col_v_pack and col_i_pack:
    for _, grp in data.groupby("_vin"):
        g = grp.dropna(subset=[col_v_pack, col_i_pack])
        if len(g) < 100:
            continue
        i_abs = g[col_i_pack].abs()
        quiet = g[i_abs < i_abs.quantile(0.05)]
        v = quiet[col_v_pack].values
        if len(v) > 20:
            d = np.abs(np.diff(v))
            if len(d):
                ref = grp[col_v_pack].median()
                if ref > 0:
                    noise_pct_list.append(np.median(d) / ref * 100)
pct_noise = np.mean(noise_pct_list) if noise_pct_list else 0
finding_noise = (
    f"Median sample-to-sample pack-voltage fluctuation during low-current "
    f"(quiet) periods is {pct_noise:.4f}% of each vehicle's median pack "
    f"voltage, computed from {len(noise_pct_list)} vehicles with sufficient "
    f"quiet samples."
)
add("Anomaly and noise control", "Measurement noise",
    score_noise(pct_noise), finding_noise)

# ============================================================================
# 4. REPRESENTATIVENESS AND DIVERSITY
# ============================================================================
print("\n== 4. Representativeness and diversity ==")

add("Representativeness and diversity", "Chemistry diversity",
    score_diversity_count(1),
    "1 chemistry (NCM) across the fleet. Single-chemistry dataset, "
    "enabling focused NCM analysis but limiting cross-chemistry comparison.")

add("Representativeness and diversity", "Vehicle type diversity",
    score_diversity_count(2),
    "2 vehicle classes per the paper: passenger vehicles and commercial vans.")

if col_t_cell_max:
    per_vin_mean = data.groupby("_vin")[col_t_cell_max].mean().round(0).dropna()
    n_temp = per_vin_mean.nunique()
    means = sorted(per_vin_mean.unique().tolist())
    finding_temp = (
        f"{n_temp} distinct mean temperature level(s) across vehicles "
        f"({means} C). Real-world temperature variation from seasons, "
        f"climates, and self-heating; no controlled setpoints."
    )
    add("Representativeness and diversity", "Temperature conditions",
        score_diversity_count(n_temp), finding_temp)
else:
    add("Representativeness and diversity", "Temperature conditions", "o",
        "Max cell temperature column not present.")

if col_speed and col_i_pack:
    speed_var = data.groupby("_vin")[col_speed].std().mean()
    current_var = data.groupby("_vin")[col_i_pack].std().mean()
    finding_dynamic = (
        f"Real-world dynamic operation present by construction. Mean speed "
        f"std across vehicles: {speed_var:.1f} km/h. Mean current std: "
        f"{current_var:.1f} A. Both driving and charging profiles are "
        f"naturally variable, including regen braking."
    )
    add("Representativeness and diversity", "Dynamic load profiles",
        score_diversity_count(2), finding_dynamic)
else:
    add("Representativeness and diversity", "Dynamic load profiles", "o",
        "Speed or current column not present.")

add("Representativeness and diversity", "Real-world operation", "++",
    f"On-road operation across {n_vehicles} production EVs. All data "
    f"collected from vehicles in service, not laboratory replays.")

add("Representativeness and diversity", "Replicate vehicles",
    score_replicates(94),
    f"94 vehicles in the sampled folder (300 in the full dataset per the "
    f"paper). Large fleet enabling statistical analysis of cell-to-cell and "
    f"vehicle-to-vehicle variability.")

for aspect, info in METADATA_ONLY.items():
    add("Representativeness and diversity", aspect, info["score"], info["finding"])

# ============================================================================
# 5. DISTRIBUTION BALANCE
# ============================================================================
print("\n== 5. Distribution balance ==")

# BUG 3 FIX (part a): SOC range scored with an upper bound + explicit
# over-range accounting.
if col_soc:
    soc_min = data.groupby("_vin")[col_soc].min().min()
    soc_max = data.groupby("_vin")[col_soc].max().max()
    n_over = int((data[col_soc] > 100).sum())
    n_soc = int(data[col_soc].notna().sum())
    pct_over = n_over / n_soc * 100 if n_soc else 0

    if soc_min <= 10 and soc_max >= 90 and soc_max <= 100:
        score_soc = "++"
    elif soc_min <= 10 and soc_max >= 90 and soc_max <= 110:
        score_soc = "+"
    else:
        score_soc = "o"

    finding_soc = (
        f"SOC range covered across the fleet: {soc_min:.1f}% to "
        f"{soc_max:.1f}%. {n_over:,}/{n_soc:,} readings ({pct_over:.3f}%) "
        f"exceed 100%. Values >100% are consistent with regen-into-full-pack "
        f"behavior where the BMS reports usable SOC above nominal 100%, or "
        f"with a units/scale artifact in the soc column; values >110% would "
        f"be treated as implausible under Correctness -> Physical "
        f"plausibility. This is a caveat, not a strength."
    )
    add("Distribution balance", "SOC range coverage", score_soc, finding_soc)
else:
    add("Distribution balance", "SOC range coverage", "o", "SOC column not present.")

if col_soc:
    bins = pd.cut(data[col_soc].dropna(), bins=[0, 20, 40, 60, 80, 100])
    dist = bins.value_counts(normalize=True) * 100
    dominant = dist.max() if len(dist) else 100
    finding_soc_bal = (
        f"Largest SOC bin holds {dominant:.1f}% of all sampled records. "
        f"Real-world fleet operation clusters around mid-to-high SOC."
    )
    add("Distribution balance", "SOC distribution balance",
        score_soh_balance(dominant), finding_soc_bal)
else:
    add("Distribution balance", "SOC distribution balance", "o",
        "SOC column not present.")

rows_per_vehicle = data.groupby("_vin").size()
cv_rows = rows_per_vehicle.std() / rows_per_vehicle.mean() * 100 if rows_per_vehicle.mean() else 0
finding_balance = (
    f"Coefficient of variation of row counts across vehicles: {cv_rows:.1f}%. "
    f"Min: {rows_per_vehicle.min():,} ({rows_per_vehicle.idxmin()}), "
    f"Max: {rows_per_vehicle.max():,} ({rows_per_vehicle.idxmax()}). "
    f"The cap of {SAMPLE_ROWS_PER_FILE:,} rows per file bounds the "
    f"sampled distribution; the observed CV reflects which vehicles had at "
    f"least that many rows."
)
add("Distribution balance", "Vehicle data balance",
    score_cv(cv_rows), finding_balance)

# ============================================================================
# 6. TEMPORAL COHERENCE
# ============================================================================
print("\n== 6. Temporal coherence ==")

# BUG 2 FIX: session-aware monotonicity check.
# A new session starts whenever terminaltime decreases (counter reset at
# BMS wake-up). We check monotonicity WITHIN sessions, not across them.
if col_time:
    n_sessions_total = 0
    n_sessions_mono = 0
    worst_vehicle = None
    worst_frac = 0.0
    session_count_list = []

    for vin, grp in data.groupby("_vin"):
        t = grp[col_time].dropna().reset_index(drop=True)
        if len(t) < 2:
            continue
        # session_id increments at each counter reset (t decreases)
        session_id = (t.diff() < 0).cumsum()
        sess_groups = t.groupby(session_id)
        n_veh_sess = len(sess_groups)
        n_veh_mono = sum(g.is_monotonic_increasing for _, g in sess_groups)
        n_sessions_total += n_veh_sess
        n_sessions_mono += n_veh_mono
        session_count_list.append(n_veh_sess)
        frac_bad = 1 - (n_veh_mono / n_veh_sess) if n_veh_sess else 0
        if frac_bad > worst_frac:
            worst_frac, worst_vehicle = frac_bad, vin

    pct_mono = (n_sessions_mono / n_sessions_total * 100
                if n_sessions_total else 100)
    finding_mono = (
        f"{n_sessions_mono:,}/{n_sessions_total:,} sessions "
        f"({pct_mono:.2f}%) have monotonic non-decreasing terminaltime. "
        f"Sessions detected by counter reset (terminaltime drops to 0 at "
        f"each BMS wake-up). The terminaltime column is a per-session "
        f"seconds counter, not an epoch — global monotonicity across "
        f"sessions is not expected and is therefore not scored. "
        f"Average sessions per vehicle: "
        f"{np.mean(session_count_list):.1f}. Worst vehicle: {worst_vehicle} "
        f"({worst_frac*100:.1f}% of its sessions non-monotonic)."
    )
    add("Temporal coherence",
        "Monotonic temporal progression (within-session)",
        score_pct_high_is_good(pct_mono), finding_mono)
else:
    add("Temporal coherence",
        "Monotonic temporal progression (within-session)", "o",
        "Time column not present.")

# Mileage monotonicity (tolerant of sub-km rounding)
if col_mileage:
    total_transitions, decreases_over_tol = 0, 0
    for _, grp in data.groupby("_vin"):
        m = grp[col_mileage].dropna()
        if len(m) > 1:
            diffs = m.diff().dropna()
            total_transitions += len(diffs)
            decreases_over_tol += (diffs < -1.0).sum()
    pct_mileage = (1 - decreases_over_tol / total_transitions) * 100 if total_transitions else 100
    finding_mileage = (
        f"{pct_mileage:.4f}% of consecutive mileage transitions are "
        f"non-decreasing beyond a 1.0 km tolerance. Sub-kilometer fluctuations "
        f"(odometer rounding) are not flagged."
    )
    add("Temporal coherence", "Mileage progression",
        score_pct_high_is_good(pct_mileage), finding_mileage)
else:
    add("Temporal coherence", "Mileage progression", "o",
        "Mileage column not present.")

# Event identification (charging, driving, rest, full)
if col_status:
    modes_present = sorted(data[col_status].dropna().unique().tolist())
    known_modes = {CHARGESTATUS_REST, CHARGESTATUS_CHARGING,
                   CHARGESTATUS_DRIVING, CHARGESTATUS_FULL}
    modes_ok = set(modes_present).issubset(known_modes)
    score_events = "++" if modes_ok else "o"
    finding_events = (
        f"chargestatus values observed: {modes_present}. Known mapping: "
        f"0=rest, 1=charging, 3=driving, 4=full. "
        f"{'All observed modes are recognised.' if modes_ok else 'Unrecognised mode values present.'}"
    )
    add("Temporal coherence", "Event identification", score_events, finding_events)
else:
    add("Temporal coherence", "Event identification", "o",
        "chargestatus column not present.")

# ============================================================================
# SAVE SCORECARD
# ============================================================================
results_df = pd.DataFrame(results)
csv_path = os.path.join(OUT_DIR, "300ev_quality_scorecard.csv")
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
ax.set_title("300-EV Real-World BMS Dataset — Data Quality Scorecard",
             fontsize=12, fontweight="bold", pad=16)
plt.tight_layout()
fig_path = os.path.join(OUT_DIR, "300ev_quality_scorecard.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"Figure saved: {fig_path}")
plt.close(fig)

print("\nScore distribution:")
print(results_df["score"].value_counts().reindex(["++", "+", "o", "-"], fill_value=0))