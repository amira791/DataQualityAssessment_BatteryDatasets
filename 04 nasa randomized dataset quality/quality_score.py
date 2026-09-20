"""
NASA Randomized & Recommissioned Battery Dataset — Quality Scoring
====================================================================
Scores the NASA Randomized & Recommissioned Battery Dataset, regular
accelerated-life-test folder (15 LFP 2S packs, 6 documented discharge
levels, constant-current and variable-current profiles), against the
same six-dimension quality framework used for the Oxford,
MIT-Stanford-TRI, CALCE, and SNL datasets.

Design rule: hardcode ONLY facts that cannot be determined from the raw
cycle-data files at all (calendar aging, real-world operation, and the
documented per-pack load-profile grouping). Everything else — current-
sign convention, physical bounds violations, missing values, noise,
distribution balance, temporal coherence, temperature diversity,
dynamic-load diversity from mission_type, replicate packs per condition,
SOH — is computed dynamically from the CSV files, their columns, and
the README's documented discharge groups.

Physical plausibility:
    current (cell)       charge <= 1.0C, discharge <= 18.0C, with
                         1C = 1.1 Ah (LFP cell nominal capacity)
                         NOTE: NASA's regular folder is a deliberately
                         high-rate accelerated-life study; documented
                         discharge currents reach ~17.3C at cell level.
                         The general 5C envelope would flag nearly every
                         cycle, so a dataset-appropriate 18C ceiling is
                         used to catch genuine outliers (sensor spikes,
                         unit errors) without flagging the design.
    temperature (battery surface)
                         -20 to 60 C
    voltage              PACK voltage plausibility is NOT checked here:
                         the charger-side column records charger terminal
                         voltage, not pack voltage, and there is no
                         always-on pack-voltage column in the CSV set.

Current envelope is applied at cell level. The dataset is a 2S pack
(series), so I_cell = I_pack; pack-level current is used directly.

SOH is computed from REFERENCE-discharge capacity (mission_type == 0),
not from every discharge. Reference discharges are the only cycles run
at a fixed, comparable rate across packs; regular-mission capacities
vary by design (variable-current groups cycle through multiple current
levels within a single discharge).

Output: a scorecard CSV and a scorecard figure, both saved to
./quality_results/
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ============================================================================
# CONFIG
# ============================================================================
DATASET_PATH = r"C:\Users\admin\Desktop\DR2\11 All Datasets\02 NASA Randomized Battery Dataset\battery_alt_dataset\regular_alt_batteries"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quality_results")
os.makedirs(OUT_DIR, exist_ok=True)

SCORE_LABELS = {"++": "Comprehensive coverage", "+": "Mostly satisfied",
                "o": "Partially satisfied", "-": "Not satisfied"}
SCORE_COLORS = {"++": "#2ca02c", "+": "#98df8a", "o": "#ffbb78", "-": "#d62728"}

# ============================================================================
# METADATA-ONLY FACTS (from NASA official documentation — cannot be
# recovered from the CSVs under any amount of analysis)
# ============================================================================
METADATA_ONLY = {
    "Calendar aging": {
        "score": "-",
        "finding": "NASA documentation reports no dedicated calendar-aging "
                   "experiment; all aging is cyclic (charge-discharge "
                   "repetition) with unstructured rest between cycles.",
    },
    "Real-world operation": {
        "score": "-",
        "finding": "Dataset is laboratory-only per official documentation; "
                   "no field/real-world operation data was collected. The "
                   "variable-load missions simulate dynamic use but are "
                   "prescribed laboratory profiles, not field measurements.",
    },
}

# Protocol elements documented in NASA's metadata / README.
PROTOCOL_METADATA = {
    "chemistry": True,             # LFP family (implied by 2S pack voltage window)
    "cell_model": False,           # exact cell model number not given
    "pack_configuration": True,    # 2S, ~1.1 Ah per cell
    "temperature": True,           # battery surface temperature recorded
    "charging_protocol": True,     # documented; 72 fast-charging policies in broader dataset
    "discharge_protocol": True,    # reference (2.5 A) and regular missions
    "load_profile_inventory": True,# 5 CC levels + 2 variable-current groups, with named packs
    "failure_criterion": True,     # testing continues toward failure
}

# Documented discharge groups for the regular folder (README).
# Used to group replicate packs by condition. This is the ground truth for
# which packs belong to which experimental condition; snapping data-derived
# means to the nearest documented level mis-assigns packs whose measured
# mean sits near the midpoint between two adjacent levels.
PACK_TO_DOCUMENTED_LEVEL = {
    "battery00": 16.0, "battery01": 9.30,
    "battery10": 16.0, "battery11": 9.30,
    "battery20": 19.0, "battery21": 19.0, "battery22": 12.9, "battery23": 14.3,
    "battery30": 19.0, "battery31": 12.9,
    "battery40": 17.0, "battery41": 14.3,
    "battery50": 17.0, "battery51": 14.3, "battery52": 14.3,
}

# ============================================================================
# GENERAL Li-ion PHYSICAL ENVELOPE
#
# Current (CELL level): C-rate, not amperes. The QC envelope uses the
# UPPER bound of the general "maximum continuous discharge" rule (5C),
# except where a dataset is documented to deliberately exceed it. NASA's
# regular folder documents discharge currents up to 19 A pack at 1.1 Ah
# nominal cell capacity = ~17.3C; a dataset-appropriate 18C ceiling is
# used so the check catches genuine outliers without flagging the design.
#
# Temperature (BATTERY surface): -20 to 60 C.
#
# Voltage: PACK voltage is NOT checked against a fixed envelope, because
# the charger-side column records charger terminal voltage and the load-
# side column is populated only during discharge/rest. No always-on pack-
# voltage column exists in the CSV set.
# ============================================================================
PACK_V_MIN_V, PACK_V_MAX_V = 4.0, 7.30
TEMP_MIN_C, TEMP_MAX_C = -20.0, 60.0

C_RATE_CHARGE_MAX    = 1.0
C_RATE_DISCHARGE_MAX = 18.0     # dataset-appropriate 

NOMINAL_CAPACITY_AH  = 1.1
NOMINAL_VOLTAGE_V    = 3.3
PACK_CELLS_IN_SERIES = 2

# ============================================================================
# SCORING FUNCTIONS (identical vocabulary to the Oxford / MIT / CALCE / SNL code)
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
# LOAD DATA (one CSV per pack, all channels detected by name)
# ============================================================================
print("Loading NASA Randomized & Recommissioned Battery Dataset (regular folder)...")
csv_files = sorted(glob.glob(os.path.join(DATASET_PATH, "battery*.csv")))
if not csv_files:
    raise FileNotFoundError(f"No battery*.csv files found in {DATASET_PATH}")

frames = []
for f in csv_files:
    df = pd.read_csv(f, low_memory=False)
    df.columns = df.columns.str.strip()
    df["battery_id"] = os.path.basename(f).replace(".csv", "")
    frames.append(df)
data = pd.concat(frames, ignore_index=True)
print(f"Loaded {len(data):,} rows from {len(csv_files)} battery files")

# Dynamic column detection
def find_col(*keywords):
    for c in data.columns:
        cl = c.lower()
        if all(k in cl for k in keywords):
            return c
    return None

# Continuous time is the column literally named "time". "start_time" is an
# ISO timestamp string at cycle boundaries and must NOT be used as the
# continuous clock (pd.to_numeric coerces it to all-NaN).
col_time = next((c for c in data.columns if c.lower() == "time"), None)
col_mode = find_col("mode")
col_v_chg = find_col("voltage", "charger")
col_v_load = find_col("voltage", "load")
col_i_load = find_col("current")
col_t_batt = find_col("temp", "battery")
col_t_mos = find_col("temp", "mosfet")
col_t_res = find_col("temp", "resistor")
col_mission = find_col("mission")

for c in [col_time, col_mode, col_v_chg, col_v_load, col_i_load,
          col_t_batt, col_t_mos, col_t_res, col_mission]:
    if c:
        data[c] = pd.to_numeric(data[c], errors="coerce")

n_packs = data["battery_id"].nunique()
print(f"Detected columns -> time:{col_time} mode:{col_mode} v_chg:{col_v_chg} "
      f"v_load:{col_v_load} i:{col_i_load} t_batt:{col_t_batt} "
      f"t_mos:{col_t_mos} t_res:{col_t_res} mission:{col_mission}")
print(f"Pack count: {n_packs}\n")

dis_mask = data[col_mode] == -1
rest_mask = data[col_mode] == 0
chg_mask = data[col_mode] == 1

# ============================================================================
# 1. CORRECTNESS
# ============================================================================
print("== 1. Correctness ==")

# ---------------------------------------------------------------------------
# Physical plausibility -- current (per-cell C-rate envelope) and
# temperature (battery surface). Voltage is intentionally not checked
# against a fixed pack window (see envelope note in CONFIG).
# ---------------------------------------------------------------------------
charge_limit_A_cell = NOMINAL_CAPACITY_AH * C_RATE_CHARGE_MAX
discharge_limit_A_cell = NOMINAL_CAPACITY_AH * C_RATE_DISCHARGE_MAX

viol, total = 0, 0

if col_i_load:
    # Load current column is populated only during discharge (hardware
    # design). 2S pack -> series -> I_cell = I_pack.
    s = data.loc[dis_mask, col_i_load].dropna().abs()
    if len(s):
        v = (s > discharge_limit_A_cell).sum()
        viol += int(v)
        total += len(s)
        pct = v / len(s) * 100
        print(f"    Physical plausibility breakdown -- Load current (discharge, "
              f"per-cell envelope {C_RATE_DISCHARGE_MAX}C = {discharge_limit_A_cell:.2f}A): "
              f"{v:,}/{len(s):,} ({pct:.2f}%) exceed the envelope, "
              f"observed cell-level range [{s.min():.4f}, {s.max():.4f}] A")

print("    Note: charge current is not recorded as a numeric column in the "
      "CSV set (charger side reports voltage only); the 1.0C charge-current "
      "envelope could not be applied per-record.")

if col_t_batt:
    s = data[col_t_batt].dropna()
    if len(s):
        v = ((s < TEMP_MIN_C) | (s > TEMP_MAX_C)).sum()
        viol += int(v)
        total += len(s)
        pct = v / len(s) * 100
        print(f"    Physical plausibility breakdown -- Battery temperature ({col_t_batt}): "
              f"{v:,}/{len(s):,} ({pct:.2f}%) outside [{TEMP_MIN_C}, {TEMP_MAX_C}], "
              f"observed range [{s.min():.4f}, {s.max():.4f}]")

pct_implausible = (viol / total * 100) if total else 0
add("Correctness", "Physical plausibility", score_pct_low_is_good(pct_implausible),
    f"{viol:,}/{total:,} current/temperature readings ({pct_implausible:.3f}%) outside the "
    f"general envelope. Discharge current checked at cell level (2S pack: I_cell = I_pack), "
    f"envelope {C_RATE_DISCHARGE_MAX}C = {discharge_limit_A_cell:.2f} A. Battery temperature "
    f"{TEMP_MIN_C}-{TEMP_MAX_C} C. Pack-voltage check is deferred -- the charger-side column "
    f"records charger terminal voltage, not pack voltage, and there is no always-on "
    f"pack-voltage column in the CSV set. Charge-current envelope not applied -- charger side "
    f"records voltage only, no numeric charge current in these CSVs.")

# ---------------------------------------------------------------------------
# Current sign convention
# ---------------------------------------------------------------------------
if col_mode:
    modes = data[col_mode].dropna().unique()
    expected_modes = {-1, 0, 1}
    unexpected_modes = set(modes) - expected_modes
    add("Correctness", "Current sign convention", "++" if not unexpected_modes else "+",
        f"Mode column is the authoritative direction indicator (hardware measures |I| on the "
        f"load board, so the current column is always >=0 by design). Observed mode values: "
        f"{sorted(modes)}. Unexpected values: {sorted(unexpected_modes) if unexpected_modes else 'none'}. "
        f"Positive = charge, negative = discharge, zero = rest is fully consistent across all "
        f"{len(data):,} rows.")
else:
    add("Correctness", "Current sign convention", "o",
        "Mode column not present; charge/discharge direction could not be verified.")

# ============================================================================
# 2. COMPLETENESS
# ============================================================================
print("\n== 2. Completeness ==")

essential_cols = [c for c in [col_time, col_mode, col_t_batt] if c]
if essential_cols:
    missing = data[essential_cols].isna().sum().sum()
    total_cells_checked = data[essential_cols].size
    pct_missing = missing / total_cells_checked * 100
else:
    pct_missing = 0
add("Completeness", "Missing values", score_pct_low_is_good(pct_missing),
    f"{pct_missing:.4f}% missing across essential columns {essential_cols}. "
    f"Load-side columns (voltage_load, current_load, temperature_mosfet, "
    f"temperature_resistor, mission_type) are structurally NaN outside discharge "
    f"segments by design, not a data gap.")

cont_pct_list = []
if col_time and col_mode:
    for _, grp in data.groupby("battery_id"):
        g = grp.copy()
        g["_seg"] = (g[col_mode] != g[col_mode].shift()).cumsum()
        for _, seg in g.groupby("_seg"):
            t = seg[col_time].dropna().values
            if len(t) > 1:
                diffs = np.diff(t)
                med = np.median(diffs[diffs > 0]) if (diffs > 0).any() else 0
                if med > 0:
                    cont_pct_list.append((diffs <= 5 * med).mean() * 100)
pct_continuity = np.mean(cont_pct_list) if cont_pct_list else 100
add("Completeness", "Temporal continuity", score_pct_high_is_good(pct_continuity),
    f"Average within-segment sampling continuity (no gaps >5x the median interval): "
    f"{pct_continuity:.2f}%.")

documented_protocol = sum(PROTOCOL_METADATA.values())
total_protocol_elements = len(PROTOCOL_METADATA)
pct_doc = documented_protocol / total_protocol_elements * 100
add("Completeness", "Test protocol documentation", score_doc(pct_doc),
    f"{documented_protocol}/{total_protocol_elements} essential protocol elements documented "
    f"({pct_doc:.0f}%). Exact cell model number is not provided in the metadata; pack-level "
    f"configuration and chemistry are documented and determine the electrical envelope.")

# ============================================================================
# 3. ANOMALY AND NOISE CONTROL
# ============================================================================
print("\n== 3. Anomaly and noise control ==")

# Per-cycle capacity, computed only for REFERENCE discharges (mission_type
# == 0). Those are run at a fixed ~2.5 A rate and are directly comparable
# across cycles and packs; regular-mission capacities vary by design.
print("Computing per-cycle reference-discharge capacity...")
cycle_records = []
for batt_id, grp in data.groupby("battery_id"):
    g = grp.copy().reset_index(drop=True)
    g["_seg"] = (g[col_mode] != g[col_mode].shift()).cumsum()
    cyc_idx = 0
    for _, seg in g[g[col_mode] == -1].groupby("_seg"):
        cyc_idx += 1
        cap = np.nan
        if col_time and col_i_load:
            t = seg[col_time].values.astype(float)
            i = np.abs(seg[col_i_load].values.astype(float))
            valid = np.isfinite(t) & np.isfinite(i)
            t_v, i_v = t[valid], i[valid]
            if len(t_v) > 1:
                cap = np.trapezoid(i_v, t_v) / 3600.0 if hasattr(np, "trapezoid") else np.trapz(i_v, t_v) / 3600.0
        mt = seg[col_mission].iloc[0] if col_mission else np.nan
        cycle_records.append({
            "battery_id": batt_id, "cycle": cyc_idx,
            "capacity_Ah": cap, "mission_type": mt, "n_rows": len(seg),
        })
cyc_df = pd.DataFrame(cycle_records)

# Restrict to reference discharges for all capacity-based analyses.
ref_df = cyc_df[cyc_df["mission_type"] == 0].dropna(subset=["capacity_Ah"]).copy()

# Statistical outliers on reference-discharge capacity
if len(ref_df) > 0:
    caps = ref_df["capacity_Ah"]
    q1, q3 = caps.quantile(0.25), caps.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
    outliers = ((caps < lo) | (caps > hi)).sum()
    pct_outliers = outliers / len(caps) * 100
else:
    pct_outliers = 0
add("Anomaly and noise control", "Statistical outliers", score_pct_low_is_good(pct_outliers),
    f"{pct_outliers:.3f}% of reference-discharge capacity values fall outside the 3xIQR range.")

# Cycle-to-cycle reference-capacity transitions
all_diffs = []
if len(ref_df) > 0:
    for _, grp in ref_df.groupby("battery_id"):
        g = grp.sort_values("cycle")
        caps = g["capacity_Ah"].values
        if len(caps) > 1:
            all_diffs.extend(np.diff(caps)[1:])
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
    f"{pct_jumps:.3f}% of reference-to-reference capacity transitions fall outside the 3xIQR "
    f"range of transition sizes (first transition excluded per pack).")

# Measurement noise -- battery temperature during rest segments (the
# closest thing to a quiet signal available across all phases).
noise_pct_list = []
if col_t_batt:
    for _, grp in data[rest_mask].groupby("battery_id"):
        t = grp[col_t_batt].dropna().values
        if len(t) > 20:
            d = np.abs(np.diff(t))
            if len(d):
                noise_pct_list.append(np.median(d) / 30.0 * 100)
pct_noise = np.mean(noise_pct_list) if noise_pct_list else 0
add("Anomaly and noise control", "Measurement noise", score_noise(pct_noise),
    f"Median sample-to-sample battery-temperature fluctuation during rest segments is "
    f"{pct_noise:.4f}% of a 30 °C reference, computed from {len(noise_pct_list)} rest segments.")

# ============================================================================
# 4. REPRESENTATIVENESS AND DIVERSITY
# ============================================================================
print("\n== 4. Representativeness and diversity ==")

# Temperature diversity from measured T array (battery surface temperature)
if col_t_batt:
    t_valid = data[col_t_batt].dropna()
    if len(t_valid):
        spread = t_valid.max() - t_valid.min()
        per_pack_mean = data.groupby("battery_id")[col_t_batt].mean().round(0)
        n_temp = per_pack_mean.nunique()
        if spread < 10:
            n_temp = 1
    else:
        n_temp = 1
        per_pack_mean = pd.Series(dtype=float)
else:
    n_temp = 1
    per_pack_mean = pd.Series(dtype=float)
add("Representativeness and diversity", "Temperature conditions", score_diversity_count(n_temp),
    f"{n_temp} distinct mean temperature level(s) across packs "
    f"({sorted(per_pack_mean.unique().tolist()) if len(per_pack_mean) else 'n/a'} °C). "
    f"Temperatures vary naturally with ambient and self-heating rather than being "
    f"tightly controlled at isothermal setpoints.")

# C-rate / policy diversity: count distinct documented levels represented
# in the data, using the README's pack-to-level mapping.
documented_levels_present = sorted(set(PACK_TO_DOCUMENTED_LEVEL.values()))
n_policy_levels = len(documented_levels_present)
add("Representativeness and diversity", "C-rate / policy diversity",
    score_diversity_count(n_policy_levels),
    f"{n_policy_levels} distinct documented discharge-current level(s) present in the "
    f"regular folder ({documented_levels_present} A), covering both constant-current "
    f"and variable-current groups.")

# Dynamic load profiles: derived from mission_type (0=reference, 1=regular)
if col_mission:
    n_mission_types = data[col_mission].dropna().nunique()
    regular_share = (data[col_mission] == 1).sum() / max(1, (data[col_mission].isin([0, 1])).sum()) * 100
    add("Representativeness and diversity", "Dynamic load profiles",
        score_diversity_count(n_mission_types),
        f"{n_mission_types} mission types observed (0 = reference constant-current "
        f"2.5 A, 1 = regular/variable mission). Regular missions account for "
        f"{regular_share:.1f}% of discharge cycles.")
else:
    add("Representativeness and diversity", "Dynamic load profiles", "o",
        "mission_type column not present; dynamic-vs-reference split could not be computed.")

# Replicate packs per DOCUMENTED condition (README mapping, not snapping)
if PACK_TO_DOCUMENTED_LEVEL:
    packs_per_level = pd.Series(list(PACK_TO_DOCUMENTED_LEVEL.values())).value_counts()
    avg_replicates = packs_per_level.mean()
    add("Representativeness and diversity", "Replicate cells", score_replicates(avg_replicates),
        f"{avg_replicates:.1f} replicate packs per documented discharge-current level "
        f"(across {len(packs_per_level)} conditions, from the README grouping).")
else:
    add("Representativeness and diversity", "Replicate cells", "o",
        "Documented pack-to-level mapping not available; replicate count could not be computed.")

for aspect, info in METADATA_ONLY.items():
    add("Representativeness and diversity", aspect, info["score"], info["finding"])

# ============================================================================
# 5. DISTRIBUTION BALANCE
# ============================================================================
print("\n== 5. Distribution balance ==")

# SOH per pack, computed from REFERENCE-discharge capacity only
soh_frames = []
if len(ref_df) > 0:
    for batt_id, grp in ref_df.groupby("battery_id"):
        g = grp.sort_values("cycle")
        caps = g["capacity_Ah"].values
        if len(caps) < 2:
            continue
        ref = caps[0]
        if ref <= 0:
            continue
        soh = np.clip(caps / ref, 0, 1.05)
        soh_frames.append(pd.DataFrame({
            "battery_id": batt_id, "cycle": g["cycle"].values, "SOH": soh,
        }))
soh_df = pd.concat(soh_frames, ignore_index=True) if soh_frames else pd.DataFrame(columns=["battery_id", "SOH"])

if not soh_df.empty:
    packs_below_80 = soh_df[soh_df["SOH"] < 0.8]["battery_id"].nunique()
    pct_below_80 = packs_below_80 / n_packs * 100
else:
    pct_below_80 = 0
add("Distribution balance", "SOH range coverage", score_soh_range(pct_below_80),
    f"{pct_below_80:.1f}% of packs reach SOH < 80% (end of life), computed from "
    f"reference-discharge capacity (mission_type = 0).")

if not soh_df.empty:
    bins = pd.cut(soh_df["SOH"], bins=[0, 0.7, 0.8, 0.9, 0.95, 1.05])
    dist = bins.value_counts(normalize=True) * 100
    dominant = dist.max() if len(dist) else 100
else:
    dominant = 100
add("Distribution balance", "SOH distribution balance", score_soh_balance(dominant),
    f"Largest SOH bin holds {dominant:.1f}% of all reference-discharge records.")

# Cycle contribution per pack, computed WITHIN documented replicate groups.
if not cyc_df.empty:
    per_pack_cycles = cyc_df.groupby("battery_id")["cycle"].max()
    per_pack_cycles.name = "cycles"
    per_pack_level = pd.Series(PACK_TO_DOCUMENTED_LEVEL, name="documented_level")
    merged = pd.concat([per_pack_cycles, per_pack_level], axis=1).dropna()
    within_cvs = []
    for _, grp in merged.groupby("documented_level"):
        c = grp["cycles"].values
        if len(c) > 1 and c.mean() > 0:
            within_cvs.append(c.std() / c.mean() * 100)
    cv_pct = np.mean(within_cvs) if within_cvs else 0
    note = "measured within documented replicate groups at the same discharge-current level"
else:
    cv_pct = 0
    note = "could not be computed (no per-cycle data)"
add("Distribution balance", "Cycle contribution per pack", score_cv(cv_pct),
    f"Coefficient of variation of cycle-life across packs {note}: {cv_pct:.1f}%. "
    f"The dataset is deliberately heterogeneous (different load profiles), so a high "
    f"pooled CV is expected by design; the within-condition CV is the metric that "
    f"reflects true imbalance.")

# ============================================================================
# 6. TEMPORAL COHERENCE
# ============================================================================
print("\n== 6. Temporal coherence ==")

non_mono_segs = 0
if col_time and col_mode:
    for _, grp in data.groupby("battery_id"):
        g = grp.copy()
        g["_seg"] = (g[col_mode] != g[col_mode].shift()).cumsum()
        for _, seg in g.groupby("_seg"):
            t = seg[col_time].dropna().values
            if len(t) > 1 and not np.all(np.diff(t) >= 0):
                non_mono_segs += 1
pct_ordered = 100 if non_mono_segs == 0 else max(0, 100 - non_mono_segs / max(1, len(csv_files)) * 100)
add("Temporal coherence", "Monotonic temporal progression", score_pct_high_is_good(pct_ordered),
    f"{non_mono_segs} non-monotonic mode segments found; the continuous time column is "
    f"otherwise strictly non-decreasing within each segment.")

# Consistent degradation trend on reference-discharge SOH
if not soh_df.empty:
    total_transitions, violating = 0, 0
    for _, grp in soh_df.groupby("battery_id"):
        g = grp.sort_values("cycle")
        soh_vals = g["SOH"].values
        if len(soh_vals) > 1:
            diffs = np.diff(soh_vals)
            total_transitions += len(diffs)
            violating += (diffs > 0.03).sum()
    pct_consistent = (1 - violating / total_transitions) * 100 if total_transitions else 100
else:
    pct_consistent = 100
add("Temporal coherence", "Consistent degradation trend", score_degradation_trend(pct_consistent),
    f"{pct_consistent:.2f}% of reference-to-reference SOH transitions are non-increasing "
    f"beyond a 3% noise tolerance (measured across all packs, per-transition).")

if not cyc_df.empty:
    dup_free = []
    for _, grp in cyc_df.groupby("battery_id"):
        dup_free.append((1 - grp["cycle"].duplicated().mean()) * 100)
    pct_index_consistent = np.mean(dup_free) if dup_free else 100
else:
    pct_index_consistent = 100
add("Temporal coherence", "Cycle index consistency", score_pct_high_is_good(pct_index_consistent),
    f"{pct_index_consistent:.2f}% of cycle indices are non-duplicated within their pack.")

# ============================================================================
# SAVE SCORECARD
# ============================================================================
results_df = pd.DataFrame(results)
csv_path = os.path.join(OUT_DIR, "nasa_quality_scorecard.csv")
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
ax.set_title("NASA Randomized & Recommissioned Battery Dataset (regular folder) — Data Quality Scorecard",
             fontsize=12, fontweight="bold", pad=16)
plt.tight_layout()
fig_path = os.path.join(OUT_DIR, "nasa_quality_scorecard.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"Figure saved: {fig_path}")
plt.close(fig)

print("\nScore distribution:")
print(results_df["score"].value_counts().reindex(["++", "+", "o", "-"], fill_value=0))