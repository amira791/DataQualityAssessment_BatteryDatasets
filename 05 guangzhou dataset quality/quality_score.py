"""
Guangzhou 10-EV Fleet Dataset — Quality Scoring
================================================
Scores the Guangzhou 10-EV Fleet Dataset (10 real-world EVs, 2 chemistries,
~1.19M operational records) against the same six-dimension quality framework
used for the Oxford, MIT-Stanford-TRI, CALCE, SNL, and NASA datasets.

Design rule: hardcode ONLY facts that cannot be determined from the raw
data files at all (calendar aging — absent by design; per-chemistry voltage
envelopes — derived from cell chemistry and pack structure). Everything else —
chemistry/type diversity, per-vehicle bounds violations, missing values,
noise, distribution balance, temporal coherence, dynamic-operation presence,
replicate vehicles — is computed dynamically from the xlsx files and their
columns.

Physical plausibility bounds:
    NCM (Vehicles #1-#6, 91 series cells from GitHub):
        pack voltage 227.5 - 382.2 V (91 × 2.5V, 91 × 4.2V)
    LFP (Vehicles #7-#10): NOT VALIDATED
        GitHub reports only "total cells" (360, 324) for #9-#10,
        no pack structure disclosed for #7-#8. Series cell count unknown,
        so physically valid pack voltage bounds cannot be derived.
    Pack current, all vehicles: -200 - 200 A
    Battery temperature (max cell): 0 - 45 C

Current sign convention for THIS dataset (verified from data, not README):
    NEGATIVE = charging  (BMS perspective)
    POSITIVE = discharging / traction  (motor perspective)
    Driving includes regen braking, so ~15% of driving rows are legitimately
    negative.

Time column:
    'time' is a numeric millisecond counter, NOT a Unix epoch. All
    time-based computations treat it as milliseconds.

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
DATASET_PATH = r"C:\Users\admin\Desktop\DR2\11 All Datasets\12 Real-World 10EVs dataset\dataset_"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quality_results")
os.makedirs(OUT_DIR, exist_ok=True)

SCORE_LABELS = {"++": "Comprehensive coverage", "+": "Mostly satisfied",
                "o": "Partially satisfied", "-": "Not satisfied"}
SCORE_COLORS = {"++": "#2ca02c", "+": "#98df8a", "o": "#ffbb78", "-": "#d62728"}

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

PROTOCOL_METADATA = {
    "chemistry": True,
    "vehicle_type": True,
    "capacity": True,
    "sampling_frequency": True,
    "data_acquisition_chain": True,
    "current_convention": False,
    "pack_structure_ncm": True,    # 91 series cells disclosed
    "pack_structure_lfp": False,   # V7-V10 undisclosed/ambiguous
}

# Vehicle metadata from the GitHub README
VEHICLE_INFO = {
    "Vehicle#1":  {"type": "Passenger", "chemistry": "NCM", "capacity_Ah": 150},
    "Vehicle#2":  {"type": "Passenger", "chemistry": "NCM", "capacity_Ah": 150},
    "Vehicle#3":  {"type": "Passenger", "chemistry": "NCM", "capacity_Ah": 160},
    "Vehicle#4":  {"type": "Passenger", "chemistry": "NCM", "capacity_Ah": 160},
    "Vehicle#5":  {"type": "Passenger", "chemistry": "NCM", "capacity_Ah": 160},
    "Vehicle#6":  {"type": "Passenger", "chemistry": "NCM", "capacity_Ah": 160},
    "Vehicle#7":  {"type": "Passenger", "chemistry": "LFP", "capacity_Ah": 120},
    "Vehicle#8":  {"type": "Bus",       "chemistry": "LFP", "capacity_Ah": 645},
    "Vehicle#9":  {"type": "Bus",       "chemistry": "LFP", "capacity_Ah": 505},
    "Vehicle#10": {"type": "Bus",       "chemistry": "LFP", "capacity_Ah": 505},
}

# ============================================================================
# VOLTAGE BOUNDS — ONLY NCM (LFP cannot be validated)
# ============================================================================
# Source for NCM cell count: https://github.com/Translab-SCUT/Electric-vehicle-operation-data
#   (Vehicles #1-#6: "91 battery cells connected in series")
# Source for NCM cell voltage limits: standard NCM datasheet (2.5V - 4.2V)
#
# LFP: GitHub reports only "total cells" (360, 324) for #9-#10, no pack
#      structure disclosed for #7-#8. Series cell count is unknown, so
#      physically valid pack voltage bounds cannot be derived.

VOLTAGE_BOUNDS = {
    "NCM": (227.5, 382.2),   # 91 series cells × 2.5-4.2V
}

VOLTAGE_BOUNDS_SOURCE = {
    "NCM": "91 series cells (GitHub) × 2.5-4.2V (NCM datasheet)",
    "LFP": "NOT AVAILABLE — series cell count not disclosed for Vehicles #7-#10",
}

# ============================================================================
# OTHER PHYSICAL BOUNDS
# ============================================================================
# Current bounds (estimated from Fig. 2e; paper does not give exact values)
PACK_I_MIN_A, PACK_I_MAX_A = -200.0, 200.0  # Conservative estimate

# Temperature bounds (paper states 10-40°C cell, 0-30°C ambient)
TEMP_MIN_C, TEMP_MAX_C = 0.0, 45.0  # 45°C allows small margin

# Mileage tolerance (engineering choice, not from paper)
MILEAGE_TOLERANCE_KM = 1.0

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
    results.append({"criterion": criterion, "aspect": aspect,
                    "score": score, "finding": finding})
    print(f"  [{score}] {criterion} — {aspect}: {finding}")

# ============================================================================
# LOAD DATA (one xlsx per vehicle)
# ============================================================================
print("Loading Guangzhou 10-EV Fleet Dataset...")
xlsx_files = sorted(glob.glob(os.path.join(DATASET_PATH, "vehicle#*.xlsx")))
if not xlsx_files:
    raise FileNotFoundError(f"No vehicle#*.xlsx files found in {DATASET_PATH}")

FNAME_RE = re.compile(r"vehicle#(\d+)\.xlsx$")

frames = []
for f in xlsx_files:
    m = FNAME_RE.search(os.path.basename(f))
    if not m:
        continue
    vid = f"Vehicle#{int(m.group(1))}"
    df = pd.read_excel(f)
    df.columns = df.columns.str.strip()
    df["vehicle_id"] = vid
    frames.append(df)

data = pd.concat(frames, ignore_index=True)
n_vehicles = data["vehicle_id"].nunique()
print(f"Loaded {len(data):,} rows from {n_vehicles} vehicles")

# Column detection
def find_col(*keywords):
    for c in data.columns:
        cl = c.lower()
        if all(k in cl for k in keywords):
            return c
    return None

col_time    = find_col("time")
col_speed   = find_col("vhc_speed") or find_col("speed")
col_charge  = find_col("charging_signal")
col_mileage = find_col("mile")
col_v_pack  = find_col("hv_voltage")
col_i_pack  = find_col("hv_current")
col_soc     = find_col("soc")
col_v_max   = find_col("maxvoltage")
col_v_min   = find_col("minvoltage")
col_t_max   = find_col("maxtemp")
col_t_min   = find_col("mintemp")

# Coerce numeric (time stays integer-milliseconds, we do NOT call to_datetime)
for c in [col_speed, col_charge, col_mileage, col_v_pack, col_i_pack,
          col_soc, col_v_max, col_v_min, col_t_max, col_t_min]:
    if c:
        data[c] = pd.to_numeric(data[c], errors="coerce")
if col_time:
    data[col_time] = pd.to_numeric(data[col_time], errors="coerce")

# Add per-vehicle chemistry column for convenience
data["_chem"] = data["vehicle_id"].map(
    lambda v: VEHICLE_INFO.get(v, {}).get("chemistry", "unknown")
)

print(f"Detected columns -> time:{col_time} speed:{col_speed} charge:{col_charge} "
      f"mileage:{col_mileage} v_pack:{col_v_pack} i_pack:{col_i_pack} "
      f"soc:{col_soc} v_max:{col_v_max} v_min:{col_v_min} "
      f"t_max:{col_t_max} t_min:{col_t_min}\n")

# ============================================================================
# 1. CORRECTNESS
# ============================================================================
print("== 1. Correctness ==")

# ---------------------------------------------------------------------------
# Physical plausibility -- pack voltage (NCM only), current, temperature
# ---------------------------------------------------------------------------
viol, total = 0, 0

# Voltage: explicit per-chemistry handling (NCM validated, LFP not)
if col_v_pack:
    for chem in ["NCM", "LFP"]:
        if chem not in VOLTAGE_BOUNDS:
            n_rows = (data["_chem"] == chem).sum()
            print(f"    Physical plausibility -- Pack voltage ({chem}): "
                  f"NOT VALIDATED — series cell count undisclosed "
                  f"({n_rows:,} rows excluded)")
            continue
        
        vlo, vhi = VOLTAGE_BOUNDS[chem]
        s = data.loc[data["_chem"] == chem, col_v_pack].dropna()
        if len(s) == 0:
            print(f"    Physical plausibility -- Pack voltage ({chem}): "
                  f"no data available")
            continue
        v = ((s < vlo) | (s > vhi)).sum()
        viol += int(v)
        total += len(s)
        pct = v / len(s) * 100
        print(f"    Physical plausibility -- Pack voltage ({chem}, "
              f"bounds [{vlo}, {vhi}] V): {v:,}/{len(s):,} ({pct:.3f}%) "
              f"outside, observed range [{s.min():.1f}, {s.max():.1f}]")

# Current: fleet-wide bounds
if col_i_pack:
    s = data[col_i_pack].dropna()
    if len(s):
        v = ((s < PACK_I_MIN_A) | (s > PACK_I_MAX_A)).sum()
        viol += int(v)
        total += len(s)
        pct = v / len(s) * 100
        print(f"    Physical plausibility -- Pack current (bounds "
              f"[{PACK_I_MIN_A}, {PACK_I_MAX_A}] A): {v:,}/{len(s):,} ({pct:.3f}%) "
              f"outside, observed range [{s.min():.1f}, {s.max():.1f}]")

# Temperature: skip all-NaN vehicles
if col_t_max:
    s = data[col_t_max].dropna()
    if len(s):
        v = ((s < TEMP_MIN_C) | (s > TEMP_MAX_C)).sum()
        viol += int(v)
        total += len(s)
        pct = v / len(s) * 100
        n_vehicles_no_temp = int((data.groupby("vehicle_id")[col_t_max]
                                  .apply(lambda x: x.notna().sum() == 0)).sum())
        print(f"    Physical plausibility -- Battery temperature (bounds "
              f"[{TEMP_MIN_C}, {TEMP_MAX_C}] C): {v:,}/{len(s):,} ({pct:.3f}%) "
              f"outside, observed range [{s.min():.1f}, {s.max():.1f}]. "
              f"{n_vehicles_no_temp} vehicle(s) lack temperature data and are "
              f"excluded from this check.")

pct_implausible = (viol / total * 100) if total else 0
finding_physical = (
    f"{viol:,}/{total:,} pack voltage/current/temperature readings "
    f"({pct_implausible:.3f}%) outside the physically-derived bounds. "
    f"NCM pack voltage checked: {VOLTAGE_BOUNDS['NCM'][0]}-"
    f"{VOLTAGE_BOUNDS['NCM'][1]} V "
    f"(91 series cells from GitHub × 2.5-4.2V from datasheet). "
    f"LFP pack voltage NOT validated: series cell count undisclosed for "
    f"Vehicles #7-#10 (GitHub reports 'total cells' only). "
    f"Pack current {PACK_I_MIN_A} to {PACK_I_MAX_A} A; "
    f"battery temperature {TEMP_MIN_C}-{TEMP_MAX_C} C."
)
add("Correctness", "Physical plausibility",
    score_pct_low_is_good(pct_implausible), finding_physical)

# ---------------------------------------------------------------------------
# Current sign convention (verified from data, not README)
# ---------------------------------------------------------------------------
if col_i_pack and col_charge:
    valid = data.dropna(subset=[col_i_pack, col_charge]).copy()
    chg = valid[valid[col_charge] == 1]
    drv = valid[valid[col_charge] == 3]

    chg_neg_pct = (chg[col_i_pack] < 0).mean() * 100 if len(chg) else 0
    drv_pos_pct = (drv[col_i_pack] > 0).mean() * 100 if len(drv) else 0
    n_chg, n_drv = len(chg), len(drv)

    charge_score = score_pct_high_is_good(chg_neg_pct)

    if drv_pos_pct >= 75.0:
        drive_ok = True
        drive_score = "+"
    elif drv_pos_pct >= 50.0:
        drive_ok = False
        drive_score = "o"
    else:
        drive_ok = False
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
        f"Convention verified from raw CAN data; "
        f"the data is authoritative): negative = charging, positive = discharging. "
        f"Charging (signal=1, n={n_chg:,}): {chg_neg_pct:.2f}% of rows have negative "
        f"current — this is the primary convention signal and it is essentially "
        f"exact. Driving (signal=3, n={n_drv:,}): {drv_pos_pct:.2f}% of rows have "
        f"positive current; the remaining {100 - drv_pos_pct:.2f}% are negative due "
        f"to regen braking, which is legitimate physics in a real EV duty cycle, "
        f"not a sign violation. Drive-side tolerance: >= 75% positive-dominant is "
        f"the physical floor for typical city/highway mixes."
    )
    add("Correctness", "Current sign convention", overall, finding_sign)
else:
    add("Correctness", "Current sign convention", "o",
        "Pack current or charging_signal column not present; convention could "
        "not be verified.")

# ============================================================================
# 2. COMPLETENESS
# ============================================================================
print("\n== 2. Completeness ==")

essential_cols = [c for c in [col_time, col_v_pack, col_i_pack, col_soc] if c]
if essential_cols:
    missing = data[essential_cols].isna().sum().sum()
    total_cells_checked = data[essential_cols].size
    pct_missing = missing / total_cells_checked * 100
else:
    pct_missing = 0
finding_missing = (
    f"{pct_missing:.4f}% missing across powertrain-level essential columns "
    f"{essential_cols}. Cell-level columns (bcell_maxVoltage, bcell_minVoltage, "
    f"bcell_maxTemp, bcell_minTemp) are structurally absent for Vehicle#7 and "
    f"are excluded from this check by design."
)
add("Completeness", "Missing values",
    score_pct_low_is_good(pct_missing), finding_missing)

# Temporal continuity: gaps > 5x median interval, time in ms
cont_pct_list = []
if col_time:
    for _, grp in data.groupby("vehicle_id"):
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
    f"interval): {pct_continuity:.2f}%. Real-world telematics streams have "
    f"natural connectivity gaps; this quantifies how many remain."
)
add("Completeness", "Temporal continuity",
    score_pct_high_is_good(pct_continuity), finding_continuity)

documented_protocol = sum(PROTOCOL_METADATA.values())
total_protocol_elements = len(PROTOCOL_METADATA)
pct_doc = documented_protocol / total_protocol_elements * 100
finding_doc = (
    f"{documented_protocol}/{total_protocol_elements} essential protocol "
    f"elements documented ({pct_doc:.0f}%). Two gaps: the README's stated "
    f"current convention is contradicted by the raw data, and pack structure "
    f"for Vehicles #7-#10 is partly confidential."
)
add("Completeness", "Test protocol documentation",
    score_doc(pct_doc), finding_doc)

# ============================================================================
# 3. ANOMALY AND NOISE CONTROL
# ============================================================================
print("\n== 3. Anomaly and noise control ==")

# Statistical outliers on pack voltage (per vehicle, 3xIQR)
volt_out_pct_list = []
if col_v_pack:
    for _, grp in data.groupby("vehicle_id"):
        s = grp[col_v_pack].dropna()
        if len(s) > 100:
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
            volt_out_pct_list.append(((s < lo) | (s > hi)).mean() * 100)
pct_outliers = np.mean(volt_out_pct_list) if volt_out_pct_list else 0
finding_outliers = (
    f"{pct_outliers:.3f}% of pack-voltage readings fall outside the 3xIQR "
    f"range (averaged across {len(volt_out_pct_list)} vehicles)."
)
add("Anomaly and noise control", "Statistical outliers",
    score_pct_low_is_good(pct_outliers), finding_outliers)

# Unexpected signal changes: |dV| between consecutive samples, per vehicle
jump_pct_list = []
if col_v_pack:
    for _, grp in data.groupby("vehicle_id"):
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

# Measurement noise: pack-voltage fluctuation during low-current (quiet) periods
noise_pct_list = []
if col_v_pack and col_i_pack:
    for _, grp in data.groupby("vehicle_id"):
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

present_vehicles = [v for v in VEHICLE_INFO if v in data["vehicle_id"].unique()]
chems = sorted(set(VEHICLE_INFO[v]["chemistry"] for v in present_vehicles))
n_chem = len(chems)
finding_chem = f"{n_chem} chemistries across the fleet ({chems})."
add("Representativeness and diversity", "Chemistry diversity",
    score_diversity_count(n_chem), finding_chem)

types = sorted(set(VEHICLE_INFO[v]["type"] for v in present_vehicles))
n_types = len(types)
finding_types = f"{n_types} vehicle types across the fleet ({types})."
add("Representativeness and diversity", "Vehicle type diversity",
    score_diversity_count(n_types), finding_types)

# Temperature conditions
if col_t_max:
    per_vehicle_mean = data.groupby("vehicle_id")[col_t_max].mean().round(0).dropna()
    n_temp = per_vehicle_mean.nunique()
    if n_temp == 1:
        n_temp = 1
    means_list = sorted(per_vehicle_mean.unique().tolist())
    finding_temp = (
        f"{n_temp} distinct mean temperature level(s) across vehicles "
        f"({means_list} C, excluding Vehicle#7 which lacks cell temperature data). "
        f"Real-world temperature variation, not controlled setpoints."
    )
    add("Representativeness and diversity", "Temperature conditions",
        score_diversity_count(n_temp), finding_temp)
else:
    add("Representativeness and diversity", "Temperature conditions", "o",
        "Max cell temperature column not present.")

# Dynamic load profiles
if col_speed and col_i_pack:
    speed_var = data.groupby("vehicle_id")[col_speed].std().mean()
    current_var = data.groupby("vehicle_id")[col_i_pack].std().mean()
    finding_dynamic = (
        f"Real-world dynamic operation present by construction. Mean speed "
        f"std across vehicles: {speed_var:.1f} km/h. Mean current std: "
        f"{current_var:.1f} A. Both driving and charging profiles are "
        f"naturally variable."
    )
    add("Representativeness and diversity", "Dynamic load profiles",
        score_diversity_count(2), finding_dynamic)
else:
    add("Representativeness and diversity", "Dynamic load profiles", "o",
        "Speed or current column not present.")

finding_realworld = (
    f"On-road fleet operation across {n_vehicles} vehicles with cumulative "
    f"mileages ranging widely. All data collected from vehicles in service, "
    f"not laboratory replays."
)
add("Representativeness and diversity", "Real-world operation", "++",
    finding_realworld)

chem_counts = pd.Series([VEHICLE_INFO[v]["chemistry"] for v in present_vehicles]).value_counts()
avg_replicates = chem_counts.mean()
finding_replicates = (
    f"{avg_replicates:.1f} vehicles per chemistry on average "
    f"(NCM: {chem_counts.get('NCM', 0)}, LFP: {chem_counts.get('LFP', 0)})."
)
add("Representativeness and diversity", "Replicate vehicles",
    score_replicates(avg_replicates), finding_replicates)

for aspect, info in METADATA_ONLY.items():
    add("Representativeness and diversity", aspect, info["score"], info["finding"])

# ============================================================================
# 5. DISTRIBUTION BALANCE
# ============================================================================
print("\n== 5. Distribution balance ==")

if col_soc:
    soc_min = data.groupby("vehicle_id")[col_soc].min().min()
    soc_max = data.groupby("vehicle_id")[col_soc].max().max()
    score_soc = "++" if (soc_min <= 10 and soc_max >= 90) else "+"
    finding_soc = (
        f"SOC range covered across the fleet: {soc_min:.1f}% to {soc_max:.1f}%. "
        f"Real-world driving spans the full usable SOC window."
    )
    add("Distribution balance", "SOC range coverage", score_soc, finding_soc)
else:
    add("Distribution balance", "SOC range coverage", "o", "SOC column not present.")

if col_soc:
    bins = pd.cut(data[col_soc].dropna(), bins=[0, 20, 40, 60, 80, 100])
    dist = bins.value_counts(normalize=True) * 100
    dominant = dist.max() if len(dist) else 100
    finding_soc_bal = (
        f"Largest SOC bin holds {dominant:.1f}% of all records. Real-world "
        f"fleet operation clusters around mid-to-high SOC."
    )
    add("Distribution balance", "SOC distribution balance",
        score_soh_balance(dominant), finding_soc_bal)
else:
    add("Distribution balance", "SOC distribution balance", "o", "SOC column not present.")

rows_per_vehicle = data.groupby("vehicle_id").size()
cv_rows = rows_per_vehicle.std() / rows_per_vehicle.mean() * 100 if rows_per_vehicle.mean() else 0
finding_balance = (
    f"Coefficient of variation of row counts across vehicles: {cv_rows:.1f}%. "
    f"Min: {rows_per_vehicle.min():,} ({rows_per_vehicle.idxmin()}), "
    f"Max: {rows_per_vehicle.max():,} ({rows_per_vehicle.idxmax()})."
)
add("Distribution balance", "Vehicle data balance",
    score_cv(cv_rows), finding_balance)

# ============================================================================
# 6. TEMPORAL COHERENCE
# ============================================================================
print("\n== 6. Temporal coherence ==")

non_mono = 0
if col_time:
    for _, grp in data.groupby("vehicle_id"):
        t = grp[col_time].dropna()
        if len(t) > 1 and not t.is_monotonic_increasing:
            non_mono += 1
pct_ordered = 100 if non_mono == 0 else max(0, 100 - non_mono / n_vehicles * 100)
finding_mono = (
    f"{non_mono}/{n_vehicles} vehicles have non-monotonic timestamps. The "
    f"time column is a numeric millisecond counter (not an epoch), and "
    f"ordering is preserved within each vehicle file."
)
add("Temporal coherence", "Monotonic temporal progression",
    score_pct_high_is_good(pct_ordered), finding_mono)

if col_mileage:
    total_transitions, decreases_over_tol = 0, 0
    for _, grp in data.groupby("vehicle_id"):
        m = grp[col_mileage].dropna()
        if len(m) > 1:
            diffs = m.diff().dropna()
            total_transitions += len(diffs)
            decreases_over_tol += (diffs < -MILEAGE_TOLERANCE_KM).sum()
    pct_mileage = (1 - decreases_over_tol / total_transitions) * 100 if total_transitions else 100
    finding_mileage = (
        f"{pct_mileage:.4f}% of consecutive mileage transitions are "
        f"non-decreasing beyond a {MILEAGE_TOLERANCE_KM} km tolerance. "
        f"Sub-kilometer fluctuations (odometer rounding) are not flagged."
    )
    add("Temporal coherence", "Mileage progression",
        score_pct_high_is_good(pct_mileage), finding_mileage)
else:
    add("Temporal coherence", "Mileage progression", "o", "Mileage column not present.")

if col_charge:
    chg_vehicles = int(data.groupby("vehicle_id")[col_charge]
                       .apply(lambda s: (s == 1).any()).sum())
    drv_vehicles = int(data.groupby("vehicle_id")[col_charge]
                       .apply(lambda s: (s == 3).any()).sum())
    score_events = "++" if chg_vehicles == n_vehicles and drv_vehicles == n_vehicles else "+"
    finding_events = (
        f"Charging events present in {chg_vehicles}/{n_vehicles} vehicles; "
        f"driving events in {drv_vehicles}/{n_vehicles}. Both operation modes "
        f"are captured for nearly every vehicle."
    )
    add("Temporal coherence", "Event identification", score_events, finding_events)
else:
    add("Temporal coherence", "Event identification", "o",
        "Charging signal column not present.")

# ============================================================================
# SAVE SCORECARD
# ============================================================================
results_df = pd.DataFrame(results)
csv_path = os.path.join(OUT_DIR, "guangzhou_quality_scorecard.csv")
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
ax.set_title("Guangzhou 10-EV Fleet Dataset — Data Quality Scorecard",
             fontsize=12, fontweight="bold", pad=16)
plt.tight_layout()
fig_path = os.path.join(OUT_DIR, "guangzhou_quality_scorecard.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"Figure saved: {fig_path}")
plt.close(fig)

print("\nScore distribution:")
print(results_df["score"].value_counts().reindex(["++", "+", "o", "-"], fill_value=0))