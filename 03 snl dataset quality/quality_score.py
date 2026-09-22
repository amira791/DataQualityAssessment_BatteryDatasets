"""
SNL Battery Dataset — Quality Scoring
======================================
Scores the Sandia National Laboratories 18650 dataset (LFP / NCA / NMC,
15/25/35 degC, 0-100 / 20-80 / 40-60 DoD, multiple discharge C-rates)
against the same six-dimension quality framework used for the Oxford,
MIT-Stanford-TRI, and CALCE datasets.

Design rule: hardcode ONLY facts that cannot be determined from the raw
cycle-data files at all (calendar aging, dynamic load profiles, real-world
operation — SNL's own documentation says these are absent, and no amount
of data analysis could confirm that). Everything else — chemistry /
temperature / DoD / C-rate diversity, physical-bounds violations, missing
values, noise, distribution balance, temporal coherence — is computed
dynamically from the CSV files and their filenames.

Physical plausibility for SNL is checked against the general Li-ion
envelope, applied uniformly across the three chemistries:
    voltage  fixed 2.5 - 4.2 V window
    current  per-cell C-rate envelope (charge <= 1.0C, discharge <= 5.0C),
             with 1C taken as each cell's rated capacity in Ah
Temperature is deliberately excluded from the plausibility check and is
used only in the anomaly/noise section, matching the stated scope.

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
DATA_DIR = r"C:\Users\admin\Desktop\DR2\11 All Datasets\10 Battery Archive Datasets\Battery Archive Data\SNL"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quality_results")
os.makedirs(OUT_DIR, exist_ok=True)

SCORE_LABELS = {"++": "Comprehensive coverage", "+": "Mostly satisfied",
                "o": "Partially satisfied", "-": "Not satisfied"}
SCORE_COLORS = {"++": "#2ca02c", "+": "#98df8a", "o": "#ffbb78", "-": "#d62728"}

# ============================================================================
# METADATA-ONLY FACTS (from SNL official documentation — cannot be
# recovered from the cycle-data CSVs under any amount of analysis)
# ============================================================================
METADATA_ONLY = {
    "Calendar aging": {
        "score": "-",
        "finding": "SNL documentation reports no dedicated calendar-aging "
                   "experiment; the study is cyclic-ageing only.",
    },
    "Dynamic load profiles": {
        "score": "-",
        "finding": "SNL documentation states cycling follows controlled "
                   "laboratory CC / CCCV protocols; no dynamic or drive-cycle "
                   "load profile is applied.",
    },
    "Real-world operation": {
        "score": "-",
        "finding": "Dataset is laboratory-only per official documentation; "
                   "no field / real-world operation data was collected.",
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
    "end_of_life_criterion": True,
}

# ============================================================================
# GENERAL Li-ion PHYSICAL ENVELOPE (dataset-agnostic, matching the Oxford
# code's approach). Applied uniformly across all chemistries in this suite
# so that physical-plausibility scores remain comparable between datasets.
#
# Voltage:
#   2.5 - 4.2 V -- common datasheet discharge / charge cutoffs for the
#   conventional 4.2 V Li-ion cell family (LCO / NMC / NCA graphite-anode
#   cells). This is a QC guard rail, not a claim about any specific cell's
#   datasheet limits.
#   Source: https://cdn-shop.adafruit.com/product-files/5035/5035_10050mAh_3.7V_A1____20210511.pdf
#
# Current:
#   Expressed as C-rate, not amperes. There is no scientifically valid
#   single ampere range for all Li-ion cells -- cells range from tiny coin
#   cells (tens of mA) to high-power 21700 cells (tens of A). Since 1C is
#   numerically equal to the cell's rated capacity in Ah, a C-rate bound
#   adapts automatically to every cell size and chemistry. The QC envelope
#   uses the UPPER bounds of the general rule so that no legitimate cell of
#   the 4.2 V family is false-flagged:
#       charge    |I| <= 1.0C
#       discharge |I| <= 5.0C
#   The actual ampere limit for a given cell is computed at runtime as
#   C_rate x that cell's own rated capacity in Ah.
#
# Temperature:
#   Deliberately excluded from the physical-plausibility check per the
#   stated scope. Temperature is still measured (in the Anomaly and noise
#   control section) as a noise / sensor-health indicator, but it is not
#   part of this envelope.
#   Source: https://cdn-shop.adafruit.com/product-files/5035/5035_10050mAh_3.7V_A1____20210511.pdf
# ============================================================================

# Plausability References: 
# For Voltage: https://data.matr.io/1/projects/5c48dd2bc625d700019f3204 

# For nominal voltage: https://iopscience.iop.org/article/10.1149/1945-7111/abae37 (max for all chemisteries)
# For C rate Charge/Discharge:  https://iopscience.iop.org/article/10.1149/1945-7111/abae37?utm_campaign=topcitedpaperusa&utm_medium=referral&utm_source=landing_page#1
#                               hhttps://iopscience.iop.org/article/10.1149/1945-7111/abae37


VOLTAGE_MIN_V, VOLTAGE_MAX_V = 2.5, 4.2
C_RATE_CHARGE_MAX    = 0.5   # upper bound of "maximum charge" C-rate rule
C_RATE_DISCHARGE_MAX = 3.0   # upper bound of "maximum continuous discharge" C-rate rule
NOMINAL_VOLTAGE_V = 3.6      # typical 18650 Li-ion operating voltage, used only to express noise as %
# Examples of sources: https://www.nature.com/articles/s41598-025-25924-2/tables/1
# https://www.nature.com/articles/s41597-025-06229-5/tables/1

# ============================================================================
# SCORING FUNCTIONS (identical vocabulary to the Oxford / MIT / CALCE code)
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
    return "o"   # n == 1 (or 0, defensively)

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
# LOAD DATA (dynamic — cell id, chemistry, temperature, DoD, C-rate all
# parsed straight from the filenames, which are part of the dataset itself)
# ============================================================================
print("Loading SNL cycle-data files (recursive across chemistry subfolders)...")
FNAME_RE = re.compile(
    r"SNL_18650_(?P<chem>[A-Za-z]+)_(?P<temp>\d+)C_(?P<dod>[\d\-]+)_"
    r"(?P<charge_c>[\d.]+)-(?P<discharge_c>[\d.]+)C_(?P<rep>[a-z])_cycle_data\.csv"
)

files = sorted(glob.glob(os.path.join(DATA_DIR, "**", "*_cycle_data.csv"), recursive=True))
if not files:
    raise FileNotFoundError(f"No *_cycle_data.csv files found under {DATA_DIR}")

frames = []
skipped = 0
for f in files:
    m = FNAME_RE.match(os.path.basename(f))
    if not m:
        skipped += 1
        continue
    df = pd.read_csv(f, low_memory=False)
    df.columns = df.columns.str.strip()
    df["cell_id"] = os.path.basename(f).replace("_cycle_data.csv", "")
    df["chemistry"] = m["chem"]
    df["temperature_C"] = int(m["temp"])
    df["dod_range"] = m["dod"]
    df["charge_c_rate"] = float(m["charge_c"])
    df["discharge_c_rate"] = float(m["discharge_c"])
    df["replicate"] = m["rep"]
    frames.append(df)

if skipped:
    print(f"  Warning: {skipped} *_cycle_data.csv files did not match the SNL filename pattern and were skipped")

data = pd.concat(frames, ignore_index=True)
n_cells = data["cell_id"].nunique()
print(f"Loaded {len(data):,} cycle records from {n_cells} cells")
print(f"  Chemistries: {sorted(data['chemistry'].unique())}")
print(f"  Temperatures: {sorted(data['temperature_C'].unique())} °C")
print(f"  DoD ranges: {sorted(data['dod_range'].unique())}")
print(f"  Discharge C-rates: {sorted(data['discharge_c_rate'].unique())}")

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
# PER-CELL RATED CAPACITY (used by the C-rate plausibility envelope)
#   SNL metadata does not state rated capacity, so it is derived per cell
#   from the median of the first three discharge-capacity readings. The
#   resulting value in Ah is what "1C" refers to for that cell.
# ============================================================================
cell_nominal_Ah = {}
if col_dchg_cap:
    cell_nominal_Ah = (
        data.groupby("cell_id")[col_dchg_cap]
            .apply(lambda s: s.dropna().head(3).median())
            .to_dict()
    )
print(f"Derived per-cell rated capacities (first 3 discharge readings, Ah):")
for cid in sorted(cell_nominal_Ah.keys())[:6]:
    print(f"    {cid}: {cell_nominal_Ah[cid]:.3f} Ah")
if len(cell_nominal_Ah) > 6:
    print(f"    ... and {len(cell_nominal_Ah) - 6} more cells")
print()

# ============================================================================
# 1. CORRECTNESS
# ============================================================================
print("== 1. Correctness ==")

# ---------------------------------------------------------------------------
# Physical plausibility -- voltage (fixed bounds, same for all cells)
# ---------------------------------------------------------------------------
viol, total = 0, 0
for label, col in [("Min_Voltage", col_min_v), ("Max_Voltage", col_max_v)]:
    if col:
        s = data[col].dropna()
        if len(s) == 0:
            continue
        v = ((s < VOLTAGE_MIN_V) | (s > VOLTAGE_MAX_V)).sum()
        viol += v
        total += len(s)
        pct = v / len(s) * 100
        print(f"    Physical plausibility breakdown -- {label} ({col}): "
              f"{v:,}/{len(s):,} ({pct:.2f}%) outside [{VOLTAGE_MIN_V}, {VOLTAGE_MAX_V}], "
              f"observed range [{s.min():.4f}, {s.max():.4f}], "
              f"5th/95th pct [{s.quantile(0.05):.4f}, {s.quantile(0.95):.4f}]")

# ---------------------------------------------------------------------------
# Physical plausibility -- current (per-cell C-rate envelope)
# ---------------------------------------------------------------------------
n_charge = n_discharge = 0
viol_charge = viol_discharge = 0
pct_charge = pct_discharge = 0.0
if col_max_i and col_min_i and cell_nominal_Ah:
    charge_limit_A    = data["cell_id"].map(cell_nominal_Ah) * C_RATE_CHARGE_MAX
    discharge_limit_A = data["cell_id"].map(cell_nominal_Ah) * C_RATE_DISCHARGE_MAX

    # Charge phase: Max_Current > 0. Flag anything above 1C.
    charge_mask = data[col_max_i] > 0
    viol_charge = (data.loc[charge_mask, col_max_i] > charge_limit_A[charge_mask]).sum()
    n_charge = int(charge_mask.sum())

    # Discharge phase: Min_Current < 0. Flag anything below -5C.
    discharge_mask = data[col_min_i] < 0
    viol_discharge = (data.loc[discharge_mask, col_min_i] < -discharge_limit_A[discharge_mask]).sum()
    n_discharge = int(discharge_mask.sum())

    viol += int(viol_charge) + int(viol_discharge)
    total += n_charge + n_discharge

    pct_charge = viol_charge / n_charge * 100 if n_charge else 0
    pct_discharge = viol_discharge / n_discharge * 100 if n_discharge else 0
    print(f"    Physical plausibility breakdown -- Max_Current (charge, "
          f"per-cell envelope {C_RATE_CHARGE_MAX}C): {viol_charge:,}/{n_charge:,} "
          f"({pct_charge:.2f}%) exceed the envelope")
    print(f"    Physical plausibility breakdown -- Min_Current (discharge, "
          f"per-cell envelope -{C_RATE_DISCHARGE_MAX}C): {viol_discharge:,}/{n_discharge:,} "
          f"({pct_discharge:.2f}%) exceed the envelope")

pct_implausible = (viol / total * 100) if total else 0
add("Correctness", "Physical plausibility", score_pct_low_is_good(pct_implausible),
    f"{viol:,}/{total:,} voltage/current readings ({pct_implausible:.3f}%) outside the general "
    f"Li-ion envelope. Voltage checked against the fixed {VOLTAGE_MIN_V}-{VOLTAGE_MAX_V} V window. "
    f"Current checked against a per-cell C-rate envelope (charge ≤ {C_RATE_CHARGE_MAX}C, "
    f"discharge ≤ {C_RATE_DISCHARGE_MAX}C), with 1C taken as each cell's rated capacity in Ah "
    f"derived from the median of its first three discharge-capacity readings. "
    f"Temperature deliberately excluded from this check.")

# ---------------------------------------------------------------------------
# Current sign convention
# ---------------------------------------------------------------------------
if col_min_i and col_max_i:
    valid_cycles = data.dropna(subset=[col_min_i, col_max_i])
    consistent = ((valid_cycles[col_max_i] > 0) & (valid_cycles[col_min_i] < 0)).sum()
    pct_sign_ok = (consistent / len(valid_cycles) * 100) if len(valid_cycles) else 0
    add("Correctness", "Current sign convention", score_pct_high_is_good(pct_sign_ok),
        f"{consistent:,}/{len(valid_cycles):,} cycles ({pct_sign_ok:.2f}%) show "
        f"positive charge current and negative discharge current, as documented.")
else:
    add("Correctness", "Current sign convention", "o",
        "Min/Max current columns not present in the cycle-data files; convention could not be "
        "verified per-record.")

# ---------------------------------------------------------------------------
# Documented vs. observed current rate
#   The SNL filename encodes the documented charge/discharge C-rate for each
#   cell. This aspect checks whether the observed per-cycle current magnitude
#   is consistent with what that documented C-rate implies, given the same
#   per-cell rated capacity used above. A mismatch is a protocol finding,
#   not a physical-plausibility violation, and is scored separately.
# ---------------------------------------------------------------------------
if col_max_i and col_min_i and cell_nominal_Ah:
    expected_charge_A    = data["charge_c_rate"]    * data["cell_id"].map(cell_nominal_Ah)
    expected_discharge_A = data["discharge_c_rate"] * data["cell_id"].map(cell_nominal_Ah)

    observed_charge_A    = data[col_max_i]
    observed_discharge_A = data[col_min_i].abs()

    ch_mask = observed_charge_A > 0
    dis_mask = observed_discharge_A > 0

    ch_far = (observed_charge_A[ch_mask] > 2.0 * expected_charge_A[ch_mask]).mean() * 100 if ch_mask.any() else 0
    dis_far = (observed_discharge_A[dis_mask] > 2.0 * expected_discharge_A[dis_mask]).mean() * 100 if dis_mask.any() else 0

    worst_far = max(ch_far, dis_far)
    add("Correctness", "Documented vs. observed current rate",
        "o" if worst_far > 50 else ("+" if worst_far > 5 else "++"),
        f"Per-cell C-rate is taken from the SNL filename and combined with the derived rated "
        f"capacity to give the expected ampere magnitude. Share of cycles whose observed current "
        f"exceeds twice the documented expectation: charge {ch_far:.1f}%, discharge {dis_far:.1f}%. "
        f"Reported as a protocol finding, not folded into physical plausibility.")
else:
    add("Correctness", "Documented vs. observed current rate", "o",
        "Insufficient data (current columns or per-cell capacities unavailable) to compare "
        "observed currents against the documented C-rate.")

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

documented_protocol = sum(PROTOCOL_METADATA.values())
total_protocol_elements = len(PROTOCOL_METADATA)
pct_doc = documented_protocol / total_protocol_elements * 100
add("Completeness", "Test protocol documentation", score_doc(pct_doc),
    f"{documented_protocol}/{total_protocol_elements} essential protocol elements documented "
    f"({pct_doc:.0f}%).")

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
    pct_outliers = outliers / len(caps) * 100 if len(caps) else 0
else:
    pct_outliers = 0
add("Anomaly and noise control", "Statistical outliers", score_pct_low_is_good(pct_outliers),
    f"{pct_outliers:.3f}% of discharge-capacity readings fall outside the 3xIQR range.")

all_diffs = []
if col_dchg_cap and col_cycle:
    for _, grp in data.groupby("cell_id"):
        g = grp.sort_values(col_cycle)
        cell_caps = g[col_dchg_cap].dropna().values
        if len(cell_caps) > 1:
            all_diffs.extend(np.diff(cell_caps)[1:])   # skip first (formation) transition
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
    f"of transition sizes (first, formation-related transition excluded per cell).")

# Measurement noise from the low-current (rest / CV-tail) portion of each
# timeseries file -- the quietest part of each SNL cycle, used here as the
# analogue of Oxford's pseudo-OCV segments.
print("\nComputing measurement noise from raw timeseries voltage (rest / CV-tail)...")
TS_FNAME_RE = re.compile(
    r"SNL_18650_(?P<chem>[A-Za-z]+)_(?P<temp>\d+)C_(?P<dod>[\d\-]+)_"
    r"(?P<charge_c>[\d.]+)-(?P<discharge_c>[\d.]+)C_(?P<rep>[a-z])_timeseries\.csv"
)
ts_files = sorted(glob.glob(os.path.join(DATA_DIR, "**", "*_timeseries.csv"), recursive=True))

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
    if len(v) < 20:
        continue
    if i_col is not None:
        i = pd.to_numeric(ts[i_col], errors="coerce").reindex(v.index)
        rest_mask = i.abs() < 0.05   # near-zero current -> rest / CV-tail: signal should be flat
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
    f"({len(voltage_noise_pct)} files contributed).")

# ============================================================================
# 4. REPRESENTATIVENESS AND DIVERSITY
# (chemistry / temperature / DoD / C-rate parsed dynamically from filenames;
#  calendar aging / dynamic load / real-world are the hardcoded metadata facts)
# ============================================================================
print("\n== 4. Representativeness and diversity ==")

n_chem = data["chemistry"].nunique()
add("Representativeness and diversity", "Chemistry diversity", score_diversity_count(n_chem),
    f"{n_chem} chemistries across the dataset ({sorted(data['chemistry'].unique())}).")

n_temp = data["temperature_C"].nunique()
add("Representativeness and diversity", "Temperature conditions", score_diversity_count(n_temp),
    f"{n_temp} temperature setpoint(s) ({sorted(data['temperature_C'].unique())} °C).")

n_dod = data["dod_range"].nunique()
add("Representativeness and diversity", "DoD diversity", score_diversity_count(n_dod),
    f"{n_dod} DoD range(s) ({sorted(data['dod_range'].unique())}).")

n_crate = data["discharge_c_rate"].nunique()
add("Representativeness and diversity", "C-rate diversity", score_diversity_count(n_crate),
    f"{n_crate} discharge C-rate(s) ({sorted(data['discharge_c_rate'].unique())}).")

# Replicate cells: average count of replicate suffixes (a, b, c, d) per
# unique (chemistry, T, DoD, charge_C, discharge_C) condition.
replicate_groups = data.groupby(
    ["chemistry", "temperature_C", "dod_range", "charge_c_rate", "discharge_c_rate"]
)["replicate"].nunique()
avg_replicates = replicate_groups.mean() if len(replicate_groups) else 0
add("Representativeness and diversity", "Replicate cells", score_replicates(avg_replicates),
    f"{avg_replicates:.1f} replicate cells per experimental condition on average "
    f"(across {len(replicate_groups)} conditions).")

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
    dominant = dist.max() if len(dist) else 100
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
    total_transitions, violating_transitions = 0, 0
    for cell_id, grp in soh_df.groupby("cell_id"):
        g = grp.sort_values(col_cycle) if col_cycle else grp
        soh_vals = g["SOH"].values
        if len(soh_vals) > 1:
            diffs = np.diff(soh_vals)
            total_transitions += len(diffs)
            violating_transitions += (diffs > 0.03).sum()   # >3% SOH recovery = violation
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
csv_path = os.path.join(OUT_DIR, "snl_quality_scorecard.csv")
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
ax.set_title("SNL Battery Dataset — Data Quality Scorecard",
             fontsize=12, fontweight="bold", pad=16)
plt.tight_layout()
fig_path = os.path.join(OUT_DIR, "snl_quality_scorecard.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"Figure saved: {fig_path}")
plt.close(fig)

print("\nScore distribution:")
print(results_df["score"].value_counts().reindex(["++", "+", "o", "-"], fill_value=0))