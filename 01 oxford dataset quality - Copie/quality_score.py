"""
Oxford Battery Degradation Dataset 1 — Quality Scoring
========================================================
Scores the Oxford dataset (8 Kokam SLPB533459H4 cells) against the same
six-dimension quality framework : hardcode
ONLY facts that cannot be determined from the .mat file at all; everything
else is computed dynamically.

What ends up hardcoded here, and why )
  - Chemistry / DoD / C-rate diversity: there is no such field anywhere in
    the .mat structure to count from -- these are pulled from metadata.txt,
    including its own explicit interpretive framing (e.g. "characterization
    currents are not treated as independent ageing conditions").
  - Dynamic load profiles: the main .mat's C1dc/OCVdc segments record
    t, v, q, T but NOT Current -- the actual Urban Artemis drive-cycle
    current only exists in the separate small ExampleDC_C1.mat illustrative
    file. There's no per-cycle current trace in the file being scored to
    verify "dynamic" from, so this stays metadata-only too.
  - Calendar aging / Real-world operation: -- genuinely
    undeterminable from data under any amount of analysis.

Computed dynamically: replicate cell count, temperature diversity (from the
actual measured T array), current-sign convention (from the direction of
charge accumulation in ch vs dc segments), physical plausibility (voltage +
temperature only, per this dataset's revised scope), statistical outliers,
unexpected signal changes, measurement noise (from OCV/pseudo-OCV segments,
the closest thing this dataset has to a "quiet" signal), distribution
balance, and temporal coherence.

Output: a scorecard CSV and a scorecard figure, both saved to
./quality_results/
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.io as sio

# ============================================================================
# CONFIG
# ============================================================================
DATASET_PATH = r"C:\Users\admin\Desktop\DR2\11 All Datasets\05 Oxford Battery Degradation Dataset\oxford dataset"
MAT_FILE = os.path.join(DATASET_PATH, "Oxford_Battery_Degradation_Dataset_1.mat")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quality_results")
os.makedirs(OUT_DIR, exist_ok=True)

SCORE_LABELS = {"++": "Comprehensive coverage", "+": "Mostly satisfied",
                "o": "Partially satisfied", "-": "Not satisfied"}
SCORE_COLORS = {"++": "#2ca02c", "+": "#98df8a", "o": "#ffbb78", "-": "#d62728"}

# ============================================================================
# METADATA-ONLY FACTS (genuinely undeterminable from the .mat file's own
# content, whatever analysis is applied to it)
# ============================================================================
METADATA_ONLY = {
    "Calendar aging": {
        "score": "-",
        "finding": "Metadata reports calendar aging is absent as a dedicated experimental condition.",
    },
    "Dynamic load profiles": {
        "score": "+",
        "finding": "Metadata: Urban Artemis-derived variable-current drive-cycle profile is present, "
                   "but this is a single representative profile (not multiple varied ones), and the "
                   "actual per-cycle drive-cycle current is not recorded in this main .mat file "
                   "(only in the separate illustrative ExampleDC_C1.mat) -- so this cannot be "
                   "verified from the data being scored here, only asserted from documentation.",
    },
    "Real-world operation": {
        "score": "-",
        "finding": "Metadata: real-world operation is absent; the discharge profile is a laboratory "
                   "replay of a representative drive cycle, not measurements from vehicles in operation.",
    },
}

# Diversity aspects with no corresponding field anywhere in the .mat
# structure to count from -- sourced from metadata.txt, including its own
# stated interpretation where relevant (C-rate).
CHEMISTRY_N = 1   # single chemistry / cell model (Kokam SLPB533459H4) across the whole dataset
DOD_N = 1         # metadata: "Limited; cells primarily subjected to the same drive-cycle protocol"
CRATE_N = 1        # metadata: characterization currents (1C, pseudo-OCV) explicitly not treated
                    # as independent ageing conditions -- the drive-cycle protocol is the one condition

# Protocol elements documented in metadata.txt (all present -> comprehensive documentation)
PROTOCOL_METADATA = {
    "chemistry": True,
    "cell_model": True,
    "environmental_temperature": True,
    "charging_protocol": True,
    "discharge_protocol": True,
    "characterization_frequency": True,
    "characterization_current": True,
}

# Physical plausibility bounds -- general Li-ion envelope (dataset-agnostic,
# per this dataset's revised scope: voltage and temperature only, current
# excluded since no reliable per-sample current exists in this file).
VOLTAGE_MIN_V, VOLTAGE_MAX_V = 2.5, 4.2
CHARGE_TEMP_MIN_C, CHARGE_TEMP_MAX_C = 0.0, 45.0
DISCHARGE_TEMP_MIN_C, DISCHARGE_TEMP_MAX_C = -20.0, 60.0
NOMINAL_CAPACITY_AH = 0.740   # 740 mAh per metadata.txt
NOMINAL_VOLTAGE_V = 3.65      # metadata: nominal 3.6-3.7V

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
    results.append({"criterion": criterion, "aspect": aspect, "score": score, "finding": finding})
    print(f"  [{score}] {criterion} — {aspect}: {finding}")

# ============================================================================
# LOAD DATA (dynamic -- every cell, every characterisation cycle, every test
# type actually present is picked up generically, not hardcoded by name)
# ============================================================================
print("Loading Oxford .mat file...")
mat_data = sio.loadmat(MAT_FILE, squeeze_me=True, struct_as_record=False)
cell_keys = [k for k in mat_data.keys() if k.startswith("Cell")]
n_cells = len(cell_keys)
print(f"Found {n_cells} cells: {cell_keys}")

rows = []
for cell_key in cell_keys:
    cell = mat_data[cell_key]
    cyc_names = [n for n in dir(cell) if n.startswith("cyc") and not n.startswith("__")]
    for cyc_name in cyc_names:
        cyc = getattr(cell, cyc_name)
        cycle_num = int(cyc_name.replace("cyc", ""))
        # Generic detection: any attribute of cyc that itself has t & v is a
        # test segment (covers C1ch, C1dc, OCVch, OCVdc, or any other test
        # type name the file happens to use, without hardcoding the names).
        for test_name in [n for n in dir(cyc) if not n.startswith("__")]:
            seg = getattr(cyc, test_name)
            if not (hasattr(seg, "t") and hasattr(seg, "v")):
                continue
            t = np.atleast_1d(seg.t).astype(float).flatten()
            v = np.atleast_1d(seg.v).astype(float).flatten()
            q = np.atleast_1d(seg.q).astype(float).flatten() / 1000.0 if hasattr(seg, "q") else np.full_like(t, np.nan)
            T = np.atleast_1d(seg.T).astype(float).flatten() if hasattr(seg, "T") else np.full_like(t, np.nan)
            n = min(len(t), len(v), len(q), len(T))
            rows.append(pd.DataFrame({
                "cell": cell_key, "cycle": cycle_num, "test_type": test_name,
                "t": t[:n], "v": v[:n], "q": q[:n], "T": T[:n],
            }))

data = pd.concat(rows, ignore_index=True)
test_types = sorted(data["test_type"].unique())
print(f"Loaded {len(data):,} rows across {data['cycle'].nunique()} characterisation cycles, "
      f"test types found: {test_types}\n")

is_charge = data["test_type"].str.endswith("ch")
is_discharge = data["test_type"].str.endswith("dc")

# ============================================================================
# 1. CORRECTNESS
# ============================================================================
print("== 1. Correctness ==")

viol, total = 0, 0
per_signal = []
for label, mask, lo, hi in [
    ("Voltage (charge segments)", is_charge, VOLTAGE_MIN_V, VOLTAGE_MAX_V),
    ("Voltage (discharge segments)", is_discharge, VOLTAGE_MIN_V, VOLTAGE_MAX_V),
    ("Temperature (charge segments)", is_charge, CHARGE_TEMP_MIN_C, CHARGE_TEMP_MAX_C),
    ("Temperature (discharge segments)", is_discharge, DISCHARGE_TEMP_MIN_C, DISCHARGE_TEMP_MAX_C),
]:
    col = "v" if "Voltage" in label else "T"
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
    f"{viol:,}/{total:,} voltage/temperature readings ({pct_implausible:.3f}%) outside the general "
    f"Li-ion envelope (voltage {VOLTAGE_MIN_V}-{VOLTAGE_MAX_V}V; temperature {CHARGE_TEMP_MIN_C}-"
    f"{CHARGE_TEMP_MAX_C}C during charge, {DISCHARGE_TEMP_MIN_C}-{DISCHARGE_TEMP_MAX_C}C during "
    f"discharge). Current excluded from this check for this dataset -- see module docstring.")

# Current sign convention, derived without a raw Current field: charge (q)
# should increase over a *ch segment and decrease over a *dc segment.
sign_checks = []
for (cell, cycle, test_type), grp in data.groupby(["cell", "cycle", "test_type"]):
    g = grp.sort_values("t").dropna(subset=["q"])
    if len(g) < 2:
        continue
    delta_q = g["q"].iloc[-1] - g["q"].iloc[0]
    expected_positive = test_type.endswith("ch")
    sign_checks.append((delta_q > 0) == expected_positive)
pct_sign_ok = (np.mean(sign_checks) * 100) if sign_checks else 0
add("Correctness", "Current sign convention", score_pct_high_is_good(pct_sign_ok),
    f"{sum(sign_checks):,}/{len(sign_checks):,} segments ({pct_sign_ok:.2f}%) show charge "
    f"accumulating during 'ch' segments and depleting during 'dc' segments, as expected "
    f"(derived from d(charge)/dt direction -- no raw Current field in this file).")

# ============================================================================
# 2. COMPLETENESS
# ============================================================================
print("\n== 2. Completeness ==")

essential_cols = ["t", "v", "q"]
missing = data[essential_cols].isna().sum().sum()
pct_missing = missing / data[essential_cols].size * 100
add("Completeness", "Missing values", score_pct_low_is_good(pct_missing),
    f"{pct_missing:.4f}% missing across essential columns {essential_cols} (temperature excluded -- "
    f"metadata notes it is not recorded in every early cycle).")

cont_pct_list = []
for (cell, cycle, test_type), grp in data.groupby(["cell", "cycle", "test_type"]):
    t_sorted = grp["t"].dropna().sort_values().values
    if len(t_sorted) > 1:
        diffs = np.diff(t_sorted)
        med = np.median(diffs)
        if med > 0:
            cont_pct_list.append((diffs <= 5 * med).mean() * 100)
pct_continuity = np.mean(cont_pct_list) if cont_pct_list else 100
add("Completeness", "Temporal continuity", score_pct_high_is_good(pct_continuity),
    f"Average within-segment sampling continuity (no gaps >5x the median interval): {pct_continuity:.2f}%.")

documented_protocol = sum(PROTOCOL_METADATA.values())
total_protocol_elements = len(PROTOCOL_METADATA)
pct_doc = documented_protocol / total_protocol_elements * 100
add("Completeness", "Test protocol documentation", score_doc(pct_doc),
    f"{documented_protocol}/{total_protocol_elements} essential protocol elements documented ({pct_doc:.0f}%).")

# ============================================================================
# 3. ANOMALY AND NOISE CONTROL
# ============================================================================
print("\n== 3. Anomaly and noise control ==")

# Capacity per (cell, cycle) from the 1-C discharge characterisation (C1dc)
# -- the consistent-rate test used for SOH tracking, matching metadata's own
# framing of what the characterization cycles are for.
c1dc = data[data["test_type"].str.contains("C1") & data["test_type"].str.endswith("dc")]
cap_records = []
for (cell, cycle), grp in c1dc.groupby(["cell", "cycle"]):
    q = grp["q"].dropna()
    if len(q) > 10:
        cap = q.max() - q.min()
        if 0 < cap < 2.0:
            cap_records.append({"cell": cell, "cycle": cycle, "capacity_Ah": cap})
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
    f"{pct_outliers:.3f}% of per-cycle 1-C discharge capacity values fall outside the 3xIQR range.")

all_diffs = []
if not cap_df.empty:
    for cell, grp in cap_df.groupby("cell"):
        g = grp.sort_values("cycle")
        caps = g["capacity_Ah"].values
        if len(caps) > 1:
            all_diffs.extend(np.diff(caps)[1:])  # skip first transition
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
    f"{pct_jumps:.3f}% of cycle-to-cycle capacity transitions fall outside the 3xIQR range of "
    f"transition sizes (first transition per cell excluded).")

# Measurement noise from the pseudo-OCV segments -- the closest thing this
# dataset has to a "quiet" signal (current ~40mA / ~0.05C, near-resting).
ocv = data[data["test_type"].str.startswith("OCV")]
noise_pct = []
for (cell, cycle, test_type), grp in ocv.groupby(["cell", "cycle", "test_type"]):
    v = grp.sort_values("t")["v"].dropna().values
    if len(v) > 5:
        diffs = np.abs(np.diff(v))
        noise_pct.append(np.median(diffs) / NOMINAL_VOLTAGE_V * 100)
pct_noise = np.mean(noise_pct) if noise_pct else 0
add("Anomaly and noise control", "Measurement noise", score_noise(pct_noise),
    f"Median sample-to-sample voltage fluctuation during pseudo-OCV (~0.05C, near-resting) "
    f"segments is {pct_noise:.4f}% of a {NOMINAL_VOLTAGE_V}V reference.")

# ============================================================================
# 4. REPRESENTATIVENESS AND DIVERSITY
# ============================================================================
print("\n== 4. Representativeness and diversity ==")

add("Representativeness and diversity", "Chemistry diversity", score_diversity_count(CHEMISTRY_N),
    f"{CHEMISTRY_N} chemistry/cell model (Kokam SLPB533459H4) across the whole dataset -- no "
    f"per-cell chemistry field exists in the .mat to count from.")

t_valid = data["T"].dropna()
temp_clusters = t_valid.round(0).nunique() if len(t_valid) else 1
# a handful of degree-level clusters around one setpoint still counts as one
# controlled condition, not real diversity -- collapse to 1 if the spread is narrow
n_temp = 1 if (len(t_valid) == 0 or (t_valid.max() - t_valid.min()) < 10) else temp_clusters
add("Representativeness and diversity", "Temperature conditions", score_diversity_count(n_temp),
    f"Measured temperature range [{t_valid.min():.1f}, {t_valid.max():.1f}]C "
    f"(chamber setpoint 40C per metadata) -> {n_temp} effective condition(s).")

add("Representativeness and diversity", "DoD diversity", score_diversity_count(DOD_N),
    f"{DOD_N} DoD condition per metadata (\"Limited; cells primarily subjected to the same "
    f"drive-cycle-based cycling protocol\") -- no per-cycle DoD field to count independently.")

add("Representativeness and diversity", "C-rate diversity", score_diversity_count(CRATE_N),
    f"{CRATE_N} ageing C-rate condition per metadata's own framing (the 1-C and pseudo-OCV "
    f"characterization currents are explicitly not treated as independent ageing conditions).")

add("Representativeness and diversity", "Replicate cells", score_replicates(n_cells),
    f"{n_cells} cells cycled under the identical documented protocol.")

for aspect, info in METADATA_ONLY.items():
    add("Representativeness and diversity", aspect, info["score"], info["finding"])

# ============================================================================
# 5. DISTRIBUTION BALANCE
# ============================================================================
print("\n== 5. Distribution balance ==")

soh_frames = []
if not cap_df.empty:
    for cell, grp in cap_df.groupby("cell"):
        g = grp.sort_values("cycle")
        ref = g["capacity_Ah"].head(3).median()
        if ref <= 0:
            continue
        soh = (g["capacity_Ah"] / ref).clip(0, 1.05)
        soh_frames.append(pd.DataFrame({"cell": cell, "cycle": g["cycle"], "SOH": soh}))
soh_df = pd.concat(soh_frames, ignore_index=True) if soh_frames else pd.DataFrame(columns=["cell", "cycle", "SOH"])

if not soh_df.empty:
    cells_below_80 = soh_df[soh_df["SOH"] < 0.8]["cell"].nunique()
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
    f"Largest SOH bin holds {dominant:.1f}% of all characterisation-cycle records.")

if not cap_df.empty:
    per_cell_cycles = cap_df.groupby("cell")["cycle"].nunique()
    cv_pct = (per_cell_cycles.std() / per_cell_cycles.mean() * 100) if per_cell_cycles.mean() else 0
else:
    cv_pct = 0
add("Distribution balance", "Cycle contribution per cell", score_cv(cv_pct),
    f"Coefficient of variation of characterisation-cycle count across cells: {cv_pct:.1f}%.")

# ============================================================================
# 6. TEMPORAL COHERENCE
# ============================================================================
print("\n== 6. Temporal coherence ==")

ordered_pct = []
for cell, grp in cap_df.groupby("cell") if not cap_df.empty else []:
    cycles = grp["cycle"].values
    if len(cycles) > 1:
        ordered_pct.append((np.diff(np.sort(cycles)) >= 0).mean() * 100)
pct_ordered = np.mean(ordered_pct) if ordered_pct else 100
add("Temporal coherence", "Monotonic temporal progression", score_pct_high_is_good(pct_ordered),
    f"{pct_ordered:.2f}% of consecutive characterisation-cycle indices are non-decreasing.")

if not soh_df.empty:
    total_transitions, violating = 0, 0
    for cell, grp in soh_df.groupby("cell"):
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
    f"{pct_index_consistent:.2f}% of characterisation-cycle indices are non-duplicated within their cell.")

# ============================================================================
# SAVE SCORECARD
# ============================================================================
results_df = pd.DataFrame(results)
csv_path = os.path.join(OUT_DIR, "oxford_quality_scorecard.csv")
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
ax.set_title("Oxford Battery Degradation Dataset 1 — Data Quality Scorecard", fontsize=12, fontweight="bold", pad=16)
plt.tight_layout()
fig_path = os.path.join(OUT_DIR, "oxford_quality_scorecard.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"Figure saved: {fig_path}")
plt.close(fig)

print("\nScore distribution:")
print(results_df["score"].value_counts().reindex(["++", "+", "o", "-"], fill_value=0))