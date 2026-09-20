"""
Guangzhou 10-EV Fleet Dataset — Exploration
=============================================
Read-only diagnostic for the Real-World 10EVs dataset. Answers the schema
questions that quality_score.py depends on, so we can verify its
assumptions before trusting its scores.

Does NOT modify anything. Does NOT write scorecards. Prints a structured
report to the console and writes it to explore_output.txt.

Sections:
    0. Load overview
    1. Schema dump (columns, dtypes, first values, NaN%)
    2. charging_signal encoding diagnosis (which value means what)
    3. Per-vehicle pack-voltage range
    4. Per-vehicle / per-column NaN breakdown
    5. Mileage monotonicity: which vehicle, how much, where
    6. Temperature: which vehicle has all-NaN bcell_maxTemp
    7. Time column diagnosis
    8. Assumption verdict table for quality_score.py
"""

import os
import glob
import re
import sys
import numpy as np
import pandas as pd

# ============================================================================
# CONFIG
# ============================================================================
DATASET_PATH = r"C:\Users\admin\Desktop\DR2\11 All Datasets\12 Real-World 10EVs dataset\dataset_"
OUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "explore_output.txt")

# Capture console + file
class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, d):
        for s in self.streams: s.write(d)
    def flush(self):
        for s in self.streams: s.flush()

_log_fh = open(OUT_FILE, "w", encoding="utf-8")
_orig_stdout = sys.stdout
sys.stdout = Tee(_orig_stdout, _log_fh)

def section(t):
    print("\n" + "=" * 78)
    print(f"  {t}")
    print("=" * 78)

def subsection(t):
    print("\n" + "-" * 78)
    print(f"  {t}")
    print("-" * 78)

# ============================================================================
# 0. LOAD
# ============================================================================
section("0. LOAD OVERVIEW")

xlsx_files = sorted(glob.glob(os.path.join(DATASET_PATH, "vehicle#*.xlsx")))
print(f"Folder: {DATASET_PATH}")
print(f"Found {len(xlsx_files)} xlsx files:")
for f in xlsx_files:
    size_mb = os.path.getsize(f) / 1e6
    print(f"    {os.path.basename(f):25s}  {size_mb:8.1f} MB")

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

print(f"\nTotal rows: {len(data):,}")
print(f"Total columns: {data.shape[1]}")
print(f"Vehicles: {data['vehicle_id'].nunique()}")

# ============================================================================
# 1. SCHEMA DUMP
# ============================================================================
section("1. SCHEMA DUMP")

print(f"{'column':25s} {'dtype':12s} {'NaN%':>8s}  first 3 non-null values")
print("-" * 100)
for c in data.columns:
    if c == "vehicle_id":
        continue
    s = data[c]
    nan_pct = s.isna().mean() * 100
    sample = s.dropna().head(3).tolist()
    print(f"{c:25s} {str(s.dtype):12s} {nan_pct:7.2f}%  {str(sample)[:50]}")

# ============================================================================
# 2. CHARGING_SIGNAL ENCODING DIAGNOSIS
# ============================================================================
section("2. CHARGING_SIGNAL ENCODING DIAGNOSIS")

if "charging_signal" in data.columns:
    print("Unique values of charging_signal:")
    print(data["charging_signal"].value_counts(dropna=False).to_string())

    print("\nMean / min / max of hv_current per charging_signal value:")
    grp = data.groupby("charging_signal")["hv_current"].agg(
        ["count", "mean", "min", "max"]
    )
    print(grp.to_string())

    print("\nInterpretation guide:")
    print("  - The value whose mean current is NEGATIVE is likely charging (current flows into pack)")
    print("  - The value whose mean current is POSITIVE is likely driving/discharge")
    print("  - If a value has mean ~0, it may be idle or unknown")

    print("\nCross-check with SOC change per charging_signal value:")
    # Per-vehicle since SOC is monotone within mode
    for sig_val in sorted(data["charging_signal"].dropna().unique()):
        sub = data[data["charging_signal"] == sig_val].dropna(subset=["bcell_soc"])
        if len(sub) > 0:
            print(f"  charging_signal = {sig_val}: mean SOC = {sub['bcell_soc'].mean():.1f}%, "
                  f"n = {len(sub):,}")

    print("\nFirst 20 rows of a charging-signal change boundary (Vehicle#1):")
    v1 = data[data["vehicle_id"] == "Vehicle#1"].head(200)
    print(v1[["time", "charging_signal", "hv_current", "bcell_soc", "vhc_speed"]].head(20).to_string(index=False))
else:
    print("charging_signal column not found.")

# ============================================================================
# 3. PER-VEHICLE PACK-VOLTAGE RANGE
# ============================================================================
section("3. PER-VEHICLE PACK-VOLTAGE RANGE")

if "hv_voltage" in data.columns:
    print(f"{'vehicle':12s} {'n':>10s} {'min':>10s} {'p01':>10s} {'p50':>10s} "
          f"{'p99':>10s} {'max':>10s}")
    print("-" * 85)
    for v in sorted(data["vehicle_id"].unique()):
        s = data[data["vehicle_id"] == v]["hv_voltage"].dropna()
        if len(s) == 0:
            print(f"{v:12s} {'no data':>10s}")
            continue
        print(f"{v:12s} {len(s):10,d} {s.min():10.1f} {s.quantile(0.01):10.1f} "
              f"{s.median():10.1f} {s.quantile(0.99):10.1f} {s.max():10.1f}")

    print("\nGrouped by chemistry (from VEHICLE_INFO mapping):")
    VEHICLE_CHEM = {
        "Vehicle#1": "NCM", "Vehicle#2": "NCM", "Vehicle#3": "NCM",
        "Vehicle#4": "NCM", "Vehicle#5": "NCM", "Vehicle#6": "NCM",
        "Vehicle#7": "LFP", "Vehicle#8": "LFP", "Vehicle#9": "LFP", "Vehicle#10": "LFP",
    }
    data["_chem"] = data["vehicle_id"].map(VEHICLE_CHEM)
    grp = data.groupby("_chem")["hv_voltage"].agg(
        ["count", "min", "median", "max"]
    )
    print(grp.to_string())

# ============================================================================
# 4. PER-VEHICLE / PER-COLUMN NaN BREAKDOWN
# ============================================================================
section("4. PER-VEHICLE / PER-COLUMN NaN BREAKDOWN")

essential_cols = [c for c in
                  ["time", "vhc_speed", "charging_signal", "vhc_totalMile",
                   "hv_voltage", "hv_current", "bcell_soc",
                   "bcell_maxVoltage", "bcell_minVoltage",
                   "bcell_maxTemp", "bcell_minTemp"]
                  if c in data.columns]

print(f"Essential columns checked: {essential_cols}\n")
print(f"{'vehicle':12s} " + " ".join(f"{c[:14]:>14s}" for c in essential_cols))
print("-" * (12 + 15 * len(essential_cols)))
for v in sorted(data["vehicle_id"].unique()):
    sub = data[data["vehicle_id"] == v]
    nan_pcts = [sub[c].isna().mean() * 100 for c in essential_cols]
    row = f"{v:12s} " + " ".join(f"{p:13.2f}%" for p in nan_pcts)
    print(row)

print("\nAggregated NaN% by column (whole dataset):")
for c in essential_cols:
    print(f"  {c:25s}: {data[c].isna().mean() * 100:6.2f}%")

# ============================================================================
# 5. MILEAGE MONOTONICITY DIAGNOSIS
# ============================================================================
section("5. MILEAGE MONOTONICITY DIAGNOSIS")

if "vhc_totalMile" in data.columns:
    print(f"{'vehicle':12s} {'monotonic?':>12s} {'#decreases':>12s} {'max decrease (km)':>20s}")
    print("-" * 60)
    for v in sorted(data["vehicle_id"].unique()):
        m = data[data["vehicle_id"] == v]["vhc_totalMile"].dropna()
        if len(m) < 2:
            continue
        diffs = m.diff().dropna()
        decreases = (diffs < 0).sum()
        max_dec = diffs.min()
        print(f"{v:12s} {str(m.is_monotonic_increasing):>12s} {decreases:12d} {max_dec:20.3f}")

    # Show context around the worst decrease
    worst_v = None
    worst_dec = 0
    for v in sorted(data["vehicle_id"].unique()):
        m = data[data["vehicle_id"] == v]["vhc_totalMile"].dropna()
        if len(m) < 2:
            continue
        d = m.diff()
        if d.min() < worst_dec:
            worst_dec = d.min()
            worst_v = v

    if worst_v:
        print(f"\nContext around worst decrease in {worst_v}:")
        sub = data[data["vehicle_id"] == worst_v].copy()
        sub["_m_diff"] = sub["vhc_totalMile"].diff()
        idx = sub["_m_diff"].idxmin()
        if pd.notna(idx):
            window = sub.loc[max(0, idx - 3): idx + 3,
                             ["time", "vhc_totalMile", "_m_diff", "charging_signal", "hv_voltage"]]
            print(window.to_string(index=False))

# ============================================================================
# 6. TEMPERATURE: WHICH VEHICLE HAS ALL-NaN bcell_maxTemp
# ============================================================================
section("6. TEMPERATURE PER-VEHICLE COVERAGE")

if "bcell_maxTemp" in data.columns:
    print(f"{'vehicle':12s} {'n':>10s} {'NaN%':>8s} {'min':>8s} {'median':>8s} {'max':>8s}")
    print("-" * 65)
    for v in sorted(data["vehicle_id"].unique()):
        sub = data[data["vehicle_id"] == v]["bcell_maxTemp"]
        n = sub.notna().sum()
        nan_pct = sub.isna().mean() * 100
        if n == 0:
            print(f"{v:12s} {n:10d} {nan_pct:7.1f}%   (all NaN)")
        else:
            print(f"{v:12s} {n:10d} {nan_pct:7.1f}% {sub.min():8.1f} {sub.median():8.1f} {sub.max():8.1f}")

# ============================================================================
# 7. TIME COLUMN DIAGNOSIS
# ============================================================================
section("7. TIME COLUMN DIAGNOSIS")

if "time" in data.columns:
    print("First 5 values:", data["time"].head(5).tolist())
    print("dtype:", data["time"].dtype)

    try:
        t = pd.to_datetime(data["time"])
        print("Successfully parsed as datetime.")
        print("Global range:", t.min(), "→", t.max())
        print("\nPer-vehicle time span:")
        for v in sorted(data["vehicle_id"].unique()):
            sub_t = pd.to_datetime(data[data["vehicle_id"] == v]["time"]).dropna()
            if len(sub_t) > 1:
                span = sub_t.max() - sub_t.min()
                dt = sub_t.sort_values().diff().dt.total_seconds().dropna()
                dt = dt[dt > 0]
                med_dt = dt.median() if len(dt) else 0
                print(f"  {v:12s}: span {span.days:3d}d  median dt = {med_dt:6.2f}s "
                      f"(~{1/med_dt:.2f} Hz)" if med_dt > 0 else f"  {v:12s}: span {span.days:3d}d")
    except Exception as e:
        print(f"Could not parse as datetime: {e}")
        t = pd.to_numeric(data["time"], errors="coerce")
        print("Numeric range:", t.min(), "→", t.max())

# ============================================================================
# 8. ASSUMPTION VERDICT TABLE
# ============================================================================
section("8. ASSUMPTION VERDICT TABLE for quality_score.py")

print("Each row is an assumption baked into quality_score.py. Verdict is based")
print("on the diagnostics above.\n")

verdicts = []

# 1. charging_signal encoding
if "charging_signal" in data.columns and "hv_current" in data.columns:
    means = data.groupby("charging_signal")["hv_current"].mean()
    if 1 in means.index and 3 in means.index:
        sig1_neg = means[1] < 0
        sig3_pos = means[3] > 0
        if sig1_neg and sig3_pos:
            verdicts.append(("signal==1 → charging, signal==3 → driving",
                             "CONFIRMED",
                             f"mean I(sig=1)={means[1]:.1f}A (neg), mean I(sig=3)={means[3]:.1f}A (pos)"))
        elif means[1] > 0 and means[3] < 0:
            verdicts.append(("signal==1 → charging, signal==3 → driving",
                             "CONTRADICTED",
                             f"signals appear SWAPPED: mean I(sig=1)={means[1]:.1f}A (pos), mean I(sig=3)={means[3]:.1f}A (neg)"))
        else:
            verdicts.append(("signal==1 → charging, signal==3 → driving",
                             "UNKNOWN",
                             f"ambiguous: means = {means.to_dict()}"))

# 2. current sign convention
if "hv_current" in data.columns:
    # Use charging_signal as ground truth
    if "charging_signal" in data.columns:
        chg = data[data["charging_signal"] == 1]["hv_current"].dropna()
        drv = data[data["charging_signal"] == 3]["hv_current"].dropna()
        if len(chg) > 0 and len(drv) > 0:
            chg_pos = (chg > 0).mean() * 100
            drv_neg = (drv < 0).mean() * 100
            if chg_pos > 50 and drv_neg > 50:
                verdicts.append(("positive = charging, negative = discharging",
                                 "CONFIRMED",
                                 f"{chg_pos:.1f}% of sig=1 rows positive, {drv_neg:.1f}% of sig=3 rows negative"))
            else:
                verdicts.append(("positive = charging, negative = discharging",
                                 "CONTRADICTED",
                                 f"only {chg_pos:.1f}% of sig=1 rows positive, {drv_neg:.1f}% of sig=3 rows negative"))

# 3. voltage bound 250–450 V fleet-wide
if "hv_voltage" in data.columns:
    per_v = data.groupby("vehicle_id")["hv_voltage"].agg(["min", "max"])
    within = ((per_v["min"] >= 250) & (per_v["max"] <= 450)).sum()
    total_v = len(per_v)
    if within == total_v:
        verdicts.append(("pack voltage 250-450 V fleet-wide",
                         "CONFIRMED",
                         f"all {total_v} vehicles within range"))
    else:
        over = per_v[per_v["max"] > 450]
        verdicts.append(("pack voltage 250-450 V fleet-wide",
                         "CONTRADICTED",
                         f"{total_v - within}/{total_v} vehicles exceed 450 V: "
                         f"{dict(over['max'].round(0))}"))

# 4. essential columns all populated
essential = [c for c in ["time", "hv_voltage", "hv_current", "bcell_soc", "bcell_maxTemp"]
             if c in data.columns]
overall_nan = {c: data[c].isna().mean() * 100 for c in essential}
worst = max(overall_nan.values())
if worst < 1:
    verdicts.append(("essential columns near-fully populated",
                     "CONFIRMED",
                     f"max NaN among essentials = {worst:.2f}%"))
else:
    verdicts.append(("essential columns near-fully populated",
                     "CONTRADICTED",
                     f"max NaN among essentials = {worst:.2f}%"))

# 5. mileage monotonic per vehicle
if "vhc_totalMile" in data.columns:
    bad = 0
    for v in sorted(data["vehicle_id"].unique()):
        m = data[data["vehicle_id"] == v]["vhc_totalMile"].dropna()
        if len(m) > 1 and not m.is_monotonic_increasing:
            bad += 1
    verdicts.append(("mileage monotonic per vehicle",
                     "CONFIRMED" if bad == 0 else "CONTRADICTED",
                     f"{bad}/{data['vehicle_id'].nunique()} vehicles non-monotonic"))

# 6. bcell_maxTemp present for all vehicles
if "bcell_maxTemp" in data.columns:
    missing_v = []
    for v in sorted(data["vehicle_id"].unique()):
        s = data[data["vehicle_id"] == v]["bcell_maxTemp"]
        if s.notna().sum() == 0:
            missing_v.append(v)
    verdicts.append(("bcell_maxTemp populated for all vehicles",
                     "CONFIRMED" if not missing_v else "CONTRADICTED",
                     f"all-NaN vehicles: {missing_v if missing_v else 'none'}"))

print(f"{'assumption':55s} {'verdict':>14s}   evidence")
print("-" * 130)
for assumption, verdict, evidence in verdicts:
    print(f"{assumption:55s} {verdict:>14s}   {evidence}")

section("END OF EXPLORATION")

sys.stdout = _orig_stdout
_log_fh.close()
print(f"\nFull report written to: {OUT_FILE}")