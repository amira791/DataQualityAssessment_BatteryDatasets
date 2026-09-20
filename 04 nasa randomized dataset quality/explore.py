"""
NASA Randomized & Recommissioned Battery Dataset — Exploration
=================================================================
Read-only diagnostic for the regular_alt_batteries folder. Answers the
schema questions that quality_score.py depends on, so we can verify its
assumptions before trusting its scores.

Does NOT modify anything. Does NOT write scorecards. Only prints a
structured report to the console and (optionally) writes a text file.

Sections:
    0. Load overview
    1. Schema dump (columns, dtypes, first values, NaN%)
    2. Column classification by mode (-1 / 0 / 1)
    3. Time column diagnosis
    4. Mode segment stats
    5. Per-pack current profile (regular vs reference discharge)
    6. Documented-level snapping (regular-mission only)
    7. Per-cycle capacity (correctly computed, using relative_time)
    8. Reference-discharge capacity trend (SOH signal)
    9. Sanity checks
   10. Assumption verdict table for quality_score.py
"""

import os
import glob
import numpy as np
import pandas as pd

# ============================================================================
# CONFIG
# ============================================================================
DATASET_PATH = r"C:\Users\admin\Desktop\DR2\11 All Datasets\02 NASA Randomized Battery Dataset\battery_alt_dataset\regular_alt_batteries"
OUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "explore_output.txt")

# Documented discharge groups for the regular folder (README)
DOCUMENTED_DISCHARGE_LEVELS_A = [9.30, 12.9, 14.3, 16.0, 17.0, 19.0]

# Capture both console and file output
import io, sys
class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
    def flush(self):
        for s in self.streams:
            s.flush()

_log_fh = open(OUT_FILE, "w", encoding="utf-8")
_orig_stdout = sys.stdout
sys.stdout = Tee(_orig_stdout, _log_fh)

def section(title):
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)

def subsection(title):
    print("\n" + "-" * 78)
    print(f"  {title}")
    print("-" * 78)

# ============================================================================
# 0. LOAD
# ============================================================================
section("0. LOAD OVERVIEW")

csv_files = sorted(glob.glob(os.path.join(DATASET_PATH, "battery*.csv")))
print(f"Folder: {DATASET_PATH}")
print(f"Found {len(csv_files)} battery CSV files:")
for f in csv_files:
    size_mb = os.path.getsize(f) / 1e6
    print(f"    {os.path.basename(f):55s}  {size_mb:7.1f} MB")

frames = []
for f in csv_files:
    df = pd.read_csv(f, low_memory=False)
    df.columns = df.columns.str.strip()
    df["battery_id"] = os.path.basename(f).replace(".csv", "")
    frames.append(df)
data = pd.concat(frames, ignore_index=True)

print(f"\nTotal rows: {len(data):,}")
print(f"Total columns: {data.shape[1]}")
print(f"Batteries: {data['battery_id'].nunique()}")

# ============================================================================
# 1. SCHEMA DUMP
# ============================================================================
section("1. SCHEMA DUMP")

print(f"{'column':35s} {'dtype':10s} {'NaN%':>8s}  first 3 non-null values")
print("-" * 100)
for c in data.columns:
    if c == "battery_id":
        continue
    s = data[c]
    nan_pct = s.isna().mean() * 100
    sample = s.dropna().head(3).tolist()
    sample_str = str(sample)[:50]
    print(f"{c:35s} {str(s.dtype):10s} {nan_pct:7.2f}%  {sample_str}")

# ============================================================================
# 2. COLUMN CLASSIFICATION BY MODE
# ============================================================================
section("2. COLUMN CLASSIFICATION BY MODE (-1 = discharge, 0 = rest, 1 = charge)")

mode_col = next((c for c in data.columns if "mode" in c.lower()), None)
if mode_col is None:
    print("No mode column found — cannot classify by phase.")
else:
    for m in [-1, 0, 1]:
        sub = data[data[mode_col] == m]
        print(f"\nMode = {m}   ({len(sub):,} rows, {len(sub)/len(data)*100:.1f}% of dataset)")
        print(f"    {'column':35s} {'NaN%':>8s}")
        for c in data.columns:
            if c == "battery_id" or c == mode_col:
                continue
            nan_pct = sub[c].isna().mean() * 100
            print(f"    {c:35s} {nan_pct:7.2f}%")

# ============================================================================
# 3. TIME COLUMN DIAGNOSIS
# ============================================================================
section("3. TIME COLUMN DIAGNOSIS")

time_like = [c for c in data.columns if "time" in c.lower()]
print(f"Time-like columns found: {time_like}\n")

for tc in time_like:
    print(f"--- Column: {tc} ---")
    s = pd.to_numeric(data[tc], errors="coerce")
    print(f"    NaN%:         {s.isna().mean()*100:.2f}%")
    print(f"    min/max:      {s.min()} / {s.max()}")
    print(f"    first 5:      {s.dropna().head(5).tolist()}")
    # Monotonicity within one representative battery
    b0 = data["battery_id"].iloc[0]
    s0 = s[data["battery_id"] == b0].dropna().values
    if len(s0) > 1:
        diffs = np.diff(s0)
        non_dec = (diffs >= 0).mean() * 100
        print(f"    within {b0}: non-decreasing on {non_dec:.2f}% of consecutive steps, "
              f"median dt = {np.median(diffs):.4f}")
    print()

# ============================================================================
# 4. MODE SEGMENT STATS
# ============================================================================
section("4. MODE SEGMENT STATS")

mode_col_present = mode_col is not None
if not mode_col_present:
    print("No mode column — skipping.")
else:
    rel_time_col = next((c for c in data.columns if "relative" in c.lower() and "time" in c.lower()), None)
    if rel_time_col is None:
        rel_time_col = next((c for c in data.columns if "time" in c.lower() and "start" not in c.lower()), None)

    print(f"Using time column for duration measurement: {rel_time_col}")
    print(f"\n{'battery':45s} {'#seg -1':>8s} {'#seg 0':>8s} {'#seg 1':>8s} "
          f"{'med rows -1':>12s} {'med dur -1':>12s}")
    print("-" * 110)

    for batt_id, grp in data.groupby("battery_id"):
        grp = grp.copy().reset_index(drop=True)
        grp["_seg"] = (grp[mode_col] != grp[mode_col].shift()).cumsum()
        n_seg = {m: 0 for m in [-1, 0, 1]}
        rows_per_seg = []
        dur_per_seg = []
        for _, seg in grp.groupby("_seg"):
            m = seg[mode_col].iloc[0]
            if m in n_seg:
                n_seg[m] += 1
            if m == -1:
                rows_per_seg.append(len(seg))
                if rel_time_col:
                    t = pd.to_numeric(seg[rel_time_col], errors="coerce").dropna().values
                    if len(t) > 1:
                        dur_per_seg.append(t[-1] - t[0])
        med_rows = int(np.median(rows_per_seg)) if rows_per_seg else 0
        med_dur = float(np.median(dur_per_seg)) if dur_per_seg else 0.0
        print(f"{batt_id:45s} {n_seg[-1]:8d} {n_seg[0]:8d} {n_seg[1]:8d} "
              f"{med_rows:12d} {med_dur:12.2f}")

# ============================================================================
# 5. PER-PACK CURRENT PROFILE
# ============================================================================
section("5. PER-PACK CURRENT PROFILE (regular vs reference discharge)")

i_col = next((c for c in data.columns if "current" in c.lower()), None)
mission_col = next((c for c in data.columns if "mission" in c.lower()), None)
if i_col is None or mode_col is None:
    print("Current or mode column missing — skipping.")
else:
    print(f"Current column: {i_col}")
    print(f"Mission column: {mission_col}")
    print()
    print(f"{'battery':45s} {'mean I (all dis)':>16s} {'mean I (mission=0)':>18s} "
          f"{'mean I (mission=1)':>18s} {'n reg':>8s}")
    print("-" * 115)

    for batt_id, grp in data.groupby("battery_id"):
        dis = grp[grp[mode_col] == -1]
        i_all = pd.to_numeric(dis[i_col], errors="coerce").dropna()
        mean_all = i_all.mean() if len(i_all) else np.nan

        if mission_col:
            m0 = dis[dis[mission_col] == 0]
            m1 = dis[dis[mission_col] == 1]
            i0 = pd.to_numeric(m0[i_col], errors="coerce").dropna()
            i1 = pd.to_numeric(m1[i_col], errors="coerce").dropna()
            mean0 = i0.mean() if len(i0) else np.nan
            mean1 = i1.mean() if len(i1) else np.nan
            n1 = len(m1)
        else:
            mean0 = mean1 = np.nan
            n1 = 0

        print(f"{batt_id:45s} {mean_all:16.3f} {mean0:18.3f} {mean1:18.3f} {n1:8d}")

# ============================================================================
# 6. DOCUMENTED-LEVEL SNAPPING (regular-mission only)
# ============================================================================
section("6. DOCUMENTED-LEVEL SNAPPING (regular-mission only)")

if i_col is None or mission_col is None or mode_col is None:
    print("Required columns missing — skipping.")
else:
    print(f"Documented levels: {DOCUMENTED_DISCHARGE_LEVELS_A}")
    print()
    print(f"{'battery':45s} {'mean I (mission=1)':>18s} {'snapped level':>14s} {'deviation':>10s}")
    print("-" * 95)

    snapped_records = []
    for batt_id, grp in data.groupby("battery_id"):
        dis = grp[grp[mode_col] == -1]
        m1 = dis[dis[mission_col] == 1]
        i1 = pd.to_numeric(m1[i_col], errors="coerce").dropna()
        if len(i1) == 0:
            continue
        mean_i = i1.mean()
        snapped = min(DOCUMENTED_DISCHARGE_LEVELS_A, key=lambda L: abs(L - mean_i))
        dev = mean_i - snapped
        snapped_records.append({
            "battery_id": batt_id,
            "mean_i": mean_i,
            "snapped": snapped,
            "deviation": dev,
        })
        print(f"{batt_id:45s} {mean_i:18.3f} {snapped:14.2f} {dev:+10.3f}")

    if snapped_records:
        snap_df = pd.DataFrame(snapped_records)
        print(f"\nGroups formed by snapping (should be ~6-7):")
        grouped = snap_df.groupby("snapped")["battery_id"].count()
        for lvl, n in grouped.items():
            print(f"    {lvl:.2f} A → {n} pack(s)")
        print(f"\nMean packs per group: {grouped.mean():.2f}")
        print(f"Within-group mean deviations:")
        for lvl, grp in snap_df.groupby("snapped"):
            print(f"    {lvl:.2f} A: mean dev = {grp['deviation'].mean():+.3f} A, "
                  f"std = {grp['deviation'].std():.3f}")

# ============================================================================
# 7. PER-CYCLE CAPACITY (correctly computed with relative time)
# ============================================================================
section("7. PER-CYCLE CAPACITY (integrate |I| dt using relative_time)")

rel_time_col = next((c for c in data.columns if "relative" in c.lower() and "time" in c.lower()), None)
if rel_time_col is None:
    rel_time_col = next((c for c in data.columns if "time" in c.lower() and "start" not in c.lower()), None)
print(f"Time column used: {rel_time_col}")
print(f"Current column used: {i_col}")

if rel_time_col is None or i_col is None or mode_col is None:
    print("Required columns missing — skipping.")
else:
    cycle_records = []
    for batt_id, grp in data.groupby("battery_id"):
        g = grp.copy().reset_index(drop=True)
        g["_seg"] = (g[mode_col] != g[mode_col].shift()).cumsum()
        cyc_idx = 0
        for _, seg in g[g[mode_col] == -1].groupby("_seg"):
            cyc_idx += 1
            t = pd.to_numeric(seg[rel_time_col], errors="coerce").values.astype(float)
            i = np.abs(pd.to_numeric(seg[i_col], errors="coerce").values.astype(float))
            # Remove NaN rows first
            valid = np.isfinite(t) & np.isfinite(i)
            t_v, i_v = t[valid], i[valid]
            cap = np.trapz(i_v, t_v) / 3600.0 if len(t_v) > 1 else np.nan
            mt = seg[mission_col].iloc[0] if mission_col else np.nan
            cycle_records.append({
                "battery_id": batt_id, "cycle": cyc_idx,
                "capacity_Ah": cap, "mission_type": mt, "n_rows": len(seg),
            })
    cyc_df = pd.DataFrame(cycle_records)

    print(f"\nTotal discharge cycles: {len(cyc_df):,}")
    print(f"Cycles with non-NaN capacity: {cyc_df['capacity_Ah'].notna().sum():,}")
    print(f"Capacity NaN%: {cyc_df['capacity_Ah'].isna().mean()*100:.2f}%")
    print()
    print(f"{'battery':45s} {'n cycles':>10s} {'cap min':>10s} {'cap median':>12s} "
          f"{'cap max':>10s} {'cap std':>10s}")
    print("-" * 105)
    for batt_id, grp in cyc_df.groupby("battery_id"):
        caps = grp["capacity_Ah"].dropna()
        if len(caps) == 0:
            print(f"{batt_id:45s} {len(grp):10d} {'ALL NaN':>10s}")
            continue
        print(f"{batt_id:45s} {len(grp):10d} {caps.min():10.4f} {caps.median():12.4f} "
              f"{caps.max():10.4f} {caps.std():10.4f}")

# ============================================================================
# 8. REFERENCE-DISCHARGE CAPACITY TREND (SOH signal)
# ============================================================================
section("8. REFERENCE-DISCHARGE CAPACITY TREND (SOH signal)")

if 'cyc_df' not in dir() or mission_col is None:
    print("No per-cycle data available — skipping.")
else:
    ref_df = cyc_df[cyc_df["mission_type"] == 0].dropna(subset=["capacity_Ah"])
    print(f"Reference discharges (mission_type=0): {len(ref_df):,} cycles across "
          f"{ref_df['battery_id'].nunique()} packs")
    print()
    print(f"{'battery':45s} {'n ref':>8s} {'first cap':>12s} {'last cap':>12s} "
          f"{'first/last':>12s}")
    print("-" * 100)
    for batt_id, grp in ref_df.groupby("battery_id"):
        g = grp.sort_values("cycle")
        caps = g["capacity_Ah"].values
        if len(caps) < 2:
            continue
        first = caps[0]
        last = caps[-1]
        ratio = first / last if last > 0 else np.nan
        print(f"{batt_id:45s} {len(caps):8d} {first:12.4f} {last:12.4f} {ratio:12.3f}")

# ============================================================================
# 9. SANITY CHECKS
# ============================================================================
section("9. SANITY CHECKS")

if mode_col is not None:
    print("Mode column consistency check:")
    print(f"    Unique mode values: {sorted(data[mode_col].dropna().unique())}")
    unexpected = set(data[mode_col].dropna().unique()) - {-1, 0, 1}
    print(f"    Unexpected values:  {sorted(unexpected) if unexpected else 'none'}")
    print()

    print("Load-side columns populated ONLY during discharge (mode=-1)?")
    for c in [i_col, "voltage_load", "temperature_mosfet", "temperature_resistor", "mission_type"]:
        if c and c in data.columns:
            dis = data[data[mode_col] == -1]
            rest = data[data[mode_col] == 0]
            chg = data[data[mode_col] == 1]
            print(f"    {c:30s}  dis NaN%: {dis[c].isna().mean()*100:6.2f}  "
                  f"rest NaN%: {rest[c].isna().mean()*100:6.2f}  "
                  f"chg NaN%: {chg[c].isna().mean()*100:6.2f}")
    print()

# ============================================================================
# 10. ASSUMPTION VERDICT TABLE
# ============================================================================
section("10. ASSUMPTION VERDICT TABLE for quality_score.py")

print("Each row is an assumption baked into quality_score.py. Verdict is based on")
print("the diagnostics above.\n")

verdicts = []

# 1. relative_time is the correct time column
rel_time_ok = rel_time_col is not None and "relative" in rel_time_col.lower()
verdicts.append((
    "col_time = relative_time (not start_time)",
    "CONFIRMED" if rel_time_ok else "CONTRADICTED",
    f"detected '{rel_time_col}'"
))

# 2. mode column is exactly {-1, 0, 1}
mode_ok = set(data[mode_col].dropna().unique()) == {-1, 0, 1} if mode_col else False
verdicts.append((
    "mode column values are exactly {-1, 0, 1}",
    "CONFIRMED" if mode_ok else "CONTRADICTED",
    f"observed {sorted(data[mode_col].dropna().unique()) if mode_col else 'n/a'}"
))

# 3. Load current is populated during discharge
if i_col and mode_col:
    dis = data[data[mode_col] == -1]
    i_nan = dis[i_col].isna().mean() * 100
    verdicts.append((
        "current column non-NaN during discharge",
        "CONFIRMED" if i_nan < 1 else "CONTRADICTED",
        f"{i_nan:.2f}% NaN during discharge"
    ))

# 4. Per-cycle capacity computes non-NaN
if 'cyc_df' in dir():
    cap_nan = cyc_df["capacity_Ah"].isna().mean() * 100
    verdicts.append((
        "per-cycle capacity is non-NaN for most cycles",
        "CONFIRMED" if cap_nan < 5 else "CONTRADICTED",
        f"{cap_nan:.2f}% NaN capacity"
    ))

# 5. Documented-level snapping produces ~6-7 groups
if 'snap_df' in dir():
    n_groups = snap_df["snapped"].nunique()
    verdicts.append((
        "replicate snapping produces 6-7 groups",
        "CONFIRMED" if n_groups >= 6 else "CONTRADICTED",
        f"{n_groups} groups formed"
    ))

# 6. Reference-discharge capacity trend is downward
if 'ref_df' in dir() and len(ref_df) > 0:
    descending_packs = 0
    total_packs = 0
    for batt_id, grp in ref_df.groupby("battery_id"):
        g = grp.sort_values("cycle")
        caps = g["capacity_Ah"].values
        if len(caps) >= 5:
            total_packs += 1
            # Simple check: last third mean < first third mean
            third = max(1, len(caps) // 3)
            if caps[-third:].mean() < caps[:third].mean():
                descending_packs += 1
    pct = descending_packs / total_packs * 100 if total_packs else 0
    verdicts.append((
        "reference-discharge capacity trends downward (degradation visible)",
        "CONFIRMED" if pct >= 80 else "CONTRADICTED",
        f"{pct:.0f}% of packs show downward trend"
    ))

print(f"{'assumption':60s} {'verdict':>14s}   evidence")
print("-" * 130)
for assumption, verdict, evidence in verdicts:
    print(f"{assumption:60s} {verdict:>14s}   {evidence}")

section("END OF EXPLORATION")

# Flush and restore
sys.stdout = _orig_stdout
_log_fh.close()
print(f"\nFull report written to: {OUT_FILE}")