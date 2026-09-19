"""
CALCE Battery Dataset — Quality Assessment

Rule-based assessment following the six-dimension framework:

1. Correctness
2. Completeness
3. Anomaly and noise control
4. Representativeness and diversity
5. Distribution balance
6. Temporal coherence

Data-level metrics are computed dynamically from the CALCE data.

Metadata-dependent aspects are explicitly hard-coded from the
dataset metadata and are NOT inferred from filenames.

Outputs are saved to:
    <dataset folder>/quality_results/

Generated files:
    - calce_quality_scores.csv
    - calce_quality_report.txt
    - calce_quality_metrics.json
    - calce_quality_datasheet.txt
    - calce_quality_scorecard.png
"""

import os
import glob
import json
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")


# ============================================================================
# CONFIGURATION
# ============================================================================

DATASET_PATH = (
    r"C:\Users\admin\Desktop\DR2\11 All Datasets"
    r"\03 CALCE Battery Dataset\dataset calce"
)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "quality_results"
)

os.makedirs(RESULTS_DIR, exist_ok=True)

SAVE_PLOT = True


# ============================================================================
# CALCE METADATA
# ============================================================================
# These values come from metadata.txt and are intentionally hard-coded.
# They must not be inferred from filenames or dynamically detected.

METADATA = {
    "dataset_name": "CALCE_CX2-16_prism_LCO",
    "battery_type": "Prismatic",
    "chemistry": "LCO",
    "temperature_conditions": 1,
    "temperature_values_C": [25],
    "dod_conditions": 1,
    "dod_description": "0-100% full cycling",
    "charge_c_rates": [0.5],
    "discharge_c_rates": [0.5],
    "replicate_cells": 1,
    "calendar_aging": False,
    "dynamic_load_profiles": False,
    "real_world_operation": False,
    "protocol_documented": True,
    "nominal_capacity_Ah": 1.35,
}

# Physical plausibility bounds based on the CALCE LCO dataset metadata.
CELL_V_MIN = 2.5
CELL_V_MAX = 4.2
TEMP_MIN = -20
TEMP_MAX = 60

# ============================================================================
# SCORE DEFINITIONS
# ============================================================================

SCORE_LABELS = {
    "++": "Comprehensive coverage",
    "+": "Mostly satisfied",
    "o": "Partially satisfied",
    "--": "Absent",
}

SCORE_ORDER = ["++", "+", "o", "--"]

SCORE_COLORS = {
    "++": "#2ca02c",
    "+": "#98df8a",
    "o": "#ffbb78",
    "--": "#d62728",
}


# ============================================================================
# HELPERS
# ============================================================================

def score_line(criterion, aspect, score, finding):
    """Store and print one assessment result."""
    print("\n" + "-" * 80)
    print(f"Criterion : {criterion}")
    print(f"Aspect    : {aspect}")
    print(f"Score     : {score} ({SCORE_LABELS[score]})")
    print(f"Finding   : {finding}")

    return {
        "criterion": criterion,
        "aspect": aspect,
        "score": score,
        "finding": finding,
    }


def clean_column_names(df):
    """Normalize CALCE column names."""
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.replace(" ", "_")
    )

    mapping = {}

    for col in df.columns:
        low = col.lower()

        if "cycle" in low and (
            "index" in low or
            "number" in low
        ):
            mapping[col] = "Cycle_Index"

        elif "discharge" in low and "capacity" in low:
            mapping[col] = "Discharge_Capacity_Ah"

        elif "charge" in low and "capacity" in low:
            mapping[col] = "Charge_Capacity_Ah"

        elif "min" in low and "voltage" in low:
            mapping[col] = "Min_Voltage_V"

        elif "max" in low and "voltage" in low:
            mapping[col] = "Max_Voltage_V"

        elif "min" in low and "current" in low:
            mapping[col] = "Min_Current_A"

        elif "max" in low and "current" in low:
            mapping[col] = "Max_Current_A"

        elif "charge" in low and "energy" in low:
            mapping[col] = "Charge_Energy_Wh"

        elif "discharge" in low and "energy" in low:
            mapping[col] = "Discharge_Energy_Wh"

    return df.rename(columns=mapping)


def detect_column(df, candidates):
    """Return first available column from a list of candidates."""
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def classify_percentage(value):
    """
    Generic percentage threshold used for:
    - physical plausibility
    - missing values
    - statistical outliers
    - unexpected signal changes
    """
    if value < 0.1:
        return "++"
    elif value < 1:
        return "+"
    elif value <= 5:
        return "o"
    else:
        return "--"


def classify_noise(value):
    """
    Measurement-noise thresholds from the framework.

    Noise is expressed as percentage of nominal capacity.
    """
    if value <= 0.5:
        return "++"
    elif value <= 1:
        return "+"
    elif value <= 2:
        return "o"
    else:
        return "--"


# ============================================================================
# LOAD DATA
# ============================================================================

print("=" * 80)
print("CALCE BATTERY DATASET — QUALITY ASSESSMENT")
print("=" * 80)

print("\nDataset:")
print(METADATA["dataset_name"])

print("\nLoading CALCE data...")

all_csv_files = sorted(
    glob.glob(os.path.join(DATASET_PATH, "*.csv"))
)

cycle_files = [
    f for f in all_csv_files
    if "cycle_data" in os.path.basename(f).lower()
    or "cycle" in os.path.basename(f).lower()
]

timeseries_files = [
    f for f in all_csv_files
    if "timeseries" in os.path.basename(f).lower()
]

print(f"Cycle files      : {len(cycle_files)}")
print(f"Timeseries files : {len(timeseries_files)}")


# ============================================================================
# READ CYCLE DATA
# ============================================================================

cycle_frames = []

for filepath in cycle_files:

    filename = os.path.basename(filepath)

    try:
        df = pd.read_csv(filepath, low_memory=False)
        df = clean_column_names(df)

        df["file_source"] = filename

        # CALCE metadata refers to the selected CX2-16 dataset.
        # We do not infer chemistry, temperature, C-rate, etc.
        # from the filename.

        df["cell_id"] = "CX2-16"

        cycle_frames.append(df)

        print(f"Loaded: {filename} -> {len(df):,} rows")

    except Exception as exc:
        print(f"WARNING: Could not load {filename}: {exc}")


if not cycle_frames:
    raise RuntimeError(
        "No CALCE cycle-data CSV files were found."
    )


cycle_data = pd.concat(
    cycle_frames,
    ignore_index=True
)

print(
    f"\nTotal records loaded: "
    f"{len(cycle_data):,}"
)


# ============================================================================
# COLUMN DETECTION
# ============================================================================

cycle_index_col = detect_column(
    cycle_data,
    ["Cycle_Index"]
)

discharge_capacity_col = detect_column(
    cycle_data,
    ["Discharge_Capacity_Ah"]
)

charge_capacity_col = detect_column(
    cycle_data,
    ["Charge_Capacity_Ah"]
)

min_voltage_col = detect_column(
    cycle_data,
    ["Min_Voltage_V"]
)

max_voltage_col = detect_column(
    cycle_data,
    ["Max_Voltage_V"]
)

min_current_col = detect_column(
    cycle_data,
    ["Min_Current_A"]
)

max_current_col = detect_column(
    cycle_data,
    ["Max_Current_A"]
)

print("\nDetected columns:")
print(f"  Cycle index        : {cycle_index_col}")
print(f"  Discharge capacity : {discharge_capacity_col}")
print(f"  Charge capacity    : {charge_capacity_col}")
print(f"  Minimum voltage    : {min_voltage_col}")
print(f"  Maximum voltage    : {max_voltage_col}")
print(f"  Minimum current    : {min_current_col}")
print(f"  Maximum current    : {max_current_col}")


# ============================================================================
# BASIC DATA STATISTICS
# ============================================================================

total_rows = len(cycle_data)
total_columns = len(cycle_data.columns)
total_cells = cycle_data["cell_id"].nunique()

print("\nBasic dataset statistics:")
print(f"  Rows       : {total_rows:,}")
print(f"  Columns    : {total_columns}")
print(f"  Cells      : {total_cells}")


# ============================================================================
# 1. CORRECTNESS
# ============================================================================

results = []

print("\n" + "=" * 80)
print("1. CORRECTNESS")
print("=" * 80)


# ----------------------------------------------------------------------------
# 1a. Physical plausibility
# ----------------------------------------------------------------------------

physical_invalid_counts = 0
physical_total_points = 0

# Voltage
voltage_invalid = 0
voltage_total = 0

if min_voltage_col:
    values = cycle_data[min_voltage_col].dropna()

    voltage_total += len(values)

    voltage_invalid += (
        (values < CELL_V_MIN) |
        (values > CELL_V_MAX)
    ).sum()

if max_voltage_col:
    values = cycle_data[max_voltage_col].dropna()

    voltage_total += len(values)

    voltage_invalid += (
        (values < CELL_V_MIN) |
        (values > CELL_V_MAX)
    ).sum()

# Capacity
capacity_invalid = 0
capacity_total = 0

if discharge_capacity_col:

    values = cycle_data[
        discharge_capacity_col
    ].dropna()

    capacity_total = len(values)

    capacity_invalid = (
        (values <= 0) |
        (values > METADATA["nominal_capacity_Ah"] * 1.5)
    ).sum()

physical_invalid_counts = (
    voltage_invalid +
    capacity_invalid
)

physical_total_points = (
    voltage_total +
    capacity_total
)

physical_invalid_pct = (
    100 * physical_invalid_counts /
    physical_total_points
    if physical_total_points > 0
    else 0
)

score_physical = classify_percentage(
    physical_invalid_pct
)

finding_physical = (
    f"{physical_invalid_counts:,}/{physical_total_points:,} "
    f"checked measurements ({physical_invalid_pct:.4f}%) "
    f"fall outside the predefined physical plausibility "
    f"bounds. Voltage bounds: {CELL_V_MIN}-{CELL_V_MAX} V; "
    f"capacity upper bound: "
    f"{METADATA['nominal_capacity_Ah'] * 1.5:.2f} Ah."
)

results.append(
    score_line(
        "Correctness",
        "Physical plausibility",
        score_physical,
        finding_physical
    )
)


# ----------------------------------------------------------------------------
# 1b. Current sign convention
# ----------------------------------------------------------------------------

if min_current_col and max_current_col:

    min_current = cycle_data[
        min_current_col
    ].dropna()

    max_current = cycle_data[
        max_current_col
    ].dropna()

    has_negative_current = (
        (min_current < 0).any()
    )

    has_positive_current = (
        (max_current > 0).any()
    )

    sign_verified = (
        has_negative_current and
        has_positive_current
    )

    if sign_verified:
        sign_score = "++"
        sign_finding = (
            "Current measurements contain both negative and "
            "positive values, consistent with the documented "
            "charge/discharge sign convention."
        )
    else:
        sign_score = "o"
        sign_finding = (
            "Current columns are available, but the expected "
            "positive/negative sign behavior could not be "
            "fully verified."
        )

else:

    sign_score = "--"

    sign_finding = (
        "Current sign convention cannot be verified because "
        "the required current columns are absent from the "
        "analyzed cycle-level data."
    )

results.append(
    score_line(
        "Correctness",
        "Current sign convention",
        sign_score,
        sign_finding
    )
)


# ============================================================================
# 2. COMPLETENESS
# ============================================================================

print("\n" + "=" * 80)
print("2. COMPLETENESS")
print("=" * 80)


# ----------------------------------------------------------------------------
# 2a. Missing values
# ----------------------------------------------------------------------------

total_values = (
    cycle_data.shape[0] *
    cycle_data.shape[1]
)

missing_values = (
    cycle_data.isna().sum().sum()
)

missing_pct = (
    100 * missing_values /
    total_values
    if total_values > 0
    else 0
)

missing_score = classify_percentage(
    missing_pct
)

columns_with_missing = (
    cycle_data.columns[
        cycle_data.isna().any()
    ].tolist()
)

finding_missing = (
    f"{missing_values:,}/{total_values:,} values are missing "
    f"({missing_pct:.4f}%). Columns containing missing values: "
    f"{columns_with_missing if columns_with_missing else 'none'}."
)

results.append(
    score_line(
        "Completeness",
        "Missing values",
        missing_score,
        finding_missing
    )
)


# ----------------------------------------------------------------------------
# 2b. Temporal continuity
# ----------------------------------------------------------------------------

if cycle_index_col:

    continuity_gaps = 0
    continuity_expected = 0

    for cell_id, group in cycle_data.groupby("cell_id"):

        cycles = (
            pd.to_numeric(
                group[cycle_index_col],
                errors="coerce"
            )
            .dropna()
            .sort_values()
            .values
        )

        if len(cycles) > 1:

            differences = np.diff(cycles)

            continuity_expected += len(differences)

            continuity_gaps += (
                differences > 1
            ).sum()

    continuity_gap_pct = (
        100 * continuity_gaps /
        continuity_expected
        if continuity_expected > 0
        else 0
    )

    continuity_score = classify_percentage(
        continuity_gap_pct
    )

    finding_continuity = (
        f"{continuity_gaps:,} gaps were detected among "
        f"{continuity_expected:,} consecutive cycle transitions "
        f"({continuity_gap_pct:.4f}% affected transitions)."
    )

else:

    continuity_score = "--"

    finding_continuity = (
        "Cycle index is unavailable; temporal continuity "
        "cannot be verified."
    )

results.append(
    score_line(
        "Completeness",
        "Temporal continuity",
        continuity_score,
        finding_continuity
    )
)


# ----------------------------------------------------------------------------
# 2c. Test protocol documentation
# ----------------------------------------------------------------------------

if METADATA["protocol_documented"]:

    documentation_score = "++"

    documentation_finding = (
        "Essential test-protocol information is documented in "
        "the dataset metadata, including battery chemistry, "
        "nominal capacity, temperature condition, DoD, and "
        "charge/discharge C-rates."
    )

else:

    documentation_score = "--"

    documentation_finding = (
        "Essential experimental protocol information is not "
        "documented in the available metadata."
    )

results.append(
    score_line(
        "Completeness",
        "Test protocol documentation",
        documentation_score,
        documentation_finding
    )
)


# ============================================================================
# 3. ANOMALY AND NOISE CONTROL
# ============================================================================

print("\n" + "=" * 80)
print("3. ANOMALY AND NOISE CONTROL")
print("=" * 80)


# ----------------------------------------------------------------------------
# 3a. Statistical outliers
# ----------------------------------------------------------------------------

if discharge_capacity_col:

    capacity = (
        cycle_data[
            discharge_capacity_col
        ]
        .dropna()
    )

    capacity = capacity[
        (capacity > 0) &
        (capacity < 2.0)
    ]

    if len(capacity) > 0:

        q1 = capacity.quantile(0.25)
        q3 = capacity.quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 3 * iqr
        upper = q3 + 3 * iqr

        outliers = (
            (capacity < lower) |
            (capacity > upper)
        )

        outlier_count = outliers.sum()

        outlier_pct = (
            100 * outlier_count /
            len(capacity)
        )

        outlier_score = classify_percentage(
            outlier_pct
        )

        outlier_finding = (
            f"Using a 3×IQR rule, {outlier_count:,}/"
            f"{len(capacity):,} discharge-capacity values "
            f"({outlier_pct:.4f}%) are statistical outliers. "
            f"IQR bounds: [{lower:.4f}, {upper:.4f}] Ah."
        )

    else:

        outlier_score = "--"

        outlier_finding = (
            "Insufficient valid discharge-capacity values "
            "for statistical outlier analysis."
        )

else:

    outlier_score = "--"

    outlier_finding = (
        "Discharge-capacity data are unavailable for "
        "statistical outlier analysis."
    )

results.append(
    score_line(
        "Anomaly and noise control",
        "Statistical outliers",
        outlier_score,
        outlier_finding
    )
)


# ----------------------------------------------------------------------------
# 3b. Unexpected signal changes
# ----------------------------------------------------------------------------
# Capacity is used as the primary degradation-related signal.
# A large cycle-to-cycle change is treated as an unexpected
# signal change. A 5% relative change threshold is used to
# identify abrupt changes.

if discharge_capacity_col and cycle_index_col:

    unexpected_changes = 0
    total_transitions = 0

    for cell_id, group in cycle_data.groupby("cell_id"):

        group = group.sort_values(
            cycle_index_col
        )

        capacities = pd.to_numeric(
            group[discharge_capacity_col],
            errors="coerce"
        ).dropna()

        if len(capacities) > 1:

            relative_changes = (
                np.abs(
                    np.diff(capacities)
                ) /
                np.maximum(
                    np.abs(capacities[:-1]),
                    1e-9
                )
            )

            total_transitions += len(
                relative_changes
            )

            unexpected_changes += (
                relative_changes > 0.05
            ).sum()

    unexpected_pct = (
        100 * unexpected_changes /
        total_transitions
        if total_transitions > 0
        else 0
    )

    unexpected_score = classify_percentage(
        unexpected_pct
    )

    unexpected_finding = (
        f"{unexpected_changes:,}/{total_transitions:,} "
        f"cycle-to-cycle capacity transitions "
        f"({unexpected_pct:.4f}%) exceed a 5% relative change."
    )

else:

    unexpected_score = "--"

    unexpected_finding = (
        "Insufficient cycle and capacity information to "
        "assess unexpected signal changes."
    )

results.append(
    score_line(
        "Anomaly and noise control",
        "Unexpected signal changes",
        unexpected_score,
        unexpected_finding
    )
)


# ----------------------------------------------------------------------------
# 3c. Measurement noise
# ----------------------------------------------------------------------------

capacity_differences = []

if discharge_capacity_col and cycle_index_col:

    for cell_id, group in cycle_data.groupby("cell_id"):

        group = group.sort_values(
            cycle_index_col
        )

        capacities = pd.to_numeric(
            group[discharge_capacity_col],
            errors="coerce"
        ).dropna().values

        if len(capacities) > 1:

            differences = np.abs(
                np.diff(capacities)
            )

            capacity_differences.extend(
                differences.tolist()
            )

if capacity_differences:

    cap_diffs = np.asarray(
        capacity_differences
    )

    # Remove the largest 5% to reduce the influence
    # of degradation jumps when estimating local variation.
    threshold = np.percentile(
        cap_diffs,
        95
    )

    noise_values = cap_diffs[
        cap_diffs <= threshold
    ]

    mean_noise_Ah = np.mean(
        noise_values
    )

    relative_noise_pct = (
        100 *
        mean_noise_Ah /
        METADATA["nominal_capacity_Ah"]
    )

    noise_score = classify_noise(
        relative_noise_pct
    )

    noise_finding = (
        f"Mean absolute cycle-to-cycle capacity variation "
        f"after excluding the upper 5% of transitions is "
        f"{mean_noise_Ah:.6f} Ah, corresponding to "
        f"{relative_noise_pct:.4f}% of nominal capacity."
    )

else:

    noise_score = "--"

    noise_finding = (
        "Capacity measurements are insufficient to estimate "
        "cycle-to-cycle measurement noise."
    )

results.append(
    score_line(
        "Anomaly and noise control",
        "Measurement noise",
        noise_score,
        noise_finding
    )
)


# ============================================================================
# 4. REPRESENTATIVENESS AND DIVERSITY
# ============================================================================
# These aspects are metadata-derived.
# They are deliberately NOT calculated from the raw data.

print("\n" + "=" * 80)
print("4. REPRESENTATIVENESS AND DIVERSITY")
print("=" * 80)


# ----------------------------------------------------------------------------
# 4a. Chemistry diversity
# ----------------------------------------------------------------------------

chemistry_score = "--"

chemistry_finding = (
    f"One documented chemistry is represented: "
    f"{METADATA['chemistry']}. No chemistry variation is "
    f"documented."
)

results.append(
    score_line(
        "Representativeness and diversity",
        "Chemistry diversity",
        chemistry_score,
        chemistry_finding
    )
)


# ----------------------------------------------------------------------------
# 4b. Temperature conditions
# ----------------------------------------------------------------------------

if METADATA["temperature_conditions"] >= 3:
    temperature_score = "++"
elif METADATA["temperature_conditions"] == 2:
    temperature_score = "+"
elif METADATA["temperature_conditions"] == 1:
    temperature_score = "o"
else:
    temperature_score = "--"

temperature_finding = (
    f"{METADATA['temperature_conditions']} documented "
    f"temperature condition(s): "
    f"{METADATA['temperature_values_C']} °C."
)

results.append(
    score_line(
        "Representativeness and diversity",
        "Temperature conditions",
        temperature_score,
        temperature_finding
    )
)


# ----------------------------------------------------------------------------
# 4c. DoD diversity
# ----------------------------------------------------------------------------

if METADATA["dod_conditions"] >= 3:
    dod_score = "++"
elif METADATA["dod_conditions"] == 2:
    dod_score = "+"
elif METADATA["dod_conditions"] == 1:
    dod_score = "--"
else:
    dod_score = "--"

dod_finding = (
    f"One documented DoD protocol is used: "
    f"{METADATA['dod_description']}."
)

results.append(
    score_line(
        "Representativeness and diversity",
        "DoD diversity",
        dod_score,
        dod_finding
    )
)


# ----------------------------------------------------------------------------
# 4d. C-rate diversity
# ----------------------------------------------------------------------------

number_of_c_rates = len(
    set(
        METADATA["charge_c_rates"] +
        METADATA["discharge_c_rates"]
    )
)

if number_of_c_rates >= 3:
    crate_score = "++"
elif number_of_c_rates == 2:
    crate_score = "+"
elif number_of_c_rates == 1:
    crate_score = "--"
else:
    crate_score = "--"

crate_finding = (
    f"Documented charge C-rate(s): "
    f"{METADATA['charge_c_rates']}C; "
    f"discharge C-rate(s): "
    f"{METADATA['discharge_c_rates']}C. "
    f"No C-rate diversity is documented."
)

results.append(
    score_line(
        "Representativeness and diversity",
        "C-rate diversity",
        crate_score,
        crate_finding
    )
)


# ----------------------------------------------------------------------------
# 4e. Replicate cells/vehicles
# ----------------------------------------------------------------------------

n_replicates = METADATA["replicate_cells"]

if n_replicates >= 10:
    replicate_score = "++"
elif n_replicates >= 5:
    replicate_score = "+"
elif n_replicates >= 2:
    replicate_score = "o"
else:
    replicate_score = "--"

replicate_finding = (
    f"{n_replicates} cell is represented in the selected "
    f"CALCE assessment subset."
)

results.append(
    score_line(
        "Representativeness and diversity",
        "Replicate cells/vehicles",
        replicate_score,
        replicate_finding
    )
)


# ----------------------------------------------------------------------------
# 4f. Calendar aging
# ----------------------------------------------------------------------------

calendar_score = (
    "o"
    if METADATA["calendar_aging"]
    else "--"
)

calendar_finding = (
    "A dedicated calendar-aging protocol is documented."
    if METADATA["calendar_aging"]
    else
    "No dedicated calendar-aging experiment is documented; "
    "the dataset focuses on cyclic laboratory aging."
)

results.append(
    score_line(
        "Representativeness and diversity",
        "Calendar aging",
        calendar_score,
        calendar_finding
    )
)


# ----------------------------------------------------------------------------
# 4g. Dynamic load profiles
# ----------------------------------------------------------------------------

dynamic_score = (
    "++"
    if METADATA["dynamic_load_profiles"]
    else "--"
)

dynamic_finding = (
    "Dynamic load profiles are documented."
    if METADATA["dynamic_load_profiles"]
    else
    "The selected CALCE dataset uses a fixed laboratory "
    "cycling protocol; no representative dynamic load profile "
    "is documented."
)

results.append(
    score_line(
        "Representativeness and diversity",
        "Dynamic load profiles",
        dynamic_score,
        dynamic_finding
    )
)


# ----------------------------------------------------------------------------
# 4h. Real-world operation
# ----------------------------------------------------------------------------

real_world_score = (
    "++"
    if METADATA["real_world_operation"]
    else "--"
)

real_world_finding = (
    "Real-world operation is represented."
    if METADATA["real_world_operation"]
    else
    "The dataset consists of laboratory cycling and does not "
    "contain real-world vehicle operation."
)

results.append(
    score_line(
        "Representativeness and diversity",
        "Real-world operation",
        real_world_score,
        real_world_finding
    )
)


# ============================================================================
# SOH COMPUTATION
# ============================================================================

print("\n" + "=" * 80)
print("SOH ANALYSIS")
print("=" * 80)

soh_df = pd.DataFrame()

if discharge_capacity_col and cycle_index_col:

    working = cycle_data[
        ["
        cell_id",
        cycle_index_col,
        discharge_capacity_col
        ]
    ].copy()

    working[discharge_capacity_col] = pd.to_numeric(
        working[discharge_capacity_col],
        errors="coerce"
    )

    working = working.dropna(
        subset=[
            cycle_index_col,
            discharge_capacity_col
        ]
    )

    working = working[
        working[discharge_capacity_col] > 0
    ]

    working = working.sort_values(
        ["
        cell_id",
        cycle_index_col
        ]
    )

    soh_records = []

    for cell_id, group in working.groupby(
        "cell_id"
    ):

        group = group.copy()

        initial_values = group[
            discharge_capacity_col
        ].head(3)

        if len(initial_values) == 0:
            continue

        initial_capacity = (
            initial_values.median()
        )

        group["SOH"] = (
            group[discharge_capacity_col] /
            initial_capacity
        )

        group["SOH"] = group["SOH"].clip(
            lower=0,
            upper=1.2
        )

        soh_records.append(
            group[
                [
                    "cell_id",
                    cycle_index_col,
                    discharge_capacity_col,
                    "SOH",
                ]
            ]
        )

    if soh_records:

        soh_df = pd.concat(
            soh_records,
            ignore_index=True
        )

        print(
            f"SOH computed for "
            f"{soh_df['cell_id'].nunique()} cell(s), "
            f"{len(soh_df):,} cycles."
        )

else:

    print(
        "SOH cannot be computed because cycle index or "
        "discharge capacity is unavailable."
    )


# ============================================================================
# 5. DISTRIBUTION BALANCE
# ============================================================================

print("\n" + "=" * 80)
print("5. DISTRIBUTION BALANCE")
print("=" * 80)


# ----------------------------------------------------------------------------
# 5a. SOH range coverage
# ----------------------------------------------------------------------------

if not soh_df.empty:

    cells_with_low_soh = (
        soh_df
        .groupby("cell_id")["SOH"]
        .min()
        .lt(0.80)
    )

    low_soh_cell_count = (
        cells_with_low_soh.sum()
    )

    number_of_soh_cells = (
        len(cells_with_low_soh)
    )

    low_soh_cell_pct = (
        100 *
        low_soh_cell_count /
        number_of_soh_cells
        if number_of_soh_cells > 0
        else 0
    )

    soh_min = soh_df["SOH"].min()
    soh_max = soh_df["SOH"].max()

    if low_soh_cell_pct >= 20:
        soh_range_score = "++"
    elif low_soh_cell_pct >= 10:
        soh_range_score = "+"
    elif low_soh_cell_pct > 0:
        soh_range_score = "o"
    else:
        soh_range_score = "--"

    soh_range_finding = (
        f"SOH range: {soh_min:.3f}-{soh_max:.3f}. "
        f"{low_soh_cell_count}/{number_of_soh_cells} cells "
        f"({low_soh_cell_pct:.1f}%) reach SOH < 80%."
    )

else:

    soh_range_score = "--"

    soh_range_finding = (
        "SOH could not be computed from the available cycle data."
    )

results.append(
    score_line(
        "Distribution balance",
        "SOH range coverage",
        soh_range_score,
        soh_range_finding
    )
)


# ----------------------------------------------------------------------------
# 5b. SOH distribution balance
# ----------------------------------------------------------------------------

if not soh_df.empty:

    bins = [
        0.0,
        0.70,
        0.80,
        0.90,
        0.95,
        1.05
    ]

    labels = [
        "<70%",
        "70-80%",
        "80-90%",
        "90-95%",
        "95-105%"
    ]

    soh_bins = pd.cut(
        soh_df["SOH"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    distribution = (
        soh_bins
        .value_counts(normalize=True)
        .sort_index()
        * 100
    )

    dominant_bin_pct = (
        distribution.max()
        if len(distribution) > 0
        else 0
    )

    if dominant_bin_pct <= 50:
        soh_balance_score = "++"
    elif dominant_bin_pct <= 70:
        soh_balance_score = "+"
    else:
        soh_balance_score = "o"

    soh_balance_finding = (
        f"Dominant SOH bin contains "
        f"{dominant_bin_pct:.2f}% of valid SOH observations. "
        f"Distribution: "
        f"{distribution.round(2).to_dict()}."
    )

else:

    soh_balance_score = "--"

    soh_balance_finding = (
        "SOH distribution could not be assessed."
    )

results.append(
    score_line(
        "Distribution balance",
        "SOH distribution balance",
        soh_balance_score,
        soh_balance_finding
    )
)


# ----------------------------------------------------------------------------
# 5c. Cycle contribution per cell
# ----------------------------------------------------------------------------

if cycle_index_col:

    cycle_counts = (
        cycle_data
        .groupby("cell_id")[cycle_index_col]
        .nunique()
    )

    if len(cycle_counts) >= 2:

        cycle_cv = (
            cycle_counts.std(ddof=1) /
            cycle_counts.mean()
        )

        if cycle_cv <= 0.10:
            cycle_balance_score = "++"
        elif cycle_cv <= 0.25:
            cycle_balance_score = "+"
        elif cycle_cv <= 0.50:
            cycle_balance_score = "o"
        else:
            cycle_balance_score = "--"

        cycle_balance_finding = (
            f"Cycle contribution per cell: "
            f"min={cycle_counts.min()}, "
            f"max={cycle_counts.max()}, "
            f"CV={cycle_cv:.4f}."
        )

    else:

        cycle_balance_score = "--"

        cycle_balance_finding = (
            "Only one cell is represented; a coefficient of "
            "variation across cells cannot be computed."
        )

else:

    cycle_balance_score = "--"

    cycle_balance_finding = (
        "Cycle index is unavailable."
    )

results.append(
    score_line(
        "Distribution balance",
        "Cycle contribution per cell",
        cycle_balance_score,
        cycle_balance_finding
    )
)


# ============================================================================
# 6. TEMPORAL COHERENCE
# ============================================================================

print("\n" + "=" * 80)
print("6. TEMPORAL COHERENCE")
print("=" * 80)


# ----------------------------------------------------------------------------
# 6a. Monotonic temporal progression
# ----------------------------------------------------------------------------

if cycle_index_col:

    total_transitions = 0
    ordered_transitions = 0

    for cell_id, group in cycle_data.groupby(
        "cell_id"
    ):

        cycles = pd.to_numeric(
            group[cycle_index_col],
            errors="coerce"
        ).dropna().sort_values().values

        if len(cycles) > 1:

            differences = np.diff(cycles)

            total_transitions += len(
                differences
            )

            ordered_transitions += (
                differences >= 0
            ).sum()

    ordered_pct = (
        100 *
        ordered_transitions /
        total_transitions
        if total_transitions > 0
        else 0
    )

    if ordered_pct >= 99.9:
        monotonic_score = "++"
    elif ordered_pct >= 99:
        monotonic_score = "+"
    elif ordered_pct >= 95:
        monotonic_score = "o"
    else:
        monotonic_score = "--"

    monotonic_finding = (
        f"{ordered_pct:.4f}% of consecutive cycle transitions "
        f"are chronologically ordered."
    )

else:

    monotonic_score = "--"

    monotonic_finding = (
        "Cycle index is unavailable."
    )

results.append(
    score_line(
        "Temporal coherence",
        "Monotonic temporal progression",
        monotonic_score,
        monotonic_finding
    )
)


# ----------------------------------------------------------------------------
# 6b. Consistent degradation trend
# ----------------------------------------------------------------------------

if not soh_df.empty:

    consistent_cells = 0
    total_soh_cells = 0

    for cell_id, group in soh_df.groupby(
        "cell_id"
    ):

        group = group.sort_values(
            cycle_index_col
        )

        soh_values = group["SOH"].values

        if len(soh_values) < 2:
            continue

        total_soh_cells += 1

        # Allow small recovery fluctuations of up to 3%.
        recoveries = np.diff(
            soh_values
        ) > 0.03

        if not recoveries.any():
            consistent_cells += 1

    consistent_pct = (
        100 *
        consistent_cells /
        total_soh_cells
        if total_soh_cells > 0
        else 0
    )

    if consistent_pct >= 99:
        degradation_score = "++"
    elif consistent_pct >= 90:
        degradation_score = "+"
    elif consistent_pct >= 70:
        degradation_score = "o"
    else:
        degradation_score = "--"

    degradation_finding = (
        f"{consistent_cells}/{total_soh_cells} cells "
        f"({consistent_pct:.2f}%) show a consistent decreasing "
        f"SOH trend without recoveries greater than 3%."
    )

else:

    degradation_score = "--"

    degradation_finding = (
        "SOH could not be computed; degradation trend cannot "
        "be assessed."
    )

results.append(
    score_line(
        "Temporal coherence",
        "Consistent degradation trend",
        degradation_score,
        degradation_finding
    )
)


# ----------------------------------------------------------------------------
# 6c. Cycle index consistency
# ----------------------------------------------------------------------------

if cycle_index_col:

    duplicate_cycle_counts = 0
    total_cycle_entries = 0

    non_integer_counts = 0

    for cell_id, group in cycle_data.groupby(
        "cell_id"
    ):

        cycles = pd.to_numeric(
            group[cycle_index_col],
            errors="coerce"
        ).dropna()

        total_cycle_entries += len(cycles)

        duplicate_cycle_counts += (
            cycles.duplicated()
        ).sum()

        non_integer_counts += (
            np.abs(cycles - np.round(cycles)) > 1e-9
        ).sum()

    inconsistent_entries = (
        duplicate_cycle_counts +
        non_integer_counts
    )

    consistency_pct = (
        100 *
        (
            total_cycle_entries -
            inconsistent_entries
        ) /
        total_cycle_entries
        if total_cycle_entries > 0
        else 0
    )

    if consistency_pct >= 99.9:
        consistency_score = "++"
    elif consistency_pct >= 99:
        consistency_score = "+"
    elif consistency_pct >= 95:
        consistency_score = "o"
    else:
        consistency_score = "--"

    consistency_finding = (
        f"Cycle-index consistency is "
        f"{consistency_pct:.4f}%. "
        f"Duplicate entries: {duplicate_cycle_counts:,}; "
        f"non-integer indices: {non_integer_counts:,}."
    )

else:

    consistency_score = "--"

    consistency_finding = (
        "Cycle index is unavailable."
    )

results.append(
    score_line(
        "Temporal coherence",
        "Cycle index consistency",
        consistency_score,
        consistency_finding
    )
)


# ============================================================================
# RESULTS DATAFRAME
# ============================================================================

results_df = pd.DataFrame(results)


# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("FINAL CALCE QUALITY SCORECARD")
print("=" * 80)

print(
    results_df[
        ["criterion", "aspect", "score"]
    ].to_string(index=False)
)

print("\nScore distribution:")

score_counts = (
    results_df["score"]
    .value_counts()
)

for score in SCORE_ORDER:

    print(
        f"  {score:>2} : "
        f"{score_counts.get(score, 0)}"
    )


# ============================================================================
# SAVE CSV RESULTS
# ============================================================================

csv_path = os.path.join(
    RESULTS_DIR,
    "calce_quality_scores.csv"
)

results_df.to_csv(
    csv_path,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================================
# COMPUTE METRICS DICTIONARY
# ============================================================================

metrics = {
    "dataset": METADATA["dataset_name"],

    "dataset_statistics": {
        "rows": int(total_rows),
        "columns": int(total_columns),
        "cells": int(total_cells),
    },

    "correctness": {
        "physical_invalid_percentage":
            float(physical_invalid_pct),

        "voltage_invalid_percentage":
            float(
                100 * voltage_invalid /
                voltage_total
                if voltage_total > 0
                else 0
            ),
    },

    "completeness": {
        "missing_values":
            int(missing_values),

        "missing_percentage":
            float(missing_pct),

        "columns_with_missing":
            columns_with_missing,
    },

    "anomaly_and_noise_control": {
        "statistical_outlier_percentage":
            float(
                outlier_pct
                if "outlier_pct" in locals()
                else 0
            ),

        "unexpected_signal_change_percentage":
            float(
                unexpected_pct
                if "unexpected_pct" in locals()
                else 0
            ),

        "relative_measurement_noise_percentage":
            float(
                relative_noise_pct
                if "relative_noise_pct" in locals()
                else 0
            ),
    },

    "representativeness_and_diversity": {
        "chemistry":
            METADATA["chemistry"],

        "temperature_conditions":
            METADATA["temperature_conditions"],

        "dod_conditions":
            METADATA["dod_conditions"],

        "charge_c_rates":
            METADATA["charge_c_rates"],

        "discharge_c_rates":
            METADATA["discharge_c_rates"],

        "replicate_cells":
            METADATA["replicate_cells"],

        "calendar_aging":
            METADATA["calendar_aging"],

        "dynamic_load_profiles":
            METADATA["dynamic_load_profiles"],

        "real_world_operation":
            METADATA["real_world_operation"],
    },

    "scores": results_df.to_dict(
        orient="records"
    ),
}


json_path = os.path.join(
    RESULTS_DIR,
    "calce_quality_metrics.json"
)

with open(
    json_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metrics,
        f,
        indent=4,
        ensure_ascii=False
    )


# ============================================================================
# TEXT REPORT
# ============================================================================

report_path = os.path.join(
    RESULTS_DIR,
    "calce_quality_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "CALCE BATTERY DATASET — QUALITY ASSESSMENT\n"
    )

    f.write("=" * 80 + "\n\n")

    f.write(
        f"Dataset: {METADATA['dataset_name']}\n"
    )

    f.write(
        "Framework: Rule-based battery dataset quality assessment\n\n"
    )

    for criterion in results_df["criterion"].unique():

        f.write(
            f"\n{criterion.upper()}\n"
        )

        f.write("-" * 80 + "\n")

        subset = results_df[
            results_df["criterion"] == criterion
        ]

        for _, row in subset.iterrows():

            f.write(
                f"{row['aspect']}: "
                f"{row['score']} "
                f"({SCORE_LABELS[row['score']]})\n"
            )

            f.write(
                f"  {row['finding']}\n\n"
            )

    f.write("\n" + "=" * 80 + "\n")
    f.write("SCORE DISTRIBUTION\n")
    f.write("=" * 80 + "\n")

    for score in SCORE_ORDER:

        f.write(
            f"{score}: "
            f"{score_counts.get(score, 0)}\n"
        )


# ============================================================================
# QUALITY DATASHEET
# ============================================================================

datasheet_path = os.path.join(
    RESULTS_DIR,
    "calce_quality_datasheet.txt"
)

with open(
    datasheet_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "CALCE DATASET QUALITY DATASHEET\n"
    )

    f.write("=" * 80 + "\n\n")

    f.write(
        f"Dataset: {METADATA['dataset_name']}\n"
    )

    f.write(
        f"Chemistry: {METADATA['chemistry']}\n"
    )

    f.write(
        f"Battery type: {METADATA['battery_type']}\n"
    )

    f.write(
        f"Nominal capacity: "
        f"{METADATA['nominal_capacity_Ah']} Ah\n"
    )

    f.write(
        f"Temperature: "
        f"{METADATA['temperature_values_C']} °C\n"
    )

    f.write(
        f"DoD: {METADATA['dod_description']}\n"
    )

    f.write(
        f"Charge C-rate: "
        f"{METADATA['charge_c_rates']}C\n"
    )

    f.write(
        f"Discharge C-rate: "
        f"{METADATA['discharge_c_rates']}C\n"
    )

    f.write(
        f"Replicate cells: "
        f"{METADATA['replicate_cells']}\n"
    )

    f.write(
        f"Calendar aging: "
        f"{METADATA['calendar_aging']}\n"
    )

    f.write(
        f"Dynamic load profiles: "
        f"{METADATA['dynamic_load_profiles']}\n"
    )

    f.write(
        f"Real-world operation: "
        f"{METADATA['real_world_operation']}\n\n"
    )

    f.write("=" * 80 + "\n")
    f.write("QUALITY SCORES\n")
    f.write("=" * 80 + "\n\n")

    for _, row in results_df.iterrows():

        f.write(
            f"{row['criterion']} | "
            f"{row['aspect']} | "
            f"{row['score']}\n"
        )


# ============================================================================
# SCORECARD VISUALIZATION
# ============================================================================

if SAVE_PLOT:

    fig_height = max(
        8,
        len(results_df) * 0.42
    )

    fig, ax = plt.subplots(
        figsize=(14, fig_height)
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(
        0,
        len(results_df)
    )

    ax.axis("off")

    col_x = {
        "criterion": 0.01,
        "aspect": 0.30,
        "score": 0.86,
    }

    ax.text(
        col_x["criterion"],
        len(results_df) + 0.25,
        "Criterion",
        fontsize=9,
        fontweight="bold"
    )

    ax.text(
        col_x["aspect"],
        len(results_df) + 0.25,
        "Aspect",
        fontsize=9,
        fontweight="bold"
    )

    ax.text(
        col_x["score"],
        len(results_df) + 0.25,
        "Score",
        fontsize=9,
        fontweight="bold"
    )

    previous_criterion = None

    for i, row in results_df.iterrows():

        y = (
            len(results_df) -
            1 -
            i
        )

        if (
            previous_criterion is not None
            and row["criterion"] != previous_criterion
        ):

            ax.axhline(
                y + 0.92,
                color="#999999",
                linewidth=0.8
            )

        previous_criterion = row["criterion"]

        ax.text(
            col_x["criterion"],
            y + 0.45,
            row["criterion"],
            fontsize=7.5,
            va="center"
        )

        ax.text(
            col_x["aspect"],
            y + 0.45,
            row["aspect"],
            fontsize=7.5,
            va="center"
        )

        score = row["score"]

        ax.add_patch(
            mpatches.FancyBboxPatch(
                (
                    col_x["score"],
                    y + 0.15
                ),
                0.08,
                0.6,
                boxstyle="round,pad=0.01",
                linewidth=0,
                facecolor=SCORE_COLORS[score]
            )
        )

        ax.text(
            col_x["score"] + 0.04,
            y + 0.45,
            score,
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="center"
        )

    legend = [
        mpatches.Patch(
            color=SCORE_COLORS[s],
            label=f"{s} — {SCORE_LABELS[s]}"
        )
        for s in SCORE_ORDER
    ]

    ax.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=4,
        fontsize=8
    )

    ax.set_title(
        "CALCE Battery Dataset — Quality Assessment",
        fontsize=12,
        fontweight="bold",
        pad=15
    )

    plt.tight_layout()

    plot_path = os.path.join(
        RESULTS_DIR,
        "calce_quality_scorecard.png"
    )

    plt.savefig(
        plot_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================================
# FINAL MESSAGE
# ============================================================================

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

print("\nResults saved to:")
print(RESULTS_DIR)

print("\nGenerated files:")

print(
    f"  - {os.path.basename(csv_path)}"
)

print(
    f"  - {os.path.basename(report_path)}"
)

print(
    f"  - {os.path.basename(json_path)}"
)

print(
    f"  - {os.path.basename(datasheet_path)}"
)

if SAVE_PLOT:
    print(
        f"  - calce_quality_scorecard.png"
    )

print("\nNo N/A scores are used.")
print(
    "Metadata-derived aspects are explicitly based on "
    "the CALCE metadata record."
)
print("=" * 80)