import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

# Cell voltage limits per chemistry (V)
CELL_V_LIMITS = {
    "NCM": (2.5, 4.2),
    "LFP": (2.0, 3.65),
}

# Vehicle metadata (from GitHub)
VEHICLE_INFO = {
    "vehicle#1":  {"chemistry": "NCM", "type": "Passenger"},
    "vehicle#2":  {"chemistry": "NCM", "type": "Passenger"},
    "vehicle#3":  {"chemistry": "NCM", "type": "Passenger"},
    "vehicle#4":  {"chemistry": "NCM", "type": "Passenger"},
    "vehicle#5":  {"chemistry": "NCM", "type": "Passenger"},
    "vehicle#6":  {"chemistry": "NCM", "type": "Passenger"},
    "vehicle#7":  {"chemistry": "LFP", "type": "Passenger"},
    "vehicle#8":  {"chemistry": "LFP", "type": "Bus"},
    "vehicle#9":  {"chemistry": "LFP", "type": "Bus"},
    "vehicle#10": {"chemistry": "LFP", "type": "Bus"},
}

# ============================================================
# FUNCTION: Estimate series cell count
# ============================================================

def estimate_series_cells(voltage_series, chemistry, percentile_low=1, percentile_high=99):
    """
    Estimate number of cells in series from pack voltage data.
    
    Parameters:
    -----------
    voltage_series : pd.Series
        Pack voltage readings (V)
    chemistry : str
        "NCM" or "LFP"
    percentile_low : float
        Lower percentile for robust min (default 1%)
    percentile_high : float
        Upper percentile for robust max (default 99%)
    
    Returns:
    --------
    dict with estimated series cell counts
    """
    v_min_cell, v_max_cell = CELL_V_LIMITS[chemistry]
    
    # Robust statistics (avoid outliers)
    v_low = np.percentile(voltage_series.dropna(), percentile_low)
    v_high = np.percentile(voltage_series.dropna(), percentile_high)
    v_min = voltage_series.min()
    v_max = voltage_series.max()
    
    # Estimate series cells
    n_from_min = v_low / v_min_cell      # Using low percentile / min cell V
    n_from_max = v_high / v_max_cell     # Using high percentile / max cell V
    
    # Also using absolute min/max
    n_from_abs_min = v_min / v_min_cell
    n_from_abs_max = v_max / v_max_cell
    
    return {
        "v_min_observed": v_min,
        "v_max_observed": v_max,
        "v_low_1pct": v_low,
        "v_high_99pct": v_high,
        "n_from_1pct_min": round(n_from_min, 1),
        "n_from_99pct_max": round(n_from_max, 1),
        "n_from_abs_min": round(n_from_abs_min, 1),
        "n_from_abs_max": round(n_from_abs_max, 1),
        "n_estimate": round((n_from_min + n_from_max) / 2, 1),
    }


# ============================================================
# MAIN: Process each vehicle
# ============================================================

def analyze_vehicle_files(data_dir=".", file_pattern="vehicle#*.csv"):
    """
    Analyze all vehicle files and estimate series cell counts.
    """
    data_path = Path(data_dir)
    files = sorted(data_path.glob(file_pattern))
    
    if not files:
        print(f"No files found matching {file_pattern} in {data_dir}")
        return
    
    results = []
    
    for f in files:
        vehicle_name = f.stem.lower()
        chemistry = VEHICLE_INFO.get(vehicle_name, {}).get("chemistry", "Unknown")
        
        if chemistry == "Unknown":
            print(f"Skipping {f.name}: unknown chemistry")
            continue
        
        print(f"\nProcessing {f.name} ({chemistry})...")
        
        # Load data (adapt column names if needed)
        df = pd.read_csv(f)
        
        # Find voltage column (adapt to your actual column name)
        voltage_col = None
        for col in ["hv_voltage", "total_voltage", "pack_voltage", "voltage"]:
            if col in df.columns:
                voltage_col = col
                break
        
        if voltage_col is None:
            print(f"  No voltage column found in {f.name}. Columns: {df.columns.tolist()}")
            continue
        
        # Estimate series cells
        result = estimate_series_cells(df[voltage_col], chemistry)
        result["vehicle"] = f.stem
        result["chemistry"] = chemistry
        result["n_rows"] = len(df)
        results.append(result)
        
        print(f"  Observed voltage range: [{result['v_min_observed']:.1f}, {result['v_max_observed']:.1f}] V")
        print(f"  1%-99% range: [{result['v_low_1pct']:.1f}, {result['v_high_99pct']:.1f}] V")
        print(f"  Estimated series cells (1%-99%): {result['n_from_1pct_min']:.1f} - {result['n_from_99pct_max']:.1f}")
        print(f"  Estimated series cells (abs min/max): {result['n_from_abs_min']:.1f} - {result['n_from_abs_max']:.1f}")
        print(f"  Best estimate: {result['n_estimate']:.0f} cells")
    
    # Summary table
    if results:
        summary = pd.DataFrame(results)
        print("\n" + "="*80)
        print("SUMMARY: Estimated Series Cell Counts")
        print("="*80)
        print(summary[["vehicle", "chemistry", "n_rows", 
                       "v_min_observed", "v_max_observed",
                       "n_from_1pct_min", "n_from_99pct_max", 
                       "n_estimate"]].to_string(index=False))
        
        return summary
    
    return None


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    # Adjust data_dir to your folder with CSV files
    summary = analyze_vehicle_files(data_dir=".", file_pattern="vehicle#*.csv")