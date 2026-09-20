import os
import glob
import pandas as pd

DATASET_PATH = r"C:\Users\admin\Desktop\DR2\11 All Datasets\10 Battery Archive Datasets\Battery Archive Data\SNL"

# Find files recursively
cycle_files = glob.glob(
    os.path.join(DATASET_PATH, "**", "*_cycle_data.csv"),
    recursive=True
)

timeseries_files = glob.glob(
    os.path.join(DATASET_PATH, "**", "*_timeseries.csv"),
    recursive=True
)

# Read one example of each
cycle_df = pd.read_csv(cycle_files[0], nrows=5)
timeseries_df = pd.read_csv(timeseries_files[0], nrows=5)

print("=" * 60)
print("SNL CYCLE_DATA COLUMNS")
print("=" * 60)

for col in cycle_df.columns:
    print(col)

print("\n" + "=" * 60)
print("SNL TIMESERIES COLUMNS")
print("=" * 60)

for col in timeseries_df.columns:
    print(col)