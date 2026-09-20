import os
import glob
import pandas as pd

DATASET_PATH = r"C:\Users\admin\Desktop\DR2\11 All Datasets\07 300-EV Real-World BMS Dataset Liu et al. (2025)\300 EVs dataset"

csv_files = sorted(
    glob.glob(os.path.join(DATASET_PATH, "vin*.csv"))
)

if not csv_files:
    raise FileNotFoundError("No vin*.csv files found.")

df = pd.read_csv(csv_files[0], nrows=5)

print("300-EV Dataset columns:")

for col in df.columns:
    print(col)