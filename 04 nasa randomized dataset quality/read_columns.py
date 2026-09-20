import os
import glob
import pandas as pd

DATASET_PATH = r"C:\Users\admin\Desktop\DR2\11 All Datasets\02 NASA Randomized Battery Dataset\battery_alt_dataset\regular_alt_batteries"

csv_files = sorted(
    glob.glob(os.path.join(DATASET_PATH, "battery*.csv"))
)

if not csv_files:
    raise FileNotFoundError("No battery*.csv files found.")

df = pd.read_csv(csv_files[0], nrows=5)

print("NASA R&R Dataset columns:")

for col in df.columns:
    print(col)