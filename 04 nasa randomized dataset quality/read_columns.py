import os
import glob
import pandas as pd

DATASET_PATH = r"C:\Users\admin\Desktop\DR2\11 All Datasets\02 NASA Randomized Battery Dataset\battery_alt_dataset\regular_alt_batteries"

csv_files = sorted(
    glob.glob(os.path.join(DATASET_PATH, "battery*.csv"))
)

print("NASA Dataset columns:\n")

for file in csv_files[:1]:
    df = pd.read_csv(file, nrows=5)

    for col in df.columns:
        print(col)