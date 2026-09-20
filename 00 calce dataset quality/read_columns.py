import glob
import os
import pandas as pd

DATA_DIR = r"C:\Users\admin\Desktop\DR2\11 All Datasets\10 Battery Archive Datasets\Battery Archive Data\CALCE\CALCE"

files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))

for f in files:
    df = pd.read_csv(f, nrows=0)

    print("\n" + "=" * 80)
    print(os.path.basename(f))
    print("=" * 80)

    for col in df.columns:
        print(col)