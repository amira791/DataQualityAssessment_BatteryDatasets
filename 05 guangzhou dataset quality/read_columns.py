import os
import glob
import pandas as pd

DATASET_PATH = r"C:\Users\admin\Desktop\DR2\11 All Datasets\12 Real-World 10EVs dataset\dataset_"

xlsx_files = sorted(
    glob.glob(os.path.join(DATASET_PATH, "vehicle*.xlsx"))
)

if not xlsx_files:
    raise FileNotFoundError("No vehicle*.xlsx files found.")

file = xlsx_files[0]

xls = pd.ExcelFile(file)

print("File:", os.path.basename(file))
print("\nSheets:")

for sheet in xls.sheet_names:
    print(f"\n--- {sheet} ---")
    
    df = pd.read_excel(file, sheet_name=sheet, nrows=5)
    
    for col in df.columns:
        print(col)