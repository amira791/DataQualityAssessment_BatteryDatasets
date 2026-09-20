import scipy.io as sio
import numpy as np
import re

MAT_FILE = r"C:\Users\admin\Desktop\DR2\11 All Datasets\05 Oxford Battery Degradation Dataset\oxford dataset\Oxford_Battery_Degradation_Dataset_1.mat"

mat_data = sio.loadmat(
    MAT_FILE,
    squeeze_me=True,
    struct_as_record=False
)

print("=" * 80)
print("OXFORD DATASET — GLOBAL STRUCTURE CHECK")
print("=" * 80)

cell_keys = sorted(
    [k for k in mat_data.keys() if k.startswith("Cell")],
    key=lambda x: int(re.search(r"\d+", x).group())
)

for cell_name in cell_keys:

    cell = mat_data[cell_name]

    cycle_names = sorted(
        [f for f in dir(cell) if re.match(r"^cyc\d+$", f)],
        key=lambda x: int(x[3:])
    )

    print("\n" + "=" * 80)
    print(cell_name)
    print("=" * 80)

    print(f"Number of characterization cycles: {len(cycle_names)}")

    if cycle_names:
        cycle_numbers = [int(x[3:]) for x in cycle_names]

        print(f"First cycle: {cycle_numbers[0]}")
        print(f"Last cycle:  {cycle_numbers[-1]}")

        missing_intervals = []
        for a, b in zip(cycle_numbers[:-1], cycle_numbers[1:]):
            if b - a != 100:
                missing_intervals.append((a, b))

        if missing_intervals:
            print("Non-100-cycle intervals:")
            print(missing_intervals)
        else:
            print("All consecutive characterization points are 100 cycles apart.")

    # Inspect first available cycle
    if cycle_names:
        cycle_name = cycle_names[0]
        cycle = getattr(cell, cycle_name)

        print(f"\nExample cycle: {cycle_name}")

        for section_name in ["C1ch", "C1dc", "OCVch", "OCVdc"]:

            if not hasattr(cycle, section_name):
                print(f"  {section_name}: MISSING")
                continue

            section = getattr(cycle, section_name)

            print(f"  {section_name}:")

            for field in ["t", "v", "q", "T"]:

                if not hasattr(section, field):
                    print(f"    {field}: MISSING")
                    continue

                values = np.asarray(
                    getattr(section, field)
                ).squeeze()

                finite = np.isfinite(values)

                if finite.any():
                    print(
                        f"    {field}: "
                        f"n={len(values)}, "
                        f"NaN={np.isnan(values).sum()}, "
                        f"min={np.nanmin(values):.4f}, "
                        f"max={np.nanmax(values):.4f}"
                    )
                else:
                    print(
                        f"    {field}: "
                        f"n={len(values)}, all values invalid"
                    )