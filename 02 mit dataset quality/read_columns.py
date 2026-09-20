import os
import h5py

DATASET_PATH = r"C:\Users\admin\Desktop\DR2\11 All Datasets\04 MIT–Stanford–TRI Fast-Charging Dataset\mit_dataset"

mat_file = next(
    os.path.join(DATASET_PATH, f)
    for f in os.listdir(DATASET_PATH)
    if f.endswith(".mat")
)

with h5py.File(mat_file, "r") as h5:

    print("Top level:")
    print(list(h5.keys()))

    batch = h5["batch"]

    print("\nBatch:")
    print(list(batch.keys()))

    cycles = batch["cycles"]
    print("\nCycles shape:", cycles.shape)

    first_cell_ref = cycles[()].flatten()[0]
    first_cell = h5[first_cell_ref]

    print("\nFirst cell fields:")
    print(list(first_cell.keys()))