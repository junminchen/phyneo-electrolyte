import pickle
from pathlib import Path

data_file = Path("../../data/data_dimer.pickle")
with open(data_file, "rb") as f:
    data = pickle.load(f)

for k in sorted(data.keys()):
    print(f"{k}: {len(data[k])} batches")
