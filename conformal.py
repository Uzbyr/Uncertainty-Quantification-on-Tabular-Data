import torch
import numpy as np
import pandas as pd

from utils import evaluate_on_openml

seed = 42
np.random.seed(seed)

# Pick the best available device
device = "cuda:0" if torch.cuda.is_available() else (
    "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
)
print("Using device:", device)

openml_datasets = [
        1479,    # hill-valley
        43946,   # Eye movements
        15,      # breast-w
        997,     # Eye balance-scale
        31,      # credit-g
        188,     # eucalyptus
]
results = {}

for dataset_id in openml_datasets:
    print(f"Dataset id: {dataset_id}")
    results[dataset_id] = evaluate_on_openml(dataset_id, device)
# print(results)

rows = []
for dataset_id, models in results.items():
    for model_name, content in models.items():
        row = {
            "dataset_id": dataset_id,
            "model": model_name,
            **content["metrics"],
            # **content["best_params"]
        }
        rows.append(row)

df = pd.DataFrame(rows)

print(df.to_string(max_cols=None, max_rows=20))
df.to_csv("results.csv")
