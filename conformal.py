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

for dataset_id in openml_datasets:
    print(f"Dataset id: {dataset_id}")
    results = evaluate_on_openml(dataset_id, device)
    df = pd.DataFrame(results)
    df.to_csv(f"results_{seed}_{dataset_id}.csv", index=False)
