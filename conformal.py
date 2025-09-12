import torch
import numpy as np
import pandas as pd

from utils import evaluate_on_openml, set_seed

# seed = 42
# sq = np.random.SeedSequence()
# seeds = sq.generate_state(20)
# print(seeds)

seeds = [
    2725058014, 3614383505, 1716993168, 2240316606,
    1044790485, 1973762987, 1608467800, 3843676287,
    2089399802, 1122245882, 2802442126, 2472966128,
    3195191841, 2231143762, 3041075460,  382812183,
    2519222555, 3315644958, 1096636570, 3733192565
]



# Pick the best available device
device = "cuda:0" if torch.cuda.is_available() else (
    "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
)
print("Using device:", device)

openml_datasets = [
        #1479,    # hill-valley
        #43946,   # Eye movements
        15,      # breast-w
        997,     # Eye balance-scale
        #31,      # credit-g
        #188,     # eucalyptus
]

metric_cols = ["accuracy", "f1_score", "cr", "cmwc", "sscs"]
for dataset_id in openml_datasets:
    results_all_seeds = []
    for seed in seeds:
        set_seed(seed)
        print(f"Dataset id: {dataset_id}")
        results = evaluate_on_openml(dataset_id, device, seed=seed)
        results_all_seeds.extend(results)
        df = pd.DataFrame(results)
        df.to_csv(f"results_{seed}_{dataset_id}.csv", index=False)
        
    df_all = pd.DataFrame(results_all_seeds)
    for c in metric_cols:
        if c in df_all.columns:
            df_all[c] = pd.to_numeric(df_all[c], errors="coerce")
    summary = (
        df_all
        .groupby(["dataset_id", "model"], dropna=False)[[c for c in metric_cols if c in df_all.columns]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join([str(x) for x in col if x]) if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    df_all.to_csv(f"results_dataset_{dataset_id}_all_seeds.csv", index=False)
    summary.to_csv(f"results_dataset_{dataset_id}_summary.csv", index=False)
    print(f"Salvos: results_dataset_{dataset_id}_all_seeds.csv e results_dataset_{dataset_id}_summary.csv")

