import os
import torch
import numpy as np
import pandas as pd

from utils import evaluate_on_openml, set_seed

# np.random.seed(42)
# sq = np.random.SeedSequence()
# seeds = sq.generate_state(20)
# print(seeds)

seeds = [
     382114703, 3773843095, 1543291710,  848852757,
     576485429, 4208438384, 2965207055, 2386776174,
    1385245016,  618360380, 3721738935, 2011376238,
    1472753441, 2269463650, 2330484666,  391807351,
     802708881, 2702360181,  363512744, 3147449495,
]

# Pick the best available device
device = torch.device("cuda:0" if torch.cuda.is_available() else (
    "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
))
print("Using device:", device)

out_folder = "results"
if not os.path.exists(out_folder): os.makedirs(out_folder, exist_ok=True)
first = True

openml_datasets = [
        1479,    # hill-valley
        43946,   # Eye movements
        15,      # breast-w
        997,     # Eye balance-scale
        31,      # credit-g
        188,     # eucalyptus
]

# metric_cols = ["accuracy", "f1_score", "cr", "mwc", "sscs"]
for dataset_id in openml_datasets:
    print(f"Dataset id: {dataset_id}")
    # results_all_seeds = []
    for seed in seeds:
        set_seed(seed)
        results = evaluate_on_openml(dataset_id, device, seed=seed)
        # results_all_seeds.extend(results)
        df = pd.DataFrame(results)
        df.to_csv(f"results/{seed}_{dataset_id}.csv", index=False)
        df.to_csv(f"results/raw.csv", index=False, header=first, mode='a')
        first = False

    # df_all = pd.DataFrame(results_all_seeds)
    # for c in metric_cols:
    #     if c in df_all.columns:
    #         df_all[c] = pd.to_numeric(df_all[c], errors="coerce")
    # summary = (
    #     df_all
    #     .groupby(["dataset_id", "model"], dropna=False)[[c for c in metric_cols if c in df_all.columns]]
    #     .agg(["mean", "std"])
    #     .reset_index()
    # )
    # summary.columns = [
    #     "_".join([str(x) for x in col if x]) if isinstance(col, tuple) else col
    #     for col in summary.columns
    # ]
    # df_all.to_csv(f"results_dataset_{dataset_id}_all_seeds.csv", index=False)
    # summary.to_csv(f"results_dataset_{dataset_id}_summary.csv", index=False)
    # print(f"Salvos: results_dataset_{dataset_id}_all_seeds.csv e results_dataset_{dataset_id}_summary.csv")

