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
device = torch.device(
    "cuda:0"
    if torch.cuda.is_available()
    else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
)
print("Using device:", device)

out_folder = "results"
if not os.path.exists(out_folder):
    os.makedirs(out_folder, exist_ok=True)
first = False

openml_datasets = [
    # 1479, 43946, 15, 997,
    # 31, 188, 1046, 1471,
    # 1476, 45060, 4534, 32
    # 45040, 45074, 1044,
    # 1053, 1459, 44122,
    # 45062, 45023, 375,
    # 4538, 45553, 41972,
    # 44130, 1496, 1507,
    # 803, 182, 42889, 44
    # 1475, 44150, 458, 30,
    # 1497, 1489, 44124,
    # 44186, 41146, 44489,
    # 1037, 42636, 1557,
    # 28, 1043, 41156 # X columns with only one distinct value
    # 40708, 40497    # X columns with only one distinct value
    # 40707, 40713, 40677,
    # 40678, 3, 46,
    # 40670,          # high gpu usage
    # 41145,          # high gpu usage
    # 45075,          # classes [-1, 1] instead of [0, 1]
    # 42178,          # y is an object instead of a category
    # 1589,           # as_frame fails -> ARFF dataset - Compressed Sparse Row
    # 41144, 41143, 40478,
    # 44091, 1487, 1548,
    # 45540, 45539, 45538,
    # 45537, 45536, 44528,
    # 36, 1067, 22,
    # 18, 14, 16, 12,
    # 41721, 41875, 41882,
    # 40664, 43442, 45648,
    # 40646, 42464, 1501,
    # 23, 1050, 54, 185,
    # 43895, 1049, 43812,
    # 1068, 1552, 1444,
    # 372,    # n_classes (y_true) and dimension of y_score is not matching -> error spliting the data
    # 1491,   # n_classes (y_true) and dimension of y_score is not matching -> error spliting the data
    # 1492,   # n_classes (y_true) and dimension of y_score is not matching -> error spliting the data
    # 1493,   # 100 classesm TabPFN supports up to 10
    # 40498,  # n_classes (y_true) and dimension of y_score is not matching -> error spliting the data
    # 40499,  # 11 classesm TabPFN supports up to 10
    # 41705   # X columns with only one distinct value -> column with one value and NaN &
              # 11 classesm TabPFN supports up to 10
    # 301,    # 2160 features, TabPFN supports up to 500
    # 20,     # 1648 features, TabPFN supports up to 500
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
        df.to_csv("results/raw.csv", index=False, header=first, mode="a")
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
