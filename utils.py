import os, random, warnings
from typing import Optional

import torch
import numpy as np

from tabpfn import TabPFNClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from skrub import TableVectorizer
from tabicl import TabICLClassifier

from sklearn.utils import Bunch
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.datasets import fetch_openml
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from rtdl_num_embeddings import PiecewiseLinearEmbeddings, compute_bins
from mapie.utils import train_conformalize_test_split
from mapie.classification import SplitConformalClassifier, CrossConformalClassifier
from mapie.metrics.classification import (classification_coverage_score,
                                          classification_mean_width_score,
                                          classification_ssc_score)


from pandas.errors import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

from TabM import TabMClassifer

DATA_DIR = '/content/MyDrive/MyDrive/Datasets/Rain_in_Australia'


def set_seed(seed: int = 42):
    """
    Sets the random seed for various libraries to ensure reproducibility.
    """
    # Set seed for Python's built-in random module
    random.seed(seed)

    # Set seed for NumPy
    np.random.seed(seed)

    # Set seed for PyTorch on CPU and CUDA
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # # Configure PyTorch to use deterministic algorithms
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # Set the PYTHONHASHSEED environment variable
    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"Random seed set to {seed}")


def load(name) -> Optional[np.ndarray]:
    p = os.path.join(DATA_DIR, name)
    return np.load(p, allow_pickle=True) if os.path.exists(p) else None


# ---- build X by concatenating [C | N] ----
def concat_features(C_part, N_part):
    parts = [p for p in (C_part, N_part) if p is not None]
    if not parts:
        raise ValueError("No features found (need at least C_* or N_*).")
    return np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]


def evaluate_classification(y_pred, y_test, y_pred_set):
    """Evaluate classification models (binary & multiclass)."""
    # if probs.shape[1] > 1:
    #     y_pred = np.argmax(probs, axis=1)
    # else:  # Binary (XGB can sometimes return shape=(n,1))
    #     probs = np.hstack([1 - probs, probs]) if probs.shape[1] == 1 else probs
    #     y_pred = (probs[:, 1] > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    # if probs.shape[1] == 2:  # Binary
    #     auc = roc_auc_score(y_test, probs[:, 1])
    # else:  # Multiclass
    #     auc = roc_auc_score(y_test, probs, multi_class="ovo", average="weighted")

    cr = classification_coverage_score(y_test, y_pred_set)
    cmwc = classification_mean_width_score(y_pred_set)
    sscs = classification_ssc_score(y_test, y_pred_set)

    # return {"accuracy": acc, "f1_score": f1, "auc": auc.item(), "cr": cr[0].item()}
    return {"accuracy": acc, "f1_score": f1, "cr": cr[0].item(), "cmwc": cmwc[0].item(), "sscs": sscs[0].item()} # type: ignore


def clean_col(col):
    return col.replace('<=', '_le_').replace('<', '_lt_').replace('[', '_').replace(']', '_')


def evaluate_on_openml(dataset_id, device, score="lac", confidence_level=0.90, seed=42):
    df: Bunch = fetch_openml(data_id=dataset_id, as_frame=True) # type: ignore
    X = df.data
    float64_cols = X.select_dtypes(np.float64).columns
    X[float64_cols] = X[float64_cols].astype(np.float32)

    if df.target.dtype.name == 'category':
        y = df.target.cat.codes.to_numpy()
    else:
        y = df.target.to_numpy()

    X_train, X_val, X_test, y_train, y_val, y_test = train_conformalize_test_split(
        X, y, train_size=0.5, conformalize_size=0.3, test_size=0.2, random_state=seed,
    )

    vectorizer = TableVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_val = vectorizer.transform(X_val)
    X_test = vectorizer.transform(X_test)

    X_train.columns = [clean_col(col) for col in X_train.columns]
    X_val.columns = [clean_col(col) for col in X_val.columns]
    X_test.columns = [clean_col(col) for col in X_test.columns]

    n_classes = len(np.unique(y))
    n_num_features = X_train.shape[1]
    cat_cardinalities = []  # already numeric after vectorizer
    num_embeddings = PiecewiseLinearEmbeddings(
        compute_bins(torch.as_tensor(SimpleImputer().fit_transform(X_train.to_numpy())), n_bins=48),
        d_embedding=16,
        activation=False,
        version='B',
    )

    results = []
    devices = None
    task_type = None
    # if "cuda" in device:
    #     devices = device.split(":")[1]
    #     task_type = "GPU"

    if n_classes == 2:
        solver = "liblinear"
    else:
        solver = "lbfgs"

    models = {
        "LightGBM": (LGBMClassifier(random_state=seed, force_col_wise=True, verbose=-100),
                        {"n_estimators": [100, 300], "max_depth": [-1, 6, 10], "max_bin": [255, 128]}),
        "CatBoost": (CatBoostClassifier(task_type=task_type, devices=devices, verbose=False, random_state=seed, allow_writing_files=False),
                        {"iterations": [200, 500], "depth": [4, 6]}),
        "XGBoost": (XGBClassifier(device=device.type, eval_metric="logloss", random_state=seed, verbosity=0),
                        {"n_estimators": [200, 500], "max_depth": [4, 6]}),
        "LogisticRegression": (
            make_pipeline(
                SimpleImputer(strategy="mean"),
                LogisticRegression(solver=solver, random_state=seed, max_iter=1000)
            ),
            None
        ),
        "TabICL": (TabICLClassifier(device=device, n_estimators=1, random_state=seed), None),
        "TabPFN": (TabPFNClassifier(device=device, n_estimators=1, random_state=seed), None),
        "TabM": (TabMClassifer(device=device, n_num_features=n_num_features,
                               cat_cardinalities=cat_cardinalities, n_classes=n_classes,
                               num_embeddings=num_embeddings, train_size=X_train.shape[0]), None),
    }

    for name, (model, param_grid) in models.items():
        print(f"Training {name}")
        params = {}
        best_model = model

        if param_grid is not None:
            search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=3,
                scoring="roc_auc_ovo_weighted",
                n_jobs=-1
            )
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            params = search.best_params_
        else:
            if name == "TabM":
                best_model.fit(np.concat([X_train, X_val], axis=0),
                               np.concat([y_train, y_val], axis=0))
            else:
                best_model.fit(X_train, y_train)

        if X_val.shape[0] > 200:
            mapie_clf = SplitConformalClassifier(
                estimator=best_model, confidence_level=confidence_level,
                prefit=True, conformity_score=score, random_state=seed,
            )
            mapie_clf.conformalize(X_val, y_val)
        else:
            mapie_clf = CrossConformalClassifier(
                estimator=best_model, confidence_level=confidence_level,
                conformity_score=score, random_state=seed,
            )
            mapie_clf.fit_conformalize(np.concat([X_train, X_val], axis=0),
                                       np.concat([y_train, y_val], axis=0))

        y_pred, y_pred_set = mapie_clf.predict_set(X_test)
        test_metrics = evaluate_classification(y_pred, y_test, y_pred_set)
            #{**results_tabicl, **results_tabpfn}
        results.append({
            "dataset_id": dataset_id,
            "seed": seed,
            "model": name,
            **params,
            **test_metrics,
        })

    return results
