import warnings
import os
import pickle
import itertools

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Lasso

from econml.orf import DMLOrthoForest

warnings.filterwarnings("ignore")

from sys import argv


# Load and process data
def load_data(
    data_type,
    task_type,
    dataset_name,
    cls_method_all,
    cls_encoding,
    label_col,
    treatment_col,
):
    data = pd.read_parquet(
        f"/home/centos/phd/5th/prepared_data/{task_type}/{dataset_name}/{data_type}_{cls_method_all}_{cls_encoding}_encoded_{dataset_name}.parquet"
    )
    X = data.drop(columns=[label_col, treatment_col])
    y = data[label_col].to_numpy()
    t = data[treatment_col].to_numpy()
    return X, y, t


def scale_data(scaler, *datasets):
    return [
        scaler.fit_transform(data) if i == 0 else scaler.transform(data)
        for i, data in enumerate(datasets)
    ]


# Save results to a pickle file
def save_results(results_dict, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(results_dict, f)


# Store results function
def store_results(
    dataset_name,
    data_type,
    results_dir,
    ate=None,
    ite=None,
    ite_lo=None,
    ite_up=None,
    cate=None,
    cate_lo=None,
    cate_up=None,
):
    results_dict = {
        (dataset_name, data_type): {
            "ate": ate,
            "ite": ite,
            "ite_lo": ite_lo,
            "ite_up": ite_up,
            "cate": cate,
            "cate_lo": cate_lo,
            "cate_up": cate_up,
        }
    }

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    file_name = os.path.join(
        results_dir, f"{dataset_name}_{data_type}_results_causal_dict.pkl"
    )
    save_results(results_dict, file_name)


# Define hyperparameters and model fitting
def run_experiment(
    N_trees,
    Min_leaf_size,
    Max_depth,
    Subsample_ratio,
    Lambda_reg,
    X_train,
    y_train,
    t_train,
    W_train,
    X_test,
    X_val,
    results_dir,
):
    for (
        n_trees,
        min_leaf_size,
        max_depth,
        subsample_ratio,
        lambda_reg,
    ) in itertools.product(
        N_trees, Min_leaf_size, Max_depth, Subsample_ratio, Lambda_reg
    ):
        print(
            f"Training with n_trees={n_trees}, min_leaf_size={min_leaf_size}, max_depth={max_depth}, subsample_ratio={subsample_ratio}, lambda_reg={lambda_reg}"
        )

        est = DMLOrthoForest(
            n_jobs=8,
            backend="threading",
            n_trees=n_trees,
            min_leaf_size=min_leaf_size,
            max_depth=max_depth,
            subsample_ratio=subsample_ratio,
            discrete_treatment=True,
            model_T=LogisticRegression(
                C=1 / (X_train.shape[0] * lambda_reg), penalty="l1", solver="saga"
            ),
            model_Y=Lasso(alpha=lambda_reg),
            model_T_final=LogisticRegression(
                C=1 / (X_train.shape[0] * lambda_reg), penalty="l1", solver="saga"
            ),
            model_Y_final=Lasso(alpha=lambda_reg),
            random_state=106,
        )

        print("Start fitting...")
        ortho_model = est.fit(y_train, t_train, X=X_train, W=W_train)
        print("Save model")
        pickle.dump(ortho_model, open("ortho_model.sav", "wb"))

        # Calculate ATE, ITE, and CATE for test and validation sets
        def evaluate_effects(X, data_type):
            ate = est.ate(X)
            ite = est.effect(X)
            ite_lo, ite_up = est.effect_interval(X, alpha=0.05)
            cate = est.const_marginal_effect(X)
            cate_lo, cate_up = est.const_marginal_effect_interval(X, alpha=0.05)
            store_results(
                dataset_name,
                data_type,
                results_dir,
                ate,
                ite,
                ite_lo,
                ite_up,
                cate,
                cate_lo,
                cate_up,
            )

        print("Calculating effects for test and validation sets...")
        evaluate_effects(X_test, "test")
        evaluate_effects(X_val, "val")


# Constants for columns and paths
case_id_col = "case_id"
activity_col = "activity"
resource_col = "resource"
timestamp_col = "timestamp"
label_col = "label"
treatment_col = "Treatment"

dataset_name = argv[1]  # "bpic2012"
task_type = "classification"
cls_method_all = "other"
cls_encoding = "index"


# Load and scale data
X_train, y_train, t_train = load_data(
    "train",
    task_type,
    dataset_name,
    cls_method_all,
    cls_encoding,
    label_col,
    treatment_col,
)
X_test, y_test, t_test = load_data(
    "test",
    task_type,
    dataset_name,
    cls_method_all,
    cls_encoding,
    label_col,
    treatment_col,
)
X_val, y_val, t_val = load_data(
    "val",
    task_type,
    dataset_name,
    cls_method_all,
    cls_encoding,
    label_col,
    treatment_col,
)

scaler = StandardScaler()
X_train, X_test, X_val = scale_data(scaler, X_train, X_test, X_val)
W_train = X_train.copy()  # Assuming W and X are the same in this context

# Hyperparameters
N_trees = [100]
Min_leaf_size = [20]
Max_depth = [30]
Subsample_ratio = [0.04]
Lambda_reg = [0.01]


# Run the experiment
results_dir = f"/home/centos/phd/5th/causal_results/{task_type}/{dataset_name}/"
run_experiment(
    dataset_name,
    N_trees,
    Min_leaf_size,
    Max_depth,
    Subsample_ratio,
    Lambda_reg,
    X_train,
    y_train,
    t_train,
    W_train,
    X_test,
    X_val,
    results_dir,
)
