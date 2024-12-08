import os
import pickle
import warnings
import numpy as np
import pandas as pd
import torch
import torchtuples as tt
from sklearn.preprocessing import StandardScaler
from pycox.models import CoxPH
from acpi import ACPI  # Assuming acpipy is the package for ACPI

# Suppress warnings
warnings.filterwarnings('ignore')
from sys import argv

# Constants
DATASET_NAME = argv[1] # "bpic2012"
TASK_TYPE = "classification"
CLS_METHOD_ALL = "other"
CLS_ENCODING = "index"
RESULTS_DIR = f"/home/centos/phd/5th/survival_results2/{TASK_TYPE}/{DATASET_NAME}/"

# Utility Functions
def save_results(results_dict, file_name):
    with open(file_name, "wb") as f:
        pickle.dump(results_dict, f)

def load_results(file_name):
    with open(file_name, "rb") as f:
        return pickle.load(f)

def store_results(results_dir, dataset_name, task_type, cls_method, cls_encoding, y_true, preds, y_lower=None, y_upper=None, alpha=None, data_type=None):
    os.makedirs(results_dir, exist_ok=True)
    results_dict = {
        (dataset_name, task_type, cls_method, cls_encoding): {
            "surv_actual": y_true,
            "surv_preds": preds,
            f"surv_lower_{alpha}": y_lower,
            f"surv_upper_{alpha}": y_upper,
        }
    }
    file_name = os.path.join(results_dir, f"{dataset_name}_{task_type}_{cls_method}_{cls_encoding}_{data_type}_{alpha}_results_survival_dict.pkl")
    save_results(results_dict, file_name)

def load_data(data_type):
    file_path = f"/home/centos/phd/5th/prepared_data/{TASK_TYPE}/{DATASET_NAME}/{data_type}_{CLS_METHOD_ALL}_{CLS_ENCODING}_encoded_{DATASET_NAME}.parquet"
    return pd.read_parquet(file_path)

def scale_data(X_train, X_cal, X_test, X_val):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    return (
        X_train_scaled.astype(np.float32),
        scaler.transform(X_cal).astype(np.float32),
        scaler.transform(X_test).astype(np.float32),
        scaler.transform(X_val).astype(np.float32)
    )

def prepare_data():
    print("Loading and preparing data...")
    train = load_data("train")
    train_sampled = train.sample(frac=0.7, random_state=42)
    cal = train.drop(train_sampled.index)

    X_train = train_sampled.drop(columns=['time_to_last_event_days_0', 'event'])
    X_cal = cal.drop(columns=['time_to_last_event_days_0', 'event'])
    X_test = load_data("test").drop(columns=['time_to_last_event_days_0', 'event'])
    X_val = load_data("val").drop(columns=['time_to_last_event_days_0', 'event'])

    get_target = lambda df: (df['time_to_last_event_days_0'].values.astype(np.float32), df['event'].values.astype(np.float32))
    y_train, y_cal = get_target(train_sampled), get_target(cal)
    y_test, y_val = get_target(load_data("test")), get_target(load_data("val"))

    print("Standardizing data...")
    return scale_data(X_train, X_cal, X_test, X_val), y_train, y_cal, y_test, y_val

def build_and_train_model(X_train_t, y_train_t):
    in_features = X_train_t.shape[1]
    net = tt.practical.MLPVanilla(in_features, [32, 32], 1, batch_norm=True, dropout=0.1)
    model = CoxPH(net, tt.optim.Adam)
    model.optimizer.set_lr(0.01)
    model.fit(X_train_t, y_train_t, batch_size=256, epochs=512, callbacks=[tt.callbacks.EarlyStopping()], verbose=True)
    return model

def get_survival_time(surv_df, prob_threshold=0.5):
    survival_times = []
    for col in surv_df.columns:
        surv_func = surv_df[col]
        time_points = surv_func.index
        survival_probs = surv_func.values
        idx = np.where(survival_probs <= prob_threshold)[0]
        survival_times.append(time_points[idx[0]] if len(idx) > 0 else np.nan)
    return pd.Series(survival_times, index=surv_df.columns)

# Main Workflow
(X_train_t, X_cal_t, X_test_t, X_val_t), y_train_t, y_cal_t, y_test_t, y_val_t = prepare_data()

model = build_and_train_model(X_train_t, y_train_t)

# Compute Baseline Hazards and Predict Survival Functions
_ = model.compute_baseline_hazards()
surv_test, surv_val = model.predict_surv_df(X_test_t), model.predict_surv_df(X_val_t)

# Get Survival Times
surv_test_time = get_survival_time(surv_test).fillna(0)
surv_val_time = get_survival_time(surv_val).fillna(0)

# ACPI Calibration and Prediction
print("Starting ACPI...")
alpha_values = np.arange(0.1, 1.0, 0.1)

for alpha in alpha_values:
    alpha = np.round(alpha, 1)
    acpi = ACPI(model_cali=model, estimator="reg")
    acpi.fit(X_cal_t, y_cal_t[0], algo="other")
    acpi.fit_calibration(X_cal_t, y_cal_t[0], quantile=1 - alpha, algo="other")

    print("Generating ACPI conformal predictions...")
    y_lower_test, y_upper_test = acpi.predict_pi(X_test_t, method="qrf", algo2="surv")
    y_lower_val, y_upper_val = acpi.predict_pi(X_val_t, method="qrf", algo2="surv")
    
    print("Saving results...")
    store_results(RESULTS_DIR, DATASET_NAME, TASK_TYPE, CLS_METHOD_ALL, CLS_ENCODING, y_test_t[0], surv_test_time, y_lower_test, y_upper_test, alpha, data_type="test")
    store_results(RESULTS_DIR, DATASET_NAME, TASK_TYPE, CLS_METHOD_ALL, CLS_ENCODING, y_val_t[0], surv_val_time, y_lower_val, y_upper_val, alpha, data_type="val")

    #break  # Remove break to run for all alpha values
