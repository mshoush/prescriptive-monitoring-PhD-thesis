import warnings

warnings.filterwarnings("ignore")


from acpi import ACPI
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, mean_squared_error

# from catboost import CatBoostClassifier, CatBoostRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import sys

sys.path.append("/home/centos/phd/5th/common_files")
from DatasetManager import DatasetManager
from catboost import Pool, CatBoostClassifier, CatBoostRegressor


from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    roc_auc_score,
    log_loss,
)
from sys import argv


# Define a function to load the best parameters
def load_best_params(params_dir, cls_method):
    best_params_df = pd.read_parquet(params_dir)
    best_params = best_params_df.iloc[0].to_dict()
    try:
        for param_name in ["n_estimators", "max_depth", "min_child_weight"]:
            best_params[param_name] = int(best_params[param_name])
    except:
        pass

    if cls_method == "randomforest":
        for param_name in ["min_samples_leaf", "min_samples_split"]:
            best_params[param_name] = int(best_params[param_name])
            if best_params['max_features']=="auto":
                best_params['max_features']=1.0

    print("Best parameters:", best_params)
    return best_params


def regression_score_function_ctb(y_true, y_pred):
    mean_predictions = y_pred[:, 0]
    std_predictions = y_pred[:, 1]

    # Calculate RMSE for mean predictions
    rmse = mean_squared_error(y_true, mean_predictions)
    # Calculate MAE for mean predictions
    mae = mean_absolute_error(y_true, mean_predictions)
    # Calculate negative log likelihood (NLL) for uncertainty estimation
    nll = -np.mean(np.log(np.sqrt(2 * np.pi * std_predictions**2))) - 0.5 * np.mean(
        ((y_true - mean_predictions) / std_predictions) ** 2
    )
    print(f"RMSE: {rmse}, MAE: {mae}, NLL: {nll}")
    return rmse


def classification_score_function_ctb(y_true, y_pred):
    print(
        f"ROC-AUC: {roc_auc_score(y_true, y_pred)}, LogLoss: {log_loss(y_true, y_pred)}"
    )
    return roc_auc_score(y_true, y_pred)


# Define a function to make predictions and calculate performance measures
def train_model(X_train, y_train, task_type, cls_method, best_params):
    if task_type == "classification":
        if cls_method == "xgboost":
            model = XGBClassifier(**best_params)
        elif cls_method == "lightgbm":
            model = LGBMClassifier(
                **best_params,
                verbosity=-1,
            )
        elif cls_method == "randomforest":
            model = RandomForestClassifier(**best_params)
        else:
            raise ValueError("Invalid cls_method for classification")
    elif task_type == "regression":
        if cls_method == "xgboost":
            model = XGBRegressor(**best_params)
        elif cls_method == "lightgbm":
            model = LGBMRegressor(
                **best_params,
                verbosity=-1,
            )
        elif cls_method == "randomforest":
            model = RandomForestRegressor(**best_params)
        else:
            raise ValueError("Invalid cls_method for regression")
    else:
        raise ValueError("Invalid task_type")

    if cls_method=="lightgbm": 
        import re
        X_train = X_train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

    model.fit(X_train, y_train)

    return model


def get_scores_classification(y_true, y_preds):
    return roc_auc_score(y_true, y_preds)


def get_scores_regression(y_true, y_preds):
    return mean_squared_error(y_true, y_preds)


def get_preds_scores_classification(model, x, y):
    preds = model.predict(x)
    preds_proba = model.predict_proba(x)
    scores = get_scores_classification(y, preds)
    return preds, preds_proba, scores


def get_preds_scores_regression(model, x, y):
    preds = model.predict(x)
    scores = get_scores_regression(y, preds)
    return preds, scores


# save results_dict as pickle file
import pickle


def save_results(results_dict, file_name):
    with open(file_name, "wb") as f:
        pickle.dump(results_dict, f)
    return None


# load results_dict from pickle file
def load_results(file_name):
    with open(file_name, "rb") as f:
        results_dict = pickle.load(f)
    return results_dict


def store_results(
    results_dir,
    dataset_name,
    task_type,
    cls_method,
    cls_encoding,
    y_tru,
    preds,
    score,
    probs_mean=None,
    type=None,
    y_pred_set=None,
    y_lower=None,
    y_upper=None,
    alpha=None,
):

    results_dict = {}

    results_dict[(dataset_name, task_type, cls_method, cls_encoding)] = {
        "actual": y_tru,
        "preds": preds,
        "probs": probs_mean,
        f"y_pred_set_{alpha}": y_pred_set,
        f"y_lower_{alpha}": y_lower,
        f"y_upper_{alpha}": y_upper,
        "score": score,
    }

    # save results_dict_classification
    import os

    name = (
        dataset_name
        + "_"
        + task_type
        + "_"
        + cls_method
        + "_"
        + cls_encoding
        + "_"
        + type
        + "_"
        + str(alpha)
    )
    file_name = os.path.join(results_dir, name + "_results_dict.pkl")
    save_results(results_dict, file_name)

    return results_dict


# Define the dataset name, task_type, cls_method, and cls_encoding
# dataset_name = "bpic2012"
# task_type = "regression"
# cls_method = "catboost"
# cls_encoding = "laststate"


case_id_col = "case_id"
activity_col = "activity"
resource_col = "resource"
timestamp_col = "timestamp"
label_col = "label"
treatment_col = "Treatment1"

dataset_ref = argv[1]  # bpic2012, bpic2017
cls_encoding = argv[2]  # laststate, agg, index, combined
cls_method = argv[3]  # catboost, 'xgboost', 'lightgbm', 'randomforest',
task_type = argv[4]  # "regression", "classification"


ref_to_task_types = {"regression": ["regression"], "classification": ["classification"]}
task_types = (
    [task_type] if task_type not in ref_to_task_types else ref_to_task_types[task_type]
)

ref_to_cls = {
    # "catboost": ["catboost"],
    "xgboost": ["xgboost"],
    "lightgbm": ["lightgbm"],
    "randomforest": ["randomforest"],
}
cls_methods = [cls_method] if cls_method not in ref_to_cls else ref_to_cls[cls_method]


dataset_ref_to_datasets = {
    "bpic2012": ["bpic2012"],
    "bpic2017": ["bpic2017"],
}

datasets = (
    [dataset_ref]
    if dataset_ref not in dataset_ref_to_datasets
    else dataset_ref_to_datasets[dataset_ref]
)


# dataset_ref_to_datasets = {
#     "bpic2012": ["bpic2012"],
#     #"bpic2017": ["bpic2017"],
# }

encoding_dict = {
    "laststate": ["static", "last"],
    "agg": ["static", "agg"],
    "index": ["static", "index"],
    "combined": ["static", "last", "agg"],
}

# task_types = ["regression",  "classification",] # ,

# cls_methods = [ 'xgboost', ]  #  'catboost',  'lightgbm', 'randomforest',
# # Initialize a dictionary to store scores
# # scores_dict = {}
# # preds_dict = {}
# # preds_proba_dict = {}

scores_training = {}
scores_testing = {}
scores_validation = {}


with_conformal = argv[5].lower() == "true" #False
without_inter =  argv[6].lower() == "true" #True

import joblib
def save_model(model, filepath):
    joblib.dump(model, filepath)

# with_conformal = argv[5] #False
# without_inter =  argv[6] #True


for dataset_name in datasets:
    for task_type in task_types:
        for cls_method in cls_methods:
            print(
                f"dataset_name: {dataset_name}, task_type: {task_type}, cls_method: {cls_method}, cls_encoding: {cls_encoding}"
            )

            # Load the best parameters
            params_dir = f"/home/centos/phd/5th/predictive_results/{task_type}/{dataset_name}/optimal_params_{cls_encoding}_{cls_method}_{dataset_name}.parquet"
            best_params = load_best_params(params_dir, cls_method)

            dataset_manager = DatasetManager(dataset_name, task_type)
            cls_method_all = "other"

            results_dir = f"/home/centos/phd/5th/predictive_results/predictions/{task_type}/{dataset_name}/"
            train = pd.read_parquet(f"/home/centos/phd/5th/prepared_data/{task_type}/{dataset_name}/train_{cls_method_all}_{cls_encoding}_encoded_{dataset_name}.parquet")
            test = pd.read_parquet(f"/home/centos/phd/5th/prepared_data/{task_type}/{dataset_name}/test_{cls_method_all}_{cls_encoding}_encoded_{dataset_name}.parquet")
            val = pd.read_parquet(f"/home/centos/phd/5th/prepared_data/{task_type}/{dataset_name}/val_{cls_method_all}_{cls_encoding}_encoded_{dataset_name}.parquet")


            if without_inter:
                results_dir = f"/home/centos/phd/5th/predictive_results/predictions_without_inter/{task_type}/{dataset_name}/"
                train = train.drop(columns=train.columns[train.columns.str.contains('nr_ongoing_cases|interval|nr_past_events|arrival_rate|case_creation_rate|case_completion_rate')])
                test = test.drop(columns=test.columns[test.columns.str.contains('nr_ongoing_cases|interval|nr_past_events|arrival_rate|case_creation_rate|case_completion_rate')])
                val = val.drop(columns=val.columns[val.columns.str.contains('nr_ongoing_cases|interval|nr_past_events|arrival_rate|case_creation_rate|case_completion_rate')])
            #else:
            #    results_dir = f"/home/centos/phd/5th/predictive_results/predictions/{task_type}/{dataset_name}/"
            #    train = pd.read_parquet(f"./../prepared_data/{task_type}/{dataset_name}/train_{cls_method_all}_{cls_encoding}_encoded_{dataset_name}.parquet")
            #    test = pd.read_parquet(f"./../prepared_data/{task_type}/{dataset_name}/test_{cls_method_all}_{cls_encoding}_encoded_{dataset_name}.parquet")
            #    val = pd.read_parquet(f"./../prepared_data/{task_type}/{dataset_name}/val_{cls_method_all}_{cls_encoding}_encoded_{dataset_name}.parquet")


            #results_dir = f"/home/centos/phd/5th/predictive_results/predictions/{task_type}/{dataset_name}/"
            import os
            # cheack if results directory exists
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)

            #train = pd.read_parquet(
            #    f"/home/centos/phd/5th/prepared_data/{task_type}/{dataset_name}/train_{cls_method_all}_{cls_encoding}_encoded_{dataset_name}.parquet"
            #)
            cat_feat_idx = np.where(
                (train.dtypes == "object")
                & ~train.columns.isin([str(dataset_manager.label_col), "Treatment"])
            )[0]

            train_sampled = train.head(int(len(train) * (70 / 100)))
            # Remove the sampled data from the original training dataset to create the calibration dataset
            cal = train.drop(train_sampled.index)
            train = train_sampled
            del train_sampled

            X_train = train.drop(columns=[dataset_manager.label_col])
            y_train = train[dataset_manager.label_col]

            X_cal = cal.drop(columns=[dataset_manager.label_col])
            y_cal = cal[dataset_manager.label_col]

            #test = pd.read_parquet(
            #    f"/home/centos/phd/5th/prepared_data/{task_type}/{dataset_name}/test_{cls_method_all}_{cls_encoding}_encoded_{dataset_name}.parquet"
            #)
            X_test = test.drop(columns=[dataset_manager.label_col])
            y_test = test[dataset_manager.label_col]

            #val = pd.read_parquet(
            #    f"/home/centos/phd/5th/prepared_data/{task_type}/{dataset_name}/val_{cls_method_all}_{cls_encoding}_encoded_{dataset_name}.parquet"
            #)
            X_val = val.drop(columns=[dataset_manager.label_col])
            y_val = val[dataset_manager.label_col]
            

            model = train_model(X_train, y_train, task_type, cls_method, best_params)
            model.save_model(model, os.path.join(results_dir, f'{dataset_name}_{task_type}_{cls_method}_{cls_encoding}.sav'))


            if without_inter:
                pass
            else:
                import matplotlib.pyplot as plt
                import shap

                ex = shap.TreeExplainer(model)
                shap_values = ex.shap_values(X_test)

                name_shap = dataset_name + '_' + task_type + '_' + cls_method + '_' + cls_encoding
                results_shap = "/home/centos/phd/5th/predictive_results/shap_plots/"
                if not os.path.exists(results_shap):
                    os.makedirs(results_shap)

                file_name = os.path.join(results_shap, name_shap + '_shap_summary_plot.pdf')
                shap.summary_plot(shap_values, X_test, show=False)
                plt.savefig(file_name, bbox_inches='tight')
                plt.close()

                if with_conformal:
                    pass
                else:
                    import sys
                    print("Terminating script...")
                    sys.exit()


            if task_type == "classification":
                estimator = "clf"
            elif task_type == "regression":
                estimator = "reg"
            
            if with_conformal: 


                alpha_values = np.arange(
                    0.1, 1.0, 0.1
                )  # Generates alpha values from 0.1 to 0.9 with a step of 0.1
    
                for alpha in alpha_values:
                    alpha = np.round(alpha, 1)
                    acpi = ACPI(model_cali=model, estimator=estimator)
                    acpi.fit(X_cal, y_cal, nonconformity_func=None)
                    acpi.fit_calibration(
                        X_cal, y_cal, nonconformity_func=None, quantile=np.round(1 - alpha, 1)
                    )

                    # print("Start ACPI...")
                    # alpha = 0.1
                    # acpi = ACPI(model_cali=model, estimator=estimator,)
                    # acpi.fit(X_cal, y_cal, nonconformity_func=None)
                    # acpi.fit_calibration(X_cal, y_cal, nonconformity_func=None, quantile=1-alpha,)

                    if task_type == "classification":
                        # train
                        preds_train, preds_proba_train, score_train = (
                            get_preds_scores_classification(model, X_train, y_train)
                        )
                        # store results train
                        store_results(
                            results_dir,
                            dataset_name,
                            task_type,
                            cls_method,
                            cls_encoding,
                            np.array(y_train),
                            preds_train,
                            score_train,
                            probs_mean=preds_proba_train,
                            type="train",
                        )
    
                        # test
                        preds_test, preds_proba_test, score_test = (
                            get_preds_scores_classification(model, X_test, y_test)
                        )
                        y_pred_set_test = acpi.predict_pi(X_test, method="qrf")
                        # store results test
                        store_results(
                            results_dir,
                            dataset_name,
                            task_type,
                            cls_method,
                            cls_encoding,
                            np.array(y_test),
                            preds_test,
                            score_test,
                            probs_mean=preds_proba_test,
                            type="test",
                            y_pred_set=y_pred_set_test,
                            alpha=alpha,
                        )
    
                        # val
                        preds_val, preds_proba_val, score_val = (
                            get_preds_scores_classification(model, X_val, y_val)
                        )
                        y_pred_set_val = acpi.predict_pi(X_val, method="qrf")
                        # store results val
                        store_results(
                            results_dir,
                            dataset_name,
                            task_type,
                            cls_method,
                            cls_encoding,
                            np.array(y_val),
                            preds_val,
                            score_val,
                            probs_mean=preds_proba_val,
                            type="val",
                            y_pred_set=y_pred_set_val,
                            alpha=alpha,
                        )
    
                    elif task_type == "regression":
                        # train
                        preds_train, score_train = get_preds_scores_regression(
                            model, X_train, y_train
                        )
                        # store results train
                        store_results(
                            results_dir,
                            dataset_name,
                            task_type,
                            cls_method,
                            cls_encoding,
                            np.array(y_train),
                            preds_train,
                            score_train,
                            type="train",
                        )
    
                        # test
                        preds_test, score_test = get_preds_scores_regression(
                            model, X_test, y_test
                        )
                        y_lower_test, y_upper_test = acpi.predict_pi(X_test, method="qrf")
                        # store results test
                        store_results(
                            results_dir,
                            dataset_name,
                            task_type,
                            cls_method,
                            cls_encoding,
                            np.array(y_test),
                            preds_test,
                            score_test,
                            type="test",
                            y_lower=y_lower_test,
                            y_upper=y_upper_test,
                            alpha=alpha,
                        )

                        # val
                        preds_val, score_val = get_preds_scores_regression(
                            model, X_val, y_val
                        )
                        y_lower_val, y_upper_val = acpi.predict_pi(X_val, method="qrf")
                        # store results val
                        store_results(
                            results_dir,
                            dataset_name,
                            task_type,
                            cls_method,
                            cls_encoding,
                            np.array(y_val),
                            preds_val,
                            score_val,
                            type="val",
                            y_lower=y_lower_val,
                            y_upper=y_upper_val,
                            alpha=alpha,
                        )
    
                    scores_training[(dataset_name, task_type, cls_method, cls_encoding)] = (
                        score_train
                    )
                    scores_testing[(dataset_name, task_type, cls_method, cls_encoding)] = (
                        score_test
                    )
                    scores_validation[
                        (dataset_name, task_type, cls_method, cls_encoding)
                    ] = score_val



            else:     
                if task_type == "classification":
                    # train 
                    preds_train, preds_proba_train, score_train = get_preds_scores_classification(model, X_train, y_train)
                    # store results train
                    store_results(results_dir, dataset_name, task_type, cls_method, cls_encoding, np.array(y_train), preds_train, score_train, probs_mean=preds_proba_train, type="train")
                            
                    # test
                    preds_test, preds_proba_test, score_test = get_preds_scores_classification(model, X_test, y_test)
                    #y_pred_set_test = acpi.predict_pi(X_test, method='qrf')
                    # store results test
                    store_results(results_dir, dataset_name, task_type, cls_method, cls_encoding, np.array(y_test), preds_test, score_test, probs_mean=preds_proba_test, type="test")
                                       
                    # val
                    preds_val, preds_proba_val, score_val = get_preds_scores_classification(model, X_val, y_val)
                    #y_pred_set_val = acpi.predict_pi(X_val, method='qrf')
                    # store results val
                    store_results(results_dir, dataset_name, task_type, cls_method, cls_encoding, np.array(y_val), preds_val, score_val, probs_mean=preds_proba_val, type="val")
                                                    
                elif task_type == "regression":
                    # train 
                    preds_train, score_train = get_preds_scores_regression(model, X_train, y_train)
                    # store results train
                    store_results(results_dir, dataset_name, task_type, cls_method, cls_encoding, np.array(y_train), preds_train, score_train, type="train")
                                       
                    # test
                    preds_test, score_test = get_preds_scores_regression(model, X_test, y_test)
                    #y_lower_test, y_upper_test = acpi.predict_pi(X_test, method='qrf')
                    # store results test
                    store_results(results_dir, dataset_name, task_type, cls_method, cls_encoding, np.array(y_test), preds_test, score_test, type="test")
                                  
                    # val
                    preds_val, score_val = get_preds_scores_regression(model, X_val, y_val)
                    #y_lower_val, y_upper_val = acpi.predict_pi(X_val, method='qrf')
                    # store results val
                    store_results(results_dir, dataset_name, task_type, cls_method, cls_encoding, np.array(y_val), preds_val, score_val, type="val")




                scores_training[(dataset_name, task_type, cls_method, cls_encoding)] = score_train
                scores_testing[(dataset_name, task_type, cls_method, cls_encoding)] = score_test
                scores_validation[(dataset_name, task_type, cls_method, cls_encoding)] = score_val

