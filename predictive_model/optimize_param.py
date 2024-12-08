#!/usr/bin/env python
# coding: utf-8

# In[12]:


import warnings
warnings.filterwarnings('ignore')

from sys import argv
import sys
sys.path.append('/home/centos/phd/5th/common_files') 
import pandas as pd
import numpy as np
import os
from sklearn.metrics import roc_auc_score, mean_squared_error
from hyperopt import Trials, STATUS_OK, tpe, fmin, hp
from catboost import Pool, CatBoostClassifier, CatBoostRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from DatasetManager import DatasetManager
import gc
import hyperopt


from catboost import CatBoostClassifier, CatBoostRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, mean_squared_error
from hyperopt import STATUS_OK

from datetime import datetime

from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, log_loss

def regression_score_function_ctb(y_true, y_pred):
    # Assuming y_pred is a tuple (mean_predictions, std_predictions)
    mean_predictions = y_pred[:, 0]
    std_predictions = y_pred[:, 1]
    #mean_predictions, std_predictions = y_pred
    # Calculate RMSE for mean predictions
    rmse = mean_squared_error(y_true, mean_predictions)
    # Calculate MAE for mean predictions
    mae = mean_absolute_error(y_true, mean_predictions)
    # Calculate negative log likelihood (NLL) for uncertainty estimation
    nll = -np.mean(np.log(np.sqrt(2 * np.pi * std_predictions ** 2))) - 0.5 * np.mean(((y_true - mean_predictions) / std_predictions) ** 2)
    print(f"RMSE: {rmse}, MAE: {mae}, NLL: {nll}")
    return rmse
    #return {'RMSE': rmse, 'MAE': mae, 'NLL': nll}

def classification_score_function_ctb(y_true, y_pred):
    # Assuming y_pred is probabilities for each class
    print(f"ROC-AUC: {roc_auc_score(y_true, y_pred)}, LogLoss: {log_loss(y_true, y_pred)}")
    #print('ROC-AUC': roc_auc_score(y_true, y_pred), 'LogLoss': log_loss(y_true, y_pred))
    return roc_auc_score(y_true, y_pred)
    #return {'ROC-AUC': roc_auc_score(y_true, y_pred), 'LogLoss': log_loss(y_true, y_pred)}





def create_and_evaluate_model(args):
    global trial_nr
    if trial_nr % 50 == 0:
        print(trial_nr)
    print("Trial %s out of %s" % (trial_nr, n_iter))
    trial_nr += 1

    score = 0
    for current_train_names, current_test_names in dataset_manager.get_idx_split_generator(dt_for_splitting, n_splits=3):
        train_idxs = case_ids.isin(current_train_names)
        X_train = X_all[train_idxs]
        y_train = y_all[train_idxs]
        X_test = X_all[~train_idxs]
        y_test = y_all[~train_idxs]

        if task_type == "classification":
            if cls_method == "catboost":
                model = CatBoostClassifier(loss_function='Logloss',
                                           learning_rate=args['learning_rate'],
                                           depth=int(args['max_depth']),
                                           subsample=args['subsample'],
                                           #bootstrap_type='Bernoulli',
                                           verbose=False,
                                           random_seed=22,
                                           posterior_sampling=True,
                                           thread_count=-1)
            elif cls_method == "xgboost":
                model = XGBClassifier(learning_rate=args['learning_rate'],
                                      max_depth=int(args['max_depth']),
                                      subsample=args['subsample'],
                                      verbosity=0,
                                      random_state=22)
            elif cls_method == "lightgbm":
                model = LGBMClassifier(learning_rate=args['learning_rate'],
                                       max_depth=int(args['max_depth']),
                                       subsample=args['subsample'],
                                       verbosity=-1,
                                       random_state=22)
            elif cls_method == "randomforest":
                model = RandomForestClassifier(n_estimators=args['n_estimators'],
                                                max_depth=int(args['max_depth']),
                                                random_state=22)
            else:
                raise ValueError("Invalid cls_method for classification")
            score_function = roc_auc_score
        elif task_type == "regression":
            if cls_method == "catboost":
                model = CatBoostRegressor(loss_function='RMSEWithUncertainty', 
                                          posterior_sampling=True,
                                          learning_rate=args['learning_rate'],
                                          depth=int(args['max_depth']),
                                          subsample=args['subsample'],
                                          #bootstrap_type='Bernoulli',
                                          verbose=False,
                                          random_seed=22,
                                          thread_count=-1)
            elif cls_method == "xgboost":
                model = XGBRegressor(learning_rate=args['learning_rate'],
                                     max_depth=int(args['max_depth']),
                                     subsample=args['subsample'],
                                     verbosity=0,
                                     random_state=22)
            elif cls_method == "lightgbm":
                model = LGBMRegressor(learning_rate=args['learning_rate'],
                                       max_depth=int(args['max_depth']),
                                       subsample=args['subsample'],
                                       verbosity=-1,
                                       random_state=22)
            elif cls_method == "randomforest":
                model = RandomForestRegressor(n_estimators=args['n_estimators'],
                                               max_depth=int(args['max_depth']),
                                               random_state=22)
            else:
                raise ValueError("Invalid cls_method for regression")
            score_function = mean_squared_error
        else:
            raise ValueError("Invalid task_type")

        if cls_method=="lightgbm":
            import re
            X_train = X_train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
                
        if cls_method!="catboost":            
            model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train, cat_features=cat_feat_idx)
        preds = model.predict(X_test)

        if cls_method!="catboost":
            score += score_function(y_test, preds)
        else:
            if task_type=="classification":
                score += classification_score_function_ctb(y_test, preds)
                #pass
            elif task_type=="regression":
                score += regression_score_function_ctb(y_test, preds)




        #score += score_function(y_test, preds)

    if task_type == "classification":
        return {'loss': -score / n_splits, 'status': STATUS_OK, 'model': model}
    elif task_type == "regression":
        return {'loss': score / n_splits, 'status': STATUS_OK, 'model': model}



case_id_col = 'case_id'
activity_col = 'activity'
resource_col = 'resource'
timestamp_col = 'timestamp'
label_col = 'label'
treatment_col = "Treatment1"

dataset_ref = argv[1] # bpic2012, bpic2017
cls_encoding = argv[2] # laststate, agg, index, combined
cls_method = argv[3] # catboost, 'xgboost', 'lightgbm', 'randomforest',
task_type = argv[4] # "regression", "classification"


ref_to_task_types = {
        "regression": ["regression"],
        "classification": ["classification"]
        }
task_types = [task_type] if task_type not in ref_to_task_types else ref_to_task_types[task_type]

ref_to_cls = {
    "catboost": ["catboost"],
    "xgboost": ["xgboost"],
    "lightgbm": ["lightgbm"],
    "randomforest": ["randomforest"],
}
cls_methods = [cls_method] if cls_method not in ref_to_cls else ref_to_cls[cls_method]



dataset_ref_to_datasets = {
    "bpic2012": ["bpic2012"],
    "bpic2017": ["bpic2017"],
}

datasets = [dataset_ref] if dataset_ref not in dataset_ref_to_datasets else dataset_ref_to_datasets[dataset_ref]



encoding_dict = {  
    "laststate": ["static", "last"],
    "agg": ["static", "agg"], 
    "index": ["static", "index"],             
    "combined": ["static", "last", "agg"]
}

for dataset_name in datasets:
    for task_type in task_types:
        for cls_method in cls_methods:
                n_iter = 25  # Update this value as needed
                trial_nr = 0
                n_splits = 3
                print(f"dataset_name: {dataset_name}, task_type: {task_type}, cls_method: {cls_method}, cls_encoding: {cls_encoding}")
                            
                params_dir = f"/home/centos/phd/5th/predictive_results_v2/{task_type}/{dataset_name}/"           
                # Check if params_dir exists, otherwise create it
                if not os.path.exists(params_dir):
                    os.makedirs(params_dir)
                
                dataset_manager = DatasetManager(dataset_name, task_type)
                #print(f"Label_col: {str(dataset_manager.label_col)}")
                
                # Load the training data
                if cls_method!="catboost":
                    cls_method_all = "other"
                else:
                    cls_method_all = cls_method
                train = pd.read_parquet(f"/home/centos/phd/5th/prepared_data/{task_type}/{dataset_name}/train_{cls_method_all}_{cls_encoding}_encoded_{dataset_name}.parquet")
                cat_feat_idx = np.where((train.dtypes == 'object') & ~train.columns.isin([str(dataset_manager.label_col), "Treatment"]))[0]
                print(cat_feat_idx)                
                
                
                # Load the prefix data
                dt_prefixes = pd.read_parquet(f"/home/centos/phd/5th/prepared_data/{task_type}/{dataset_name}/train_prefixes_{dataset_name}.parquet")
                
                y_all = train[dataset_manager.label_col]
                if task_type == "classification":
                    y_all = y_all.astype(int)  # Ensure the target variable is integer type for classification
                elif task_type == "regression":
                    y_all = y_all.astype(float)  # Ensure the target variable is float type for regression
                else:
                    raise ValueError("Invalid task_type")
                                
                X_all = train.drop([str(dataset_manager.label_col)], axis=1)
                
                case_ids = dt_prefixes.groupby(dataset_manager.case_id_col).first()["orig_case_id"].reset_index(drop=True)
                dt_for_splitting = pd.DataFrame({dataset_manager.case_id_col: case_ids, dataset_manager.label_col: y_all}).drop_duplicates().reset_index(drop=True)
                
                #print('Optimizing parameters...')
                
                if cls_method == "catboost":
                    space = {
                        'learning_rate': hp.uniform("learning_rate", 0.01, 0.8),
                        'one_hot_max_size': hp.quniform('one_hot_max_size', 4, 255, 1),
                        'subsample': hp.uniform("subsample", 0.5, 1),
                        'max_depth': hp.quniform('max_depth', 6, 16, 1),
                        #'colsample_bytree': hp.uniform("colsample_bytree", 0.5, 1),
                        'bagging_temperature': hp.uniform('bagging_temperature', 0.0, 100),
                        'random_strength': hp.uniform('random_strength', 0.0, 100),
                        'l2_leaf_reg': hp.loguniform('l2_leaf_reg', 0, np.log(10)),
                        'n_estimators': hp.choice('n_estimators', [250, 500, 1000]),
                        #'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1)
                    }
                elif cls_method == "xgboost":
                    space = {
                        'learning_rate': hp.uniform("learning_rate", 0.01, 0.8),
                        'max_depth': hp.quniform('max_depth', 2, 10, 1),
                        'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
                        'subsample': hp.uniform("subsample", 0.5, 1),
                        'colsample_bytree': hp.uniform("colsample_bytree", 0.5, 1),
                        'gamma': hp.uniform("gamma", 0, 10),
                        'n_estimators': hp.choice('n_estimators', [100, 250, 500]),
                    }
                elif cls_method == "lightgbm":
                    space = {
                        'learning_rate': hp.uniform("learning_rate", 0.01, 0.8),
                        'max_depth': hp.quniform('max_depth', 2, 10, 1),
                        'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
                        'subsample': hp.uniform("subsample", 0.5, 1),
                        'colsample_bytree': hp.uniform("colsample_bytree", 0.5, 1),
                        'bagging_fraction': hp.uniform("bagging_fraction", 0.5, 1),
                        'feature_fraction': hp.uniform("feature_fraction", 0.5, 1),
                        'n_estimators': hp.choice('n_estimators', [100, 250, 500]),
                    }
                elif cls_method == "randomforest":
                    space = {
                        'n_estimators': hp.choice('n_estimators', [100, 250, 500]),
                        'max_depth': hp.quniform('max_depth', 2, 10, 1),
                        'min_samples_split': hp.quniform('min_samples_split', 2, 20, 1),
                        'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 10, 1),
                        'max_features': hp.choice('max_features', ['sqrt', 'log2', None]),
                    }
                else:
                    print("No Valid cls_method")
                    
                trials = Trials()
                start_time = datetime.now()
                best = fmin(create_and_evaluate_model, space, algo=tpe.suggest, max_evals=n_iter, trials=trials)

                best_params = hyperopt.space_eval(space, best)
                best_trial = trials.best_trial
                end_time = datetime.now()
                time_taken_dict = {}
                time_taken = (end_time - start_time).total_seconds()
                time_taken_dict["time_taken"] = time_taken

                #time_taken = (end_time - start_time).total_seconds()
                
                best_params_df = pd.DataFrame([best_params])
                best_trial_df = pd.DataFrame([best_trial]).astype(str)
                best_time_taken_df = pd.DataFrame([time_taken_dict]).astype(str)
                # Convert column names to strings
                #best_time_taken_df.columns = best_time_taken_df.columns.astype(str)


                #print(best_params_df)
                outfile_optimal_params = os.path.join(params_dir, f"optimal_params_{cls_encoding}_{cls_method}_{dataset_name}.parquet")
                outfile_best_trial = os.path.join(params_dir, f"best_trial_{cls_encoding}_{cls_method}_{dataset_name}.parquet")
                outfile_time_taken = os.path.join(params_dir, f"time_taken_{cls_encoding}_{cls_method}_{dataset_name}.parquet")

                print("save best param")
                best_params_df.to_parquet(outfile_optimal_params)
                print("save best trial")
                best_trial_df.to_parquet(outfile_best_trial)
                print("save time taken")
                best_time_taken_df.to_parquet(outfile_time_taken)




# In[ ]:


