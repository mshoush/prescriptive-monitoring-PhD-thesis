import warnings
warnings.filterwarnings('ignore')

from acpi import ACPI
import os
from sys import argv

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, mean_squared_error
#from catboost import CatBoostClassifier, CatBoostRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import sys
sys.path.append('/home/centos/phd/5th/common_files') 
from DatasetManager import DatasetManager
from catboost import Pool, CatBoostClassifier, CatBoostRegressor
# importing the joblib libraray
import joblib 



from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, log_loss

# Define a function to load the best parameters
def load_best_params(params_dir, cls_method):
    best_params_df = pd.read_parquet(params_dir)
    best_params = best_params_df.iloc[0].to_dict()
    try: 
        for param_name in ["n_estimators", "max_depth", "min_child_weight"]:
            best_params[param_name] = int(best_params[param_name])
    except:
        pass
    
    if cls_method=="randomforest":
        for param_name in ["min_samples_leaf", "min_samples_split"]:
            best_params[param_name] = int(best_params[param_name])
    
    #print("Best parameters:", best_params)
    return best_params


def regression_score_function_ctb(y_true, y_pred):
    mean_predictions = y_pred[:, 0] 
    std_predictions = y_pred[:, 1]
    
    # Calculate RMSE for mean predictions
    rmse = mean_squared_error(y_true, mean_predictions)
    # Calculate MAE for mean predictions
    mae = mean_absolute_error(y_true, mean_predictions)
    # Calculate negative log likelihood (NLL) for uncertainty estimation
    nll = -np.mean(np.log(np.sqrt(2 * np.pi * std_predictions ** 2))) - 0.5 * np.mean(((y_true - mean_predictions) / std_predictions) ** 2)
    #print(f"RMSE: {rmse}, MAE: {mae}, NLL: {nll}")
    return rmse

def classification_score_function_ctb(y_true, y_pred):
    #print(f"ROC-AUC: {roc_auc_score(y_true, y_pred)}, LogLoss: {log_loss(y_true, y_pred)}")
    return roc_auc_score(y_true, y_pred)


from catboost import CatBoostClassifier, CatBoostRegressor
import numpy as np

class Ensemble(object):
    def __init__(self, esize=10, iterations=1000, lr=0.1, random_strength=0, border_count=128, depth=6, seed=100, best_param=None, task_type='classification'):
        self.seed = seed
        self.esize = esize
        self.depth = depth
        self.iterations = iterations
        self.lr = lr
        self.random_strength = random_strength
        self.border_count = border_count
        self.best_param = best_param
        self.ensemble = []
        self.models = []
        self.task_type = task_type
        print(self.best_param)
        for e in range(self.esize):
            if self.task_type == 'classification':
                model = CatBoostClassifier(**self.best_param, 
                                            loss_function='Logloss',
                                            posterior_sampling=True,
                                            langevin=True,
                                            random_seed=self.seed + e,
                                            verbose=False)
                #score_function = classification_score_function_ctb
            elif self.task_type == 'regression':
                model = CatBoostRegressor(**self.best_param, 
                                           loss_function='RMSEWithUncertainty', 
                                           posterior_sampling=True, 
                                           random_seed=self.seed + e,
                                           verbose=False)
                #score_function = regression_score_function_ctb
            else:
                raise ValueError("Invalid task_type")
            self.ensemble.append(model)  

    def fit(self, X_train, y_train, cat_feat_idx, eval_set=None):
        for count, model in enumerate(self.ensemble, 1):
            print(f"\nFitting model {count}...")
            model.fit(X_train, y_train, cat_features=cat_feat_idx, early_stopping_rounds=10,)
            self.models.append(model)

    def save_model(self, filepath):
        joblib.dump(self.ensemble, filepath)

    def predict_proba(self, x):
        probs = [model.predict_proba(x) for model in self.ensemble]
        return np.stack(probs)

    def predict(self, x):
        preds = [model.predict(x) for model in self.ensemble]
        return np.stack(preds)    

    def predict_regression(self, x):
        ens_preds = [model.predict(x)  for model in self.ensemble] # mean prediction
        return ens_preds
    
    
    def get_acpi(self, X_cal, y_cal, cat_feat_idx, estimator, alpha):
        acpi_list = []
        print("Start ACPI...")
        print(f"\nExperimenting with alpha = {alpha}")
        for count, model in enumerate(self.models, 1):
            print(f"Start ACPI...{count}")
            acpi = ACPI(model_cali=model, estimator=estimator)                
            acpi.fit(X_cal, np.array(y_cal), nonconformity_func=None, categorical_features=cat_feat_idx) 
            acpi.fit_calibration(X_cal, np.array(y_cal), nonconformity_func=None, quantile=np.round(1 - alpha, 1), categorical_features=cat_feat_idx, n_iter_qrf=10)
                    
            acpi_list.append(acpi)
            if count == 3:
                break

        return acpi_list
                


# Uncertainties for classification
def entropy_of_expected(probs, epsilon=1e-10):
    mean_probs = np.mean(probs, axis=0)
    log_probs = -np.log(mean_probs + epsilon)
    return np.sum(mean_probs * log_probs, axis=1)


# Data uncer: avg(entropy of indviduals)
def expected_entropy(probs, epsilon=1e-10):
    log_probs = -np.log(probs + epsilon)

    return np.mean(np.sum(probs * log_probs, axis=2), axis=0)


# Knowledge uncer
def mutual_information(probs, epsilon):
    eoe = entropy_of_expected(probs, epsilon)
    exe = expected_entropy(probs, epsilon)
    return eoe - exe  # knowldge_ucer = total_uncer - data_uncer


def uncertainty_classification(probs, epsilon=1e-10):
    #print(f"Ensemble size: {len(probs)}\n")
    mean_probs = np.mean(probs, axis=0)  # avg ensamble prediction
    conf = np.max(mean_probs, axis=1)  # max avg ensamble prediction: predicted class

    eoe = entropy_of_expected(probs, epsilon)
    exe = expected_entropy(probs, epsilon)
    mutual_info = eoe - exe

    uncertainty = {'confidence': conf,
                   'total_uncer': eoe,  # Total uncer: entropy of the avg predictions
                   'data_uncer': exe,  # Data uncer: avg(entropy of indviduals)
                   'knowledge_uncer': mutual_info,  # Knowledge uncer
                   }    

    return uncertainty
    



def get_preds_uncertainty_regression(preds_t):       
    preds = np.mean(preds_t, axis=0)[:, 0]
    data_uncer = np.mean(preds_t, axis=0)[:, 1]  # Average estimated data uncertainty
    knowledge_uncer = np.var(preds_t, axis=0)[:, 0]  # Estimated knowledge uncertainty
    total_uncer = np.sum(np.array([data_uncer, knowledge_uncer]), axis=0)  # Total uncertainty
    return preds, data_uncer, knowledge_uncer, total_uncer


def get_reliability_regression(preds):
    deviation = (preds - np.mean(preds, axis=0)) / np.std(preds, axis=0)
    reliability_estimate = 1 - (deviation / np.max(deviation))
    return deviation, reliability_estimate

def get_preds_uncertainty_classification(preds_e, probs):
    preds = (np.mean(preds_e, axis=0) >= 0.5).astype(int)
    data_uncer = uncertainty_classification(probs)['data_uncer']
    knowledge_uncer = uncertainty_classification(probs)['knowledge_uncer']
    total_uncer = uncertainty_classification(probs)['total_uncer']
    probs_mean = np.mean(probs, axis=0)
    return preds, data_uncer, knowledge_uncer, total_uncer, probs_mean


def get_reliability_classification(preds):
    deviation = np.subtract(1, preds) / 1    
    reliability = np.count_nonzero(np.transpose(deviation), axis=1)/deviation.shape[0]
    return  deviation, reliability



# save results_dict as pickle file
import pickle
def save_results(results_dict, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(results_dict, f)
    return None

# load results_dict from pickle file
def load_results(file_name):
    with open(file_name, 'rb') as f:
        results_dict = pickle.load(f)
    return results_dict


def store_results(results_dir, dataset_name, task_type, cls_method, cls_encoding, \
                y_tru, preds, data_uncer, knowledge_uncer, total_uncer, reliability_estimate, deviation, score, probs_mean=None, type=None, \
                y_pred_set=None, y_lower=None, y_upper=None, alpha=None):
    
    results_dict = {}

    results_dict[(dataset_name, task_type, cls_method, cls_encoding)] = {
        'actual': y_tru,
        'preds': preds,
        'probs': probs_mean,
        'data_uncer': data_uncer,
        'knowledge_uncer': knowledge_uncer,
        'total_uncer': total_uncer,
        'reliability_estimate': reliability_estimate,
        'deviation': deviation,
        'score': score,   
        f'y_pred_set_{alpha}': y_pred_set,
        f'y_lower_{alpha}': y_lower,
        f'y_upper_{alpha}': y_upper,        
        'score': score
    }

    # save results_dict_classification
    import os
    name = dataset_name + '_' + task_type + '_' + cls_method + '_' + cls_encoding + "_" + type + "_" + str(alpha)
    file_name = os.path.join(results_dir, name + '_results_dict.pkl')
    save_results(results_dict, file_name)


    return results_dict

# def store_results(results_dir, dataset_name, task_type, cls_method, cls_encoding, \
#                 y_tru, preds, data_uncer, knowledge_uncer, total_uncer, reliability_estimate, deviation, score, probs_mean=None, type=None):
    
#     results_dict = {}

#     results_dict[(dataset_name, task_type, cls_method, cls_encoding)] = {
#         'actual': y_tru,
#         'preds': preds,
#         'probs': probs_mean,
#         'data_uncer': data_uncer,
#         'knowledge_uncer': knowledge_uncer,
#         'total_uncer': total_uncer,
#         'reliability_estimate': reliability_estimate,
#         'deviation': deviation,
#         'score': score
#     }

#     # save results_dict_classification
#     import os
#     name = dataset_name + '_' + task_type + '_' + cls_method + '_' + cls_encoding + "_" + type
#     file_name = os.path.join(results_dir, name + '_results_dict.pkl')
#     save_results(results_dict, file_name)


#     return results_dict

def get_pred_interval(X_test, acpi_list, cat_feat_idx):    
    preds_intervals_lower_all = [acpi.predict_pi(X_test, method='qrf', categorical_features=cat_feat_idx)[0] for acpi in acpi_list]
    preds_intervals_upper_all = [acpi.predict_pi(X_test, method='qrf', categorical_features=cat_feat_idx)[1] for acpi in acpi_list]
    preds_intervals_lower_avg = np.mean(preds_intervals_lower_all, axis=0)
    preds_intervals_upper_avg = np.mean(preds_intervals_upper_all, axis=0)
    
    return preds_intervals_lower_avg, preds_intervals_upper_avg





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

with_conformal = argv[5].lower() == "true" #False
without_inter =  argv[6].lower() == "true" #True


ref_to_task_types = {"regression": ["regression"], "classification": ["classification"]}
task_types = (
    [task_type] if task_type not in ref_to_task_types else ref_to_task_types[task_type]
)

ref_to_cls = {
    "catboost": ["catboost"],
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



encoding_dict = {
    "laststate": ["static", "last"],
    "agg": ["static", "agg"],
    "index": ["static", "index"],
    "combined": ["static", "last", "agg"],
}

scores_training = {}
scores_testing = {}
scores_validation = {}

#with_conformal = False
#without_inter = True

for dataset_name in datasets:
    for task_type in task_types:
        for cls_method in cls_methods:
            print(f"dataset_name: {dataset_name}, task_type: {task_type}, cls_method: {cls_method}, cls_encoding: {cls_encoding}")
            
            dataset_manager = DatasetManager(dataset_name, task_type)
            results_dir = f"/home/centos/phd/5th/predictive_results/predictions/{task_type}/{dataset_name}/"
            
            # cheack if results directory exists
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)       

            # Load the best parameters
            params_dir = f"/home/centos/phd/5th/predictive_results/{task_type}/{dataset_name}/optimal_params_{cls_encoding}_{cls_method}_{dataset_name}.parquet"
            best_params = load_best_params(params_dir, cls_method)
            

            if cls_method!="catboost":
                cls_method_all = "other"    
            else:
                cls_method_all = cls_method
                if "colsample_bytree" in best_params.keys():
                    del best_params["colsample_bytree"]
                    del best_params["min_child_weight"]

            
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

            # cheack if results directory exists
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)


                    
            #train = pd.read_parquet(f"/home/centos/phd/5th/prepared_data/{task_type}/{dataset_name}/train_{cls_method_all}_{cls_encoding}_encoded_{dataset_name}.parquet")
            cat_feat_idx = np.where((train.dtypes == 'object') & ~train.columns.isin([str(dataset_manager.label_col), "Treatment"]))[0]
                            
            train_sampled = train.head(int(len(train)*(70/100))) 
            # Remove the sampled data from the original training dataset to create the calibration dataset
            cal = train.drop(train_sampled.index)                
            train = train_sampled
            del train_sampled

            
            X_train = train.drop(columns=[dataset_manager.label_col])
            y_train = train[dataset_manager.label_col]
            
            X_cal = cal.drop(columns=[dataset_manager.label_col])
            y_cal = cal[dataset_manager.label_col]
            

            #test = pd.read_parquet(f"/home/centos/phd/5th/prepared_data/{task_type}/{dataset_name}/test_{cls_method_all}_{cls_encoding}_encoded_{dataset_name}.parquet")
            X_test = test.drop(columns=[dataset_manager.label_col])
            y_test = test[dataset_manager.label_col]
            
            #val = pd.read_parquet(f"/home/centos/phd/5th/prepared_data/{task_type}/{dataset_name}/val_{cls_method_all}_{cls_encoding}_encoded_{dataset_name}.parquet")
            X_val = val.drop(columns=[dataset_manager.label_col])                
            y_val = val[dataset_manager.label_col]
            
            en_size=20
            ens = Ensemble(esize=en_size, iterations=1000, lr=0.1, depth=6, seed=2, random_strength = 100, best_param=best_params, task_type=task_type)
            ens.fit(X_train, y_train, cat_feat_idx)

            # Save the final ensemble model
            ens.save_model(os.path.join(results_dir, f'{dataset_name}_{task_type}_{cls_method}_{cls_encoding}.sav'))
            #ens.save_model('ensemble_model.pkl')
            #ens.save_model(os.path.join(results_dir, f'{dataset_name}_{task_type}_{cls_method}_{cls_encoding}_ensemble_model.pkl'))
           

            #if without_inter:
            #    pass
            #else:

            #    import shap
            #    import matplotlib.pyplot as plt
            #    # Explain all the predictions in the test set
            #    shap_values_all = []
            #    for model in ens.models:
            #        shap_values = shap.TreeExplainer(model).shap_values(X_test)
            #        shap_values_all.append(shap_values)
            #    shap_values_all = np.mean(shap_values_all, axis=0)


            #    # Save SHAP summary plot
            #    name_shap = f"{dataset_name}_{task_type}_{cls_method}_{cls_encoding}"
            #    results_shap = "/home/centos/phd/5th/predictive_results/shap_plots/"
            #    if not os.path.exists(results_shap):
            #        os.makedirs(results_shap)
            #    file_name = os.path.join(results_shap, f"{name_shap}_shap_summary_plot.pdf")
    
            #    shap.summary_plot(shap_values_all, X_test, show=False)
            #    plt.savefig(file_name, bbox_inches='tight')
            #    plt.close()

            #    if with_conformal:
            #        pass
            #    else:
            #        import sys
            #        print("Terminating script...")
            #        sys.exit()
            
                        
            
            if task_type == "classification":
                estimator='clf'                                 
            elif task_type == "regression":
                estimator='reg'
            
            if with_conformal:
                alpha_values = np.arange(
                    0.1, 1.0, 0.1
                )  # Generates alpha values from 0.1 to 0.9 with a step of 0.1
                for alpha in alpha_values:
                    alpha = np.round(alpha, 1)
                    acpi_list = ens.get_acpi(X_cal, y_cal, cat_feat_idx, estimator, alpha)
                    print(f"Done! ACPI with alpha: {alpha}")             
                    # Predictions and Uncertainties for Classification
                    if task_type == "classification":
                        probs_train = ens.predict_proba(X_train)
                        probs_test = ens.predict_proba(X_test)
                        probs_val = ens.predict_proba(X_val)                        
                        preds_train_e = ens.predict(X_train)
                        preds_test_e = ens.predict(X_test)
                        preds_val_e = ens.predict(X_val)                    
                        # train
                        preds_train, data_uncer_train, knowledge_uncer_train, total_uncer_train, probs_train_mean =  get_preds_uncertainty_classification(preds_train_e, probs_train)
                        deviation_train, reliability_estimate_train = get_reliability_classification(preds_train_e)
                        score_train = classification_score_function_ctb(y_train, preds_train)  
                        print(f"Score train: {score_train}")
                        # store results train
                        store_results(results_dir, dataset_name, task_type, cls_method, cls_encoding, np.array(y_train), preds_train, data_uncer_train, knowledge_uncer_train, total_uncer_train,reliability_estimate_train, deviation_train, score_train, probs_mean=probs_train_mean, type="train")                                                               
                        # test
                        preds_test, data_uncer_test, knowledge_uncer_test, total_uncer_test, probs_test_mean = get_preds_uncertainty_classification(preds_test_e, probs_test)
                        deviation_test, reliability_estimate_test = get_reliability_classification(preds_test_e)
                        score_test = classification_score_function_ctb(y_test, preds_test)
                        print(f"Score test: {score_test}")
                        # ACPI
                        preds_sets_all_test = [acpi.predict_pi(X_test, method='qrf', categorical_features=cat_feat_idx) for acpi in acpi_list]
                        preds_sets_avg_test = np.round(np.mean([array.astype(int) for array in preds_sets_all_test], axis=0)).astype(bool)                                  
                        # store results test
                        store_results(results_dir, dataset_name, task_type, cls_method, cls_encoding, np.array(y_test), preds_test, data_uncer_test, knowledge_uncer_test, total_uncer_test, reliability_estimate_test, deviation_test, score_test, probs_mean=probs_test_mean, type="test", y_pred_set=preds_sets_avg_test, alpha=alpha)                                   
                        # val
                        preds_val, data_uncer_val, knowledge_uncer_val, total_uncer_val, probs_val_mean = get_preds_uncertainty_classification(preds_val_e, probs_val)
                        deviation_val, reliability_estimate_val = get_reliability_classification(preds_val_e)
                        score_val = classification_score_function_ctb(y_val, preds_val)
                        print(f"Score val: {score_val}")
                        # ACPI
                        preds_sets_all_val = [acpi.predict_pi(X_val, method='qrf', categorical_features=cat_feat_idx) for acpi in acpi_list]
                        preds_sets_avg_val = np.round(np.mean([array.astype(int) for array in preds_sets_all_val], axis=0)).astype(bool)
                        # store results val
                        store_results(results_dir, dataset_name, task_type, cls_method, cls_encoding, np.array(y_val), preds_val, data_uncer_val, knowledge_uncer_val, total_uncer_val, reliability_estimate_val, deviation_val, score_val, probs_mean=probs_val_mean, type="val", y_pred_set=preds_sets_avg_val, alpha=alpha)                        
                        scores_training[(dataset_name, task_type, cls_method, cls_encoding)] = score_train
                        scores_testing[(dataset_name, task_type, cls_method, cls_encoding)] = score_test
                        scores_validation[(dataset_name, task_type, cls_method, cls_encoding)] = score_val                    
                    elif task_type == "regression":
                        preds_train_e = ens.predict_regression(X_train)
                        preds_test_e = ens.predict_regression(X_test)  
                        preds_val_e = ens.predict_regression(X_val)                       
                        preds_train, data_uncer_train, knowledge_uncer_train, total_uncer_train = get_preds_uncertainty_regression(preds_train_e)
                        deviation_train, reliability_estimate_train = get_reliability_regression(preds_train)
                        score_train = regression_score_function_ctb(y_train, np.mean(preds_train_e, axis=0))        
                        print(f"Score train: {score_train}")
                        # store results train
                        store_results(results_dir, dataset_name, task_type, cls_method, cls_encoding, np.array(y_train), preds_train, data_uncer_train, knowledge_uncer_train, total_uncer_train, reliability_estimate_train, deviation_train, score_train, type="train")                                                
                        # test
                        preds_test, data_uncer_test, knowledge_uncer_test, total_uncer_test = get_preds_uncertainty_regression(preds_test_e)
                        deviation_test, reliability_estimate_test = get_reliability_regression(preds_test)
                        score_test = regression_score_function_ctb(y_test, np.mean(preds_test_e, axis=0))
                        print(f"Score test: {score_test}")
                        # ACPI
                        preds_intervals_lower_avg_test, preds_intervals_upper_avg_test = get_pred_interval(X_test, acpi_list, cat_feat_idx)    
                        # store results test
                        store_results(results_dir, dataset_name, task_type, cls_method, cls_encoding, np.array(y_test), preds_test, data_uncer_test, knowledge_uncer_test, total_uncer_test, reliability_estimate_test, deviation_test, score_test, type="test", y_lower=preds_intervals_lower_avg_test, y_upper=preds_intervals_upper_avg_test, alpha=alpha)                                       
                        preds_val, data_uncer_val, knowledge_uncer_val, total_uncer_val = get_preds_uncertainty_regression(preds_val_e)
                        deviation_val, reliability_estimate_val = get_reliability_regression(preds_val)
                        score_val = regression_score_function_ctb(y_val, np.mean(preds_val_e, axis=0))
                        print(f"Score val: {score_val}")
                        # ACPI
                        preds_intervals_lower_avg_val, preds_intervals_upper_avg_val = get_pred_interval(X_val, acpi_list, cat_feat_idx)
                        # store results val 
                        store_results(results_dir, dataset_name, task_type, cls_method, cls_encoding, np.array(y_val), preds_val, data_uncer_val, knowledge_uncer_val, total_uncer_val, reliability_estimate_val, deviation_val, score_val, type="val", y_lower=preds_intervals_lower_avg_val, y_upper=preds_intervals_upper_avg_val, alpha=alpha)   
                        scores_training[(dataset_name, task_type, cls_method, cls_encoding)] = score_train
                        scores_testing[(dataset_name, task_type, cls_method, cls_encoding)] = score_test
                        scores_validation[(dataset_name, task_type, cls_method, cls_encoding)] = score_val                                                
                    else:
                        raise ValueError("Invalid task_type")

            else:
                if task_type == "classification":
                    probs_train = ens.predict_proba(X_train)
                    probs_test = ens.predict_proba(X_test)
                    probs_val = ens.predict_proba(X_val)

                    preds_train_e = ens.predict(X_train)
                    preds_test_e = ens.predict(X_test)
                    preds_val_e = ens.predict(X_val)

                    # train
                    preds_train, data_uncer_train, knowledge_uncer_train, total_uncer_train, probs_train_mean =  get_preds_uncertainty_classification(preds_train_e, probs_train)
                    deviation_train, reliability_estimate_train = get_reliability_classification(preds_train_e)
                    score_train = classification_score_function_ctb(y_train, preds_train)

                    print(f"Score train: {score_train}")
                    # store results train
                    store_results(results_dir, dataset_name, task_type, cls_method, cls_encoding, np.array(y_train), preds_train, data_uncer_train, knowledge_uncer_train, total_uncer_train,reliability_estimate_train, deviation_train, score_train, probs_mean=probs_train_mean, type="train")

                    # test
                    preds_test, data_uncer_test, knowledge_uncer_test, total_uncer_test, probs_test_mean = get_preds_uncertainty_classification(preds_test_e, probs_test)
                    deviation_test, reliability_estimate_test = get_reliability_classification(preds_test_e)
                    score_test = classification_score_function_ctb(y_test, preds_test)
                    print(f"Score test: {score_test}")
                    # ACPI
                    #preds_sets_all_test = [acpi.predict_pi(X_test, method='qrf', categorical_features=cat_feat_idx) for acpi in acpi_list]
                    #preds_sets_avg_test = np.round(np.mean([array.astype(int) for array in preds_sets_all_test], axis=0)).astype(bool)
                    
                    # store results test
                    store_results(results_dir, dataset_name, task_type, cls_method, cls_encoding, np.array(y_test), preds_test, data_uncer_test, knowledge_uncer_test, total_uncer_test, reliability_estimate_test, deviation_test, score_test, probs_mean=probs_test_mean, type="test")
                    # val
                    preds_val, data_uncer_val, knowledge_uncer_val, total_uncer_val, probs_val_mean = get_preds_uncertainty_classification(preds_val_e, probs_val)
                    deviation_val, reliability_estimate_val = get_reliability_classification(preds_val_e)
                    score_val = classification_score_function_ctb(y_val, preds_val)
                    print(f"Score val: {score_val}")
                    # ACPI
                    #preds_sets_all_val = [acpi.predict_pi(X_val, method='qrf', categorical_features=cat_feat_idx) for acpi in acpi_list]
                    #preds_sets_avg_val = np.round(np.mean([array.astype(int) for array in preds_sets_all_val], axis=0)).astype(bool)
                        
                    # store results val
                    store_results(results_dir, dataset_name, task_type, cls_method, cls_encoding, np.array(y_val), preds_val, data_uncer_val, knowledge_uncer_val, total_uncer_val, reliability_estimate_val, deviation_val, score_val, probs_mean=probs_val_mean, type="val")

                                      

                    scores_training[(dataset_name, task_type, cls_method, cls_encoding)] = score_train
                    scores_testing[(dataset_name, task_type, cls_method, cls_encoding)] = score_test
                    scores_validation[(dataset_name, task_type, cls_method, cls_encoding)] = score_val

                elif task_type == "regression":
                    preds_train_e = ens.predict_regression(X_train)
                    preds_test_e = ens.predict_regression(X_test)
                    preds_val_e = ens.predict_regression(X_val)

                    preds_train, data_uncer_train, knowledge_uncer_train, total_uncer_train = get_preds_uncertainty_regression(preds_train_e)
                    deviation_train, reliability_estimate_train = get_reliability_regression(preds_train)
                    score_train = regression_score_function_ctb(y_train, np.mean(preds_train_e, axis=0))

                    print(f"Score train: {score_train}")
                    # store results train
                    store_results(results_dir, dataset_name, task_type, cls_method, cls_encoding, np.array(y_train), preds_train, data_uncer_train, knowledge_uncer_train, total_uncer_train, reliability_estimate_train, deviation_train, score_train, type="train")
                    # test
                    preds_test, data_uncer_test, knowledge_uncer_test, total_uncer_test = get_preds_uncertainty_regression(preds_test_e)
                    deviation_test, reliability_estimate_test = get_reliability_regression(preds_test)
                    score_test = regression_score_function_ctb(y_test, np.mean(preds_test_e, axis=0))
                    print(f"Score test: {score_test}")
                    # ACPI
                    #preds_intervals_lower_avg_test, preds_intervals_upper_avg_test = get_pred_interval(X_test, acpi_list, cat_feat_idx)
                    
                    # store results test
                    store_results(results_dir, dataset_name, task_type, cls_method, cls_encoding, np.array(y_test), preds_test, data_uncer_test, knowledge_uncer_test, total_uncer_test, reliability_estimate_test, deviation_test, score_test, type="test",)

                    preds_val, data_uncer_val, knowledge_uncer_val, total_uncer_val = get_preds_uncertainty_regression(preds_val_e)
                    deviation_val, reliability_estimate_val = get_reliability_regression(preds_val)
                    score_val = regression_score_function_ctb(y_val, np.mean(preds_val_e, axis=0))
                    print(f"Score val: {score_val}")
                    # ACPI
                    #preds_intervals_lower_avg_val, preds_intervals_upper_avg_val = get_pred_interval(X_val, acpi_list, cat_feat_idx)
                    # store results val 
                    store_results(results_dir, dataset_name, task_type, cls_method, cls_encoding, np.array(y_val), preds_val, data_uncer_val, knowledge_uncer_val, total_uncer_val, reliability_estimate_val, deviation_val, score_val, type="val")
                    scores_training[(dataset_name, task_type, cls_method, cls_encoding)] = score_train
                    scores_testing[(dataset_name, task_type, cls_method, cls_encoding)] = score_test
                    scores_validation[(dataset_name, task_type, cls_method, cls_encoding)] = score_val
                else:
                    raise ValueError("Invalid task_type")
                
