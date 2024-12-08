import pandas as pd
import numpy as np
from hyperopt import fmin, tpe, hp, Trials
from tqdm import tqdm
import threading
import random
import time
import gc
from hyperopt.fmin import generate_trials_to_calculate


# show all columns in pandas
pd.set_option("display.max_columns", None)


import os
import random
import numpy as np


def create_results_directory(log_name, iteration, result_dir, t_dur=1, t_dist="fixed"):
    if t_dist == "normal":
        treatment_duration = int(random.uniform(1, t_dur))
        folder = f"{result_dir}/results_normal_{iteration}/{log_name}/"
    elif t_dist == "fixed":
        treatment_duration = t_dur
        folder = f"{result_dir}/results_fixed_{iteration}/{log_name}/"
    else:
        treatment_duration = int(np.random.exponential(t_dur, size=1))
        folder = f"{result_dir}/results_exp_{iteration}/{log_name}/"

    results_dir = f"{folder}"

    if not os.path.exists(os.path.join(results_dir)):
        os.makedirs(os.path.join(results_dir))

    return results_dir, treatment_duration, t_dist


import pandas as pd
import numpy as np
from hyperopt import fmin, tpe, hp, Trials, space_eval, atpe
from hyperopt.fmin import generate_trials_to_calculate
from tqdm import tqdm
import threading
import random
import time
import gc
import pickle
from env_manager import EnvManager
from online_utils import *
from filter_cases import FilterCases
from functools import partial
import os
from sys import argv
from itertools import product
import warnings

# Show all columns in pandas
pd.set_option("display.max_columns", None)

# Suppress numpy warnings
np.warnings = warnings



def generate_param_combinations(space):
    """Generate parameter combinations from a given search space."""
    keys, values = zip(*space.items())
    for v in product(*values):
        yield dict(zip(keys, v))




import pandas as pd
import numpy as np
from hyperopt import fmin, tpe, hp, Trials, space_eval, atpe
from hyperopt.fmin import generate_trials_to_calculate
from tqdm import tqdm
import threading
import random
import time
import gc
import pickle
from env_manager import EnvManager
from online_utils import *
from filter_cases import FilterCases
from functools import partial
import os
from sys import argv
from itertools import product
import warnings
import datetime

# Show all columns in pandas
pd.set_option('display.max_columns', None)

# Suppress numpy warnings
np.warnings = warnings

# Function to get the max value from a column in df
def get_mean_ub(df, col):
    return df[col].mean()

# Function to get the max value from a column in df
def get_max_ub(df, col):
    return df[col].max()

def generate_param_combinations(space):
    """Generate parameter combinations from a given search space."""
    keys, values = zip(*space.items())
    for v in product(*values):
        yield dict(zip(keys, v))


def main(dataset_name, rules, outcome_gain, treatment_cost, optimization, netgainType, resources_list):
    best_params = {}
    sample = "sample1"
    data_types = ["val", "test"] if optimization == "WithOptimization" else ["test"]
    default_threshold_parameters = {
         "neg_proba": 0.5,
        "uncer_threshold": 0.5,
        "confidence_level": 0.5,  
              
        "causal_uncer_thre": 0.0,
        "remtime_uncer_thre": 10,
        "reliability_thre": 0.5,
        "remtime_thre": 10,
        "causal_thre": 0,      
          
        "weight_need": 1,
        "weight_effect": 1,
        "weight_urgency": 1,
        "weight_uncertainty": 1,
    }   


    # for VM
    results_dir_base = "/home/centos/phd/5th/results_all/online/"
    data_dir_base = "/home/centos/phd/5th/results_all/online/"
   
    treated_cases = {}
    netGain = {}
    optimization_results = {}

    iteration = 1
    results_directory, treatment_duration, t_dist = create_results_directory(dataset_name, iteration, result_dir=results_dir_base, t_dur=1, t_dist="fixed")

    for resources in resources_list:
        availble_resources = list(range(1, resources + 1))
        print(f"Available resources: {availble_resources}")
    
        for rule_name in rules: 
            
            for data_type in data_types: 
                treated_cases[rule_name] = []
                print(f"\nProcessing data type: {data_type}, rule: {rule_name} ...")
    
                file_name = f"{data_type}_{dataset_name}_all_online.parquet"
                data_dir = f"{data_dir_base}/{data_type}/{dataset_name}/"    
                results_dir = f"{results_dir_base}/{data_type}/{dataset_name}/"

                data = EnvManager(data_type, dataset_name, results_dir, file_name, data_dir)    
                data_f = data.return_df()
                
                # Compute urgency and causal upper bounds
                alpha_value, alpha_col = get_alpha_col(default_threshold_parameters['confidence_level'])
                mean_urgency_ub = get_mean_ub(data_f, f"surv_upper_{alpha_value}")
                max_causal_ub = get_max_ub(data_f, f"causal_upper_{alpha_value}")
                
                
                
                if data_type == "val":
                    print(f"Running Hyperopt optimization for rule: {rule_name} ...")
                    space_w = {
                    'weight_need': hp.choice('weight_need', [0.5, 1.0, 2.0]),
                    'weight_effect': hp.choice('weight_effect', [0.5, 1.0, 2.0]),
                    'weight_urgency': hp.choice('weight_urgency', [0.5, 1.0, 2.0]),
                    'weight_uncertainty': hp.choice('weight_uncertainty', [0.5, 1.0, 2.0]),  
                    }

                    space_p, space_dict = create_search_space_filter(mean_urgency_ub, max_causal_ub, rule_name=rule_name)
                    #combined_space = {**space_p}
                    combined_space = {**space_w, **space_p}
                    early_stop_fn_ = min(len(combined_space.keys())*2+1, 20) #* 2 + 1
                    print(f"LEN of len(combined_space.keys()): {len(combined_space.keys())}")

                    # Record start time of optimization
                    start_time = datetime.datetime.now()

                    init_vals = [{k: default_threshold_parameters[k] for k in combined_space.keys()}]
                    trials = generate_trials_to_calculate(init_vals)
                    SEED = 22
                                                
                    from hyperopt.early_stop import no_progress_loss
                    
                    tried_params = set()
                    count = [0]
                                
                    partial_objective = partial(objective, tried_params=tried_params, count=count, trials=trials, rule_name=rule_name, data=data, data_f=data_f, outcome_gain=outcome_gain, treatment_cost=treatment_cost, netgainType=netgainType, resources=resources)
                    best_params = fmin(fn=partial_objective, space=combined_space, trials=trials, algo=tpe.suggest, max_evals=25, early_stop_fn=no_progress_loss(early_stop_fn_), rstate=np.random.default_rng(SEED))
                    best_params = space_eval(combined_space, best_params)
                    
                    # Record end time of optimization
                    end_time = datetime.datetime.now()
                    time_taken_to_optimize = (end_time - start_time).total_seconds() / 60 

                    best_trial_index = trials.best_trial['tid']
                    best_trial = trials.trials[best_trial_index]

                    optimization_results[rule_name] = {
                        "best_params": best_params,
                        "best_trial": best_trial,
                        "elapsed_time": time_taken_to_optimize
                    }

                    best_params = {**default_threshold_parameters, **best_params}
                    continue
                
                neg_proba_thre = np.round(best_params.get('neg_proba', default_threshold_parameters['neg_proba']), 2)
                total_uncer_thre = np.round(best_params.get('uncer_threshold', default_threshold_parameters['uncer_threshold']), 2)
                confidence_level = np.round(best_params.get('confidence_level', default_threshold_parameters['confidence_level']), 2)
                
                causal_uncer_thre = np.round(best_params.get('causal_uncer_thre', default_threshold_parameters['causal_uncer_thre']), 2)
                remtime_uncer_thre = np.round(best_params.get('remtime_uncer_thre', default_threshold_parameters['remtime_uncer_thre']), 2)
                reliability_thre = np.round(best_params.get('reliability_thre', default_threshold_parameters['reliability_thre']), 2)
                remtime_thre = np.round(best_params.get('remtime_thre', default_threshold_parameters['remtime_thre']), 2)
                causal_thre = np.round(best_params.get('causal_thre', default_threshold_parameters['causal_thre']), 2)
                
            
        
                weight_need = np.round(best_params.get('weight_need', default_threshold_parameters['weight_need']), 2)
                weight_effect = np.round(best_params.get('weight_effect', default_threshold_parameters['weight_effect']), 2)
                weight_urgency = np.round(best_params.get('weight_urgency', default_threshold_parameters['weight_urgency']), 2)
                weight_uncertainty = np.round(best_params.get('weight_uncertainty', default_threshold_parameters['weight_uncertainty']), 2)
                
                alpha_value, alpha_col = get_alpha_col(confidence_level)
                mean_urgency_ub = get_mean_ub(data_f, f"surv_upper_{alpha_value}")
                max_causal_ub = get_max_ub(data_f, f"causal_upper_{alpha_value}")
                # mean_urgency_ub, mean_causal_ub
                print(mean_urgency_ub)
                print(max_causal_ub)
                print(f"Params: \nNeg_proba_thre: {neg_proba_thre}\nTotal_uncer_thre: {total_uncer_thre}\nConfidence_level: {confidence_level}\n")
                print(f"params: \nCausal_uncer_thre: {causal_uncer_thre}\nRemtime_uncer_thre: {remtime_uncer_thre}\nReliability_thre: {reliability_thre}\nRemtime_thre: {remtime_thre}\nCausal_thre: {causal_thre}")
                
                candidate_cases = {}
                treated_cases_ids = []
                net_gain = 0
                
            
                while not data.finished:
                    event = data.get_event_by_event()
                    event = FilterCases.filter_cases(event, neg_proba_thre, total_uncer_thre, confidence_level, 
                                                     causal_thre, reliability_thre, remtime_thre,
                                                     causal_uncer_thre, remtime_uncer_thre,                                                     
                                                     rule_name)

                    if event is not None:
                        alpha_value, alpha_col = get_alpha_col(confidence_level)

                        if event["case_id"].iloc[0] in treated_cases_ids:
                            if event["case_id"].iloc[0] in candidate_cases:
                                del candidate_cases[event["case_id"].iloc[0]]
                            continue
                        else:
                            candidate_cases = process_event(event, candidate_cases, outcome_gain, treatment_cost, alpha_value, treated_cases_ids)
                        
                        if candidate_cases:
                            best_case_key = rank_candidate_cases(candidate_cases, rule_name, weight_need, weight_effect, weight_urgency, weight_uncertainty)

                            if best_case_key and availble_resources:
                                if best_case_key in treated_cases_ids:
                                    print(f"Case already treated: {best_case_key}")
                                    continue
                                else:
                                    selected_res = availble_resources.pop(0)
                                    treated_case = candidate_cases.pop(best_case_key)
                                    treated_cases[rule_name].append(treated_case)
                                    treated_cases_ids.append(best_case_key)                                   
                                    allocate_res(selected_res, t_dist, availble_resources, treatment_duration)
                            else:
                                pass
                                #print(f"No resource available or no best case found for rule: {rule_name}")
                        else:
                            pass
                            # print(f"No candidate case found for rule: {rule_name}")
                    else:
                        pass
                        # print(f"Condition does not hold")
                    
            print(f"Processing finished for rule: {rule_name}")
            print(f"len of treated cases: {len(treated_cases_ids)}")
            netGain = np.sum([l[-1] for l in treated_cases[rule_name]])
            # append net gain to treated_cases
            treated_cases[rule_name].append(netGain)            
                     
            print(f"Net Gain: {netGain}")
            netGain=0

        return treated_cases, optimization_results


if __name__ == "__main__":
    from sys import argv
    import datetime

    datasets = [argv[1]]
    results_folder = argv[2]
    resources_list = [int(argv[3])]
    iterations = int(argv[4])
    ratios = [float(argv[5])]
    optimizations = [argv[6]]
    
    #datasets = [argv[1]]
    #results_folder = argv[2]
    #resources_list = [int(argv[1])]
    #iterations = int(argv[3])  
    print(f"Data: {datasets}, result_folder: {results_folder}, resource_list: {resources_list},terations: {iterations} ")
    

    # ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    netgainType = "all"

    
    outcome_gain = 50
    
    rules = [
        "predictive", # IP1
        "causal", # IP2
        "reliability", # IP3
        "remtime", # IP4
        "urgency", # IP5
        "totalUncer", # IP6
        "predictive_conformal", # IP7
        "causal_conformal", # IP8
        "remtime_conformal", # IP9
        "urgency_conformal", # IP10
        "predictive_conformal_totalUncer", # IP11
        "predictive_conformal_causal_conformal", # IP12
        "predictive_conformal_urgency_conformal", # IP13
        "causal_conformal_urgency_conformal", # IP14
        "predictive_conformal_causal_conformal_urgency_conformal", # IP15               
    ]
    
    #optimizations = ["WithoutOptimization", "WithOptimization"] # WithOptimization,  WithoutOptimization

    for optimization in optimizations:
        print(f"Optimization: {optimization}")
        
        for dataset_name in datasets:
            print(f"Starting with {dataset_name}...")
            for resource in resources_list:
                for ratio in ratios:
                #for resource in resources_list:
                    print(f"Resource: {resource}")
                    treatment_cost = ratio * outcome_gain
                    
                    treated_cases, optimization_results = main(dataset_name, rules, outcome_gain, treatment_cost, optimization, netgainType, resources_list)
                    
                    results_dir = f"/home/centos/phd/5th/results_all/{results_folder}/{dataset_name}/{iterations}/{resource}/{optimization}"
                    os.makedirs(results_dir, exist_ok=True)
                    
                    # Save treated cases
                    with open(f'{results_dir}/treated_cases_{netgainType}_{optimization}_{outcome_gain}_{treatment_cost}_{ratio}_{dataset_name}.pickle', 'wb') as f:
                        pickle.dump(treated_cases, f)
                
                    # Save optimization results
                    with open(f'{results_dir}/optimization_results_{netgainType}_{optimization}_{outcome_gain}_{treatment_cost}_{ratio}_{dataset_name}.pickle', 'wb') as f:
                        pickle.dump(optimization_results, f)
                    
                    print("\n+++++++++++++++++\n")













