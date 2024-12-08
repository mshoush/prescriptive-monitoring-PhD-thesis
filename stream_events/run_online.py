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
pd.set_option('display.max_columns', None)
import pickle
from env_manager import EnvManager
from online_utils import *
from filter_cases import FilterCases
from hyperopt import fmin, hp, tpe, Trials, space_eval, atpe
from functools import partial
import os
from sys import argv
from hyperopt.fmin import generate_trials_to_calculate
import numpy as np
from itertools import product

import numpy as np
import warnings
np.warnings = warnings


def generate_param_combinations(space):
    keys, values = zip(*space.items())
    for v in product(*values):
        yield dict(zip(keys, v))
        

def main(dataset_name, rules, outcome_gain, treatment_cost, optimization, netgainType):
    best_params = {}
    # Initialize dataset parameters
    dataset_name = dataset_name
    sample = "sample1"
    data_types = ["val", "test",] if optimization == "WithOptimization" else ["test"]
    default_threshold_parameters = {
        "neg_proba": 0.5, # avg: 0.6
        "uncer_threshold": 0.5, # avg: 0.5
        "confidence_level": 0.5, # avg: 0.5
    }   

    # Define file and directory paths
    sample = "sample1"
    results_dir_base = "/home/centos/phd/5th/results_all/online/"
    data_dir_base = "/home/centos/phd/5th/results_all/online/"
   
    treated_cases = {}  # Dictionary to store treated cases for all rules    
    netGain = {}
    optimization_results = {}  # Dictionary to store optimization results for all rules
    
    for rule_name in rules: 
        for data_type in data_types: 
            print(f"Data type: {data_type}")
 
            file_name = f"{data_type}_{dataset_name}_{sample}_online.parquet"
            data_dir = f"{data_dir_base}/{data_type}/{dataset_name}/"    
            results_dir = f"{results_dir_base}/{data_type}/{dataset_name}/"

            data = EnvManager(data_type, dataset_name, results_dir, file_name, data_dir)    
            data_f = EnvManager(data_type, dataset_name, results_dir, file_name, data_dir).return_df() 
            
            if data_type == "val":
                best_params = None
                print(f"Hyperopt optimization for rule: {rule_name} ...")
                space_p, space_dict = create_search_space_filter(rule_name=rule_name)
                combined_space = {**space_p}
                early_stop_fn_ = len(combined_space.keys()) * 20 + 1
                print(f"early_stop_fn_ = {early_stop_fn_}")
                import datetime
                
                # Record the start time of the optimization process
                start_time = datetime.datetime.now()

                init_vals = [{k: default_threshold_parameters[k] for k in combined_space.keys()}]
                
                print("")
                print("Initial Values:")
                print(init_vals)
                trials = generate_trials_to_calculate(init_vals)
                SEED = 22 # 20
                                              
                from hyperopt.early_stop import no_progress_loss
                
                tried_params = set()
                count = [0] 
                              
                partial_objective = partial(objective, tried_params=tried_params, count=count, trials=trials, rule_name=rule_name, data=data, data_f=data_f, outcome_gain=outcome_gain, treatment_cost=treatment_cost, netgainType=netgainType)
                best_params = fmin(fn=partial_objective, space=combined_space, trials=trials, algo=atpe.suggest, max_evals=25, early_stop_fn=no_progress_loss(early_stop_fn_), rstate=np.random.default_rng(SEED))
                best_params = space_eval(combined_space, best_params)
                
                end_time = datetime.datetime.now()

                # Calculate the time taken to optimize parameters
                time_taken_to_optimize = (end_time - start_time).total_seconds() /60 

                
                best_trial_index = trials.best_trial['tid']
                best_trial = trials.trials[best_trial_index]
                
                best_trial_time = best_trial['book_time']
                first_trial_time = trials.trials[0]['book_time']
                
                elapsed_time = trials.trials[0]['book_time']
                optimization_results[rule_name] = {
                    "best_params": best_params,
                    "best_trial":  best_trial , # trials.trials[0],
                    "elapsed_time": time_taken_to_optimize
                }
                
                best_params = {**default_threshold_parameters, **best_params}
                continue
            neg_proba_thre = np.round(best_params.get('neg_proba', default_threshold_parameters['neg_proba']), 2)
            total_uncer_thre = np.round(best_params.get('uncer_threshold', default_threshold_parameters['uncer_threshold']), 2)
            confidence_level = np.round(best_params.get('confidence_level', default_threshold_parameters['confidence_level']), 2)

            print(f"Params: \nNeg_proba_thre: {neg_proba_thre}\nTotal_uncer_thre: {total_uncer_thre}\nConfidence_level: {confidence_level}\n")
            
            print(f"Running rule: {rule_name}")
            candidate_cases = {}
            net_gain = 0

            while not data.finished:
                event = data.get_event_by_event()
                event = FilterCases.filter_cases(event, neg_proba_thre, total_uncer_thre, confidence_level, rule_name)

                if event is not None:
                    alpha_value, alpha_col = get_alpha_col(confidence_level)
                    case_id, gain_per_case = process_event(event, candidate_cases, outcome_gain, treatment_cost, alpha_value)
                    if case_id is not None:
                        candidate_cases[case_id].append(gain_per_case)
                        net_gain += gain_per_case
            
            if candidate_cases:  # Check if there are any treated cases  
                print(f"Processing finished for rule: {rule_name}\n")
                treated_cases[rule_name] = candidate_cases
                treated_cases[rule_name]['netGain'] = net_gain  # Append net gain to treated cases dictionary
            else:
                print(f"No treated cases found for rule: {rule_name}\n")
                treated_cases[rule_name] = candidate_cases
                   
        print(netgainType)
        print(f"netGain: {net_gain}")

        netGain[rule_name] =  get_netGain(treated_cases, data_f, rule_name, outcome_gain, treatment_cost, netgainType=netgainType) 


    return treated_cases, optimization_results, netGain

if __name__ == "__main__":
    ratio = float(argv[1])
    datasets = argv[2]
    netgainType = argv[3]
    results_folder =  argv[4] # "treated_cases_local_V2"
    
    
    # netgainType = "all" # treated, all, not_treated
    # datasets = ["bpic2017"] # "bpic2012", 
    # ratios = [ 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] # 0.1,
    
    

    outcome_gain = 50 
    # Example usage:
    search_space, space_dict = create_search_space_filter(rule_name="predictive")
    rules = list(space_dict.keys())  
    print(f"Rules: {rules}")
    
    # rules = ['predictive', 'predictive_conformal', "predictive_conformal_totalUncer"]
    
    optimizations = ["WithoutOptimization",  "WithOptimization",] # "WithOptimization", WithoutOptimization # "WithoutOptimization", 
    
    for optimization in optimizations:
        print(f"{optimization}")    
        
        for dataset_name in [datasets]:  # "bpic2012"
            print(f"Starting with {dataset_name}...")
            #for ratio in ratios:
            print(f"ratio: {ratio}")
            treatment_cost = ratio * outcome_gain
            
            treated_cases, optimization_results, netGain = main(dataset_name, rules, outcome_gain, treatment_cost, optimization, netgainType)
            
            #results_dir = f"/home/mshoush/5th/results_all/{results_folder}/{dataset_name}"
            results_dir = f"/home/centos/phd/5th/results_all/{results_folder}/{dataset_name}"
            os.makedirs(results_dir, exist_ok=True)
            
            # Save results
            with open(f'{results_dir}/treated_cases_{netgainType}_{optimization}_{outcome_gain}_{treatment_cost}_{ratio}_{dataset_name}.pickle', 'wb') as f:
                pickle.dump(treated_cases, f)
            
            # Save optimization results
            with open(f'{results_dir}/optimization_results_{netgainType}_{optimization}_{outcome_gain}_{treatment_cost}_{ratio}_{dataset_name}.pickle', 'wb') as f:
                pickle.dump(optimization_results, f)
            
            # save net gain
            with open(f'{results_dir}/netGain_{optimization}_{netgainType}_{outcome_gain}_{treatment_cost}_{ratio}_{dataset_name}.pickle', 'wb') as f:
                pickle.dump(netGain, f)
            print("\n+++++++++++++++++\n")
