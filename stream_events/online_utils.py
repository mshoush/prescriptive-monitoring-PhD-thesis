# analysis_utils.py

import os
import random
import numpy as np
import threading
import time


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


def sum_gain_treated_cases(treated_cases):
    sum_gain = {
        "TN": 0,  # True Negative
        "FP": 0,  # False Positive
        "TP": 0,  # True Positive
        "FN": 0,  # False Negative
        "gain": 0,
    }

    for case in treated_cases.values():
        actual = case[3]  # Assuming actual outcomes are in the 4th column
        predicted = case[4]  # Assuming predicted outcomes are in the 5th column
        gain = case[-1]  # Assuming gain values are in the second-to-last column

        if actual == 0 and predicted == 0:  # True Negative
            sum_gain["TN"] += gain
        elif actual == 0 and predicted == 1:  # False Positive
            sum_gain["FP"] += gain
        elif actual == 1 and predicted == 1:  # True Positive
            sum_gain["TP"] += gain
        elif actual == 1 and predicted == 0:  # False Negative
            sum_gain["FN"] += gain

        sum_gain["gain"] += gain

    return sum_gain


def accuracy(tp, tn, fp, fn):
    return (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0


def precision(tp, fp):
    return tp / (tp + fp) if (tp + fp) != 0 else 0


def recall(tp, fn):
    return tp / (tp + fn) if (tp + fn) != 0 else 0


def f_score(precision, recall):
    return (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) != 0
        else 0
    )


def auc(tp, fp, fn, tn):
    return tp / (tp + fn) + tn / (tn + fp) if (tp + fn) != 0 and (tn + fp) != 0 else 0


def get_alpha_col(confidence_level):
    alpha_value = np.round((1 - confidence_level), 1)
    alpha_col = "alpha_" + str(alpha_value)
    return alpha_value, alpha_col


def allocate_res(s_res, dist, nr_res, treatment_duration):
    # print("Allocate resource")
    t = threading.Thread(
        daemon=True,
        target=block_and_release_res,
        args=(s_res, dist, nr_res, treatment_duration),
    )
    t.start()


def block_and_release_res(s_res, dist, nr_res, treatment_duration):
    # print("Block resource")
    time.sleep(treatment_duration)
    nr_res.append(s_res)


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


from hyperopt import hp


from hyperopt import hp

# mean_urgency_ub, mean_causal_ub
def create_search_space_filter(mean_urgency_ub, max_causal_ub, rule_name="predictive_conformal"):
    space_dict = {
        # IP1: Predictive
        "predictive": {
            "neg_proba": [
                0.3,
                0.8,
            ]
        },
        # IP2: Causal
        "causal": {
            "causal_thre": [
                0.1,
                max_causal_ub,
            ]
        },
        # IP3: Reliability
        "reliability": {
            "reliability_thre": [
                0.1,
                0.9,
            ]
        },
        # IP4: Remaining Time
        "remtime": {
            "remtime_thre": [
                1,
                mean_urgency_ub,
            ]
        },
        # IP5: Urgency
        "urgency": {
            "remtime_thre": [
                1,
                mean_urgency_ub,
            ]
        },
        # IP6: Total Uncertainty
        "totalUncer": {
            "uncer_threshold": [
                0.1,
                0.9,
            ]
        },
        # IP7: Predictive + Conformal
        "predictive_conformal": {
            "neg_proba": [
                0.3,
                0.8,
            ],
            "confidence_level": [
                0.1,
                0.9,
            ]
        },
        # IP8: Causal + Conformal
        "causal_conformal": {
            "causal_thre": [
                0.1,
                max_causal_ub,
            ],
            "causal_uncer_thre": [
                0.1,
                max_causal_ub,
            ]
        },
        # IP9: Remaining Time + Conformal
        "remtime_conformal": {
            "remtime_thre": [
                 1,
                mean_urgency_ub,
            ],
            "remtime_uncer_thre": [
                 1,
                mean_urgency_ub,
            ]
        },
        # IP10: Urgency + Conformal
        "urgency_conformal": {
            "remtime_thre": [
                 1,
                mean_urgency_ub,
            ],
            "remtime_uncer_thre": [
                 1,
                mean_urgency_ub,
            ]
        },
        # IP11: Predictive + Conformal + Total Uncertainty
        "predictive_conformal_totalUncer": {
            "neg_proba": [
                0.3,
                0.8,
            ],
            "confidence_level": [
                0.1,
                0.9,
            ],
            "uncer_threshold": [
                0.1,
                0.9,
            ]
        },
        # IP12: Predictive + Conformal + Causal Conformal
        "predictive_conformal_causal_conformal": {
            "neg_proba": [
                0.3,
                0.8,
            ],
            "confidence_level": [
                0.1,
                0.9,
            ],
            "causal_thre": [
                0.1,
                max_causal_ub,
            ],
            "causal_uncer_thre": [
                0.1,
                max_causal_ub,
            ]
        },
        # IP13: Predictive + Conformal + Urgency Conformal
        "predictive_conformal_urgency_conformal": {
            "neg_proba": [
                0.3,
                0.8,
            ],
            "confidence_level": [
                0.1,
                0.9,
            ],
            "remtime_thre": [
                 1,
                mean_urgency_ub,
            ],
            "remtime_uncer_thre": [
                 1,
                mean_urgency_ub,
            ]
        },
        # IP14: Causal Conformal + Urgency Conformal
        "causal_conformal_urgency_conformal": {
            "causal_thre": [
                0.1,
                max_causal_ub,
            ],
            "causal_uncer_thre": [
                0.1,
                max_causal_ub,
            ],
            "remtime_thre": [
                 1,
                mean_urgency_ub,
            ],
            "remtime_uncer_thre": [
                 1,
                mean_urgency_ub,
            ]
        },
        # IP15: Predictive + Conformal + Causal Conformal + Urgency Conformal
        "predictive_conformal_causal_conformal_urgency_conformal": {
            "neg_proba": [
                0.3,
                0.8,
            ],
            "confidence_level": [
                0.1,
                0.9,
            ],
            "causal_thre": [
                0.1,
                max_causal_ub,
            ],
            "causal_uncer_thre": [
                0.1,
                max_causal_ub,
            ],
            "remtime_thre": [
                 1,
                mean_urgency_ub,
            ],
            "remtime_uncer_thre": [
                 1,
                mean_urgency_ub,
            ]
        },
    }

    search_space = {}
    params = space_dict.get(rule_name, {})

    for param_name, param_range in params.items():
        param_space = hp.quniform(f"{param_name}", param_range[0], param_range[1], 0.1)
        search_space[param_name] = param_space

    return search_space, space_dict


# def create_search_space_filter(rule_name="predictive_conformal"):

#     space_dict = {
#         # IP1
#         "predictive": {"neg_proba": [0.3, 0.8,]
#                        },
                       
#         # IP2
#         "predictive_conformal": {"neg_proba": [0.3, 0.8,],
#                                  "confidence_level": [0.1, 0.9,],
#                                  },
        
#         "predictive_totalUncer": {"neg_proba": [0.3, 0.8,],
#                                   "uncer_threshold": [0.1, 0.9, ],
#         },
#         # IP3
#         "predictive_conformal_totalUncer": {
#             "neg_proba": [
#                 0.3,
#                 0.8,
#             ],
#             "confidence_level": [
#                 0.1,
#                 0.9,
#             ],
#             "uncer_threshold": [
#                 0.1,
#                 0.9,
#             ],
#         },
#         "conformal_only": {
#             "confidence_level": [
#                 0.1,
#                 0.9,
#             ]
#         },
#         "totalUncer_only": {
#             "uncer_threshold": [
#                 0.1,
#                 0.9,
#             ]
#         },
#         "conformal_totalUncer": {
#             "confidence_level": [
#                 0.1,
#                 0.9,
#             ],
#             "uncer_threshold": [
#                 0.1,
#                 0.9,
#             ],
#         },
#     }

#     search_space = {}
#     params = space_dict.get(rule_name, {})

#     for param_name, param_range in params.items():
#         param_space = hp.quniform(f"{param_name}", param_range[0], param_range[1], 0.1)
#         search_space[param_name] = param_space

#     return search_space, space_dict


def get_alpha_col(confidence_level):
    alpha_value = np.round((1 - confidence_level), 1)
    alpha_col = "alpha_" + str(alpha_value)
    return alpha_value, alpha_col


def calculate_gain_per_Case(true_outcome, outcome_gain, treatment_cost):
    if true_outcome == 1:
        return outcome_gain - treatment_cost
    elif true_outcome == 0:
        return -treatment_cost  # -0.5 * treatment_cost - treatment_cost


def process_event(
    event, candidate_cases, outcome_gain, treatment_cost, alpha_value, treated_cases_ids
):
    
    # Extract necessary details from the event data
    case_id = event["case_id"].iloc[0]
    actual_clf = float(event["actual_clf"].iloc[0])
    predicted_clf = float(event["preds_clf"].iloc[0])
    predicted_proba_0 = float(event['probs_clf_0'].iloc[0])  # Probability of negative outcome
    predicted_proba_1 = float(event['probs_clf_1'].iloc[0])
    proba_conformal = float(event[f'y_pred_set_{alpha_value}_clf'].iloc[0])
    proba_uncer = float(event['total_uncer_clf'].iloc[0])
    causal_ite = float(event['causal_ite'].iloc[0])
    reliability_estimate_clf = float(event['reliability_estimate_clf'].iloc[0])
    preds_reg = float(event['preds_reg'].iloc[0])
    surv_pred = float(event['surv_pred'].iloc[0])

    # Causal conformal bounds
    causal_conformal_up = float(event[f'causal_upper_{alpha_value}'].iloc[0])
    causal_conformal_lo = float(event[f'causal_lower_{alpha_value}'].iloc[0])
    causal_uncer = (causal_conformal_up + causal_conformal_lo) / 2

    # Remaining time conformal bounds
    remtime_conformal_up = float(event[f'y_upper_{alpha_value}_reg'].iloc[0])
    remtime_conformal_lo = float(event[f'y_lower_{alpha_value}_reg'].iloc[0])
    remtime_uncer = (remtime_conformal_up + remtime_conformal_lo) / 2

    # Urgency conformal bounds
    urgency_conformal_up = float(event[f'surv_upper_{alpha_value}'].iloc[0])
    urgency_conformal_lo = float(event[f'surv_lower_{alpha_value}'].iloc[0])
    urgency_uncer = (urgency_conformal_up + urgency_conformal_lo) / 2


    case_id = event["case_id"].iloc[0]
    activity = event["activity"].iloc[0]
    timestamp = str(event["timestamp"].iloc[0])

    actual_clf = float(event["actual_clf"].iloc[0])
    predicted_clf = float(event["preds_clf"].iloc[0])
    predicted_proba_0 = float(event["probs_clf_0"].iloc[0])
    predicted_proba_1 = float(event["probs_clf_1"].iloc[0])
    proba_conformal = float(event[f"y_pred_set_{alpha_value}_clf"].iloc[0])
    proba_uncer = float(event["total_uncer_clf"].iloc[0])

    y0 = float(event["y0"].iloc[0])  # outcome proba if not treated
    y00 = 0 if y0 < 0.5 else 1
    y1 = float(event["y1"].iloc[0])  # outcome proba if treated
    y11 = 0 if y1 < 0.5 else 1
    t = float(event["t"].iloc[0])

    y = np.where((y11 == 1) & (y00 == 0), 1, 0).item()  # for persudable cases

    gain_per_case = calculate_gain_per_Case(y, outcome_gain, treatment_cost)
    

    candidate_cases[case_id] = [
        case_id, # 0
        activity, # 1
        timestamp, # 2
        actual_clf, # 3
        predicted_clf, # 4
        predicted_proba_0, # 5
        proba_uncer, # 6
        proba_conformal, # 7
        y0, # 8
        causal_uncer, # 9
        y1, # 10        
        t, # 11
        urgency_uncer, # 12
        y, # 13
        gain_per_case, # 14
    ]

    return candidate_cases


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


def rank_candidate_cases(
    candidate_cases,
    rule_name,
    weight_need=1,
    weight_effect=1,
    weight_urgency=1,
    weight_uncertainty=1,
):
    """
    Rank candidate cases based on modified criteria with weights.
    
    candidate_cases[case_id] = [
        case_id, # 0
        activity, # 1
        timestamp, # 2
        actual_clf, # 3
        predicted_clf, # 4
        predicted_proba_0, # 5
        proba_uncer, # 6
        proba_conformal, # 7
        y0, # 8
        causal_uncer, # 9
        y1, # 10        
        t, # 11
        urgency_uncer, # 12
        y, # 13
        gain_per_case, # 14
    ]

    """

    if candidate_cases:
        best_case_key = None
        best_case_score = float("-inf")

        # print(candidate_cases)
        for case_key, case in candidate_cases.items():
            need_proba = case[7] # proba_conformal
            effect = case[9]
            urgency = case[12]
            uncertainty = case[6]
            
            # need_proba = case[5]  # Assuming need probability is in the 6th column
            # effect = case[9]  # Assuming effect is in the 10th column
            # urgency = case[8]
            # uncertainty = (
            #     case[6] + case[10]
            # )  # Assuming uncertainty is in the 7th and 11th columns

            score = (
                ((weight_need * need_proba) + (weight_effect * effect))
                - (weight_urgency * urgency)
                - (weight_uncertainty * uncertainty)
            )

            if score > best_case_score:
                best_case_score = score
                best_case_key = case_key

        return best_case_key
    else:
        return None


import pandas as pd
from env_manager import EnvManager
from online_utils import *
from filter_cases import FilterCases
from hyperopt import fmin, hp, tpe, Trials, STATUS_OK
from functools import partial

from filter_cases import FilterCases


def get_netGain(
    treated_cases, data_f, rule_name, outcome_gain, treatment_cost, netgainType="all"
):
    netGain = 0
    # Print total number of unique cases
    total_cases = len(data_f["case_id"].unique())
    print(f"Total number of cases: {total_cases}")

    if len(treated_cases[rule_name].keys()) != 0:
        print(f"Number of treated cases: {len(treated_cases[rule_name].keys())}")

        y = np.array(
            [
                treated_cases[rule_name][case][-2]
                for case in list(treated_cases[rule_name].keys())[:-1]
            ]
        )

        netGain = np.sum((y * outcome_gain) - (treatment_cost))
        print("Net gain treated: " + str(netGain))
    else:
        print("Net gain Not treated: " + str(netGain))
        netGain = 0

    print(f"Total net gain: {netGain}\n")

    return netGain


# def objective(params, tried_params, count, trials, rule_name, data, data_f, outcome_gain, treatment_cost, netgainType = "all"):
#     count[0] += 1
#     print(f"count: {count[0]}")

#     treated_cases = {}
#     default_threshold_parameters = {
#         "neg_proba": 0.5,
#         "uncer_threshold": 0.5,
#         "confidence_level": 0.9
#     }

#     params = {**default_threshold_parameters, **params}

#     for param in params.keys():
#         params[param] = np.round(params[param], 2)

#     # Convert params to a hashable tuple
#     params_tuple = tuple(sorted(params.items()))

#     if params_tuple in tried_params:
#         return {'loss': 9999999999, 'status': STATUS_OK}

#     # Add the hashable tuple to tried_params
#     tried_params.add(params_tuple)

#     print(f"tried_params: {tried_params}")

#     if len(trials.trials) == 0:
#         pass
#     else:
#         print(trials.trials[-1]['misc']['vals'])

#     print("Objective")
#     print(params)
#     neg_proba_thre = params['neg_proba']
#     total_uncer_thre = params['uncer_threshold']
#     confidence_level = params['confidence_level']

#     candidate_cases = {}
#     netGain = 0


#     print(f"neg_proba_thre: {neg_proba_thre}, total_uncer_thre: {total_uncer_thre}, confidence_level: {confidence_level}")

#     while not data.finished:
#         event = data.get_event_by_event()
#         event = FilterCases.filter_cases(event, neg_proba_thre, total_uncer_thre, confidence_level, rule_name)

#         if event is not None:
#             alpha_value, alpha_col = get_alpha_col(confidence_level)
#             case_id, gain_per_case = process_event(event, candidate_cases, outcome_gain, treatment_cost, alpha_value)

#             if case_id is not None:
#                 candidate_cases[case_id].append(gain_per_case)
#                 netGain += gain_per_case

#     #print(f"candidate_cases: {candidate_cases}")
#     if candidate_cases:  # Check if there are any treated cases
#         print(f"Processing finished for rule: {rule_name}")
#         treated_cases[rule_name] = candidate_cases

#     else:
#         print(f"No treated cases found for rule: {rule_name}")
#         treated_cases[rule_name] = candidate_cases
#         netGain=0


#     print(f"netGain: {netGain}\n")
#     data = data.reset()
#     candidate_cases = {}

#     return {'loss': -netGain, 'status': STATUS_OK}


def objective(
    params,
    tried_params,
    count,
    trials,
    rule_name,
    data,
    data_f,
    outcome_gain,
    treatment_cost,
    netgainType="all",
    resources=None,
):
    count[0] += 1
    print(f"count: {count[0]}")

    # Default threshold parameters
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

    # Merge default parameters with provided params
    params = {**default_threshold_parameters, **params}

    # Round parameters for consistency
    for param in params.keys():
        params[param] = np.round(params[param], 2)

    # Convert params to a hashable tuple to avoid duplicate evaluations
    params_tuple = tuple(sorted(params.items()))

    if params_tuple in tried_params:
        return {"loss": 9999999999, "status": STATUS_OK}

    # Add params to the set of tried params
    tried_params.add(params_tuple)

    print(f"tried_params: {tried_params}")

    # Print the last set of parameter values from trials
    if len(trials.trials) > 0:
        print(trials.trials[-1]["misc"]["vals"])

    print("Objective")
    print(params)

    
    #  "neg_proba": 0.5,
    #     "uncer_threshold": 0.5,
    #     "confidence_level": 0.5,  
              
    #     "causal_uncer_thre": 0.0,
    #     "remtime_uncer_thre": 10,
    #     "reliability_thre": 0.5,
    #     "remtime_thre": 10,
    #     "causal_thre": 0,      
          
    #     "weight_need": 1,
    #     "weight_effect": 1,
    #     "weight_urgency": 1,
    #     "weight_uncertainty": 1,
    
    
    # Extract relevant parameters for filtering cases
    neg_proba_thre = params["neg_proba"]
    total_uncer_thre = params["uncer_threshold"]
    confidence_level = params["confidence_level"]
    
    weight_need = params["weight_need"]
    weight_effect = params["weight_effect"]
    weight_urgency = params["weight_urgency"]
    weight_uncertainty = params["weight_uncertainty"]
    
    
    causal_thre = params["causal_thre"]
    reliability_thre = params["reliability_thre"]
    remtime_thre = params["remtime_thre"]
    causal_uncer_thre = params["causal_uncer_thre"]
    remtime_uncer_thre = params["remtime_uncer_thre"]

    print(
        f"neg_proba_thre: {neg_proba_thre}, total_uncer_thre: {total_uncer_thre}, confidence_level: {confidence_level}, \
        causal_thre: {causal_thre}, reliability_thre: {reliability_thre}, remtime_thre: {remtime_thre}, \
        causal_uncer_thre: {causal_uncer_thre}, remtime_uncer_thre: {remtime_uncer_thre}"
    )

    candidate_cases = {}
    treated_cases = {}
    treated_cases_ids = []
    netGain = 0
    availble_resources = list(
        range(1, resources + 1)
    )  # Assuming 2 available resources in this example
    print(f"availble_resources objective: {availble_resources}")

    # Process events and filter cases
    while not data.finished:
        event = data.get_event_by_event()
        event = FilterCases.filter_cases(
            event, neg_proba_thre, total_uncer_thre, confidence_level, 
            causal_thre, reliability_thre, remtime_thre,
            causal_uncer_thre, remtime_uncer_thre,
            rule_name
        )

        if event is not None:
            alpha_value, alpha_col = get_alpha_col(confidence_level)

            if event["case_id"].iloc[0] not in treated_cases_ids:
                candidate_cases = process_event(
                    event,
                    candidate_cases,
                    outcome_gain,
                    treatment_cost,
                    alpha_value,
                    treated_cases_ids,
                )

            # Rank and treat candidate cases
            if candidate_cases and availble_resources:
                best_case_key = rank_candidate_cases(
                    candidate_cases,
                    rule_name,
                    weight_need,
                    weight_effect,
                    weight_urgency,
                    weight_uncertainty,
                )

                if best_case_key and availble_resources:
                    selected_res = availble_resources.pop(0)
                    treated_case = candidate_cases.pop(best_case_key)
                    treated_cases[rule_name] = treated_case
                    treated_cases_ids.append(best_case_key)
                    allocate_res(
                        selected_res, "fixed", availble_resources, 1
                    )  # Assume fixed treatment duration of 1

                netGain += treated_case[
                    -1
                ]  # Accumulate the net gain from each treated case

    print(f"netGain: {netGain}\n")

    # Reset the environment for the next iteration
    data = data.reset()
    candidate_cases = {}

    # Return the negative of netGain since hyperopt minimizes the objective function
    return {"loss": -netGain, "status": STATUS_OK}
