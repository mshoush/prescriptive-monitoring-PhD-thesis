# filter_cases.py

import numpy as np

class FilterCases:

    def __init__(self):
        pass
    
    @staticmethod
    def get_alpha_col(confidence_level):
        alpha_value = np.round((1 - confidence_level), 1)
        alpha_col = "alpha_" + str(alpha_value)
        return alpha_value, alpha_col


    @staticmethod
    def filter_cases(event, neg_proba_thre, totalUncer_thre, confidence_level,
                     causal_thre, reliability_thre, remtime_thre,
                     causal_uncer_thre, remtime_uncer_thre, 
                     rule="predictive"):
        """
        Filter cases based on rule
        """
        
        # get alpha value and alpha col    
        confidence = confidence_level
        alpha_value, alpha_col = FilterCases.get_alpha_col(confidence)
        
        
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
        
        # Filter logic based on the rule
        if rule == "predictive":  # IP1
            if predicted_proba_0 > neg_proba_thre:
                return event

        elif rule == 'causal':  # IP2
            if causal_ite > causal_thre:
                return event
            # # Placeholder for causal rule logic
            # pass

        elif rule == 'reliability':  # IP3
            if reliability_estimate_clf > reliability_thre:
                return event
            # # Placeholder for reliability rule logic
            # pass

        elif rule == 'remtime':  # IP4
            if preds_reg < remtime_thre:
                return event
            # # Placeholder for remaining time rule logic
            # pass

        elif rule == 'urgency':  # IP5
            if surv_pred < remtime_thre:
                return event
            # # Placeholder for urgency rule logic
            # pass

        elif rule == 'totalUncer':  # IP6
            if proba_uncer <= totalUncer_thre:
                return event

        elif rule == 'predictive_conformal':  # IP7
            if predicted_proba_0 > neg_proba_thre and proba_conformal == 1.0:
                return event

        elif rule == 'causal_conformal':  # IP8
            if causal_ite > causal_thre and causal_uncer > causal_uncer_thre:
                return event
                
            # # Placeholder for causal conformal rule logic
            # pass

        elif rule == 'remtime_conformal':  # IP9
            if preds_reg < remtime_thre and remtime_uncer < remtime_uncer_thre:
                return event
            # # Placeholder for remaining time conformal rule logic
            # pass

        elif rule == 'urgency_conformal':  # IP10
            if surv_pred < remtime_thre and urgency_uncer < remtime_uncer_thre:
                return event
            # # Placeholder for urgency conformal rule logic
            # pass

        elif rule == 'predictive_conformal_totalUncer':  # IP11
            if predicted_proba_0 > neg_proba_thre and proba_conformal == 1.0 and proba_uncer <= totalUncer_thre:
                return event

        elif rule == 'predictive_conformal_causal_conformal':  # IP12
            if predicted_proba_0 > neg_proba_thre and proba_conformal == 1.0 and causal_ite > causal_thre and causal_uncer > causal_uncer_thre:
                return event
            # # Placeholder for predictive_conformal_causal_conformal rule logic
            # pass

        elif rule == 'predictive_conformal_urgency_conformal':  # IP13
            if predicted_proba_0 > neg_proba_thre and proba_conformal == 1.0 and surv_pred < remtime_thre and urgency_uncer < remtime_uncer_thre:
                return event
            # # Placeholder for predictive_conformal_urgency_conformal rule logic
            # pass

        elif rule == 'causal_conformal_urgency_conformal':  # IP14
            if causal_ite > causal_thre and causal_uncer > causal_uncer_thre and surv_pred < remtime_thre and urgency_uncer < remtime_uncer_thre:
                return event
            # # Placeholder for causal_conformal_urgency_conformal rule logic
            # pass

        elif rule == 'predictive_conformal_causal_conformal_urgency_conformal':  # IP15
            if predicted_proba_0 > neg_proba_thre and proba_conformal == 1.0 and causal_ite > causal_thre and causal_uncer > causal_uncer_thre and surv_pred < remtime_thre and urgency_uncer < remtime_uncer_thre: 
                return event
            
            # # Placeholder for predictive_conformal_causal_conformal_urgency_conformal rule logic
            # pass

        else:
            raise ValueError("Not a Valid Rule")

        return None  # If the event doesn't pass the filter, return None

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        # # Extract event details
        # case_id = event["case_id"].iloc[0]
        # actual_clf = float(event["actual_clf"].iloc[0])
        # # outcome prediction
        # predicted_clf = float(event["preds_clf"].iloc[0])
        # predicted_proba_0 = float(event['probs_clf_0'].iloc[0]) # proba of negative outcome
        # predicted_proba_1 = float(event['probs_clf_1'].iloc[0])
        # # conformal prediction
        # proba_conformal = float(event[f'y_pred_set_{alpha_value}_clf'].iloc[0])
        # # print(f"Proba_conformal: {proba_conformal}")
        # # total uncertainty
        # proba_uncer = float(event['total_uncer_clf'].iloc[0])
        
        # # causal rule
        # causal_ite = float(event['causal_ite'].iloc[0])
        
        # # reliability rule
        # reliability_estimate_clf = float(event['reliability_estimate_clf'].iloc[0])
        
        # # remtime rule
        # preds_reg = float(event['preds_reg'].iloc[0])
        
        # # urgency rule
        # surv_pred = float(event['surv_pred'].iloc[0])
        
        # # totalUncer rule
        # proba_uncer = float(event['total_uncer_clf'].iloc[0])
        
        
        # # causal conformal rule
        # causal_conformal_up = float(event[f'causal_upper_{alpha_value}'].iloc[0])
        # causal_conformal_lo = float(event[f'causal_lower_{alpha_value}'].iloc[0])
        
        # # remtime conformal rule # IP10
        # remtime_conformal_up = float(event[f'y_upper_{alpha_value}_reg'].iloc[0])
        # remtime_conformal_lo = float(event[f'y_lower_{alpha_value}_reg'].iloc[0])
        
        # # urergency conformal rule # IP11
        # urgency_conformal_up = float(event[f'surv_upper_{alpha_value}'].iloc[0])
        # urgency_conformal_lo = float(event[f'surv_lower_{alpha_value}'].iloc[0])
        
        # # predictive_conformal_causal_conformal # IP12
        # # predictive_conformal_urgency_conformal # IP13
        # # causal_conformal_urgency_conformal # IP14
        # # predictive_conformal_causal_conformal_urgency_conformal # IP15
        
        
        
        
        # # Define filtering logic for each rule
        # if rule == "predictive": # IP1
        #     if predicted_proba_0 > neg_proba_thre:
        #         return event
        # elif rule == 'causal': # IP2
        #     # Add causal rule logic here
        #     pass
        # elif rule == 'reliability': # IP3
        #     # Add reliability rule logic here
        #     pass
        # elif rule == 'remtime': # IP4
        #     # Add remtime rule logic here
        #     pass
        # elif rule == 'urgency': # IP5
        #     # Add urgency rule logic here
        #     pass
        # elif rule == 'totalUncer': # IP6
        #     if proba_uncer <= totalUncer_thre:
        #         return event
        # elif rule == 'predictive_conformal': # IP7
        #     if predicted_proba_0 > neg_proba_thre and proba_conformal == 0.0:
        #         return event
        # elif rule == 'causal_conformal': # IP8
        #     # Add causal_conformal rule logic here
        #     pass
        # elif rule == 'remtime_conformal': # IP9
        #     # Add remtime_conformal rule logic here
        #     pass
        # elif rule == 'urgency_conformal': # IP10
        #     # Add urgency_conformal rule logic here
        #     pass
        # elif rule == 'predictive_conformal_totalUncer': # IP11
        #     if predicted_proba_0 > neg_proba_thre and proba_conformal == 0.0 and proba_uncer <= totalUncer_thre:
        #         return event
        # elif rule == 'predictive_conformal_causal_conformal': # IP12
        #     # Add predictive_conformal_causal_conformal rule logic here
        #     pass
        # elif rule == 'predictive_conformal_urgency_conformal': # IP13
        #     # Add predictive_conformal_urgency_conformal rule logic here
        #     pass
        # elif rule == 'causal_conformal_urgency_conformal': # IP14
        #     # Add causal_conformal_urgency_conformal rule logic here
        #     pass
        # elif rule == 'predictive_conformal_causal_conformal_urgency_conformal': # IP15
        #     # Add predictive_conformal_causal_conformal_urgency_conformal rule logic here
        #     pass
        # else:
        #     raise ValueError("Not a Valid Rule")
        
        # # If the event doesn't pass the filter, return None
        # return None
        

        
        # # filter cases
        # if rule == "predictive":
        #     if predicted_proba_0 > neg_proba_thre:
        #         return event 
        #     else:
        #         return None
        # elif rule == 'predictive_conformal':
        #     if predicted_proba_0 > neg_proba_thre and proba_conformal == 0.0:
        #         return event 
        #     else:
        #         return None
        # elif rule == 'predictive_totalUncer':
        #     if predicted_proba_0 > neg_proba_thre and proba_uncer <= totalUncer_thre:
        #         return event 
        #     else:
        #         return None
        # elif rule == 'predictive_conformal_totalUncer':
        #     if predicted_proba_0 > neg_proba_thre and proba_conformal == 0.0 and proba_uncer <= totalUncer_thre:
        #         return event 
        #     else:
        #         return None
        # elif rule == 'conformal_only':
        #     if proba_conformal == 0.0:
        #         return event 
        #     else:
        #         return None
        # elif rule == 'totalUncer_only':
        #     if proba_uncer <= totalUncer_thre:
        #         return event 
        #     else:
        #         return None
        # elif rule == "conformal_totalUncer":
        #     if proba_conformal == 0.0 and proba_uncer <= totalUncer_thre:
        #         return event 
        #     else:
        #         return None
        # else:
        #     raise ValueError("Not a Valid Rule")
