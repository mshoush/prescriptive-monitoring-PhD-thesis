#!/bin/bash

# Define the parameter values
dataset_ref="bpic2012 bpic2017" # "bpic2012 bpic2017"
cls_encoding="laststate agg index combined" # laststate agg index combined
cls_method="lightgbm randomforest xgboost" # lightgbm randomforest catboost xgboost
task_type="classification regression"  #regression

# Split the parameter values into arrays
IFS=" " read -r -a dataset_refs <<< "$dataset_ref"
IFS=" " read -r -a cls_encodings <<< "$cls_encoding"
IFS=" " read -r -a cls_methods <<< "$cls_method"
IFS=" " read -r -a task_types <<< "$task_type"

# Iterate over all combinations of parameters
for dr in "${dataset_refs[@]}"; do
    #for tt in "${task_types[@]}"; do
    for cm in "${cls_methods[@]}"; do
	for ce in "${cls_encodings[@]}"; do
            #for cm in "${cls_methods[@]}"; do
	    for tt in "${task_types[@]}"; do
		echo "Task Type: $tt"
		# Check if cls_method is catboost
		if [[ "$cm" == "catboost" ]]; then
		    python predictive_model/predictive_conformal_catboost.py "$dr" "$ce" "$cm" "$tt" &
		    sleep 60
		else
		    python predictive_model/predictive_conformal.py "$dr" "$ce" "$cm" "$tt" &
		fi
		sleep 60
            done
        done		
	wait
    done    
    wait
done
# Wait for all background processes to finish
wait

