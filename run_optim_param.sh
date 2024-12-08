#!/bin/bash

# Define the parameter values
dataset_ref="bpic2017" # "bpic2012 bpic2017"
cls_encoding="laststate" # agg index combined
cls_method="catboost" # lightgbm randomforest catboost xgboost
task_type="classification" # classification regression

# Split the parameter values into arrays
IFS=" " read -r -a dataset_refs <<< "$dataset_ref"
IFS=" " read -r -a cls_encodings <<< "$cls_encoding"
IFS=" " read -r -a cls_methods <<< "$cls_method"
IFS=" " read -r -a task_types <<< "$task_type"

# Iterate over all combinations of parameters
for dr in "${dataset_refs[@]}"; do
    for ce in "${cls_encodings[@]}"; do
        for cm in "${cls_methods[@]}"; do
            for tt in "${task_types[@]}"; do
                # Run the Python script in the background with the current parameters
                python predictive_model/optimize_param.py "$dr" "$ce" "$cm" "$tt" &
		sleep 60
            done
        done
    done
done

# Wait for all background processes to finish
wait

