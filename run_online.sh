#!/bin/bash

# Define the list of ratios and datasets
#ratios=("1.0" "0.5" "0.1")
ratios=("0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "1.0")
datasets=("bpic2017" "bpic2012")
#netgainType=("all" "treated" "not_treated")
netgainType=( "all")
resultsFolder=("treated_cases_vm_V2")



# Loop over ratios
for ratio in "${ratios[@]}"; do
    # Loop over datasets
    for dataset in "${datasets[@]}"; do
	for netgaintype in "${netgainType[@]}"; do
		echo "Starting with; $dataset, and Ratio: $ratio, and type: $netgaintype"
		# Activate the Python virtual environment if necessary
		# source /path/to/your/python/virtual/environment/bin/activate
		# Run the Python script with the current ratio and dataset
		python  stream_events/run_online.py "$ratio" "$dataset" "$netgaintype" "$resultsFolder" > out2.txt;
	done
        # Deactivate the Python virtual environment if activated
        # deactivate
    done
done

