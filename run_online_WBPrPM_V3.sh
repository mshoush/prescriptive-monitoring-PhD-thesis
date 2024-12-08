#!/bin/bash

# Define an array of datasets, ratios, and optimizations
datasets=("bpic2017" "bpic2017")
ratios=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9) # 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
optimizations=("WithOptimization" "WithoutOptimization")

# Function to process a dataset
process_dataset() {
    dataset="$1"
    echo "Start with $dataset..."

    # Set resource levels based on the dataset name
    if [ "$dataset" == "bpic2012" ]; then
        resources_list=(1 4 6 12)
	# resources_list=(1)
        y=1
    elif [ "$dataset" == "bpic2017" ]; then
        y=2
	# resources_list=(1)
        resources_list=(1 3 4 8)
    else
        echo "Unknown dataset: $dataset"
        return
    fi

    # Get the number of available CPUs
    num_cpus=$(nproc --all)

    # Calculate the number of CPUs to allocate for this dataset (up to a maximum of 8)
    cpus_per_dataset=$((num_cpus / ${#datasets[@]}))  # Divide by the number of datasets

    if [ "$cpus_per_dataset" -gt 8 ]; then
        cpus_per_dataset=8
    elif [ "$cpus_per_dataset" -gt 4 ]; then
        cpus_per_dataset=4
    elif [ "$cpus_per_dataset" -gt 2 ]; then
        cpus_per_dataset=2
    fi

    # Calculate the start and end CPU indexes for this dataset
    if [ "$y" -eq 1 ]; then
        start_cpu=0
    else
        x=$((y - 1))
        start_cpu=$((cpus_per_dataset * x))
    fi

    # Calculate the end CPU index for this dataset
    end_cpu=$((start_cpu + cpus_per_dataset - 1))
    echo "start_cpu: $start_cpu"
    echo "end_cpu: $end_cpu"

    # Iterate over dataset iterations sequentially
    for iteration in {2..2}; do
        echo "Starting Iteration... $iteration"

        # Iterate over resources in parallel
        for resources in "${resources_list[@]}"; do
            echo "Resources: $resources"

            # Iterate over ratios in parallel
            for ratio in "${ratios[@]}"; do
                echo "Ratio: $ratio"

                # Iterate over optimizations in parallel
                for optimization in "${optimizations[@]}"; do
                    echo "Optimization: $optimization"

                    # Run the taskset command in the background for each resource, ratio, and optimization combination
                    taskset -c $start_cpu-$end_cpu python ./stream_events/run_online_WBPrPM_V2.py "$dataset" "results_all_WBPrPM_v5" "$resources" "$iteration" "$ratio" "$optimization" &
                    # sleep 60s;
                done
            done
        done

        # Wait for all background processes of this iteration to finish
        wait
    done
}

# Iterate over datasets in parallel
for d in "${datasets[@]}"; do
    process_dataset "$d" &
done

# Wait for all dataset processes to finish
wait

