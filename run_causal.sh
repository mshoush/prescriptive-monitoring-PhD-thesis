#!/bin/bash

# Define the path to your Python script
PYTHON_SCRIPT="/home/centos/phd/5th/causal_model/causal_model.py"

# Run the script for both datasets in parallel
echo "Start with BPIC2012 ..."
python $PYTHON_SCRIPT bpic2012 &
echo "Start with BPIC2017 ..."
python $PYTHON_SCRIPT bpic2017 &

# Wait for all background jobs to complete
wait

echo "Both experiments have completed."

