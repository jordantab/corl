#!/bin/bash

# This script runs eval_model.py with specified command line arguments
# Usage: ./run_eval_model.sh <dataset_path> <device>

# Assigning command line arguments to variables
DATASET_PATH=${1:-"./examples/test_python.json"}  # Default path if not provided
DEVICE=${2:-"cpu"}  # Default to "cpu" if not provided

# Running the Python script with parameters using 'time' for profiling
echo "Running eval_model.py with dataset at '$DATASET_PATH' and on device '$DEVICE'"
time python eval_model.py --file_path "$DATASET_PATH" --device "$DEVICE"
