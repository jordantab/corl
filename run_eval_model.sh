#!/bin/bash

# This script runs eval_model.py with specified command line arguments
# Usage: ./run_eval_model.sh <dataset_path> <device>

# Assigning command line arguments to variables
DATASET_PATH=${1:-"./testset/python_leq_60_tokens_test.json"}
DEVICE=${2:-"cuda"}  # Default to "cuda" if not provided -- override with "cpu" locally

# Running the Python script with parameters using 'time' for profiling
echo "Running eval_model.py with dataset at '$DATASET_PATH' and on device '$DEVICE'"
time python eval_model.py --file_path "$DATASET_PATH" --device "$DEVICE" > generated_code.txt
