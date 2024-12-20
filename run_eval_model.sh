#!/bin/bash

# This script runs eval_model.py with specified command line arguments
# Usage: ./run_eval_model.sh <dataset_path> <device>

# Assigning command line arguments to variables
CHECKPOINT=${1:-"meta-llama/Meta-Llama-3-8B-Instruct"}
DATASET_PATH=${2:-"./testset/python_leq_60_tokens_test.json"}
DEVICE=${3:-"cuda"}  # Default to "cuda" if not provided -- override with "cpu" locally

# Running the Python script with parameters using 'time' for profiling
echo "Running eval_model.py with dataset at '$DATASET_PATH' and on device '$DEVICE'"
time python eval_model.py --checkpoint "$CHECKPOINT" --file_path "$DATASET_PATH" --device "$DEVICE" > generated_code.txt
