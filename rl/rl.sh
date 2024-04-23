#!/bin/bash

# Set the path to your Python script
PYTHON_SCRIPT="rl_run.py"
#PYTHON_SCRIPT="rl_eval.py"

# Set parameters
MODEL_NAME="Salesforce/codet5p-2b"
NUM_EPISODES=10
DATASET_PATH="../datasets/cpp/cpp_leq_105_tokens_train.json"

# Run the Python script with the specified parameters
python $PYTHON_SCRIPT --model_name $MODEL_NAME --num_episodes $NUM_EPISODES --dataset_path $DATASET_PATH
