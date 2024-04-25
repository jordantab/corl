#!/bin/bash

# Set the path to your Python script
PYTHON_SCRIPT="rl_run.py"

# Set parameters
MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
NUM_EPISODES=10
DATASET_PATH="./examples/test_python.json"
MAX_LENGTH=256
R1=-1.0
R2=0.0
R3=0.5
R4=1.0

# Run the Python script with the specified parameters
python $PYTHON_SCRIPT \
    --model_name $MODEL_NAME \
    --num_episodes $NUM_EPISODES \
    --dataset_path $DATASET_PATH \
    --max_length $MAX_LENGTH \
    --R1 $R1 \
    --R2 $R2 \
    --R3 $R3 \
    --R4 $R4