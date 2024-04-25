"""
Default configurations for the RL-LLM training pipeline.
"""

import os

DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

# Fine-tuning defaults
DEFAULT_OUTDIR = "models/dataset_tuned_checkpoint"

# Data preprocessing
DEFAULT_MAX_LEN = 25
DEFAULT_PADDING = "max_length"
DEFAULT_TEST_SIZE = 1
DEFAULT_INSTRUCTION_DATASET = "examples/sample1.json"

# Training specific settings
DEFAULT_EPOCHS = 3
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_LEARNING_RATE_WARMUP_STEPS = 30
DEFAULT_BATCH_SIZE = 1
DEFAULT_GRAD_ACC_STEPS = 16
DEFAULT_LOCAL_RANK = -1
DEFAULT_DEEPSPEED = None
DEFAULT_FP16 = False

# Platform specific configs
DEFAULT_DEVICE = "cuda"  # for GPU usage or "cpu" for CPU usage
DEFAULT_NUM_PROCS = os.cpu_count()
