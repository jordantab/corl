#!/bin/bash

# This script runs tune.py using configurations suitable for deployment
# For locally testing tune.py, use "tune_local.sh".

time python3 tune.py --device "cuda" \
                --max-len 105 \
                --inst-dataset "datasets/cpp_leq_105_tokens.json" \
                # TODO:
                # --outdir "<somewhere-outside-the-repo>""

# Other flags:
#   --outdir OUTDIR       The output directory to save the fine-tuned model
#   --max-len MAX_LEN     The maximum length of input and output sequences
#   --padding PADDING     The padding strategy for input sequences
#   --test-size TEST_SIZE
#                         The fraction of the dataset to use for testing
#   --inst-dataset INST_DATASET
#                         The path to the instruction dataset
#   --epochs EPOCHS       The number of training epochs
#   --lr LR               The learning rate
#   --lr-warmup-steps LR_WARMUP_STEPS
#                         The number of learning rate warmup steps
#   --batch-size-per-replica BATCH_SIZE_PER_REPLICA
#                         The batch size per GPU/TPU replica
#   --grad-acc-steps GRAD_ACC_STEPS
#                         The number of gradient accumulation steps
#   --local-rank LOCAL_RANK
#                         The local rank for distributed training
#   --deepspeed DEEPSPEED
#                         The path to the DeepSpeed configuration file
#   --fp16                Whether to use mixed precision (FP16) training
#   --device DEVICE       The device to use for training (e.g., 'cpu', 'cuda')
#   --num-procs NUM_PROCS
#                         The number of CPU cores to use for data preprocessing
