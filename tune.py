"""
Performs initial fine-tuning step on the dataset using an instruction dataset.

NOTE: all commandline args are strictly optional, and will
override default configuration values specified in config.py.

Usage:
    python3 tune.py [--lr-warmup-steps] [--batch-size-per-replica]
                    [--grad-acc-steps] [--local-rank] [--deepspeed]
                    [--fp16] [--device] [--num-procs]

Apply instruction-tuning to a pretrained model

options:
  -h, --help            show this help message and exit
  --model MODEL         The pre-trained model to fine-tune
  --outdir OUTDIR       The output directory to save the fine-tuned model
  --max-len MAX_LEN     The maximum length of input and output sequences
  --padding PADDING     The padding strategy for input sequences
  --test-size TEST_SIZE
                        The fraction of the dataset to use for testing
  --inst-dataset INST_DATASET
                        The path to the instruction dataset
  --epochs EPOCHS       The number of training epochs
  --lr LR               The learning rate
  --lr-warmup-steps LR_WARMUP_STEPS
                        The number of learning rate warmup steps
  --batch-size-per-replica BATCH_SIZE_PER_REPLICA
                        The batch size per GPU/TPU replica
  --grad-acc-steps GRAD_ACC_STEPS
                        The number of gradient accumulation steps
  --local-rank LOCAL_RANK
                        The local rank for distributed training
  --deepspeed DEEPSPEED
                        The path to the DeepSpeed configuration file
  --fp16                Whether to use mixed precision (FP16) training
  --device DEVICE       The device to use for training (e.g., 'cpu', 'cuda')
  --num-procs NUM_PROCS
                        The number of CPU cores to use for data preprocessing
"""

import argparse
import copy
import os

import numpy as np
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

import config
import datasets


def prompt_input(instruction, _input):
    """
    Returns a generic prompt suited to the provided instruction and input data.

    Args:
        instruction (string): Directive for what the model should generate.
        _input (string): Context-specific data for the task, if applicable.
    Returns:
        string: A complete prompt for the model.
    """
    if _input == "":
        return (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n### Response:"
        )
    else:
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n### Input:\n{_input}\n\n### Response:"
        )


def create_preprocessor(tokenizer, max_len):
    """
    Creates a function to preprocess a dataset using the provided tokenizer.

    Args:
        tokenizer (PreTrainedTokenizer): tokenizer to use during preprocessing.
        max_len (int): max length for inputs and labels.
    Returns:
        func(Dataset) -> Any: preprocessor function.
    """

    def preprocessor(examples):
        # build complete prompts and target outputs
        sources = [
            prompt_input(instruction, _input)
            for instruction, _input in zip(examples["instruction"], examples["input"])
        ]
        targets = [
            source + output + tokenizer.eos_token
            for source, output in zip(sources, examples["output"])
        ]

        # tokenize prompts and create labels from target outputs
        model_inputs = tokenizer(
            sources, max_length=max_len, padding="max_length", truncation=True
        )
        labels = tokenizer(
            targets, max_length=max_len, padding="max_length", truncation=True
        )
        model_inputs["decoder_input_ids"] = copy.deepcopy(labels["input_ids"])

        # removes prefix prompts from output tokens
        # ensures loss is computed on the relevant output data
        eos_token_id = tokenizer.eos_token_id
        for x, y in zip(model_inputs["input_ids"], labels["input_ids"]):
            label_prefix_len = x.index(eos_token_id) if eos_token_id in x else len(x)
            y[:label_prefix_len] = [-100] * label_prefix_len

            if eos_token_id in y:
                pad_len = len(y) - y.index(eos_token_id) - 1
                if pad_len > 0:
                    y[y.index(eos_token_id) + 1 :] = [-100] * pad_len

        # shift labels to the right and add decoder start token id
        # ensures that the model is trained to predict the next token
        decoder_start_id = tokenizer.eos_token_id
        for z in model_inputs["decoder_input_ids"]:
            z[1:] = z[:-1]
            z[0] = decoder_start_id

        # prepares model_inputs specifically for evaluation on seq2seq
        model_inputs["labels"] = copy.deepcopy(labels["input_ids"])
        model_inputs["decoder_attention_mask"] = labels["attention_mask"]
        return model_inputs

    return preprocessor


def tokenize_dataset(tokenizer, tuning_data, max_len, procs):
    """
    Tokenizes the provided dataset with the provided tokenizer.

    Args:
        tokenizer (PreTrainedTokenizer): tokenizer to process tuning data.
        tuning_data (Dataset): dataset to be tokenized.
        max_len (int): max length for model inputs and labels.
        procs (int): number of cpu cores to use during tokenization.
    Returns:
    """
    preprocessor = create_preprocessor(tokenizer, max_len)
    train_data = tuning_data.map(
        preprocessor,
        batched=True,
        remove_columns=tuning_data.column_names,
        num_proc=procs,
        load_from_cache_file=False,
    )
    print(f"  ==> Loaded {len(train_data)} samples")
    # TODO: cache training data samples
    return train_data


def get_model_size(model):
    """
    Reports the number of trainable parameters.

    Args:
        model (PreTrainedModel): Model with training params.
    Returns:
        string: formatted string reporting trainable params.
    """
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_size = sum([np.prod(p.size()) for p in model_parameters])
    return "{}M".format(round(model_size / 1e6))


def freeze_self_attn_params(model):
    """
    Freezes self-attention parameters in the decoder and enables gradient
    computation for cross-attention parameters. This prepares the model
    to be fine-tuned for sequence to sequence generation.

    Args:
        model (PreTrainedModel): Model to have params frozen.
    Returns:
        None
    """
    param_count = model.num_parameters()
    trainable_count = get_model_size(model)
    print(f"Total params: {param_count}, trainable params: {trainable_count}")

    for param in model.decoder.parameters():
        param.requires_grad = False

    num_decoder_layers = model.decoder.config.n_layer
    for i in range(num_decoder_layers):
        each_decoder_layer = model.decoder.transformer.h[i]
        if hasattr(each_decoder_layer, "crossattention"):
            for param in each_decoder_layer.crossattention.parameters():
                param.requires_grad = True
            each_decoder_layer.crossattention.to(torch.float32)

        if hasattr(each_decoder_layer, "alpha_xattn"):
            each_decoder_layer.alpha_xattn.requires_grad = True

    unfrozen_param_count = model.num_parameters()
    unfrozen_trainable_count = get_model_size(model)
    print(
        f"Unfrozen params: {unfrozen_param_count}, Unfrozen trainable: {unfrozen_trainable_count}"
    )


def fine_tune(model, train_data, conf):
    """
    Commences training to fine tune the model. Saves the model to the outdir
    specified by the provided configuration.

    Args:
        model (PreTrainedModel): Model to tune.
        train_data (datasets.Dataset): Instruction dataset for fine tuning.
        conf (Namespace): Configuration namespace.
    Returns:
        None
    """
    print(f"Starting training loop")

    training_args = TrainingArguments(
        report_to=None,
        output_dir=conf.outdir,
        overwrite_output_dir=False,
        do_train=True,
        save_strategy="epoch",
        num_train_epochs=conf.epochs,
        per_device_train_batch_size=conf.batch_size_per_replica,
        gradient_accumulation_steps=conf.grad_acc_steps,
        learning_rate=conf.lr,
        weight_decay=0.0,
        warmup_steps=conf.lr_warmup_steps,
        save_total_limit=2,
        dataloader_drop_last=True,
        dataloader_num_workers=4,
        local_rank=conf.local_rank,
        deepspeed=conf.deepspeed,
        fp16=conf.fp16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
    )

    trainer.train()

    if conf.local_rank in [0, -1]:
        final_checkpoint_dir = os.path.join(conf.outdir, "final_checkpoint")
        model.save_pretrained(final_checkpoint_dir)
        print(f"  ==> Finish training and save to {final_checkpoint_dir}")


def parse_args():
    """
    Parses commandline flags, which are only used to override default values
    defined in config.py

    Args: None
    Returns:
        Cmdline Args Namespace
    """
    parser = argparse.ArgumentParser(
        description="Apply instruction-tuning to a pretrained model"
    )
    # general configs
    parser.add_argument(
        "--model",
        default=config.DEFAULT_MODEL,
        type=str,
        help="The pre-trained model to fine-tune",
    )
    parser.add_argument(
        "--outdir",
        default=config.DEFAULT_OUTDIR,
        type=str,
        help="The output directory to save the fine-tuned model",
    )

    # dataset configs
    parser.add_argument(
        "--max-len",
        default=config.DEFAULT_MAX_LEN,
        type=int,
        help="The maximum length of input and output sequences",
    )
    parser.add_argument(
        "--padding",
        default=config.DEFAULT_PADDING,
        type=str,
        help="The padding strategy for input sequences",
    )
    parser.add_argument(
        "--test-size",
        default=config.DEFAULT_TEST_SIZE,
        type=float,
        help="The fraction of the dataset to use for testing",
    )
    parser.add_argument(
        "--inst-dataset",
        default=config.DEFAULT_INSTRUCTION_DATASET,
        type=str,
        help="The path to the instruction dataset",
    )

    # training parameters
    parser.add_argument(
        "--epochs",
        default=config.DEFAULT_EPOCHS,
        type=int,
        help="The number of training epochs",
    )
    parser.add_argument(
        "--lr",
        default=config.DEFAULT_LEARNING_RATE,
        type=float,
        help="The learning rate",
    )
    parser.add_argument(
        "--lr-warmup-steps",
        default=config.DEFAULT_LEARNING_RATE_WARMUP_STEPS,
        type=int,
        help="The number of learning rate warmup steps",
    )
    parser.add_argument(
        "--batch-size-per-replica",
        default=config.DEFAULT_BATCH_SIZE,
        type=int,
        help="The batch size per GPU/TPU replica",
    )
    parser.add_argument(
        "--grad-acc-steps",
        default=config.DEFAULT_GRAD_ACC_STEPS,
        type=int,
        help="The number of gradient accumulation steps",
    )
    parser.add_argument(
        "--local-rank",
        default=config.DEFAULT_LOCAL_RANK,
        type=int,
        help="The local rank for distributed training",
    )
    parser.add_argument(
        "--deepspeed",
        default=config.DEFAULT_DEEPSPEED,
        type=str,
        help="The path to the DeepSpeed configuration file",
    )
    parser.add_argument(
        "--fp16",
        default=config.DEFAULT_FP16,
        action="store_true",
        help="Whether to use mixed precision (FP16) training",
    )

    # platform configuration
    parser.add_argument(
        "--device",
        default=config.DEFAULT_DEVICE,
        type=str,
        help="The device to use for training (e.g., 'cpu', 'cuda')",
    )
    parser.add_argument(
        "--num-procs",
        default=config.DEFAULT_NUM_PROCS,
        type=int,
        help="The number of CPU cores to use for data preprocessing",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # TODO: currently uses the entire dataset
    train_data = datasets.load_dataset("json", data_files=args.inst_dataset)["train"]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenized_data = tokenize_dataset(
        tokenizer, train_data, args.max_len, args.num_procs
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(args.device)

    freeze_self_attn_params(model)
    fine_tune(model, tokenized_data, args)


if __name__ == "__main__":
    main()
