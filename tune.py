import copy
import datasets
import os
import torch
import numpy as np
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)

# Fine-tuning defaults
DEFAULT_MODEL = "Salesforce/codet5p-2b"
DEFAULT_OUTDIR = "models/dataset_tuned_checkpoint"

# Data preprocessing
DEFAULT_MAX_LEN = 25
DEFAULT_PADDING = "max_length"
DEFAULT_TEST_SIZE = 1

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
DEFAULT_DEVICE = "cpu"  # for GPU usage or "cpu" for CPU usage
DEFAULT_NUM_PROCS = os.cpu_count()

DATASET = "examples/sample1.json"


def prompt_input(instruction, _input):
    """
    Returns a generic prompt suited to the provided instruction and input data.

    Parameters:
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


def tokenize_dataset(
    tokenizer, tuning_data, max_len=DEFAULT_MAX_LEN, procs=DEFAULT_NUM_PROCS
):
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
    Freezes self-attention parameters in the decoder and enables
    gradient computation for cross-attention parameters.
    This prepares the model to be fine-tuned for sequence
    to sequence generation.

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


def fine_tune(model, train_data):
    print(f"Starting training loop")

    training_args = TrainingArguments(
        report_to=None,
        output_dir=DEFAULT_OUTDIR,
        overwrite_output_dir=False,
        do_train=True,
        save_strategy="epoch",
        num_train_epochs=DEFAULT_EPOCHS,
        per_device_train_batch_size=DEFAULT_BATCH_SIZE,
        gradient_accumulation_steps=DEFAULT_GRAD_ACC_STEPS,
        learning_rate=DEFAULT_LEARNING_RATE,
        weight_decay=0.0,
        warmup_steps=DEFAULT_LEARNING_RATE_WARMUP_STEPS,
        save_total_limit=2,
        dataloader_drop_last=True,
        dataloader_num_workers=4,
        local_rank=DEFAULT_LOCAL_RANK,
        deepspeed=DEFAULT_DEEPSPEED,
        fp16=DEFAULT_FP16,
        # TODO: Implement logging
        # logging_dir=args.save_dir,
        # logging_first_step=True,
        # logging_steps=args.log_freq,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
    )

    trainer.train()

    if DEFAULT_LOCAL_RANK in [0, -1]:
        final_checkpoint_dir = os.path.join(DEFAULT_OUTDIR, "final_checkpoint")
        model.save_pretrained(final_checkpoint_dir)
        print(f"  ==> Finish training and save to {final_checkpoint_dir}")


def main():
    # TODO: currently uses the entire dataset
    train_data = datasets.load_dataset("json", data_files=DATASET)["train"]
    print(f"Dataset:\n{train_data}")

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    tokenized_data = tokenize_dataset(tokenizer, train_data)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        DEFAULT_MODEL,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    freeze_self_attn_params(model)
    fine_tune(model, tokenized_data)


if __name__ == "__main__":
    main()
