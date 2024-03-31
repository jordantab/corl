import argparse
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from datasets import load_dataset

DEFAULT_MODEL = "Salesforce/codet5-small"
DEFAULT_OUTDIR = "models/"
DEFAULT_LOGDIR = "logs/"
TRAIN_DATA = "datasets/example_train/"
EVAL_DATA = "datasets/example_eval/"


def parse_args():
    """
    Parses commandline args.
    Example usage:
        python3 train.py -m NaiveBayes -d datasets/feature-envy.arff
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--train_path",
        required=True,
        help=f"Path to training data to be used for finetuning",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=DEFAULT_MODEL,
        help=f"Pretrained model name, (Default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "-e",
        "--eval_path",
        required=True,
        help=f"Path to evaluation data to be used for finetuning",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        default=DEFAULT_OUTDIR,
        help=f"Directory to save model after finetuning, (Default: {DEFAULT_OUTDIR})",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        default=DEFAULT_LOGDIR,
        help=f"Directory to write logs from funentuing, (Default: {DEFAULT_LOGDIR})",
    )
    return parser.parse_args()


def prepare_data(tokenizer, dataset):
    """
    Encodes raw text data using the provided tokenizer.
    """

    def tokenize_func(examples):
        return tokenizer(examples["input_code"], padding="max_length", truncation=True)

    return dataset.map(tokenize_func, batched=True)


def fine_tune(model_name, train_path, eval_path, outdir, logdir):
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # arbitrary upper bounds for now
    max_source_length = 2048
    max_target_length = 2048

    data_files = {"train": train_path, "test": eval_path}
    dataset = prepare_data(tokenizer, load_dataset("csv", data_files=data_files))

    task_prefix = "Optimize the following python function:\n"
    input_sequences = [
        task_prefix + sequence for sequence in dataset["train"]["input_code"]
    ]
    print(f"input_sequences: {input_sequences}")
    print()

    output_sequences = [
        task_prefix + sequence for sequence in dataset["train"]["target_code"]
    ]
    print(f"target_sequences: {output_sequences}")
    print()

    encoding = tokenizer(
        [task_prefix + sequence for sequence in input_sequences],
        padding="longest",
        max_length=max_source_length,
        truncation=True,
        return_tensors="pt",
    )

    input_ids, attention_mask = encoding.input_ids, encoding.attention_mask

    # encode the targets
    target_encoding = tokenizer(
        output_sequences,
        padding="longest",
        max_length=max_target_length,
        truncation=True,
        return_tensors="pt",
    )
    labels = target_encoding.input_ids

    # replace padding token id's of the labels by -100 so it's ignored by the loss
    labels[labels == tokenizer.pad_token_id] = -100

    # forward pass
    # loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
    output_sequences = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=2048,
    )

    print(tokenizer.batch_decode(output_sequences, skip_special_tokens=True))


def main():
    args = parse_args()

    # Fine-tune the model
    fine_tune(args.model, args.train_path, args.eval_path, args.outdir, args.logdir)


if __name__ == "__main__":
    main()
