"""
filter_dataset.py

Usage:
    python filter_dataset.py --infile <input.json> \
                             --outfile <output.json> \
                             --token_limit <limit> \
                             --model <model_name>

This script filters a JSON dataset based on token length limits.
The JSON file should contain a list of entries, each with 'instruction' and 'input' fields.
The script uses the tokenizer of the specified model (default is the model specified in config.py)
to count the number of tokens in the 'instruction' and 'input' fields of each entry.
Entries with a token count exceeding the specified limit are excluded from the output file.

Arguments:
    --infile: Path to the input JSON file.
    --outfile: Path to the output JSON file.
    --token_limit: Maximum token length to be allowed for both input and output.
    --model: Model to be used for tokenizing. Default is the model specified in config.py.
"""

import json
import argparse
from transformers import AutoTokenizer
import config

MAX_LENGTH = 1024 * 20


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Filter a JSON dataset based on token length limits."
    )
    parser.add_argument("--infile", type=str, help="Path to the input JSON file.")
    parser.add_argument("--outfile", type=str, help="Path to the output JSON file.")
    parser.add_argument(
        "--token_limit",
        type=int,
        help="Maximum token length for both input and output.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=config.DEFAULT_MODEL,
        help="Name of the pre-trained model for tokenization.",
    )
    return parser.parse_args()


def load_dataset(file):
    """Load the JSON dataset from the specified file."""
    with open(file, "r") as f:
        data = json.load(f)
    return data


def filter_dataset(data, tokenizer, token_limit):
    """Filter the dataset based on the token length limit."""
    filtered_data = []

    for entry in data:
        instruction = entry.get("instruction", "")
        input_text = entry.get("input", "")
        output_text = entry.get("output", "")

        # Combine instruction and input text
        combined_input_text = instruction
        if input_text:
            combined_input_text += f"\n{input_text}"

        # Tokenize and calculate token lengths
        input_tokens = tokenizer.encode(
            combined_input_text,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        output_tokens = tokenizer.encode(
            output_text,
            truncation=True,
            max_length=MAX_LENGTH,
        )

        input_length = len(input_tokens)
        output_length = len(output_tokens)

        if input_length <= token_limit and output_length <= token_limit:
            filtered_data.append(entry)

    return filtered_data


def save_filtered_dataset(data, file):
    """Save the filtered dataset to the specified file."""
    with open(file, "w") as f:
        json.dump(data, f, indent=4)


def main():
    args = parse_args()
    data = load_dataset(args.infile)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    filtered_data = filter_dataset(data, tokenizer, args.token_limit)
    save_filtered_dataset(filtered_data, args.outfile)
    print(f"Filtered dataset saved to {args.outfile}")
    print(f"Number of samples in the original dataset: {len(data)}")
    print(f"Number of samples in the filtered dataset: {len(filtered_data)}")


if __name__ == "__main__":
    main()
