"""
count_tokens.py

Usage:
    python count_tokens.py --file <file.json> [--model <model_name>]

This script analyzes the number of tokens in the input and output text of each entry in a JSON file.
The JSON file should contain a list of entries, each with 'instruction', 'input', and 'output' fields.
The script uses the tokenizer of the specified model (default is the model specified in config.py).

Arguments:
    --file: Path to the JSON file for token analysis.
    --model: Model to be used for tokenizing. Default is the model specified in config.py.
"""

import argparse
import json
from transformers import AutoTokenizer
import numpy as np
from scipy import stats

import config

MAX_LENGTH = 1024 * 20


def parse_args():
    """
    Parses command-line arguments.

    Returns:
        An argparse.Namespace object with the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--file",
        help="JSON file for token analysis",
    )
    parser.add_argument(
        "--model", default=config.DEFAULT_MODEL, help="Model to be used for training"
    )
    return parser.parse_args()


def analyze(file, tokenizer):
    """
    Analyzes the number of tokens in the input and output text of each entry in a JSON file.

    Arguments:
        file: Path to the JSON file.
        tokenizer: A tokenizer from the transformers library.

    Returns:
        A tuple of two lists: input_lengths and output_lengths.
        Each list contains the number of tokens in the input and output text of each entry, respectively.
    """
    input_lengths = []
    output_lengths = []
    samples = 0

    with open(file, "r") as f:
        data = json.load(f)
        for entry in data:
            samples += 1

            instruction = entry.get("instruction", "")
            input_text = entry.get("input", "")
            output_text = entry.get("output", "")

            # Combine the instruction and input text
            combined_input_text = instruction
            if input_text:
                combined_input_text += f"\n{input_text}"

            input_tokens = tokenizer.encode(
                combined_input_text, max_length=MAX_LENGTH, truncation=True
            )
            input_lengths.append(len(input_tokens))

            output_tokens = tokenizer.encode(
                output_text, max_length=MAX_LENGTH, truncation=True
            )
            output_lengths.append(len(output_tokens))

    print_statistics(input_lengths, "input")
    print_statistics(output_lengths, "output")
    print(f"Total number of samples: {samples}")


def print_statistics(token_lengths, label):
    print(f"\nStatistics for {label} token lengths:")
    token_lengths_array = np.array(token_lengths)
    mean = np.mean(token_lengths_array)
    median = np.median(token_lengths_array)
    mode = stats.mode(token_lengths_array).mode

    min_val = np.min(token_lengths_array)
    max_val = np.max(token_lengths_array)
    percentile_25 = np.percentile(token_lengths_array, 25)
    percentile_50 = np.percentile(token_lengths_array, 50)
    percentile_75 = np.percentile(token_lengths_array, 75)

    print(f"Mean: {mean}")
    print(f"Median: {median}")
    print(f"Mode: {mode}")
    print(f"Min: {min_val}")
    print(f"Max: {max_val}")
    print(f"25th Percentile: {percentile_25}")
    print(f"50th Percentile (Median): {percentile_50}")
    print(f"75th Percentile: {percentile_75}")


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    analyze(args.file, tokenizer)


if __name__ == "__main__":
    main()
