"""
Processes the csv data found in datasets/improvement_pairs_additional_metadata.csv.

Our csv file contains a variety of data that may be useful for other steps, but
the initial finetuning on the dataset only requires a few columns. Furthermore,
there is already an accepted format used for fine-tuning a model for sequence to
sequence generation. For example, see the alpaca instruction tuning dataset:
https://github.com/sahil280114/codealpaca/blob/master/data/code_alpaca_2k.json

All this script does is create a JSON instruction dataset file from our dataset.

Input (csv) format:
    user_id, problem_id, language, submission_id_v0, submission_id_v1, cpu_time_v0,
    cpu_time_v1, memory_v0, memory_v1, status_v0, status_v1, improvement_frac, code_v0,
    code_v1, code_v0_loc, code_v1_loc

Output (json) format:
    {
        "instruction": "Provide an optimized version of the following code snippet.",
        "input": <code_v0>,
        "output": <code_v1>
    }

To run with defaults, just specify an outfile:


    python3 process_csv.py --outfile examples/sample1.json


This script can also be used to create datasets of limited size for local testing
and development using the --samples flag. The default file processed is
datasets/improvement_pairs_additional_metadata, however other files with the same
columns can be used as well with the --infile flag:


    python3 process_csv.py --infile other_dataset.csv --outfile outfile.json --samples 5


Additionally, you can filter the dataset by language using the --language flag:


    python3 process_csv.py --language "Python" --outfile examples/sample1.json


By default, the language is set to "C++".
Run the script with --help to print detailed usage.
"""

import argparse
import csv
import json

INSTRUCTION = "Provide an optimized version of the following code snippet."
MAX_FIELD_MEM = 2 << 20  # 1 MB


def process_csv(infile, outfile, samples, language):
    """
    Processes the CSV data and creates a JSON instruction dataset file.

    Args:
        infile (str): Path to the input CSV file.
        outfile (str): Path to the output JSON file.
        samples (int): Limits number of samples if set.

    Returns:
        None
    """
    data = []
    with open(infile, "r", newline="", encoding="utf-8") as csvfile:
        csv.field_size_limit(MAX_FIELD_MEM)
        reader = csv.DictReader(csvfile, delimiter="\t")
        count = 0
        for row in reader:
            if count == samples:
                break
            if row["language"].lower() == language.lower():
                data.append(
                    {
                        "instruction": INSTRUCTION,
                        "input": row["code_v0"],
                        "output": row["code_v1"],
                    }
                )
                count += 1

    with open(outfile, "w", encoding="utf-8") as jsonfile:
        json.dump(data, jsonfile, indent=4)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--infile",
        default="datasets/improvement_pairs_additional_metadata.csv",
        help="Path to the input CSV file (default: datasets/improvement_pairs_additional_metadata.csv)",
    )
    parser.add_argument("--outfile", required=True, help="Path to the output JSON file")
    parser.add_argument(
        "--samples",
        type=int,
        default=-1,
        help="Number of samples to generate (optional)",
    )
    parser.add_argument(
        "--language", type=str, default="C++", help="Selected language of code snippets"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    process_csv(args.infile, args.outfile, args.samples, args.language)


if __name__ == "__main__":
    main()
