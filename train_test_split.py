"""
train_test_split.py

Usage:
    python train_test_split.py --infile <input.json> \
                               --train <train.json> \
                               --test <test.json> \
                               [--test_size <float>]

This script splits a JSON dataset into a train and test set, ensuring that all
instances of a particular problem_id are either in the training set or the test set.
The JSON file should contain a list of entries, each with a 'problem_id' field.
The script uses the GroupShuffleSplit function from sklearn.model_selection to perform the split.

Arguments:
    --infile: Path to the input JSON file.
    --train: Path to the output train JSON file.
    --test: Path to the output test JSON file.
    --test_size: Proportion of the dataset to include in the test split. Default is 0.2.
"""

import argparse
import json
from sklearn.model_selection import GroupShuffleSplit

RANDOM_STATE = 22
DEFAULT_TEST_SIZE = 0.2


def split_dataset(input_file, train_file, test_file, test_size=0.2):
    with open(input_file, "r") as f:
        data = json.load(f)

    # Extract problem_id from each data entry
    problem_ids = [entry["problem_id"] for entry in data]

    # Create a GroupShuffleSplit object
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)

    # Get the indices of the train and test sets
    train_indices, test_indices = next(gss.split(data, groups=problem_ids))

    # Use the indices to create the train and test sets
    train_data = [data[i] for i in train_indices]
    test_data = [data[i] for i in test_indices]

    with open(train_file, "w") as f:
        json.dump(train_data, f, indent=4)

    with open(test_file, "w") as f:
        json.dump(test_data, f, indent=4)


def main():
    parser = argparse.ArgumentParser(
        description="Split a JSON file into a train and test set."
    )
    parser.add_argument("--infile", required=True, help="Path to the input JSON file.")
    parser.add_argument(
        "--train", required=True, help="Path to the output train JSON file."
    )
    parser.add_argument(
        "--test", required=True, help="Path to the output test JSON file."
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="Proportion of the dataset to include in the test split.",
    )

    args = parser.parse_args()

    split_dataset(args.infile, args.train, args.test, args.test_size)


if __name__ == "__main__":
    main()
