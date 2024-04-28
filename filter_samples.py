import argparse
import json
from unit_tests.run_code import run_tcs
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", help="Input JSON file path")
    parser.add_argument("--outfile", help="Output JSON file path")
    return parser.parse_args()


def filter_working(dataset):
    accepted_samples = []
    for problem in dataset:

        slow_code = problem["input"]
        fast_code = problem["output"]

        verdict_slow, _ = run_tcs(slow_code, problem["problem_id"])
        verdict_fast, _ = run_tcs(fast_code, problem["problem_id"])

        print(f"Ran sample {problem['problem_id']}")

        # evaluate generated code
        if verdict_slow == "Accepted" and verdict_fast == "Accepted":
            print(f"Keeping {problem['problem_id']}")
            accepted_samples.append(problem)
        else:
            print(f"Discarding {problem['problem_id']}")
    return accepted_samples


def main():
    args = parse_args()

    with open(args.infile, "r") as f:
        data = json.load(f)

    filtered = filter_working(data)
    with open(args.outfile, "w") as f:
        json.dump(filtered, f, indent=4)


if __name__ == "__main__":
    main()