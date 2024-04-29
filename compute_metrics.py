"""
compute_metrics.py

This script processes model results and computes metrics.

Usage:
    python compute_metrics.py <input_path>

Arguments:
    input_path      Path to the JSON file with model results.
"""

import json
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process model results and compute metrics."
    )
    parser.add_argument(
        "input_path", type=str, help="Path to the JSON file with model results."
    )
    return parser.parse_args()


def process_results(input_path):
    with open(input_path, "r") as file:
        data = json.load(file)

    count_optimized_slow = 0
    count_optimized_fast = 0
    total_slow_vs_gen = 0
    total_fast_vs_gen = 0
    total_lines_reduced_slow = 0
    total_lines_reduced_fast = 0
    count_valid_results = 0

    for result in data:
        if result["verdict"] == 1:
            slow = result["slow"]
            fast = result["fast"]
            generated = result["generated"]
            num_lines_slow = result["num_lines_slow"]
            num_lines_fast = result["num_lines_fast"]
            num_lines_generated = result["num_lines_generated"]

            if generated < slow:
                count_optimized_slow += 1
                total_slow_vs_gen += (slow - generated) / slow
                total_lines_reduced_slow += (
                    num_lines_slow - num_lines_generated
                ) / num_lines_slow

            if generated < fast:
                count_optimized_fast += 1
                total_fast_vs_gen += (fast - generated) / fast
                total_lines_reduced_fast += (
                    num_lines_fast - num_lines_generated
                ) / num_lines_fast

            count_valid_results += 1

    metrics = {
        "percent_valid": (100 * count_valid_results / len(data)),
        "avg_speedup_slow": (
            (100 * total_slow_vs_gen / count_valid_results)
            if count_valid_results
            else 0
        ),
        "avg_speedup_fast": (
            (100 * total_fast_vs_gen / count_valid_results)
            if count_valid_results
            else 0
        ),
        "valid_percent_optimized_slow": (
            (100 * count_optimized_slow / count_valid_results)
            if count_valid_results
            else 0
        ),
        "valid_percent_optimized_fast": (
            (100 * count_optimized_fast / count_valid_results)
            if count_valid_results
            else 0
        ),
        "total_percent_optimized_slow": (
            (100 * count_optimized_slow / len(data)) if data else 0
        ),
        "total_percent_optimized_fast": (
            (100 * count_optimized_fast / len(data)) if data else 0
        ),
        "avg_lines_reduced_slow": (
            (100 * total_lines_reduced_slow / count_valid_results)
            if count_valid_results
            else 0
        ),
        "avg_lines_reduced_fast": (
            (100 * total_lines_reduced_fast / count_valid_results)
            if count_valid_results
            else 0
        ),
    }

    return metrics


def save_metrics(metrics, input_path):
    output_dir = "metrics"
    os.makedirs(output_dir, exist_ok=True)

    base_filename = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_dir, f"{base_filename}_metrics.json")

    with open(output_path, "w") as file:
        json.dump(metrics, file, indent=4)

    print(f"Metrics saved to {output_path}")


def main():
    args = parse_args()
    metrics = process_results(args.input_path)
    save_metrics(metrics, args.input_path)


if __name__ == "__main__":
    main()
