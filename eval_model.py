"""
eval_model.py

Usage:
    python eval_model.py [--file_path <path_to_dataset>] [--device <device>]

This script evaluates the performance of a transformer model in optimizing Python code snippets in terms
of runtime and memory usage. It uses specific test cases from a JSON file, generates optimized code using a transformer
model, and compares these optimizations against baseline implementations to measure improvements.
Each test case in the JSON file should contain an 'input' (slow code) and 'output' (fast code), along with a 'problem_id'.

Arguments:
    --file_path: Optional override for the path to the dataset.
    --device: Specify 'cpu' or 'cuda' to set the device for model computation. Defaults to 'cpu'.
"""

import argparse
import json
import torch
import transformers
from unit_tests.run_code import run_tcs
import os
import config

os.environ["TOKENIZERS_PARALLELISM"] = "true"


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
        "--file_path",
        default=config.DEFAULT_TEST_DATASET_PATH,
        help="Path to the dataset.",
    )
    parser.add_argument(
        "--device",
        default=config.DEFAULT_DEVICE,
        help="Specify 'cpu' or 'cuda' to set the device for model computation.",
    )
    return parser.parse_args()


def eval_model(checkpoint, dataset, device):
    results = []

    for problem in dataset:
        problem_id = problem["problem_id"]
        slow_code = problem["input"]
        fast_code = problem["output"]

        _, runtime_slow = run_tcs(slow_code, problem_id)
        _, runtime_fast = run_tcs(fast_code, problem_id)
        generated_code = generate_code(checkpoint, slow_code, device)
        verdict, runtime_generated = run_tcs(generated_code, problem_id)

        verdict_num = 1 if verdict == "Accepted" else 0

        results.append(
            {
                "problem_id": problem_id,
                "slow": runtime_slow,
                "fast": runtime_fast,
                "generated": runtime_generated,
                "verdict": verdict_num,
            }
        )

    return results


def generate_code(checkpoint, slow_code, device):
    pipeline = transformers.pipeline(
        "text-generation",
        model=checkpoint,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map=device,
    )
    messages = [
        {
            "role": "system",
            "content": "Provide an optimized version of the following code snippet. Only provide the code, no need to provide any description. ",
        },
        {"role": "user", "content": slow_code},
    ]

    prompt = pipeline.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    outputs = pipeline(
        prompt,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    return outputs[0]["generated_text"][len(prompt) :]


def calculate_model_metrics(
    outputs_slow, outputs_fast, outputs_memory_slow, outputs_memory_fast
):
    pct_opt_slow = sum(outputs_slow) / len(outputs_slow) if outputs_slow else 0
    pct_opt_fast = sum(outputs_fast) / len(outputs_fast) if outputs_fast else 0
    pct_opt_memory_slow = (
        sum(outputs_memory_slow) / len(outputs_memory_slow)
        if outputs_memory_slow
        else 0
    )
    pct_opt_memory_fast = (
        sum(outputs_memory_fast) / len(outputs_memory_fast)
        if outputs_memory_fast
        else 0
    )
    return (
        pct_opt_slow,
        pct_opt_fast,
        pct_opt_memory_slow,
        pct_opt_memory_fast,
    )


def main():
    args = parse_args()
    checkpoint = config.DEFAULT_MODEL
    file_path = args.file_path
    device = args.device

    with open(file_path, "r") as json_file:
        data = json.load(json_file)

    results = eval_model(checkpoint, data, device)

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    dataset_name = os.path.splitext(os.path.basename(args.file_path))[0]
    results_filename = f"{dataset_name}_results.json"
    results_path = os.path.join(results_dir, results_filename)

    with open(results_path, "w") as outfile:
        json.dump(results, outfile, indent=4)

    print(f"Evaluation completed. Results saved to '{results_path}'.")


if __name__ == "__main__":
    main()
