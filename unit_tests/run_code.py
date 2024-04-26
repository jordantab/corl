"""
The only function you need is run_tcs(code, problem_id), which will return a tuple(Verdict, runtime)
Code: the code sample you want to run as a string
problem_id: the problem id, which corresponds to the problems in the dataset folder
verdict: "Accepted", "Wrong Answer", "Time Limit Exceeded", "Compilation Error", "Runtime Error"
runtime: the execution time of the program
"""

import subprocess
import tempfile
import os
import pandas as pd
import time
import glob
import concurrent.futures
from typing import Tuple


DEFAULT_DATASET = "~/data/improvement_pairs_additional_metadata.csv"
PUBLIC_TEST_CASES_FOLDER = "datasets/public_test_cases/"
HIDDEN_TEST_CASES_FOLDER = "datasets/generated_test_cases/"
MAX_TIMEOUT = 5


def run_python_code_with_file_input(
    code: str, input_file_path: str
) -> Tuple[str, float, str]:
    # Create a temporary directory to hold the Python script
    with tempfile.TemporaryDirectory() as temp_dir:
        python_file_path = os.path.join(temp_dir, "code.py")

        # Write the Python code to a file
        with open(python_file_path, "w") as python_file:
            python_file.write(code)

        # Run the Python script with input redirected from the input file
        try:
            start_time = time.time()
            with open(input_file_path, "r") as input_file:
                run_process = subprocess.run(
                    ["python3", python_file_path],
                    stdin=input_file,
                    capture_output=True,
                    text=True,
                    timeout=MAX_TIMEOUT,
                )
            end_time = time.time()
            print(f"time to run all test cases: {end_time - start_time:.2f} seconds")
            if run_process.returncode != 0:
                # Handle runtime errors
                return "Runtime Error", -1, ""
            return "Accepted", (end_time - start_time), run_process.stdout
        except subprocess.TimeoutExpired:
            return "Time Limit Exceeded", MAX_TIMEOUT, ""


def eval_output(output: str, expected_output_file: str) -> bool:
    with open(expected_output_file, "r") as expected_file:
        expected_output = expected_file.read()
        return output.strip() == expected_output.strip()


def run_single_test_case(code: str, input_file: str) -> Tuple[str, float, str]:
    expected_output_file = input_file.replace("input", "output")
    verdict, runtime, actual_output = run_python_code_with_file_input(code, input_file)
    if verdict != "Accepted":
        return verdict, runtime, input_file
    elif not eval_output(actual_output, expected_output_file):
        return "Wrong Answer", runtime, input_file
    return "Accepted", runtime, input_file


def run_tcs(code: str, problem_id: int) -> Tuple[str, float]:
    # Example paths for test cases, these need to be defined or configured
    sample_output_folder = f"{PUBLIC_TEST_CASES_FOLDER}{problem_id}"
    hidden_output_folder = f"{HIDDEN_TEST_CASES_FOLDER}{problem_id}"
    
    # Check if the code compiles
    with tempfile.NamedTemporaryFile(suffix=".py") as temp_file:
        temp_file.write(code.encode())
        temp_file.flush()
        try:
            subprocess.check_output(["python", temp_file.name], stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError:
            print("Compilation Error")
            return "Compilation Error", -1
    
    start_time = time.time()
    folders = [sample_output_folder, hidden_output_folder]
    test_cases = []
    execution_time = 0

    for folder in folders:
        input_files = glob.glob(os.path.join(os.path.expanduser(folder), "input.*.txt"))
        for input_file in input_files:
            test_cases.append((code, input_file))

    # print('all testcases', folders)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(lambda p: run_single_test_case(*p), test_cases)

    for verdict, runtime, input_file in results:
        if verdict != "Accepted":
            print(f"Failed on test case {input_file}")
            return verdict, 2 if verdict == "Time Limit Exceeded" else -1
        execution_time += runtime

    end_time = time.time()
    print(f"time to run all test cases: {end_time - start_time:.2f} seconds")
    return "Accepted", execution_time / len(test_cases)


def load_dataset(dataset=DEFAULT_DATASET):
    df = pd.read_csv(dataset, sep="\t")
    return df


if __name__ == "__main__":
    sample_code = df.at[17, "code_v0"]
    python_df = df[df["language"] == "Python"]
    print(run_tcs(sample_code, 3352))
