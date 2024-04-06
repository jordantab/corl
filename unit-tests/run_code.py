import subprocess
import tempfile
import os
import pandas as pd
import time
import glob
import concurrent.futures
from typing import Tuple


DEFAULT_DATASET = "../datasets/improvement_pairs_additional_metadata.csv"
PUBLIC_TEST_CASES_FOLDER = "../datasets/codenet/public_test_cases/"
HIDDEN_TEST_CASES_FOLDER = "../datasets/codenet2/generated_test_cases/"


def run_cpp_code_with_file_input(code: str, input_file_path: str) -> Tuple[str, float]:
    # Create a temporary directory to hold the C++ file and executable
    with tempfile.TemporaryDirectory() as temp_dir:
        cpp_file_path = os.path.join(temp_dir, "code.cpp")
        executable_path = os.path.join(temp_dir, "code")
        
        # Write the C++ code to a file
        with open(cpp_file_path, "w") as cpp_file:
            cpp_file.write(code)
        
        # Compile the C++ code
        compile_process = subprocess.run(["g++", cpp_file_path, "-o", executable_path], capture_output=True, text=True)
        if compile_process.returncode != 0:
            # Compilation failed
            return f"Compilation Failed:\n{compile_process.stderr}", -1
        
        # Run the compiled executable with input redirected from the input file
        try:
            start_time = time.time()
            with open(input_file_path, 'r') as input_file:
                run_process = subprocess.run(executable_path, stdin=input_file, capture_output=True, text=True, universal_newlines=True, timeout=2)
                if run_process.returncode != 0:
                    # Runtime error
                    return f"Runtime Error:\n{run_process.stderr}", -1
            end_time = time.time()
            return run_process.stdout, (end_time - start_time)
        except subprocess.TimeoutExpired:
            return -1, -1

def eval_output(output: str, expected_output_file: str) -> bool:
    with open(expected_output_file, 'r') as expected_file:
        expected_output = expected_file.read()
        return output.strip() == expected_output.strip()
    
def run_single_test_case(code, input_file):
    expected_output_file = input_file.replace('input', 'output')
    actual_output, runtime = run_cpp_code_with_file_input(code, input_file)
    if runtime == -1 or not eval_output(actual_output, expected_output_file):
        return False, input_file
    return True, input_file
    
def run_tcs(code: str, problem_id: int) -> bool:
    sample_output_folder = f"{PUBLIC_TEST_CASES_FOLDER}p{problem_id:05d}"
    hidden_output_folder = f"{HIDDEN_TEST_CASES_FOLDER}p{problem_id:05d}"
    start_time = time.time()
    folders = [sample_output_folder, hidden_output_folder]
    test_cases = []

    for folder in folders:
        input_files = glob.glob(os.path.join(folder, "input.*.txt"))
        for input_file in input_files:
            test_cases.append((code, input_file))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(lambda p: run_single_test_case(*p), test_cases)
    
    for result, input_file in results:
        if not result:
            print(f'Failed on test case {input_file}')
            return False
    end_time = time.time()
    print(f"total time for all test cases: {end_time - start_time:.2f} seconds")
    return True

def load_dataset(dataset=DEFAULT_DATASET):
    df = pd.read_csv(dataset, sep="\t")
    return df


if __name__ == "__main__":
    df = load_dataset()
    sample_code = df.at[3, 'code_v1']
    print(run_tcs(sample_code, 849))