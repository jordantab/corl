import subprocess
import tempfile
import os
import pandas as pd
import time
import glob


def run_cpp_code_with_file_input(code: str, input_file_path: str) -> str:
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
            return f"Compilation Failed:\n{compile_process.stderr}"
        
        # Run the compiled executable with input redirected from the input file
        try:
            start_time = time.time()
            with open(input_file_path, 'r') as input_file:
                run_process = subprocess.run(executable_path, stdin=input_file, capture_output=True, text=True, universal_newlines=True, timeout=2)
                if run_process.returncode != 0:
                    # Runtime error
                    return f"Runtime Error:\n{run_process.stderr}"
            end_time = time.time()
            return run_process.stdout, (end_time - start_time)
        except subprocess.TimeoutExpired:
            return -1, -1

def eval_output(output: str, expected_output_file: str) -> bool:
    with open(expected_output_file, 'r') as expected_file:
        expected_output = expected_file.read()
        return output.strip() == expected_output.strip()
    

def run_tcs(code: str, sample_output_folder: str, hidden_output_folder) -> bool:
    for folder in [sample_output_folder, hidden_output_folder]:
        input_files = glob.glob(os.path.join(folder, "input.*.txt"))
        for input_file in input_files:
            # print('processing:', input_file)
            expected_output_file = input_file.replace('input', 'output')  # Assuming expected output file naming convention
            actual_output, time = run_cpp_code_with_file_input(code, input_file)
            # print(time)
            if actual_output is None or not eval_output(actual_output, expected_output_file):
                print('Failed on test case', input_file)
                return False
    
    return True


if __name__ == "__main__":
    df = pd.read_csv("../datasets/improvement_pairs_additional_metadata.csv", sep="\t")
    # Assuming 'input.txt' is a file in the current directory containing the input for the C++ program
    # input_file_path = '../datasets/codenet/public_test_cases/p00849/input.0.txt'
    sample_code = df.at[3, 'code_v1']
    # output, tot_time = run_cpp_code_with_file_input(sample_code, input_file_path)
    # print(f"total time: {tot_time:.2f} seconds")
    # print(eval_output(output, '../datasets/codenet/public_test_cases/p00849/output.0.txt'))
    print(run_tcs(sample_code, "../datasets/codenet/public_test_cases/p00849/", "../datasets/codenet2/generated_test_cases/p00849/"))