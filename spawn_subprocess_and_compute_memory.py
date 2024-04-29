# spawn_and_compute_memory.py
import subprocess

MEMPROF_SCRIPT = "memprof_subprocess.py"
COMMAND = "python3 count_tokens.py --file datasets/cpp/cpp_leq_105_tokens_test.json"


def main():
    # Run the command
    cmd_list = ["python3", MEMPROF_SCRIPT] + COMMAND.split()
    completed_process = subprocess.run(cmd_list, capture_output=True)

    # Get the stdout, decode it to a string, strip whitespace, and convert to an integer
    print(f"stdout: {completed_process.stdout}")
    max_rss = int(completed_process.stdout.decode().strip())

    print(f"Maximum memory usage: {max_rss} bytes")


if __name__ == "__main__":
    main()
