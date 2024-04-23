import json
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from unit_tests.run_code import run_tcs


def eval_model(checkpoint, dataset):
    outputs_slow = []
    outputs_fast = []
    outputs_memory_slow = []
    outputs_memory_fast = []

    # import model
    device = "cpu"
    # TODO: figure out max_length output or max_new_tokens
    max_length_output = 105
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        checkpoint,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)

    for problem in dataset:
        # print("running new problem")
        # get slow_code, fast_code
        slow_code = problem["input"]
        fast_code = problem["output"]

        # calculate slow_runtime, fast_runtime
        # verdict_slow, runtime_slow, memory_slow = run_tcs(
        #     slow_code, problem["problem_id"]
        # )
        # verdict_fast, runtime_fast, memory_fast = run_tcs(
        #     fast_code, problem["problem_id"]
        # )

        verdict_fast, runtime_fast = run_tcs(fast_code, problem["problem_id"])
        print("verdict_fast", verdict_fast, runtime_fast)

        verdict_slow, runtime_slow = run_tcs(slow_code, problem["problem_id"])
        print("verdict_slow", verdict_slow)

        # generate problem statement
        problem_statement = problem["instruction"] + problem["input"]

        # encode problem (replace with actual problem input)
        encoding = tokenizer(problem_statement, return_tensors="pt").to(device)
        encoding["decoder_input_ids"] = encoding["input_ids"].clone()

        # generate code
        outputs = model.generate(**encoding, max_length=max_length_output)
        generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("generated_code ", generated_code, " done")

        # TODO: update second argument here
        # run generated code
        # verdict, runtime, new_memory = run_tcs(
        #     generated_code, problem
        # )
        verdict, runtime = run_tcs(generated_code, problem["problem_id"])

        print("verdict", verdict)

        # evaluate generated code
        if verdict == "Accepted":
            if runtime < runtime_slow:
                outputs_slow.append(1)
            if runtime < runtime_fast:
                outputs_fast.append(1)
            # if new_memory < memory_slow:
            #     outputs_memory_slow.append(1)
            # if new_memory < memory_fast:
            #     outputs_memory_fast.append(1)
            else:
                outputs_slow.append(0)
                outputs_fast.append(0)
                outputs_memory_slow.append(0)
                outputs_memory_fast.append(0)
        else:
            outputs_slow.append(0)
            outputs_fast.append(0)
            outputs_memory_slow.append(0)
            outputs_memory_fast.append(0)

    pct_opt_slow, pct_opt_fast, pct_opt_memory_slow, pct_opt_memory_fast = (
        calculate_model_metrics(
            outputs_slow, outputs_fast, outputs_memory_slow, outputs_memory_fast
        )
    )

    return pct_opt_slow, pct_opt_fast, pct_opt_memory_slow, pct_opt_memory_fast


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
    # args = parse_args()

    # # load sample dataset
    # test_dataset = datasets.load_dataset("json", data_files=args.inst_dataset)["train"]
    # print(test_dataset)
    checkpoint = "Salesforce/codet5p-2b"
    file_path = "./examples/test.json"

    # Open the file in read mode
    with open(file_path, "r") as json_file:
        # Load the JSON data from the file
        data = json.load(json_file)
    print(eval_model(checkpoint, data))


if __name__ == "__main__":
    main()
