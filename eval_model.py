import json
import torch
import transformers
from unit_test.run_code import run_tcs
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def eval_model(checkpoint, dataset):
    outputs_slow, outputs_fast, outputs_memory_slow, outputs_memory_fast = []

    for problem in dataset:
        # get slow_code, fast_code
        slow_code = problem["input"]
        fast_code = problem["output"]

        # calculate slow_runtime, fast_runtime
        # verdict_fast, runtime_fast = run_tcs(fast_code, problem["problem_id"])
        # verdict_slow, runtime_slow = run_tcs(slow_code, problem["problem_id"])

        # generate problem statement
        generated_code = generate_code(checkpoint, problem["input"])
        print(generated_code)

        # run generated code
        # verdict, runtime = run_tcs(generated_code, problem["problem_id"])
        verdict = "Accepted"

        # evaluate generated code
        if verdict == "Accepted":

            # if runtime < runtime_slow:
            #     outputs_slow.append(1)
            # if runtime < runtime_fast:
            #     outputs_fast.append(1)
            # if new_memory < memory_slow:
            #     outputs_memory_slow.append(1)
            # if new_memory < memory_fast:
            #     outputs_memory_fast.append(1)
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


def generate_code(checkpoint, slow_code):
    pipeline = transformers.pipeline(
        "text-generation",
        model=checkpoint,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",  # change to device?
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
    checkpoint = "meta-llama/Meta-Llama-3-8B-Instruct"
    file_path = "./examples/test_python.json"

    # Open the file in read mode
    with open(file_path, "r") as json_file:
        # Load the JSON data from the file
        data = json.load(json_file)
    print(eval_model(checkpoint, data))


if __name__ == "__main__":
    main()
