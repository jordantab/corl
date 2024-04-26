import json
import torch
from transformers import AutoTokenizer
import transformers
from unit_test.run_code import run_tcs
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def eval_model(checkpoint, dataset):
    outputs_slow = []
    outputs_fast = []
    outputs_memory_slow = []
    outputs_memory_fast = []

    # import model
    device = "cuda"
    max_length_output = 200
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # model = AutoModelForCausalLM.from_pretrained(
    #     checkpoint, torch_dtype=torch.bfloat16, device_map=device
    # )

    for problem in dataset:
        # get slow_code, fast_code
        slow_code = problem["input"]
        fast_code = problem["output"]

        problem_statement = problem["instruction"] + "\n" + problem["input"]

        print("problem_statement", problem_statement)

        llm_model = transformers.pipeline(
            "text-generation",
            model=checkpoint,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map=device,
        )

        print(llm_model("How are you?"))
        return


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


def pirate_example(model_id):
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    messages = [
        {
            "role": "system",
            "content": "You are a pirate chatbot who always responds in pirate speak!",
        },
        {"role": "user", "content": "Who are you?"},
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
    print(outputs[0]["generated_text"][len(prompt) :])


def optimize_code_test_1(model_id):
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    slow_fibonacci_code = 'def fibonacci_recursive(n): return 0 if n == 0 else 1 if n == 1 else fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2); print(f"The {n}th Fibonacci number (slow way): {fibonacci_recursive(10)}")'

    messages = [
        {
            "role": "system",
            "content": "Provide an optimized version of the following code snippet. Only provide the code, no need to provide any description. ",
        },
        {"role": "user", "content": slow_fibonacci_code},
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
    print(outputs[0]["generated_text"][len(prompt) :])


def optimize_code_test_2(model_id):
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    cpp_sample = "#include <bits/stdc++.h>\n\nusing namespace std;\n\nint main() {\n\n\tset<int> s; int a;\n\n\tfor(int i = 0; i < 3; i++) {\n\n\t\tcin >> a;\n\n\t\ts.insert(a);\n\n\t}\n\n\tcout << s.size() << endl;\n\n\treturn 0;\n\n}"

    messages = [
        {
            "role": "system",
            "content": "Provide an optimized version of the following code snippet. Only provide the code, no need to provide any description. ",
        },
        {"role": "user", "content": cpp_sample},
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
    print(outputs[0]["generated_text"][len(prompt) :])


def main():
    torch.cuda.empty_cache()
    checkpoint = "meta-llama/Meta-Llama-3-8B-Instruct"
    file_path = "./examples/test_python.json"

    # Open the file in read mode
    with open(file_path, "r") as json_file:
        # Load the JSON data from the file
        data = json.load(json_file)
    # optimize_code_test_1(checkpoint)
    optimize_code_test_2(checkpoint)
    # pirate_example(checkpoint)
    # print(eval_model(checkpoint, data))


if __name__ == "__main__":
    main()
