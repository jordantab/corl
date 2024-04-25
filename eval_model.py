import json
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)
import transformers
from unit_tests.run_code import run_tcs
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
    """
    model = AutoModelForSeq2SeqLM.from_pretrained(
        checkpoint,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    """
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, torch_dtype=torch.bfloat16, device_map=device
    )

    for problem in dataset:
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

        # verdict_fast, runtime_fast = run_tcs(fast_code, problem["problem_id"])
        # print("verdict_fast", verdict_fast, runtime_fast)

        # verdict_slow, runtime_slow = run_tcs(slow_code, problem["problem_id"])
        # print("verdict_slow", verdict_slow)

        # generate problem statement
        problem_statement = problem["instruction"] + "\n" + problem["input"]

        print("problem_statement", problem_statement)

        # Tokenize the problem statement for the encoder
        """
        encoder_inputs = tokenizer(
                "def print_hello_world():",
            return_tensors="pt",
            padding="max_length",
            max_length=105,
            truncation=True,
        ).to(device)
        """

        # messages = [
        #     {
        #         "role": "system",
        #         "content": "Please provide an optimized version of the following code in python",
        #     },
        #     {"role": "user", "content": problem["input"]},
        # ]

        # encoder_inputs = tokenizer.apply_chat_template(
        #     messages, add_generation_prompt=True, return_tensors="pt"
        # ).to(device)

        # terminators = [
        #     tokenizer.eos_token_id,
        #     tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        # ]

        # outputs = model.generate(
        #     encoder_inputs,
        #     max_new_tokens=256,
        #     eos_token_id=terminators,
        #     do_sample=True,
        #     temperature=0.6,
        #     top_p=0.9,
        # )
        # response = outputs[0][encoder_inputs.shape[-1] :]
        # generated_code = tokenizer.decode(response, skip_special_tokens=True)
        # print("gen code\n", generated_code, "\ndone")
        llm_model = transformers.pipeline(
            "text-generation",
            model=checkpoint,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map=device,
        )

        print(llm_model("How are you?"))
        return

        """
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
        generated_code = outputs[0]["generated_text"][len(prompt) :]
        print("generated_code\n", generated_code, "\ndone")

        attention_mask = encoder_inputs["attention_mask"]

        # List of potential start tokens
        start_tokens = [
            tokenizer.bos_token_id,
            tokenizer.cls_token_id,
            tokenizer.pad_token_id,
        ]

        for token_id in start_tokens:
            if token_id is not None:
                print({tokenizer.decode([token_id])})
                # Prepare decoder_input_ids with the start token
                decoder_input_ids = torch.tensor([[token_id]], dtype=torch.long).to(
                    device
                )

        # Generate code
        generated_tokens = model.generate(
            input_ids=encoder_inputs["input_ids"],
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            max_length=max_length_output,
        )

        generated_code = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

        # # encode problem (replace with actual problem input)
        # encoding = tokenizer(problem_statement, return_tensors="pt").to(device)
        # encoding["decoder_input_ids"] = encoding["input_ids"].clone()

        # # generate code
        # outputs = model.generate(**encoding, max_length=max_length_output)
        # generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print("generated_code ", generated_code, " done")

        # TODO: update second argument here
        # run generated code
        verdict, runtime = run_tcs(generated_code, problem["problem_id"])

        print("verdict", verdict)

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
"""


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
    checkpoint = "meta-llama/Meta-Llama-3-8B"
    file_path = "./examples/test_python.json"

    # Open the file in read mode
    with open(file_path, "r") as json_file:
        # Load the JSON data from the file
        data = json.load(json_file)
    print(eval_model(checkpoint, data))


if __name__ == "__main__":
    main()
