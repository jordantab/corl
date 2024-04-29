import gradio as gr
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# Ensure TOKENIZERS_PARALLELISM is set for transformers
os.environ["TOKENIZERS_PARALLELISM"] = "true"

global model, tokenizer, pipeline


def load_checkpoint(checkpoint):

    global model, tokenizer

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        load_in_8bit=True,  # Optional: Load in 8-bit mode for better performance
        device_map="auto",  # Optional: Load the model on the appropriate device
    )

    # Specify the paths to the checkpoint files
    model_path = "models/checkpoints/python_leq_60_tokens_finetune/meta_model_0.pt"
    adapter_path = "models/checkpoints/python_leq_60_tokens_finetune/adapter_0.pt"

    # Load the base model state dictionary
    model_state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(model_state_dict, strict=False)

    # Load the adapter state dictionary and apply it to the model
    adapter_state_dict = torch.load(adapter_path, map_location="cpu")
    model.load_state_dict(adapter_state_dict, strict=False)

    return model, tokenizer


def tokenize_code(code):
    """
    Tokenize the code using the tokenizer and prepare it for the model.

    Args:
        code (str): The code to tokenize.

    Returns:
        torch.Tensor: The tokenized code ready for input into the model.
    """
    print("Tokenizing code...")
    # Ensure to return_tensors='pt' to get PyTorch tensors directly
    inputs = tokenizer.encode_plus(
        code,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_length,
        add_special_tokens=True,
    )
    input_ids = inputs["input_ids"].to(device)  # Move to the correct device
    print(f"Tokenized code: {input_ids}")
    return input_ids


def generate_code(input_code, pipeline, temperature=0.6):
    """
    Generate optimized code based on the input code tokens.

    Args:
        input_code (str): The input code.
        pipeline: The text generation pipeline object.
        temperature (float): Temperature setting for generation.

    Returns:
        list: The generated optimized code tokens.
    """

    print("Generating optimized code...")
    print(f"Input code: {input_code}")

    messages = [
        {
            "role": "system",
            "content": "Provide an optimized version of the following code snippet. Only provide the code, no need to provide any description.",
        },
        {"role": "user", "content": input_code},
    ]

    prompt = pipeline.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    print(f"prompt getting passed in is: {prompt}")

    outputs = pipeline(
        prompt,
        max_new_tokens=256,  # Adjusted from args.max_length to a static value
        eos_token_id=terminators,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
    )
    generated_code = outputs[0]["generated_text"][len(prompt) :]
    print(f"Generated optimized code tokens: {generated_code}")

    return generated_code


def generate_response(input_text, history):
    """
    Generate text based on the input using the loaded model and tokenizer.
    wrapper class essential for gradio, cannot change input format
    """

    global pipeline

    print(input_text)
    print(pipeline)

    output_text = generate_code(input_text, pipeline)
    print(f"Output: {output_text}")
    return output_text


def main():
    """
    Main function to load model, tokenizer and run Gradio interface.
    """

    global pipeline

    checkpoint = "meta-llama/Meta-Llama-3-8B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using model: {checkpoint}, device: {device}")

    print("\nLoading tokenizer and base model")
    model, tokenizer = load_checkpoint("hi")
    print("Loaded tokenizer and base model")

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="cuda",
    )
    print("\nSet up pipeline!")
    model = pipeline.model
    print("Updated model from pipeline")

    # Set up Gradio interface
    print("\nSetting up Gradio interface...")
    iface = gr.ChatInterface(fn=generate_response)

    # iface = gr.Interface(
    #    fn=lambda text: generate_text(model, tokenizer, text),
    #    inputs=gr.inputs.Textbox(lines=2, placeholder="Enter text here..."),
    #    outputs="text",
    #    title="OptRL Code Generator",
    #    description="Generate text using a fine-tuned Llama3 model."
    # )

    # Launch the interface WITH PUBLIC LINK
    iface.launch(share=True)


if __name__ == "__main__":
    main()
