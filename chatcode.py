import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# Ensure TOKENIZERS_PARALLELISM is set for transformers
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def load_model(model_path):
    """
    Load the model with checkpoint files.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        load_in_8bit=True,  # Optional: Load in 8-bit mode for better performance
        device_map="auto"  # Optional: Load the model on the appropriate device
    )
    
    # Load state dictionaries from checkpoint files
    model_state_dict = torch.load(f"{model_path}/meta_model_0.pt", map_location="cpu")
    adapter_state_dict = torch.load(f"{model_path}/adapter_0.pt", map_location="cpu")
    
    # Apply the loaded state dictionaries to the model
    model.load_state_dict(model_state_dict, strict=False)
    model.load_state_dict(adapter_state_dict, strict=False)

    return model

def generate_text(model, tokenizer, input_text):
    """
    Generate text based on the input using the loaded model and tokenizer.
    """
    # Encode the input text
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    
    # Generate output from the model
    output = model.generate(input_ids, max_length=100)
    
    # Decode and return the output text
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return output_text

def main():
    """
    Main function to load model, tokenizer and run Gradio interface.
    """
    model_path = "models/checkpoints/python_leq_160_tokens/"  # Replace with your actual model directory path
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load the model
    model = load_model(model_path)

    # Set up Gradio interface
    iface = gr.Interface(
        fn=lambda text: generate_text(model, tokenizer, text),
        inputs=gr.inputs.Textbox(lines=2, placeholder="Enter text here..."),
        outputs="text",
        title="OptRL Code Generator",
        description="Generate text using a fine-tuned Llama3 model."
    )
    
    # Launch the interface
    iface.launch()

if __name__ == "__main__":
    main()
