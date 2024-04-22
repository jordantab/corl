import torch
import torch.optim as optim
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import random
import argparse
import matplotlib.pyplot as plt
import json
import os


# Utility functions
def parse_args():
    parser = argparse.ArgumentParser(description="Train a model for code optimization.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Salesforce/codet5p-2b",
        help="Model checkpoint for initialization.",
    )
    parser.add_argument(
        "--num_episodes", type=int, default=10, help="Number of training episodes."
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="Path to the dataset file."
    )
    return parser.parse_args()


def load_dataset(file_path):
    """
    Load and return the dataset from a JSON file.
    """
    with open(file_path, "r") as file:
        return json.load(file)


# Load arguments and initialize components
args = parse_args()

checkpoint = args.model_name
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-5)

max_length = 256

# Define rewards
R1, R2, R3, R4 = -1.0, 0.0, 0.5, 1.0

# Load dataset
dataset = load_dataset(args.dataset_path)


# Util functions for reward calculating
def compile_code(code):
    """
    Compile the code and return a random compilation result (True for success, False for failure).

    Args:
        code (str): The code to compile.

    Returns:
        bool: True if the compilation is successful, False otherwise.
    """
    result = random.choice([True, False])
    print(f"Compilation result: {result}")
    return result


def pass_unit_tests(code):
    """
    Run unit tests and return a random test result (True for passing, False for failing).

    Args:
        code (str): The code to test.

    Returns:
        bool: True if the unit tests pass, False otherwise.
    """
    result = random.choice([True, False])
    print(f"Unit test result: {result}")
    return result


def execution_time(code):
    """
    Return a random execution time value (in seconds).

    Args:
        code (str): The code to measure execution time.

    Returns:
        float: The execution time in seconds.
    """
    time = random.uniform(0.1, 1.0)
    print(f"Execution time: {time:.2f} seconds")
    return time


def tokenize_code(code):
    """
    Tokenize the code using the tokenizer.

    Args:
        code (str): The code to tokenize.

    Returns:
        list: The tokenized code as a list of token IDs.
    """
    print("Tokenizing code...")
    tokens = tokenizer.tokenize(code)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    print(f"Tokenized code: {token_ids}")
    return token_ids


def get_reward(generated_code, reference_code):
    """
    Calculate the reward based on the generated code and reference code.

    Args:
        generated_code (str): The generated code.
        reference_code (str): The reference code.

    Returns:
        float: The reward value.
    """
    print("Calculating reward...")
    if not compile_code(generated_code):
        print("Reward: Cannot compile (R1)")
        return R1
    elif not pass_unit_tests(generated_code):
        print("Reward: Failed unit tests (R2)")
        return R2
    elif pass_unit_tests(generated_code) and execution_time(
        generated_code
    ) < execution_time(reference_code):
        print("Reward: Passed unit tests and improved execution time (R4)")
        return R4
    else:
        print("Reward: Passed unit tests but no improvement in execution time (R3)")
        return R3


def generate_code(input_code_tokens, temperature=1.0):
    """
    Generate optimized code based on the input code tokens.

    Args:
        input_code_tokens (list): The tokenized input code.

    Returns:
        list: The generated optimized code tokens.
    """
    print("Generating optimized code...")
    # Decode tokens to code
    input_code = tokenizer.decode(input_code_tokens, skip_special_tokens=True)

    # Create an optimization prompt
    optimization_prompt = (
        f"Optimize the following code for performance and readability:\n\n{input_code}, "
        f"only return the optimized code."
    )

    # Encode the prompt
    encoding = tokenizer(optimization_prompt, return_tensors="pt").to(device)
    encoding["decoder_input_ids"] = encoding["input_ids"].clone()

    # Generate optimized code
    outputs = model.generate(
        **encoding,
        max_length=max_length,
        pad_token_id=tokenizer.eos_token_id,
        temperature=temperature
    )
    generated_code_tokens = outputs[0].tolist()
    print(f"Generated optimized code tokens: {generated_code_tokens}")

    return generated_code_tokens


def calculate_score(generated_code_tokens, reference_code_tokens):
    """
    Calculate the score of the generated code based on Equation 9.
    This score represents how likely the model is to generate that particular code sample.

    Args:
        generated_code_tokens (list): The token IDs of the generated code.
        reference_code_tokens (list): The token IDs of the reference code.

    Returns:
        float: The calculated score.
    """
    print("Calculating score...")
    log_probs = []
    for token_id in generated_code_tokens:
        reference_code = tokenizer.decode(
            reference_code_tokens, skip_special_tokens=True
        )
        reference_encoding = tokenizer(reference_code, return_tensors="pt").to(device)
        logits = model(reference_encoding["input_ids"]).logits
        log_prob = torch.log_softmax(logits[0, -1], dim=-1)[token_id].item()
        log_probs.append(log_prob)

    score = sum(log_probs) / len(generated_code_tokens)
    print(f"Calculated sample likelihood score: {score:.4f}")
    return score


def train(model, tokenizer, optimizer, num_episodes, dataset):
    """
    Train the model using episodes from the provided dataset.

    Args:
        model (torch.nn.Module): The model to be trained.
        tokenizer (Tokenizer): The tokenizer for encoding and decoding code.
        optimizer (torch.optim.Optimizer): The optimizer for model training.
        num_episodes (int): Number of training episodes.
        dataset (list): List of dataset entries.
    """
    model.train()
    losses = []  # List to store loss values for visualization

    for episode in range(num_episodes):
        print(f"Episode {episode + 1}/{num_episodes}")
        try:
            entry = dataset[episode % len(dataset)]
            prompt = f"{entry['instruction']} \n\n {entry['input']}"
            input_code_tokens = tokenize_code(prompt)

            candidate_samples = []
            num_samples = 5
            rewards = []

            for i in range(num_samples):
                print(f"Generating candidate sample {i + 1}/{num_samples}")
                generated_code_tokens = generate_code(input_code_tokens, temperature=0.8)
                candidate_samples.append(generated_code_tokens)

                generated_code = tokenizer.decode(
                    generated_code_tokens, skip_special_tokens=True
                )
                reward = get_reward(generated_code, entry["input"])
                rewards.append(reward)

            ranking_loss = 0
            for i in range(len(candidate_samples)):
                for j in range(len(candidate_samples)):
                    if rewards[i] < rewards[j]:
                        log_prob_i = calculate_score(
                            candidate_samples[i], input_code_tokens
                        )
                        log_prob_j = calculate_score(
                            candidate_samples[j], input_code_tokens
                        )
                        ranking_loss += max(0, log_prob_i - log_prob_j)

            best_sample_idx = rewards.index(max(rewards))
            best_sample = candidate_samples[best_sample_idx]
            tuning_loss = -calculate_score(best_sample, input_code_tokens)

            # Combine losses and update model parameters
            loss = ranking_loss + tuning_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())  # Collect loss for plotting
            print(f"Combined loss: {loss.item():.4f}")

        except Exception as e:
            print(f"Error during training in episode {episode + 1}: {str(e)}")
            continue  # Continue training despite the error

    # Save the model only after all episodes
    os.mkdir("../checkpoints")
    model_checkpoint_path = "../checkpoints/codet5_model.pt"
    torch.save(model.state_dict(), model_checkpoint_path)
    print(f"Model checkpoint saved at: {model_checkpoint_path}")

    # Plot training losses
    plt.plot(losses)
    plt.title("Training Loss per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.show()

    print("Training completed.")


# Start training
train(model, tokenizer, optimizer, args.num_episodes, dataset)
