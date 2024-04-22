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
    Tokenize the code using the tokenizer and prepare it for the model.

    Args:
        code (str): The code to tokenize.

    Returns:
        torch.Tensor: The tokenized code ready for input into the model.
    """
    print("Tokenizing code...")
    # Ensure to return_tensors='pt' to get PyTorch tensors directly
    inputs = tokenizer.encode_plus(code, return_tensors="pt", truncation=True, max_length=max_length, add_special_tokens=True)
    input_ids = inputs['input_ids'].to(device)  # Move to the correct device
    print(f"Tokenized code: {input_ids}")
    return input_ids



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


def generate_code(input_code_tokens, temperature=1.0, do_sample=True):
    """
    Generate optimized code based on the input code tokens.

    Args:
        input_code_tokens (torch.Tensor): The tokenized input code.
        temperature (float): Temperature setting for generation.
        do_sample (bool): Whether to sample based on the temperature.

    Returns:
        list: The generated optimized code tokens.
    """
    print("Generating optimized code...")
    outputs = model.generate(
        input_ids=input_code_tokens,
        max_length=max_length,
        pad_token_id=tokenizer.eos_token_id,
        temperature=temperature,
        do_sample=do_sample
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
    total_losses = []  # List to store total loss values for visualization
    reward_scores = []  # List to store reward values
    ranking_losses = []  # List to store ranking loss values
    tuning_losses = []  # List to store tuning loss values

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
                generated_code_tokens = generate_code(input_code_tokens, temperature=0.8, do_sample=True)
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
            total_loss = ranking_loss + tuning_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Collect data for plotting
            total_losses.append(total_loss.item())
            reward_scores.append(sum(rewards) / len(rewards))  # Average reward for simplicity
            ranking_losses.append(ranking_loss)
            tuning_losses.append(tuning_loss)
            print(f"Combined loss: {total_loss.item():.4f}")

        except Exception as e:
            print(f"Error during training in episode {episode + 1}: {str(e)}")
            continue  # Continue training despite the error

    # Save the model only after all episodes
    checkpoints_dir = "../checkpoints"
    os.makedirs(checkpoints_dir, exist_ok=True)
    model_checkpoint_path = os.path.join(checkpoints_dir, "codet5_model.pt")
    torch.save(model.state_dict(), model_checkpoint_path)
    print(f"Model checkpoint saved at: {model_checkpoint_path}")

    # Plot and save the metrics
    plot_directory = "../plots"
    os.makedirs(plot_directory, exist_ok=True)
    metrics = {'Total Loss': total_losses, 'Reward Scores': reward_scores, 'Ranking Losses': ranking_losses,
               'Tuning Losses': tuning_losses}
    for metric_name, values in metrics.items():
        print(f"{metric_name} is {values}")
        plt.figure()
        plt.plot(values)
        plt.title(f"{metric_name} per Episode")
        plt.xlabel("Episode")
        plt.ylabel(metric_name)
        plt.savefig(os.path.join(plot_directory, f"{metric_name.replace(' ', '_').lower()}.png"))
        plt.close()

    print("Training completed.")


# Start training
train(model, tokenizer, optimizer, args.num_episodes, dataset)
