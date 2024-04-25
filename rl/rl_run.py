import torch
import torch.optim as optim
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import random
import argparse
import matplotlib.pyplot as plt
import json
import os
import traceback

from unit_test.run_code import run_tcs

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
    parser.add_argument(
        "--max_length", type=int, default=256, help="Maximum length of the input code."
    )
    parser.add_argument(
        "--R1", type=float, default=-1.0, help="Reward for compilation failure."
    )
    parser.add_argument(
        "--R2", type=float, default=0.0, help="Reward for unit test failure."
    )
    parser.add_argument(
        "--R3", type=float, default=0.5, help="Reward for unit test success with no improvement."
    )
    parser.add_argument(
        "--R4", type=float, default=1.0, help="Reward for unit test success with improvement."
    )
    return parser.parse_args()


def load_dataset(file_path):
    """
    Load and return the dataset from a JSON file.
    """
    with open(file_path, "r") as file:
        return json.load(file)


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


def get_reward(generated_code, reference_code, problem_id):
    """
    Calculate the reward based on the generated code and reference code.

    Args:
        generated_code (str): The generated code.
        reference_code (str): The reference code.
        problem_id (int): The problem ID.

    Returns:
        float: The reward value.
    """
    print("Calculating reward...")

    # Run the generated code and get the verdict, runtime, and memory usage
    verdict, runtime = run_tcs(generated_code, problem_id)

    # Run the reference code and get the verdict, runtime, and memory usage
    reference_verdict, reference_runtime = run_tcs(reference_code, problem_id)

    if not compile_code(generated_code):
        print("Reward: Cannot compile (R1)")
        return args.R1
    elif verdict == "Accepted":
        if runtime < reference_runtime:
            print("Reward: Passed unit tests and improved execution time (R4)")
            return args.R4
        else:
            print("Reward: Passed unit tests but no improvement in execution time (R3)")
            return args.R3
    else:
        print("Reward: Failed unit tests (R2)")
        return args.R2


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

    # Check if input_code_tokens is a tensor and not a list
    if not isinstance(input_code_tokens, torch.Tensor):
        raise TypeError("input_code_tokens must be a PyTorch Tensor")

    print(f"Input code tokens: {input_code_tokens}")

    outputs = model.generate(
        input_ids=input_code_tokens,
        max_length=args.max_length,
        pad_token_id=tokenizer.eos_token_id,
        temperature=temperature,
        do_sample=do_sample,
    )
    generated_code_tokens = outputs[0].tolist()
    print(f"Generated optimized code tokens: {generated_code_tokens}")

    return generated_code_tokens


def calculate_score(generated_code_tokens, input_code_tokens):
    """
    Calculate the score of the generated code based on Equation 9.
    This score represents how likely the model is to generate that particular code sample.

    Args:
        generated_code_tokens (list): The token IDs of the generated code.
        input_code_tokens (torch.Tensor): The token IDs of the input code.

    Returns:
        float: The calculated score.
    """
    print("Calculating score...")
    log_probs = []

    if isinstance(generated_code_tokens, list):
        generated_code_tokens = torch.tensor(
            generated_code_tokens, dtype=torch.long, device=device
        )

    # Get the model's output logits for the input code tokens
    logits = model(input_ids=input_code_tokens).logits

    # Loop over each token_id in the generated code and calculate log probability
    for token_id in generated_code_tokens:
        log_prob = torch.log_softmax(logits[0], dim=-1)
        if token_id.item() >= log_prob.size(0):
            continue  # skip if token_id is out of range of logits
        log_prob = log_prob[token_id.item()].item()
        log_probs.append(log_prob)

    score = sum(log_probs) / len(generated_code_tokens) if generated_code_tokens else 0
    print(f"Calculated sample likelihood score: {score:.4f}")
    return score


def train(model, tokenizer, optimizer, num_episodes, dataset):

    model.train()
    total_losses = []
    reward_scores = []
    ranking_losses = []
    tuning_losses = []

    for episode in range(num_episodes):
        print(f"Episode {episode + 1}/{num_episodes}")
        episode_total_losses = []
        episode_reward_scores = []
        episode_ranking_losses = []
        episode_tuning_losses = []

        for entry in dataset:
            try:
                prompt = f"{entry['instruction']} \n {entry['input']}"
                pid = entry["problem_id"]
                print(f"Prompt: {prompt}")
                input_code_tokens = tokenize_code(prompt)

                candidate_samples = []
                num_samples = 1
                rewards = []

                for i in range(num_samples):
                    print(f"Generating candidate sample {i + 1}/{num_samples}")
                    generated_code_tokens = generate_code(
                        input_code_tokens, temperature=0.8, do_sample=True
                    )
                    candidate_samples.append(generated_code_tokens)

                    generated_code = tokenizer.decode(
                        generated_code_tokens, skip_special_tokens=True
                    )
                    print(f"Generated code: {generated_code}")

                    reward = get_reward(generated_code, entry["input"], pid)
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
                tuning_loss = -calculate_score(best_sample, entry["input"])

                # Collect data for the current entry
                episode_total_losses.append(ranking_loss + tuning_loss)
                episode_reward_scores.append(sum(rewards) / len(rewards))
                episode_ranking_losses.append(ranking_loss)
                episode_tuning_losses.append(tuning_loss)

            except Exception as e:
                print(f"Error during training for entry: {str(e)}")
                traceback.print_exc()
                continue

        # Calculate average metrics for the episode
        avg_total_loss = sum(episode_total_losses) / len(episode_total_losses)
        avg_reward_score = sum(episode_reward_scores) / len(episode_reward_scores)
        avg_ranking_loss = sum(episode_ranking_losses) / len(episode_ranking_losses)
        avg_tuning_loss = sum(episode_tuning_losses) / len(episode_tuning_losses)

        # Update model parameters
        optimizer.zero_grad()
        avg_total_loss.backward()
        optimizer.step()

        # Collect data for plotting
        total_losses.append(avg_total_loss.item())
        reward_scores.append(avg_reward_score)
        ranking_losses.append(avg_ranking_loss)
        tuning_losses.append(avg_tuning_loss)
        print(f"Average combined loss for episode: {avg_total_loss.item():.4f}")

    # Save the model only after all episodes
    checkpoints_dir = "../checkpoints"
    os.makedirs(checkpoints_dir, exist_ok=True)
    model_checkpoint_path = os.path.join(checkpoints_dir, "codet5_model.pt")
    torch.save(model.state_dict(), model_checkpoint_path)
    print(f"Model checkpoint saved at: {model_checkpoint_path}")

    # Plot and save the metrics
    plot_directory = "../plots"
    os.makedirs(plot_directory, exist_ok=True)
    metrics = {
        "Average Total Loss": total_losses,
        "Average Reward Scores": reward_scores,
        "Average Ranking Losses": ranking_losses,
        "Average Tuning Losses": tuning_losses,
    }
    for metric_name, values in metrics.items():
        print(f"{metric_name}: {values}")
        plt.figure()
        plt.plot(values)
        plt.title(f"{metric_name} per Episode")
        plt.xlabel("Episode")
        plt.ylabel(metric_name)
        plt.savefig(
            os.path.join(plot_directory, f"{metric_name.replace(' ', '_').lower()}.png")
        )
        plt.close()

    # Calculate and print overall average metrics
    overall_avg_total_loss = sum(total_losses) / len(total_losses)
    overall_avg_reward_score = sum(reward_scores) / len(reward_scores)
    overall_avg_ranking_loss = sum(ranking_losses) / len(ranking_losses)
    overall_avg_tuning_loss = sum(tuning_losses) / len(tuning_losses)

    print("\nOverall Average Metrics:")
    print(f"Total Loss: {overall_avg_total_loss:.4f}")
    print(f"Reward Score: {overall_avg_reward_score:.4f}")
    print(f"Ranking Loss: {overall_avg_ranking_loss:.4f}")
    print(f"Tuning Loss: {overall_avg_tuning_loss:.4f}")

    print("Training completed.")


if __name__ == "__main__":
    args = parse_args()
    checkpoint = args.model_name
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        checkpoint,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    dataset = load_dataset(args.dataset_path)

    # Start training
    train(
        model,
        tokenizer,
        optimizer,
        args.num_episodes,
        dataset
    )