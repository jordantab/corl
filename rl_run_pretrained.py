import torch
import torch.optim as optim
from torch.optim.adam import Adam
from torch.nn.functional import log_softmax
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import argparse
import matplotlib.pyplot as plt
import json
import os
import traceback
import transformers
from unit_tests.run_code import run_tcs, run_python_code_with_file_input

os.environ["TOKENIZERS_PARALLELISM"] = "true"


# Utility functions
def parse_args():
    parser = argparse.ArgumentParser(description="Train a model for code optimization.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Model checkpoint for initialization.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the model checkpoint file",
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
        "--R3",
        type=float,
        default=0.5,
        help="Reward for unit test success with no improvement.",
    )
    parser.add_argument(
        "--R4",
        type=float,
        default=1.0,
        help="Reward for unit test success with improvement.",
    )
    return parser.parse_args()


def load_dataset(file_path):
    """
    Load and return the dataset from a JSON file.
    """
    with open(file_path, "r") as file:
        return json.load(file)


# Util functions for reward calculating
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

    if verdict == "Compilation Error":
        print("Reward: Cannot compile (R1)")
        return args.R1
    elif verdict == "Accepted":
        # Run the reference code and get the verdict, runtime, and memory usage
        reference_verdict, reference_runtime = run_tcs(reference_code, problem_id)

        if runtime < reference_runtime:
            print("Reward: Passed unit tests and improved execution time (R4)")
            return args.R4
        else:
            print("Reward: Passed unit tests but no improvement in execution time (R3)")
            return args.R3
    else:
        print("Reward: Failed unit tests (R2)")
        return args.R2


def generate_code(input_code, temperature=0.6):
    """
    Generate optimized code based on the input code tokens.

    Args:
        input_code (str): The input code.
        temperature (float): Temperature setting for generation.

    Returns:
        list: The generated optimized code tokens.
    """
    print("Generating optimized code...")
    print(f"Input code: {input_code}")

    messages = [
        {
            "role": "system",
            "content": "Provide an optimized version of the following code snippet. Only provide the code, no need to provide any description. ",
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

    outputs = pipeline(
        prompt,
        max_new_tokens=args.max_length,
        eos_token_id=terminators,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
    )
    generated_code = outputs[0]["generated_text"][len(prompt) :]
    print(f"Generated optimized code tokens: {generated_code}")

    return generated_code


def calculate_score(generated_code, input_code):
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
    
    print('\n===========================================')
    print(f"input code: {input_code}")
    
    print('\n===========================================')
    print(f"generated code: {generated_code}")
    
    # Ensure input_code_tokens is a tensor, on the correct device, and has a batch dimension
    # if not isinstance(input_code, torch.Tensor):
    #     input_code = torch.tensor(input_code, dtype=torch.long, device=device)
    # input_code = input_code.to(device)
    # if input_code.dim() == 1:
    #     input_code = input_code.unsqueeze(0)
        
    input_code_tokens = tokenize_code(input_code)
    generated_code_tokens = tokenize_code(generated_code)
    
    print('\n===========================================')
    print(f"input code tokens: {input_code_tokens}")
    
    print('\n===========================================')
    print(f"generated code tokens: {generated_code_tokens}")
    

    # Generate logits from the model
    logits = model(input_ids=input_code_tokens).logits
    print

    # Convert generated_code_tokens to a tensor if it's a list
    # if isinstance(generated_code_tokens, list):
    #     generated_code_tokens = torch.tensor(generated_code_tokens, dtype=torch.long, device=device)

    # Calculate the log probability for each token in the generated code
    for token_id in generated_code_tokens:
        # data_list = data.to('cuda').tolist()
        data = token_id.to('cuda').tolist()
        for token in data:
            token_logits = log_softmax(logits[0], dim=-1)  # Compute softmax over the logits of the first (and only) batch item
            if token >= token_logits.size(1):
                continue  # Skip if token is out of the valid range for logits
            log_prob = token_logits[0, token]  # Access log probability of the specific token
            log_probs.append(log_prob.item())

    # Calculate the average log probability (score) if there are any valid log probabilities
    score = sum(log_probs) / len(log_probs) if log_probs else 0
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

        print("dataset size" + str(len(dataset)))
        for entry in dataset:
            print(f"this entry is: {entry}")
            try:
                input_code = entry["input"]
                pid = entry["problem_id"]
                print("pid: ",pid)
                # input_code_tokens = tokenize_code(input_code)

                candidate_samples = []
                num_samples = 3
                rewards = []

                for i in range(num_samples):
                    print(f"\nGenerating candidate sample {i + 1}/{num_samples}")
                    generated_code = generate_code(input_code, temperature=0.8)
                    
                    
                    # generated_code_tokens = tokenizer.encode(
                    #     generated_code, add_special_tokens=True
                    # )
                    candidate_samples.append(generated_code)
                    print(f"\nGenerated code: {generated_code}")

                    # if isinstance(generated_code, list):
                    #     generated_code = generated_code[0]

                    reward = get_reward(generated_code, entry["input"], pid)
                    print(f"reward is {reward}")
                    rewards.append(reward)

                ranking_loss = 0
                for i in range(len(candidate_samples)):
                    for j in range(len(candidate_samples)):
                        if rewards[i] < rewards[j]:
                            log_prob_i = calculate_score(
                                candidate_samples[i], input_code
                            )
                            log_prob_j = calculate_score(
                                candidate_samples[j], input_code
                            )
                            ranking_loss += max(0, log_prob_i - log_prob_j)

                print(f"ranking loss is: {ranking_loss}")

                best_sample_idx = rewards.index(max(rewards))
                best_sample = candidate_samples[best_sample_idx]
                tuning_loss = -calculate_score(best_sample, entry["input"])

                print(f"tuning loss is: {tuning_loss}")

                # Collect data for the current entry
                episode_total_losses.append(ranking_loss + tuning_loss)
                # avg rewards across all the samples(for that entry) generated per episode
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
        
        print(avg_total_loss, avg_reward_score, avg_ranking_loss, avg_tuning_loss)

        # Update model parameters
        loss_tensor = torch.tensor(avg_total_loss, requires_grad=True)
        optimizer.zero_grad()
        loss_tensor.backward()
        optimizer.step()

        # Collect data for plotting
        total_losses.append(avg_total_loss)
        reward_scores.append(avg_reward_score)
        ranking_losses.append(avg_ranking_loss)
        tuning_losses.append(avg_tuning_loss)
        print(f"Average combined loss for episode: {avg_total_loss:.4f}")

    # Save the model checkpoint only after all episodes
    checkpoints_dir = "./models/rl-checkpoint"
    os.makedirs(checkpoints_dir, exist_ok=True)
    model_checkpoint_path = os.path.join(checkpoints_dir, "llama-rl-trained.pt")
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
    # save the metrics data to a csv file
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(plot_directory, "metrics.csv"), index=False)
    print(f"Metrics data saved at: {os.path.join(plot_directory, 'metrics.csv')}")

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
    print("load args")
    checkpoint = args.model_name
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("load tokenizer and base model")
    # model = AutoModelForCausalLM.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    # comment this line if you want to use the model directly from huggingface
    # model_path = os.path.join(args.checkpoint_path, "meta_model_0.pt")
    # adaptor_path = os.path.join(args.checkpoint_path, "adapter_0.pt")

    # # Load the model state dict
    # model_state_dict = torch.load(model_path)
    # model.load_state_dict(model_state_dict)

    # print("Model state dict loaded!")

    # # Load the adaptor state dict
    # adaptor_state_dict = torch.load(adaptor_path)

    # # Apply the adaptor state dict to the model
    # for name, param in adaptor_state_dict.items():
    #     if name in model.state_dict():
    #         model.state_dict()[name].copy_(param)

    # print("Adaptor state dict loaded!")
    
    # pipeline = transformers.pipeline(
    #     "text-generation", model=model, tokenizer=tokenizer, device=device
    # )
    
    
    # the below commented part is for using a pretrained model from huggingface
    pipeline = transformers.pipeline(
        "text-generation",
        model=checkpoint,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    print("set up pipeline")
    model = pipeline.model
    print("get model from pipeline")

    optimizer = Adam(model.parameters(), lr=1e-5)
    print("set up optimizer, loading dataset")
    dataset = load_dataset(args.dataset_path)
    print(
        f"Loaded dataset with {len(dataset)} entries, input code is: {dataset[0]['input']}"
    )

    # Start training
    train(model, tokenizer, optimizer, args.num_episodes, dataset)
