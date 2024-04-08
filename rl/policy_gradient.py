import torch
import torch.optim as optim
from ray.rllib.algorithms.ppo import PPO
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import random
import gym
import numpy as np
import matplotlib.pyplot as plt

checkpoint = "Salesforce/codet5p-2b"
device = "cpu"  # or "cuda" for GPU
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint,
                                              torch_dtype=torch.float32,
                                              low_cpu_mem_usage=True,
                                              trust_remote_code=True).to(device)
# Define the optimizer for the CodeT5 model
optimizer = optim.Adam(model.parameters(), lr=1e-5)

max_length = 256
obs_space = gym.spaces.Box(low=0, high=1, shape=(max_length,), dtype=np.float32)
vocab_size = tokenizer.vocab_size
action_space = gym.spaces.Discrete(vocab_size)

R1 = -1.0  # Negative reward for code that cannot be compiled
R2 = 0.0   # Zero reward for code that fails unit tests
R3 = 0.5   # Positive reward for code that passes unit tests but doesn't improve execution time
R4 = 1.0   # Higher positive reward for code that passes unit tests and improves execution time

# Placeholder functions

def compile_code(code):
    """
    Compile the code and return a random compilation result (True for success, False for failure).

    Args:
        code (str): The code to compile.

    Returns:
        bool: True if the compilation is successful, False otherwise.
    """
    return random.choice([True, False])

def pass_unit_tests(code):
    """
    Run unit tests and return a random test result (True for passing, False for failing).

    Args:
        code (str): The code to test.

    Returns:
        bool: True if the unit tests pass, False otherwise.
    """
    return random.choice([True, False])

def execution_time(code):
    """
    Return a random execution time value (in seconds).

    Args:
        code (str): The code to measure execution time.

    Returns:
        float: The execution time in seconds.
    """
    return random.uniform(0.1, 1.0)

def tokenize_code(code):
    """
    Tokenize the code using the tokenizer.

    Args:
        code (str): The code to tokenize.

    Returns:
        list: The tokenized code as a list of token IDs.
    """
    tokens = tokenizer.tokenize(code)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    return token_ids

def get_reward(lang, generated_code_ids, reference_code_ids, tokenizer):
    """
    Calculate the reward based on the generated code and reference code.

    Args:
        lang (str): The programming language.
        generated_code_ids (list): The token IDs of the generated code.
        reference_code_ids (list): The token IDs of the reference code.
        tokenizer (AutoTokenizer): The tokenizer.

    Returns:
        float: The reward value.
    """
    generated_code = tokenizer.decode(generated_code_ids, skip_special_tokens=True)
    reference_code = tokenizer.decode(reference_code_ids, skip_special_tokens=True)

    if not compile_code(generated_code):
        return R1
    elif not pass_unit_tests(generated_code):
        return R2
    elif pass_unit_tests(generated_code) and execution_time(generated_code) < execution_time(reference_code):
        return R4
    else:
        return R3

def generate_code(state, action):
    """
    Generate the next state of the code based on the current state and action.

    Args:
        state (list): The current state of the code as token IDs.
        action (int): The action to take (not used in this adapted function).

    Returns:
        tuple: A tuple containing the next state, reward, done flag, and an empty info dict.
    """
    # Assuming state is already in the correct format; otherwise, adjust accordingly
    encoding = tokenizer(" ".join([tokenizer.decode(token_id) for token_id in state]), return_tensors="pt").to(device)
    encoding['decoder_input_ids'] = encoding['input_ids'].clone()

    outputs = model.generate(**encoding, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    generated_code_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    next_state_token_ids = tokenizer.encode(generated_code_text, return_tensors="pt").tolist()[0]
    reference_code_token_ids = state  # Assuming the reference code is the input state

    reward = get_reward(lang, next_state_token_ids, reference_code_token_ids, tokenizer)

    if not compile_code(generated_code_text) or not pass_unit_tests(generated_code_text):
        done = True
    else:
        done = False

    return next_state_token_ids, reward, done, {}

def calculate_score(generated_code_ids, reference_code_ids):
    """
    Calculate the score of the generated code based on Equation 9.
    This score represents how likely the model is to generate that particular code sample.

    Args:
        generated_code_ids (list): The token IDs of the generated code.
        reference_code_ids (list): The token IDs of the reference code.

    Returns:
        float: The calculated score.
    """
    log_probs = []
    for token in generated_code_ids:
        log_prob = torch.log(agent.get_policy().model.forward(torch.FloatTensor(reference_code_ids))[token])
        log_probs.append(log_prob)
    score = sum(log_probs) / len(generated_code_ids)
    return score


def train(config, reporter):
    """
    Train the PPO agent.

    Args:
        config (dict): The configuration for the PPO agent.
        reporter (func): The reporter function for logging custom metrics.
    """
    episode_rewards = []
    episode_scores = []

    for episode in range(num_episodes):
        state = input_code_tokens
        done = False
        episode_reward = 0

        # Generate candidate code samples
        candidate_samples = []
        num_samples = 5
        for _ in range(num_samples):
            # Generate code using greedy or random sampling
            generated_code_ids, _, _, _ = generate_code(state, None)
            candidate_samples.append(generated_code_ids)

        # Calculate rewards and scores for candidate samples
        rewards = [get_reward(tokenizer.decode(sample), tokenizer.decode(state)) for sample in candidate_samples]
        scores = [calculate_score(sample, state) for sample in candidate_samples]

        # Compute ranking loss (L_rank)
        # encourages the model to assign higher scores to code samples with higher rewards
        # calculated by comparing the scores of samples with different rewards.
        ranking_loss = 0
        for i in range(len(candidate_samples)):
            for j in range(len(candidate_samples)):
                if rewards[i] < rewards[j]:
                    ranking_loss += max(0, scores[i] - scores[j])

        # Compute tuning loss (L_tuning)
        # maximize the likelihood of generating the best code sample(samples with the highest reward)
        # calculated as negative log probability of the tokens in the best sample
        best_sample_idx = np.argmax(rewards)
        best_sample = candidate_samples[best_sample_idx]
        tuning_loss = -sum(
            [torch.log(model(torch.tensor(state[:i]).unsqueeze(0))[0, token]) for i, token in enumerate(best_sample)])

        # Combine losses and update LLM model parameters
        loss = ranking_loss + tuning_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        episode_rewards.append(rewards[best_sample_idx])
        episode_scores.append(scores[best_sample_idx])

    # Plot the episode rewards
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_episodes + 1), episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Episode Rewards')
    plt.show()

    # Plot the episode scores
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_episodes + 1), episode_scores)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Episode Scores')
    plt.show()

    # Save the fine-tuned CodeT5 model checkpoint
    model_checkpoint_path = "../checkpoints"
    torch.save(model.state_dict(), model_checkpoint_path)
    print(f"CodeT5 model checkpoint saved at: {model_checkpoint_path}")

    agent.stop()

    agent.stop()

# Define the configuration
config = {
    "env": None,
    "framework": "torch",
    "num_gpus": 0,
    "num_workers": 1,
    "model": {
        "fcnet_hiddens": [256],
        "fcnet_activation": "relu",
    },
    "observation_space": obs_space,
    "action_space": action_space,  # Add this line
}

# Train the agent
num_episodes = 10
lang = 'python'  # Specify the programming language

# Input Data
input_code = '''
def hello_world():
    print("Hello, World!")
'''

input_code2 = '''
def calculate_average(numbers):
    total = sum(numbers)
    count = len(numbers)
    average = total / count
    return average
'''

input_code3 = '''
def find_max(numbers):
    if not numbers:
        return None
    max_num = numbers[0]
    for num in numbers:
        if num > max_num:
            max_num = num
    return max_num
'''

input_code_tokens = tokenize_code(input_code)
input_code_tokens2 = tokenize_code(input_code2)
input_code_tokens3 = tokenize_code(input_code3)

# Create the PPO agent
agent = PPO(config=config)

# Train the agent
train(config, reporter=None)

# Save the checkpoint
checkpoint_path = agent.save()
print(f"Checkpoint saved at: {checkpoint_path}")

# Terminate the agent
agent.stop()