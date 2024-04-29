import json
from transformers import (
    PreTrainedTokenizerFast,
    LlamaForCausalLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

# Load the dataset
with open("examples/test_python.json", "r") as f:
    dataset = json.load(f)

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

# Initialize the tokenizer and model
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)


# Tokenize the dataset
def tokenize(examples):
    inputs = [
        f"Optimize the following code: {example['slow_code']}" for example in examples
    ]
    targets = [example["fast_code"] for example in examples]
    model_inputs = tokenizer(
        inputs, max_length=1024, truncation=True, return_tensors="pt"
    )
    labels = tokenizer(
        targets, max_length=1024, truncation=True, return_tensors="pt"
    ).input_ids

    model_inputs["labels"] = labels
    return model_inputs


tokenized_dataset = tokenize(dataset)

# Split the dataset into train and validation sets
train_dataset = tokenized_dataset[: int(0.8 * len(tokenized_dataset))]
val_dataset = tokenized_dataset[int(0.8 * len(tokenized_dataset)) :]

# Set up the fine-tuning arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="path/to/output_dir",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=5,
    predict_with_generate=True,
)

# Set up the trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

# Fine-tune the model
trainer.train()

# Evaluate the model
metrics = trainer.evaluate()
print(f"Evaluation metrics: {metrics}")

# Use the fine-tuned model for code optimization
slow_code = "Inefficient code sample"
input_ids = tokenizer(
    f"Optimize the following code: {slow_code}", return_tensors="pt"
).input_ids
output_ids = model.generate(input_ids, max_length=1024, early_stopping=True)
optimized_code = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f"Optimized code: {optimized_code}")
