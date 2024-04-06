from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration
import torch
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-large")
model = AutoModelForSeq2SeqLM.from_pretrained('Salesforce/codet5-large', torch_dtype=torch.float16,
                                              low_cpu_mem_usage=True, trust_remote_code=True)

#model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-large")
text = "covert from one currency to another"
input_ids = tokenizer(text, return_tensors="pt").input_ids

# simply generate a single sequence
generated_ids = model.generate(input_ids, max_length=8)
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))