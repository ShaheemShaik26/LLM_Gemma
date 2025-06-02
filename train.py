import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType

# Load Dataset
dataset = load_dataset("json", data_files="data/dsa_dbms_dataset.jsonl", split="train")

# Format the dataset for instruction tuning
def format_prompt(example):
    return {
        "text": f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    }

dataset = dataset.map(format_prompt)

# Load model + tokenizer
model_id = "google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", load_in_4bit=True)

# Tokenize the dataset
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize, batched=True)

# Apply LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    task_type=TaskType.CAUSAL_LM,
    lora_dropout=0.1,
    bias="none"
)
model = get_peft_model(model, lora_config)

# Training setup
training_args = TrainingArguments(
    output_dir="./finetuned_gemma",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    report_to="none"
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
model.save_pretrained("finetuned_gemma")
tokenizer.save_pretrained("finetuned_gemma")
