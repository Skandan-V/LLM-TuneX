import json
import os
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer

# Load model and tokenizer
model_id = "codellama/CodeLlama-34b-Instruct-hf"  # Change this as needed
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Prepare the dataset
def load_dataset(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def tokenize_function(examples):
    return tokenizer(examples["prompt"], padding="max_length", truncation=True, max_length=512)

# Load train and validation datasets
train_data = load_dataset('./data/train_data.json')
valid_data = load_dataset('./data/valid_data.json')

# Tokenize datasets
train_encodings = tokenize_function(train_data)
valid_encodings = tokenize_function(valid_data)

# Create a dataset class
import torch

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

# Create dataset objects
train_dataset = CustomDataset(train_encodings)
valid_dataset = CustomDataset(valid_encodings)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained('./model')
tokenizer.save_pretrained('./model')

print("Training complete and model saved!")
