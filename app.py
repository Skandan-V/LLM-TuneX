import streamlit as st
import json
import os
import requests
from datetime import datetime
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer

# Define constants
MODEL_FILE = "models.json"

# Load or initialize model list
def load_models():
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "r") as f:
            return json.load(f)
    return []

def save_models(models):
    with open(MODEL_FILE, "w") as f:
        json.dump(models, f, indent=4)

# Fine-tune the model
def fine_tune_model(model_name, dataset_url, output_dir):
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load dataset
    # Assuming the dataset is in a JSONL format
    dataset = requests.get(dataset_url).json()  # Adjust as needed based on the dataset structure
    # Process the dataset to fit the required format for the Trainer

    # Create TrainingArguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        num_train_epochs=3,
        save_steps=10_000,
        save_total_limit=2,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,  # This should be a Dataset object, adapt as needed
    )

    # Train model
    trainer.train()

    return model_name

# Streamlit UI
st.title("LLama Model Fine-Tuning on Together AI")

# Input fields
model_name = st.text_input("Enter the base model name (e.g., 'huggingface/llama-7b'):")
dataset_url = st.text_input("Enter the dataset URL:")
output_dir = st.text_input("Output directory for the fine-tuned model:", value="fine_tuned_model")

if st.button("Fine-Tune Model"):
    if model_name and dataset_url:
        st.write("Fine-tuning the model...")
        try:
            fine_tuned_model = fine_tune_model(model_name, dataset_url, output_dir)
            st.success(f"Model {fine_tuned_model} fine-tuned successfully.")
            models = load_models()
            models.append({
                "model_name": fine_tuned_model,
                "date": datetime.now().isoformat(),
                "output_dir": output_dir
            })
            save_models(models)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.error("Please provide both the model name and dataset URL.")

# Display fine-tuned models
st.subheader("Fine-Tuned Models")
models = load_models()
if models:
    for m in models:
        st.write(f"Model: {m['model_name']} | Date: {m['date']} | Output Directory: {m['output_dir']}")
else:
    st.write("No models have been fine-tuned yet.")
