import streamlit as st
import json
import os
import requests
from datetime import datetime
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
import pandas as pd

# Define constants
MODEL_FILE = "models.json"
API_URL = "https://api.together.ai/v1/models"  # Example API URL

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
    response = requests.get(dataset_url)
    data = response.text.splitlines()
    dataset = [json.loads(line) for line in data]  # Adjust as needed based on the dataset structure

    # Create a DataFrame from the dataset
    df = pd.DataFrame(dataset)

    # Convert DataFrame to Dataset object compatible with Trainer
    train_dataset = df[['input', 'output']]  # Adjust columns based on your dataset

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
        train_dataset=train_dataset,  # This should be a Dataset object, adapt as needed
    )

    # Train model
    trainer.train()

    return model_name

# Check dataset format
def verify_dataset_structure(dataset):
    required_keys = {"input", "output"}  # Adjust based on your dataset's structure
    return all(required_keys.issubset(set(item.keys())) for item in dataset)

# Fetch available models from Together AI
def fetch_available_models(api_type):
    headers = {
        'Authorization': f'Bearer {api_type}'
    }
    response = requests.get(API_URL, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Failed to fetch models from the API. Please check your API key.")

# Streamlit UI
st.title("LLama Model Fine-Tuning on Together AI")

# Input fields for API connection
api_type = st.text_input("Enter the Together AI API key (type='password'):")

if st.button("Connect to API"):
    available_models = fetch_available_models(api_type)
    if available_models:
        st.session_state.models = available_models
        st.success("Connected to the API successfully!")
    else:
        st.error("No models available or connection failed.")

# Model selection
if "models" in st.session_state:
    model_name = st.selectbox("Select a model:", [model['name'] for model in st.session_state.models])

    # Dataset input
    dataset_file = st.file_uploader("Upload your dataset in JSONL format:", type=["jsonl"])
    dataset_url = st.text_input("Or enter the dataset URL:")

    if dataset_file:
        dataset_content = dataset_file.getvalue().decode("utf-8")
        dataset = [json.loads(line) for line in dataset_content.splitlines()]
        if verify_dataset_structure(dataset):
            output_dir = st.text_input("Output directory for the fine-tuned model:", value="fine_tuned_model")

            if st.button("Fine-Tune Model"):
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
            st.error("Dataset structure is invalid. Ensure it contains 'input' and 'output' fields.")
    elif dataset_url:
        response = requests.get(dataset_url)
        data = response.text.splitlines()
        dataset = [json.loads(line) for line in data]
        if verify_dataset_structure(dataset):
            output_dir = st.text_input("Output directory for the fine-tuned model:", value="fine_tuned_model")

            if st.button("Fine-Tune Model"):
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
            st.error("Dataset structure is invalid. Ensure it contains 'input' and 'output' fields.")
    
    # Display fine-tuned models
    st.subheader("Fine-Tuned Models")
    models = load_models()
    if models:
        for m in models:
            st.write(f"Model: {m['model_name']} | Date: {m['date']} | Output Directory: {m['output_dir']}")
    else:
        st.write("No models have been fine-tuned yet.")
