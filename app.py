import streamlit as st
import json
import os
from datetime import datetime
from some_ai_library import fine_tune_model  # Replace this with your actual fine-tuning function

# File to store fine-tuned model information
MODEL_FILE = "models.json"

# Load fine-tuned models from JSON file
def load_models():
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "r") as f:
            return json.load(f)
    return []

# Save fine-tuned models to JSON file
def save_models(models):
    with open(MODEL_FILE, "w") as f:
        json.dump(models, f, indent=4)

# Fine-tune model and save it
def fine_tune_and_save(model_id, dataset_url, additional_params):
    fine_tune_id = fine_tune_model(model_id=model_id, dataset=dataset_url, **additional_params)  # Placeholder for actual fine-tune function
    models = load_models()
    models.append({
        "fine_tune_id": fine_tune_id,
        "base_model_id": model_id,
        "dataset_url": dataset_url,
        "params": additional_params,
        "timestamp": datetime.now().isoformat()
    })
    save_models(models)
    return fine_tune_id

# Streamlit UI
st.title("Automated Fine-Tuning Model Manager")

# Load the fine-tuned models list
models = load_models()

# Section to start fine-tuning a new model
st.header("Fine-Tune a New Model")
model_id = st.text_input("Enter Model ID")
dataset_url = st.text_input("Dataset URL")
learning_rate = st.number_input("Learning Rate", min_value=1e-6, max_value=1e-1, value=1e-3)
epochs = st.number_input("Epochs", min_value=1, max_value=100, value=5)

if st.button("Start Fine-Tuning"):
    if model_id and dataset_url:
        fine_tune_id = fine_tune_and_save(
            model_id,
            dataset_url,
            {"learning_rate": learning_rate, "epochs": epochs}
        )
        st.success(f"Started fine-tuning. Fine-tune ID: {fine_tune_id}")
    else:
        st.error("Please enter both model ID and dataset URL.")

# Display previously fine-tuned models
st.header("Previously Fine-Tuned Models")
if models:
    for idx, model in enumerate(models):
        st.subheader(f"Model {idx + 1}")
        st.write(f"**Fine-Tune ID**: {model['fine_tune_id']}")
        st.write(f"**Base Model ID**: {model['base_model_id']}")
        st.write(f"**Dataset URL**: {model['dataset_url']}")
        st.write(f"**Parameters**: {model['params']}")
        st.write(f"**Timestamp**: {model['timestamp']}")
        
        # Continue training
        if st.button(f"Continue Training {model['fine_tune_id']}"):
            fine_tune_id = fine_tune_and_save(
                model["fine_tune_id"],
                model["dataset_url"],
                model["params"]
            )
            st.success(f"Continued training. New fine-tune ID: {fine_tune_id}")

        # Delete model
        if st.button(f"Delete Model {model['fine_tune_id']}"):
            models.pop(idx)
            save_models(models)
            st.success(f"Deleted model {model['fine_tune_id']}")
else:
    st.write("No fine-tuned models found.")

