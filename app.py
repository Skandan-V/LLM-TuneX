import streamlit as st
import requests
import json
import os

# Constants
MODEL_FILE = "models.json"

# Function to connect to the Together AI API and retrieve available models
def get_models(api_key):
    response = requests.get('https://api.together.ai/models', headers={'Authorization': f'Bearer {api_key}'})
    if response.status_code == 200:
        return response.json()  # Assuming the response is JSON
    else:
        st.error("Failed to retrieve models.")
        return []

# Function to validate the JSONL dataset structure
def validate_dataset(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            try:
                json_line = json.loads(line)
                # Check if the required keys are in the dataset
                if not all(key in json_line for key in ["prompt", "completion"]):
                    return False
            except json.JSONDecodeError:
                return False
    return True

# Function to fine-tune the model (Placeholder for actual fine-tuning logic)
def fine_tune_model(model_name, dataset_url):
    st.success(f"Fine-tuning model '{model_name}' with dataset from {dataset_url}...")
    # Here you'd add the actual fine-tuning logic using the API

# Streamlit UI
st.title("LLM TuneX - Fine-tune Your LLM")

# API Key input
api_key = st.text_input("Enter your Together AI API key:", type="password")
if st.button("Connect"):
    st.session_state.models = get_models(api_key)
    if st.session_state.models:
        st.success("Connected to the API successfully!")
        
        # Display available models
        model_names = [model.get('name', 'Unknown Model') for model in st.session_state.models]
        model_name = st.selectbox("Select a model:", model_names)

        # Dataset URL or file upload
        dataset_option = st.radio("Choose dataset input method:", ("Enter URL", "Upload File"))

        if dataset_option == "Enter URL":
            dataset_url = st.text_input("Enter the dataset URL:")
        else:
            uploaded_file = st.file_uploader("Upload your JSONL file:", type=["jsonl"])
            if uploaded_file is not None:
                # Save uploaded file temporarily
                with open("temp_dataset.jsonl", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                dataset_url = "temp_dataset.jsonl"

        if st.button("Start Training"):
            # Verify dataset input
            if dataset_option == "Enter URL":
                if dataset_url:
                    st.info(f"Starting training with dataset from: {dataset_url}")
                    fine_tune_model(model_name, dataset_url)
                else:
                    st.error("Please provide a valid dataset URL.")
            else:
                # Validate uploaded file
                if validate_dataset("temp_dataset.jsonl"):
                    st.info(f"Starting training with uploaded dataset.")
                    fine_tune_model(model_name, "temp_dataset.jsonl")
                else:
                    st.error("Uploaded file is not in the correct format. Please ensure it is a valid JSONL file with 'prompt' and 'completion' keys.")

# Additional options for listing tuned models, etc., can be added here
