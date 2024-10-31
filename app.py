import streamlit as st
import requests
import pandas as pd
import json
import os
from transformers import TrainingArguments

# Constants
MODEL_FILE = "models.json"
API_URL = "https://api.together.ai/v1/models"  # Example API URL

# Function to connect to the Together AI API and retrieve available models
def get_models(api_key):
    try:
        response = requests.get(API_URL, headers={'Authorization': f'Bearer {api_key}'})
        response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)

        # Debug: print the raw response text
        st.write("Raw API Response:", response.text)  # Show the raw response for debugging

        models_data = response.json()  # Assuming the response is JSON
        return models_data  # Return the entire model data for inspection

    except requests.exceptions.HTTPError as err:
        st.error(f"HTTP error occurred: {err}")
        return []
    except requests.exceptions.RequestException as err:
        st.error(f"Error occurred while connecting to the API: {err}")
        return []
    except json.JSONDecodeError:
        st.error("Failed to decode JSON response from the API.")
        return []

# Function to validate dataset format
def validate_dataset(df):
    required_columns = ['input', 'output']  # Adjust based on your model's requirements
    for column in required_columns:
        if column not in df.columns:
            return False, f"Missing required column: {column}"
    return True, ""

# Streamlit UI
st.title("LLM TuneX - Fine-tune Your LLM")

# API Key input
api_key = st.text_input("Enter your Together AI API key:", type="password")
if st.button("Connect"):
    st.session_state.models = get_models(api_key)
    if st.session_state.models:
        st.success("Connected to the API successfully!")

        # Debug: Check the structure of the models data
        st.write("Models Data Structure:", st.session_state.models)

        # Extract model display names
        model_names = [model.get('display_name', 'Unknown Model') for model in st.session_state.models]
        
        # Check for available models
        if not model_names:
            st.error("No models found in the response.")
        else:
            model_name = st.selectbox("Select a model:", model_names)

            # Dataset input
            dataset_file = st.file_uploader("Upload your dataset (JSONL format):", type=["jsonl"])
            if dataset_file is not None:
                df = pd.read_json(dataset_file, lines=True)

                # Validate dataset format
                is_valid, error_message = validate_dataset(df)
                if not is_valid:
                    st.error(error_message)
                else:
                    st.success("Dataset format is valid.")

                    # Start training process (this is just an example; you should adjust based on your needs)
                    output_dir = "output"  # Define your output directory
                    training_args = TrainingArguments(
                        output_dir=output_dir,
                        per_device_train_batch_size=4,
                        num_train_epochs=3,
                    )

                    # Assuming further code for the training process here...
                    st.success("Training started with selected model and dataset.")
