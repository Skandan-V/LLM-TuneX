

# LLM-TuneX

LLM-TuneX is a streamlined tool designed for fine-tuning large language models on Together AI’s infrastructure. The application offers an easy-to-use interface for uploading datasets, training models, and managing fine-tuned versions of models with built-in reuse capabilities.

## Features
- **Dataset Upload**: Supports uploading datasets in `.jsonl` format via URL or local file upload.
- **Automated Model Training**: Fine-tunes models automatically using Together AI's API with adjustable hyperparameters.
- **Model Management**: View and select from a list of previously trained models for continued training.
- **Customizable Training Parameters**: Allows users to modify parameters like epochs, batch size, and learning rate.
- **User-Friendly UI**: Interactive interface with status updates, logs, and usage instructions.

## Installation

Clone the repository and install the dependencies:

- git clone https://github.com/Skandan-V/LLM-TuneX.git
- cd LLM-TuneX
- pip install -r requirements.txt 

## Usage

To launch the application:
```
streamlit run app.py
```

### 1. Interface Overview
| Section                  | Description                                                                                  |
|--------------------------|----------------------------------------------------------------------------------------------|
| **Dataset Upload**       | Allows dataset uploads in `.jsonl` format from a URL or local file.                           |
| **Model Training**       | Configures hyperparameters and initiates model training with status and progress updates.     |
| **Model Management**     | Displays previously trained models for selection and further training.                       |

### 2. Dataset Upload
You can upload datasets from:
- **URL**: Paste a dataset URL in `.jsonl` format.
- **Local File**: Choose a `.jsonl` file from your local system.

### 3. Training a Model
1. After uploading a dataset, select model parameters:
    - **Learning Rate**: Set the model’s learning rate for optimal convergence.
    - **Epochs**: Number of times the model will iterate over the dataset.
    - **Batch Size**: Size of the dataset batches for each training step.
2. Click on **Start Training**. Training logs and model progress will be displayed in real-time.

### 4. Model Management
The app saves all trained models and displays them in a dedicated section for easy selection. Users can re-train an existing model by selecting it from the list.

## Configuration

| Parameter      | Description                                                                 | Default   |
|----------------|-----------------------------------------------------------------------------|-----------|
| **Learning Rate** | Adjusts the speed of model convergence.                                   | `0.001`   |
| **Epochs**        | Controls how many complete passes the model makes over the dataset.      | `3`       |
| **Batch Size**    | Determines the number of samples in each training batch.                 | `8`       |

## Together AI Configuration

Update your Together AI credentials in the `.env` file:
```env
API_KEY="your_api_key_here"
MODEL_ID="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
```

## Example Workflow
1. **Upload Dataset**: Use a `.jsonl` dataset URL.
2. **Select Model**: Choose a model or previously trained version from the list.
3. **Fine-tune Model**: Adjust parameters and start training.
4. **Monitor Training**: View logs and status in real-time.

## Sample Code
```python
# Example of training setup using Together AI API
response = requests.post(
    "https://api.together.ai/v1/model/train",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={
        "model_id": MODEL_ID,
        "dataset_url": dataset_url,
        "hyperparameters": {
            "learning_rate": 0.001,
            "epochs": 3,
            "batch_size": 8
        }
    }
)
print(response.json())
```

## Contributing

Feel free to contribute to LLM-TuneX by submitting pull requests. For major changes, please open an issue first to discuss potential modifications.

## License

This project is licensed under the MIT License.
```

