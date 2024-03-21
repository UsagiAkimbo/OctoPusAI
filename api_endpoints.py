from fastapi import APIRouter, HTTPException, Depends, File, UploadFile, FastAPI, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.security import APIKeyHeader, APIKey
from typing import Dict
from typing import Optional, List
from models import create_cnn_model, create_lstm_model, create_gru_model, create_rnn_model
from data_preprocessing import load_and_preprocess_dataset, preprocess_image_for_cnn
from tensorflow.keras.models import load_model
import numpy as np
import uvicorn
from security import get_api_key
from mlflow.tracking import MlflowClient
import openai
import config
from config import dataset_keywords, training_parameters_map, dataset_nn_mapping
from models import create_cnn_model, create_lstm_model, create_gru_model, train_rnn_model, train_cnn_model, train_lstm_model, train_gru_model, train_rnn_model
from train_and_evaluate import train_and_evaluate
from db_utils import get_shap_values
from explainability import generate_shap_explanations
from ml_flow_utils import log_model_details
from SystemMonitor import system_monitor

router = APIRouter()
app = FastAPI()

# Setup API key security
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=True)

# Serve static files, including SHAP plots
app.mount("/shap_plots", StaticFiles(directory="shap_plots"), name="shap_plots")

async def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key == config.API_KEY:
        return api_key
    else:
        raise HTTPException(status_code=403, detail="Could not validate credentials")

@router.post("/analyze-prompt/", dependencies=[Depends(get_api_key)])
async def analyze_prompt(prompt: str):
    """
    Endpoint to analyze the user's prompt and suggest neural network types, datasets, and parameters.
    """
    # This is a placeholder function. Replace it with your actual logic to analyze the prompt.
    analysis = {"analysis": "Your analysis here based on the prompt", "recommended_datasets": ["mnist", "cifar10"]}
    return analysis

@router.post("/train-model/", dependencies=[Depends(get_api_key)])
async def train_model(dataset_name: str, model_type: str):
    """
    Endpoint to train a model based on the dataset name and model type.
    """
    try:
        if model_type not in ['CNN', 'LSTM', 'GRU', 'RNN']:
            raise HTTPException(status_code=400, detail=f"Unsupported model type: {model_type}")

        # Load and preprocess dataset
        (x_train, y_train), (x_test, y_test) = await load_and_preprocess_dataset(dataset_name)

        # Select model based on type
        if model_type == 'CNN':
            model = create_cnn_model()
        elif model_type == 'LSTM':
            model = create_lstm_model()
        elif model_type == 'GRU':
            model = create_gru_model()
        else:  # RNN
            model = create_rnn_model()

        # Train the model
        model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

        # Log model details to MLflow
        client = MlflowClient()
        run = client.create_run(experiment_id="1")  # Assume default experiment ID
        client.log_param(run.info.run_id, "model_type", model_type)
        client.log_param(run.info.run_id, "dataset_name", dataset_name)
        # More MLflow logging can be added here

        return {"message": f"Model of type {model_type} trained successfully on {dataset_name}."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload-image/", dependencies=[Depends(get_api_key)])
async def analyze_image(file: UploadFile = File(...), model_path: str = Depends(get_api_key)):
    """
    Endpoint to analyze an uploaded image using a pre-trained model.
    """
    try:
        image = preprocess_image_for_cnn(await file.read())
        model = load_model(model_path)  # Load your pre-trained model
        predictions = model.predict(image)
        # Process predictions to return a meaningful response
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/train-evaluate/")
async def train_evaluate_endpoint(background_tasks: BackgroundTasks, model_type: str, dataset_name: str):
    if model_type not in ['CNN', 'LSTM']:  # Extend this list based on your models
        raise HTTPException(status_code=400, detail=f"Unsupported model type: {model_type}")
    background_tasks.add_task(train_and_evaluate, model_type, dataset_name)
    return {"message": f"Training and evaluation for {model_type} on {dataset_name} initiated in the background."}

@app.post("/generate-explanations/")
async def generate_explanations(model_type: str, dataset_name: str):
    # Your logic to call generate_shap_explanations
    # E.g., generate_shap_explanations(model, data_sample, model_type, "shap_plots/plot.png")
    return {"message": "SHAP explanation generated.", "plot_url": "/shap_plots/plot.png"}

@router.get("/shap-values/{model_type}/{dataset_name}")
async def get_shap_values_endpoint(model_type: str, dataset_name: str):
    shap_values = get_shap_values(model_type, dataset_name)
    if shap_values:
        return shap_values
    return {"error": "SHAP values not found for the specified model and dataset."}

@router.post("/train")
async def train_model(model_type: str, dataset_name: str, background_tasks: BackgroundTasks):
    """
    Endpoint to initiate model training.
    Model type and dataset name are required parameters.
    """
    # Validate model type and dataset name if necessary
    if model_type not in ["CNN", "LSTM", "GRU"]:
        raise HTTPException(status_code=400, detail="Unsupported model type")
    
    # Add model training task to background to not block API
    background_tasks.add_task(train_and_evaluate, model_type, dataset_name)
    return {"message": f"Training task for {model_type} on {dataset_name} dataset started"}

@router.get("/evaluate/{model_type}")
async def evaluate_model(model_type: str, dataset_name: Optional[str] = None):
    """
    Endpoint to evaluate a trained model.
    Model type is required and dataset name is optional.
    """
    # This is a placeholder for your model evaluation logic
    # For now, we'll just return a mock response
    return {"model_type": model_type, "dataset_name": dataset_name, "accuracy": "mock_accuracy"}

@router.get("/shap-explanation/{model_type}")
async def get_shap_explanation(model_type: str, sample_index: int):
    """
    Endpoint to fetch SHAP explanation visualization for a given model type and sample index.
    """
    # This assumes generate_shap_explanations function exists and returns a path or URL to the visualization
    try:
        explanation_path = generate_shap_explanations(model_type, sample_index)
        return {"model_type": model_type, "explanation_path": explanation_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/initialize-brain/")
async def api_initialize_brain(brain_type: str, api_key: str = Depends(get_api_key)):
    # Assume validation of API key is done within get_api_key
    new_brain = await task_manager.initialize_brain(brain_type)
    if new_brain:
        return {"message": "Brain initialized successfully."}
    else:
        return {"error": "Failed to initialize brain."}
    
@router.get("/tasks")
async def get_tasks():
    system_monitor.log_activity("Fetching tasks")
    # Fetch tasks logic
    return {"tasks": []}

@router.post("/tasks")
async def create_task(task: Task):
    try:
        # Create task logic
        system_monitor.log_activity(f"Task created: {task.description}")
    except Exception as e:
        system_monitor.log_error(f"Error creating task: {e}")
        raise HTTPException(status_code=400, detail="Error creating task")
    
@router.post("/analyze-prompt/")
@router.post("/analyze-prompt/")
async def analyze_user_prompt_with_openai(prompt: str, api_key: str = Depends(get_api_key)):
    """
    Analyze the user's prompt to suggest neural network types, datasets, and parameters based on OpenAI's response.
    """
    try:
        openai.api_key = config.OPENAI_API_KEY
        response = openai.Completion.create(
            engine="davinci",
            prompt=f"Given the following task: '{prompt}'. What types of neural networks are best suited for analyzing this request, and what available datasets can be used for training those neural networks? Suggest any specific parameters or hyperparameters.",
            max_tokens=100,
            n=1,
            stop=None,
            temperature=0.5
        )
        analysis_text = response.choices[0].text.strip()

        recommended_datasets = set()
        recommended_model_types = set()
        parameters = {}

        # Parse datasets and models from the analysis using dataset_nn_mapping
        for keyword, dataset_info in dataset_keywords.items():
            if keyword in analysis_text.lower():
                for dataset in dataset_info:
                    recommended_datasets.add(dataset)
                    recommended_model_types.update(dataset_nn_mapping.get(dataset, []))
        
        # Parse for training parameters - using training_parameters_map
        for parameter, value in training_parameters_map.items():
            if parameter in analysis_text.lower():
                parameters[parameter] = value

        return {
            "analysis": analysis_text,
            "recommended_datasets": list(recommended_datasets),
            "recommended_model_types": list(recommended_model_types),
            "parameters": parameters
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Additional endpoints can be added here following the same pattern
