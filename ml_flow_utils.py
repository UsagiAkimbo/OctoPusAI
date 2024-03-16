import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = 'your_mlflow_tracking_uri_here'  # Update with your MLflow tracking URI

# Set the MLflow tracking URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def start_mlflow_run(experiment_name='OctoPusAI_Experiments'):
    """
    Start an MLflow run in a given experiment.
    
    :param experiment_name: The name of the MLflow experiment.
    :return: The MLflow run context.
    """
    mlflow.set_experiment(experiment_name)
    return mlflow.start_run()

def log_model_details(run_id, model, model_key, training_info, dataset_name):
    """
    Log model details, including parameters, metrics, and the model itself to MLflow.

    :param run_id: The run ID of the MLflow experiment run.
    :param model: The trained model instance.
    :param model_key: The key identifying the model type (e.g., 'CNN', 'LSTM').
    :param training_info: A dictionary containing training metrics.
    :param dataset_name: The name of the dataset used for training.
    """
    with mlflow.start_run(run_id=run_id):
        mlflow.log_params({"model_type": model_key, "dataset": dataset_name})
        mlflow.log_metrics({"accuracy": training_info['accuracy']})
        mlflow.keras.log_model(model, "models/" + model_key)  # Adjust the artifact path as needed

def log_artifacts(run_id, artifact_paths):
    """
    Log additional artifacts (e.g., plots, data files) to the MLflow run.

    :param run_id: The run ID of the MLflow experiment run.
    :param artifact_paths: List of local paths to artifacts to log.
    """
    with mlflow.start_run(run_id=run_id):
        for artifact_path in artifact_paths:
            mlflow.log_artifact(artifact_path)
