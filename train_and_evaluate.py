import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
import mlflow
from models import get_model
from data_preprocessing import load_and_preprocess_dataset
from explainability import generate_shap_explanations
import config  # Ensure this contains necessary configurations
from db_utils import save_shap_values
from explainability import generate_shap_explanations

# Ensure MLflow is set up correctly
mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)

def train_model(model, x_train, y_train, x_val, y_val, model_key, dataset_name):
    """
    Trains the model, logs the training process with MLflow, and returns the trained model.
    """
    callbacks = [EarlyStopping(monitor='val_loss', patience=3)]
    with mlflow.start_run():
        history = model.fit(x_train, y_train, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE,
                            validation_data=(x_val, y_val), callbacks=callbacks)
        mlflow.log_params({"model_type": model_key, "dataset": dataset_name})
        mlflow.log_metrics({"accuracy": max(history.history['val_accuracy'])})
        mlflow.keras.log_model(model, model_key)
    return model, history

async def train_and_evaluate(model_type, dataset_name):
    """
    Asynchronous function to load and preprocess dataset, train the model, and generate SHAP explanations.
    """
    try:
        (x_train, y_train), (x_test, y_test) = await load_and_preprocess_dataset(dataset_name)
    except Exception as e:
        print(f"Error loading or preprocessing dataset {dataset_name}: {e}")
        raise
    
    # Dynamically create the model based on the type
    model = get_model(model_type, input_shape=(224, 224, 3), num_classes=10)  # Adjust parameters as needed
        
    if model is None:
        print(f"Unsupported model type: {model_type}")
        raise ValueError(f"Unsupported model type: {model_type}")

    x_val, y_val = x_test[:len(x_test)//2], y_test[:len(y_test)//2]
    x_test, y_test = x_test[len(x_test)//2:], y_test[len(y_test)//2:]

    model, _ = train_model(model, x_train, y_train, x_val, y_val, model_type, dataset_name)

    # SHAP explanations
    sample_indices = np.random.choice(x_test.shape[0], 100, replace=False)
    data_sample = x_test[sample_indices]
    generate_shap_explanations(model, data_sample, model_type)
    shap_values = generate_shap_explanations(model, data_sample, model_type)
    save_shap_values(model_type, dataset_name, shap_values)
    

    # Optionally: Evaluate the model on the test set and log results
    test_scores = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test loss: {test_scores[0]}, Test accuracy: {test_scores[1]}")
    
    return model

