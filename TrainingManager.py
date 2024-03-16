import tensorflow as tf
from tensorflow.keras.metrics import SparseCategoricalAccuracy, MeanSquaredError
from typing import Dict, List
from models import get_model
from data_preprocessing import load_and_preprocess_dataset
import mlflow
from ml_flow_utils import log_training_details, fetch_experiment_results
from db_utils import save_model_performance, fetch_model_performance
from communication import communication_bus
import asyncio
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import numpy as np

class TrainingManager:
    def __init__(self):
        self.training_sessions = {}
        communication_bus.register_listener("assign_training", self.assign_training)
        communication_bus.register_listener("cross_train_model", self.manage_cross_training)
        communication_bus.register_listener("training_completed", self.handle_training_completed)

    async def assign_training(self, data: Dict):
        model_type = data['model_type']
        dataset_name = data['dataset_name']
        # Load and preprocess dataset
        (x_train, y_train), (x_test, y_test) = await load_and_preprocess_dataset(dataset_name)
        # Dynamically create model
        model = get_model(model_type, input_shape=x_train.shape[1:], num_classes=len(y_train[0]))
        # Train model
        await self.train_model(model, x_train, y_train, x_test, y_test, model_type, dataset_name)

    async def train_model(self, model, x_train, y_train, x_test, y_test, model_type, dataset_name):
        # Define a general set of training parameters
        training_params = {
            'epochs': 10,
            'batch_size': 32,
            'validation_data': (x_test, y_test),
        }

        # Customize training parameters or logic based on model_type
        if model_type == 'CNN':
            # For CNNs, you might want to use a specific optimizer or adjust epochs
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        elif model_type == 'LSTM' or model_type == 'GRU':
            # LSTM and GRU might benefit from different settings
            model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
            training_params['epochs'] = 15
        elif model_type in ['BERT', 'Transformer', 'GPT']:
            # For transformer-based models, training logic might be significantly different
            # This might involve using a different training loop, for instance
            print(f"Training {model_type} requires a custom training loop.")
            await self.train_transformer_model(model_type, dataset_name, x_train, y_train, x_val, y_val)
            return

        # Execute the training
        history = model.fit(
            x_train, y_train,
            epochs=training_params['epochs'],
            batch_size=training_params['batch_size'],
            validation_data=training_params['validation_data']
        )

        # Log training with MLflow or another tool
        with mlflow.start_run():
            mlflow.log_params({"model_type": model_type, "dataset_name": dataset_name, "epochs": training_params['epochs']})
            mlflow.log_metrics({"accuracy": max(history.history['val_accuracy'])})
            mlflow.keras.log_model(model, "models")

        print(f"Model {model_type} trained on {dataset_name} dataset.")
        
    async def train_transformer_model(self, model_type, dataset_name, x_train, y_train, x_val, y_val):
        """
        A custom training loop for transformer models like BERT, Transformer, GPT.
        """
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_type)

        # Tokenize the input (this part highly depends on your dataset structure)
        train_encodings = tokenizer(x_train, truncation=True, padding=True)
        val_encodings = tokenizer(x_val, truncation=True, padding=True)

        # Convert to TensorFlow datasets (or PyTorch datasets, if using PyTorch)
        train_dataset = tf.data.Dataset.from_tensor_slices((
            dict(train_encodings),
            y_train
        ))
        val_dataset = tf.data.Dataset.from_tensor_slices((
            dict(val_encodings),
            y_val
        ))

        # Load model
        model = TFAutoModelForSequenceClassification.from_pretrained(model_type, num_labels=len(np.unique(y_train)))

        # Compile the model
        optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        # Train the model
        model.fit(train_dataset.shuffle(1000).batch(16), epochs=3, batch_size=16,
                  validation_data=val_dataset.batch(16))

        # Save the model
        save_path = f"./models/{model_type}"
        model.save_pretrained(save_path)

        # Log training details with MLflow or another tool
        # This part remains similar to your existing logging logic
        # Example:
        with mlflow.start_run():
            mlflow.log_params({"model_type": model_type, "dataset_name": dataset_name})
            mlflow.log_artifact(save_path, "models")

        print(f"Transformer model {model_type} trained on {dataset_name} dataset.")

    async def manage_cross_training(self, model_types: List[str], dataset_names: List[str]):
        # Logic to handle cross-training of models across different datasets
        for model_type in model_types:
            for dataset_name in dataset_names:
                (x_train, y_train), (x_test, y_test) = await load_and_preprocess_dataset(dataset_name)
                # Dynamically create model based on type
                model = get_model(model_type, input_shape=x_train.shape[1:], num_classes=len(y_train[0]))
                # Cross-train model
                print(f"Cross-training {model_type} model with {dataset_name} dataset.")
                await self.train_model(model, x_train, y_train, x_test, y_test, model_type, dataset_name)
                # Send a message about the cross-training completion
                await communication_bus.send_message("cross_training_completed", {"model_type": model_type, "dataset_name": dataset_name})
                
    async def handle_training_completed(self, data: Dict):
        """
        Handle post-training activities such as further processing or initiating cross-training.
        """
        model_type = data['model_type']
        dataset_name = data['dataset_name']
        print(f"Post-training activities for {model_type} on {dataset_name}.")

        # Fetch experiment results from MLflow
        experiment_results = fetch_experiment_results(model_type, dataset_name)

        # Evaluate model performance across different datasets
        for dataset in ['dataset1', 'dataset2']:
            if dataset != dataset_name:
                (x_train, y_train), (x_test, y_test) = await load_and_preprocess_dataset(dataset)
                model = get_model(model_type)
                performance = self.evaluate_model(model, x_test, y_test)
                save_model_performance(model_type, dataset, performance)
                print(f"Model {model_type} evaluated on {dataset} with performance {performance}.")

        # Example: Optimizing hyperparameters based on performance
        best_performance = fetch_model_performance(model_type)
        if best_performance['accuracy'] < threshold:
            print("Optimizing hyperparameters for better performance.")
            # Placeholder for hyperparameter optimization logic

        # Notify the system or user about the post-training completion
        await communication_bus.send_message("post_training_completed", {"model_type": model_type, "dataset_name": dataset_name, "status": "success", "action": "evaluation and optimization"})
        
    async def evaluate_model(self, model, model_type, x_test, y_test):
        """
        Dynamically evaluates the model on the test dataset and returns performance metrics
        based on the model type.
        """
        if model_type in ['CNN', 'ResNet', 'EfficientNet']:
            # For classification models, calculate accuracy
            metric = SparseCategoricalAccuracy()
            metric.update_state(y_test, model.predict(x_test))
            accuracy = metric.result().numpy()
            return {"accuracy": accuracy}

        elif model_type in ['LSTM', 'GRU', 'BERT']:
            # Assuming binary classification for simplicity; adjust as needed for your project
            predictions = model.predict(x_test)
            predictions = np.round(predictions).astype(int)
            accuracy = np.mean(predictions == y_test)
            return {"accuracy": accuracy}

        elif model_type == 'DNN':
            # For regression models like DNN on Boston Housing, calculate mean squared error
            metric = MeanSquaredError()
            metric.update_state(y_test, model.predict(x_test))
            mse = metric.result().numpy()
            return {"mse": mse}

        elif model_type in ['Random Forest', 'XGBoost']:
            # Placeholder for non-Keras models; requires custom evaluation logic
            # You might need to use Scikit-learn metrics here
            return {"custom_metric": 0.0}

        else:
            raise ValueError(f"Unsupported model type for evaluation: {model_type}")

    # Additional methods as needed for the TrainingManager's functionality
