from dotenv import load_dotenv
import os
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

load_dotenv()

# Fetch secret from Azure Key Vault
def get_secret_from_key_vault(secret_name):
    key_vault_url = os.getenv("AZURE_KEY_VAULT_URL")
    credential = DefaultAzureCredential()
    client = SecretClient(vault_url=key_vault_url, credential=credential)
    retrieved_secret = client.get_secret(secret_name)
    return retrieved_secret.value

training_parameters_map = {
    # Common Training Parameters
    "dataset": ["dataset", "data collection", "training set"],
    "batch size": ["batch size", "mini-batch", "batch"],
    "epochs": ["epochs", "cycles", "iterations"],

    # Common Hyperparameters
    "learning rate": ["learning rate", "step size"],
    "activation function": ["activation function", "ReLU", "sigmoid", "tanh"],
    "optimizer": ["optimizer", "Adam", "SGD", "RMSprop"],

    # CNN-Specific Hyperparameters
    "filter size": ["filter size", "kernel size"],
    "number of filters": ["number of filters", "feature detectors"],
    "pooling size": ["pooling size", "max pooling", "average pooling"],

    # RNN/LSTM/GRU-Specific Hyperparameters
    "sequence length": ["sequence length", "time steps"],
    "hidden units": ["hidden units", "LSTM units", "GRU cells"],

    # Model Architecture and Design
    "number of layers": ["layers", "depth"],
    "dropout rate": ["dropout rate", "dropout"],
    "regularization": ["regularization", "L1", "L2"],
    
    # Random Forest-Specific Parameters & Hyperparameters
    "number of trees": ["number of trees", "estimators", "n_estimators"],
    "max depth": ["max depth", "tree depth", "depth limit"],
    "min samples split": ["min samples split", "minimum samples for split"],
    "min samples leaf": ["min samples leaf", "minimum samples in leaf"],

    # XGBoost-Specific Parameters & Hyperparameters
    "booster": ["booster", "tree booster", "linear booster"],
    "learning rate": ["learning rate", "eta"],
    "max depth xgb": ["max depth", "maximum depth of trees"],
    "subsample": ["subsample", "sample of training instance"],
    "colsample bytree": ["colsample bytree", "subsample ratio of columns"],
    "lambda": ["lambda", "L2 regularization term on weights"],
    "alpha": ["alpha", "L1 regularization term on weights"],

    # Regression-Specific Parameters & Hyperparameters
    "loss function": ["loss function", "MSE", "MAE", "Huber loss"],
    "output activation": ["output activation", "linear"],
    "normalization": ["feature normalization", "data scaling"],
}

dataset_keywords = {
    # Extended Dataset to Neural Network Mappings
    'mnist': ["digits", "handwritten digits", "MNIST"],
    'fashion_mnist': ["fashion", "clothing", "apparel", "Fashion MNIST"],
    'cifar10': ["objects", "animals", "vehicles", "CIFAR-10"],
    'cifar100': ["fine-grained objects", "100 classes", "CIFAR-100"],
    'imdb': ["movie reviews", "sentiment analysis", "text", "IMDB"],
    'boston_housing': ["housing", "prices", "regression", "Boston Housing"],
    'coarse_grained_cifar100': ["superclass", "coarse CIFAR-100", "superclass labels"],
    'caltech101': ["images", "objects", "Caltech 101", "categories"],
    'caltech256': ["more images", "more objects", "Caltech 256", "more categories"],
    'celeba': ["celebrity faces", "attributes", "CelebA", "facial attributes"],
    'flowers': ["flower photos", "flower species", "102 categories", "Flowers"],
    'svhn_cropped': ["street view house numbers", "SVHN", "digits recognition", "cropped"],
    'oxford_iiit_pet': ["pet images", "breeds", "Oxford-IIIT Pet", "cats and dogs"],
    'eurosat': ["satellite images", "land cover", "EuroSAT", "remote sensing"],
    'horses_or_humans': ["horses", "humans", "binary classification", "Horses or Humans"],
    'rock_paper_scissors': ["hand gestures", "Rock Paper Scissors", "game"],
}

dataset_nn_mapping = {
    'mnist': ['CNN'],  # MNIST is suitable for basic image classification tasks with CNNs.
    'fashion_mnist': ['CNN'],  # Fashion MNIST is also best suited for CNNs due to its nature as an image classification dataset.
    'cifar10': ['CNN', 'ResNet', 'EfficientNet'],  # CIFAR-10, with its more complex images, may benefit from more advanced CNN architectures like ResNet or EfficientNet.
    'cifar100': ['CNN', 'ResNet', 'EfficientNet'],  # CIFAR-100's larger number of classes also makes it suitable for advanced CNNs.
    'imdb': ['LSTM', 'GRU', 'BERT'],  # IMDB, being text data, is suitable for LSTM, GRU, and transformer models like BERT for sentiment analysis.
    'boston_housing': ['DNN', 'Random Forest', 'XGBoost'],  # Boston Housing, a regression problem, can be approached with DNNs or ensemble methods like Random Forest and XGBoost.
    'conll2003': ['LSTM', 'GRU', 'BERT'],  # CoNLL-2003, often used for named entity recognition, fits LSTM, GRU, or BERT models.
    'svhn': ['CNN', 'ResNet'],  # The Street View House Numbers (SVHN) dataset, being another image classification task, is well suited for CNNs and ResNet.
    'squad': ['BERT', 'Transformer', 'GPT'],  # The Stanford Question Answering Dataset (SQuAD) is best approached with advanced NLP models like BERT or GPT.
    'wikitext': ['LSTM', 'Transformer', 'GPT-3'],  # Wikitext, a language modeling dataset, is suitable for LSTM, Transformer, and GPT-3 for generating coherent text.
}

dataset_preprocessing_map = {
    'mnist': {
        "preprocessing_function": "standardize_images",
        "input_shape": (28, 28, 1),
        "num_classes": 10
    },
    'fashion_mnist': {
        "preprocessing_function": "standardize_images",
        "input_shape": (28, 28, 1),
        "num_classes": 10
    },
    'cifar10': {
        "preprocessing_function": "standardize_images",
        "input_shape": (32, 32, 3),
        "num_classes": 10
    },
    'cifar100': {
        "preprocessing_function": "standardize_images",
        "input_shape": (32, 32, 3),
        "num_classes": 100
    },
    'imdb': {
        "preprocessing_function": "tokenize_and_pad_text",
        "max_words": 10000,
        "maxlen": 500,
        "num_classes": 2  # Binary sentiment classification
    },
    'boston_housing': {
        "preprocessing_function": "normalize_features",
        "num_classes": None  # Regression problem
    },
    'coarse_grained_cifar100': {
        "preprocessing_function": "standardize_images",
        "input_shape": (32, 32, 3),
        "num_classes": 20  # Assuming 20 superclasses for CIFAR-100
    },
    'caltech101': {
        "preprocessing_function": "resize_and_standardize_images",
        "input_shape": (224, 224, 3),
        "num_classes": 101
    },
    'caltech256': {
        "preprocessing_function": "resize_and_standardize_images",
        "input_shape": (224, 224, 3),
        "num_classes": 256
    },
    'celeba': {
        "preprocessing_function": "resize_and_standardize_images",
        "input_shape": (218, 178, 3),  # Original CelebA image size
        "num_classes": None  # Depends on the attribute being predicted
    },
    'flowers': {
        "preprocessing_function": "resize_and_standardize_images",
        "input_shape": (224, 224, 3),
        "num_classes": 102
    },
    'svhn_cropped': {
        "preprocessing_function": "standardize_images",
        "input_shape": (32, 32, 3),
        "num_classes": 10
    },
    'oxford_iiit_pet': {
        "preprocessing_function": "resize_and_standardize_images",
        "input_shape": (224, 224, 3),
        "num_classes": 37  # 37 pet breeds
    },
    'eurosat': {
        "preprocessing_function": "standardize_images",
        "input_shape": (64, 64, 3),
        "num_classes": 10  # Assuming 10 types of land cover
    },
    'horses_or_humans': {
        "preprocessing_function": "resize_and_standardize_images",
        "input_shape": (300, 300, 3),
        "num_classes": 2
    },
    'rock_paper_scissors': {
        "preprocessing_function": "resize_and_standardize_images",
        "input_shape": (300, 300, 3),
        "num_classes": 3
    },
}

# Database connection string from environment or Key Vault
DATABASE_CONNECTION_STRING = os.getenv("DATABASE_CONNECTION_STRING") or get_secret_from_key_vault("DatabaseConnectionString")

# Securely stored API key for authentication
API_KEY = os.getenv("OCTOPUSAI_API_KEY")

# MLflow tracking URI for model logging
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

# OpenAI API key for NLP tasks
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# TaskManager specific configurations
MAX_CONCURRENT_TASKS = int(os.getenv("MAX_CONCURRENT_TASKS", "10"))
TASK_ASSIGNMENT_RULES = os.getenv("TASK_ASSIGNMENT_RULES", "default")

# Model training default parameters
DEFAULT_EPOCHS = int(os.getenv("DEFAULT_EPOCHS", "10"))
DEFAULT_BATCH_SIZE = int(os.getenv("DEFAULT_BATCH_SIZE", "32"))

# System health monitoring parameters
RESOURCE_UTILIZATION_THRESHOLD = float(os.getenv("RESOURCE_UTILIZATION_THRESHOLD", "0.75"))
PERFORMANCE_METRIC_ALERT_THRESHOLD = float(os.getenv("PERFORMANCE_METRIC_ALERT_THRESHOLD", "0.9"))
