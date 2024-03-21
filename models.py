import tensorflow as tf
import ml_flow_utils
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, GRU, SimpleRNN, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16, ResNet50, EfficientNetB0
from transformers import TFBertModel, TFGPT2Model, GPT2Config, TFAutoModelForSequenceClassification
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def get_model(model_type, **kwargs):
    if model_type == 'CNN':
        return create_cnn_model(**kwargs)
    elif model_type == 'LSTM':
        return create_lstm_model(**kwargs)
    elif model_type == 'GRU':
        return create_gru_model(**kwargs)
    elif model_type == 'RNN':
        return create_rnn_model(**kwargs)
    elif model_type == 'ResNet':
        return ResNet50(**kwargs)  # Simplified; you might need a wrapper to adjust inputs/outputs
    elif model_type == 'EfficientNet':
        return EfficientNetB0(**kwargs)  # Simplified; adjustments may be needed
    elif model_type == 'BERT':
        return TFBertModel.from_pretrained('bert-base-uncased', **kwargs)
    elif model_type == 'Transformer':
        # Assuming you have a custom transformer model function
        return create_transformer_model(**kwargs)
    elif model_type == 'GPT':
        config = GPT2Config.from_pretrained('gpt2', **kwargs)
        return TFGPT2Model(config)
    elif model_type == 'DNN':
        # Assuming a function for simple dense neural networks
        return create_dnn_model(**kwargs)
    elif model_type in ['Random Forest', 'XGBoost']:
        # These models don't fit directly into a TensorFlow pipeline;
        # they would need separate handling for training and inference.
        raise NotImplementedError(f"{model_type} model type requires a different handling approach.")
    else:
        raise ValueError(f"Model type {model_type} not supported.")

def create_cnn_model(input_shape=(224, 224, 3), num_classes=10):
    """
    Create a Convolutional Neural Network (CNN) model based on VGG16 architecture for image analysis.
    :param input_shape: Shape of the input images.
    :param num_classes: Number of classes for classification.
    :return: Compiled CNN model.
    """
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    x = Flatten()(base_model.output)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_lstm_model(input_length=100, vocab_size=10000, num_classes=10):
    """
    Create an LSTM model for text sequence analysis.
    :param input_length: Maximum length of input sequences.
    :param vocab_size: Size of the vocabulary.
    :param num_classes: Number of classes for classification.
    :return: Compiled LSTM model.
    """
    model = Sequential([
        Embedding(vocab_size, 128, input_length=input_length),
        LSTM(128, dropout=0.2, recurrent_dropout=0.2),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_gru_model(input_length=100, vocab_size=10000, num_classes=10):
    """
    Create a GRU model for text sequence analysis.
    :param input_length: Maximum length of input sequences.
    :param vocab_size: Size of the vocabulary.
    :param num_classes: Number of classes for classification.
    :return: Compiled GRU model.
    """
    model = Sequential([
        Embedding(vocab_size, 128, input_length=input_length),
        GRU(128, dropout=0.2, recurrent_dropout=0.2),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_rnn_model(input_length=100, vocab_size=10000, num_classes=10):
    """
    Create a Simple RNN model for text or sequence data processing.
    :param input_length: Maximum length of input sequences.
    :param vocab_size: Size of the vocabulary.
    :param num_classes: Number of classes for classification.
    :return: Compiled RNN model.
    """
    model = Sequential([
        Embedding(vocab_size, 128, input_length=input_length),
        SimpleRNN(128, dropout=0.2, recurrent_dropout=0.2),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_dnn_model(input_shape=(28*28,), num_classes=10):
    model = Sequential([
        Input(shape=input_shape),
        Dense(512, activation='relu'),
        Dropout(0.2),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_transformer_model(pretrained_model_name='bert-base-uncased', num_labels=2):
    model = TFAutoModelForSequenceClassification.from_pretrained(pretrained_model_name, num_labels=num_labels)
    return model

def train_cnn_model(x_train, y_train, x_val, y_val):
    """
    Train a CNN model and log the details using MLflow.

    :param x_train: Training features.
    :param y_train: Training labels.
    :param x_val: Validation features.
    :param y_val: Validation labels.
    :return: Trained CNN model.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=x_train.shape[1:]),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(len(y_train[0]), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))
    
    # Log the model and training details to MLflow
    run_id = ml_flow_utils.start_mlflow_run().info.run_id
    training_info = {"accuracy": max(history.history['val_accuracy'])}
    ml_flow_utils.log_model_details(run_id, model, 'CNN', training_info, 'Your Dataset Name')
    
    return model

def train_lstm_model(x_train, y_train, x_val, y_val, input_length=100, vocab_size=10000, num_classes=10):
    """
    Train an LSTM model and log the details using MLflow.

    :param x_train: Training features.
    :param y_train: Training labels.
    :param x_val: Validation features.
    :param y_val: Validation labels.
    :return: Trained LSTM model.
    """
    model = create_lstm_model(input_length=input_length, vocab_size=vocab_size, num_classes=num_classes)
    
    history = model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))
    
    # Log the model and training details to MLflow
    run_id = ml_flow_utils.start_mlflow_run().info.run_id
    training_info = {"accuracy": max(history.history['val_accuracy'])}
    ml_flow_utils.log_model_details(run_id, model, 'LSTM', training_info, 'Your Dataset Name')
    
    return model

def train_gru_model(x_train, y_train, x_val, y_val, input_length=100, vocab_size=10000, num_classes=10):
    """
    Train a GRU model and log the details using MLflow.

    :param x_train: Training features.
    :param y_train: Training labels.
    :param x_val: Validation features.
    :param y_val: Validation labels.
    :return: Trained GRU model.
    """
    model = create_gru_model(input_length=input_length, vocab_size=vocab_size, num_classes=num_classes)
    
    history = model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))
    
    # Log the model and training details to MLflow
    run_id = ml_flow_utils.start_mlflow_run().info.run_id
    training_info = {"accuracy": max(history.history['val_accuracy'])}
    ml_flow_utils.log_model_details(run_id, model, 'GRU', training_info, 'Your Dataset Name')
    
    return model

def train_rnn_model(x_train, y_train, x_val, y_val, input_length=100, vocab_size=10000, num_classes=10):
    """
    Train a Simple RNN model and log the details using MLflow.

    :param x_train: Training features.
    :param y_train: Training labels.
    :param x_val: Validation features.
    :param y_val: Validation labels.
    :return: Trained RNN model.
    """
    model = create_rnn_model(input_length=input_length, vocab_size=vocab_size, num_classes=num_classes)
    
    history = model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))
    
    # Log the model and training details to MLflow
    run_id = ml_flow_utils.start_mlflow_run().info.run_id
    training_info = {"accuracy": max(history.history['val_accuracy'])}
    ml_flow_utils.log_model_details(run_id, model, 'RNN', training_info, 'Your Dataset Name')
    
    return model
