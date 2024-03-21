import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.datasets import mnist, cifar10, cifar100, fashion_mnist, imdb, boston_housing
from tensorflow.keras.preprocessing import image, sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import mlflow
from mlflow.sklearn import log_model
from xgboost import XGBRegressor
import numpy as np
import logging
from config import dataset_preprocessing_map

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def load_and_preprocess_dataset(dataset_name, **kwargs):
    input_shape = kwargs.get('input_shape', (224, 224, 3)) # Default shape for CNNs
    num_classes = kwargs.get('num_classes', 10) # Default for classification tasks
    max_words = kwargs.get('max_words', 10000) # For text datasets
    maxlen = kwargs.get('maxlen', 500) # For text datasets    

    def normalize_img(image, label):
        resized_image = tf.image.resize(image, input_shape[:2])
        return tf.cast(resized_image, tf.float32) / 255., tf.one_hot(label, depth=num_classes)

    try:
        if dataset_name == 'mnist' or dataset_name == 'fashion_mnist':
            dataset = mnist if dataset_name == 'mnist' else fashion_mnist
            (x_train, y_train), (x_test, y_test) = dataset.load_data()
            x_train = x_train.reshape((-1,) + input_shape + (1,)).astype('float32') / 255
            x_test = x_test.reshape((-1,) + input_shape + (1,)).astype('float32') / 255
            y_train, y_test = to_categorical(y_train, num_classes), to_categorical(y_test, num_classes)
        elif dataset_name in ['cifar10', 'cifar100']:
            dataset = cifar10 if dataset_name == 'cifar10' else cifar100
            (x_train, y_train), (x_test, y_test) = dataset.load_data()
            x_train, x_test = x_train.astype('float32') / 255, x_test.astype('float32') / 255
            y_train, y_test = to_categorical(y_train, num_classes), to_categorical(y_test, num_classes)
        elif dataset_name == 'imdb':
            (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words)
            x_train, x_test = sequence.pad_sequences(x_train, maxlen=maxlen), sequence.pad_sequences(x_test, maxlen=maxlen)
        elif dataset_name == 'svhn_cropped':
            # Load and preprocess SVHN dataset
            ds_train, ds_test = tfds.load('svhn_cropped', split=['train', 'test'], as_supervised=True)
            ds_train = ds_train.map(normalize_img).cache().shuffle(10000).batch(128).prefetch(tf.data.experimental.AUTOTUNE)
            ds_test = ds_test.map(normalize_img).batch(128).prefetch(tf.data.experimental.AUTOTUNE)
            return ds_train, ds_test
        elif dataset_name == "boston_housing":
            (x_train, y_train), (x_val, y_val) = boston_housing.load_data()
            
            # Normalize features as it's a common practice for regression problems
            mean = x_train.mean(axis=0)
            std = x_train.std(axis=0)
            x_train = (x_train - mean) / std
            x_val = (x_val - mean) / std
            
            # Split the training data for validation
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

            # Dynamically pass parameters based on training_parameters_map
            model, history = await train_model(x_train, y_train, x_val, y_val, "boston_housing", **kwargs)

            # Evaluate the model
            test_loss, test_mae = model.evaluate(x_test, y_test, verbose=2)
            print(f"Test MAE: {test_mae}")

            return model, history
        elif dataset_name == 'caltech101':
            ds_train, ds_info = tfds.load('caltech101', split='train', with_info=True, as_supervised=True)
            ds_test = tfds.load('caltech101', split='test', as_supervised=True)
            ds_train = ds_train.map(normalize_img).cache().shuffle(10000).batch(128).prefetch(tf.data.experimental.AUTOTUNE)
            ds_test = ds_test.map(normalize_img).batch(128).prefetch(tf.data.experimental.AUTOTUNE)
            return ds_train, ds_test
        elif dataset_name == "random_forest":
            rf_param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
            # Train Random Forest
            print("Training Random Forest...")
            rf_model = RandomForestRegressor(random_state=42)
            train_model_with_grid_search_cv(rf_model, rf_param_grid, x_train, y_train, x_test, y_test)
            # Note: This is an example. You might need a specific dataset for Random Forest.
            model, history = await train_model(x_train, y_train, x_val, y_val, "random_forest", **kwargs)

        elif dataset_name == "xgboost":
            xgb_param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [6, 10],
                'learning_rate': [0.01, 0.1],
                'subsample': [0.8, 1],
                'colsample_bytree': [0.8, 1]
            }
            # Train XGBoost
            print("Training XGBoost...")
            xgb_model = XGBRegressor(random_state=42)
            train_model_with_grid_search_cv(xgb_model, xgb_param_grid, x_train, y_train, x_test, y_test)
            # Note: This is an example. You might need a specific dataset for XGBoost.
            model, history = await train_model(x_train, y_train, x_val, y_val, "xgboost", **kwargs)
        else:
            raise ValueError(f"Dataset {dataset_name} is not supported.")

        logger.info(f"{dataset_name} loaded and preprocessed successfully.")
        return (x_train, y_train), (x_test, y_test)

    except Exception as e:
        logger.error(f"Failed to load or preprocess {dataset_name}: {e}")
        raise
    
async def train_model(x_train, y_train, x_val, y_val, model_type, **kwargs):
    if model_type == "boston_housing":
        model = get_regression_model(input_shape=x_train.shape[1:], **kwargs)
    elif model_type == "random_forest":
        model = get_random_forest_model(**kwargs)
    elif model_type == "xgboost":
        model = get_xgboost_model(**kwargs)
    
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=kwargs.get('epochs', 100), batch_size=kwargs.get('batch_size', 32))
    return model, history

def preprocess_image_for_cnn(image_path, target_size=(224, 224)):
    """
    Update the preprocess_image_for_cnn function to allow dynamic resizing based on the model requirements.
    :param image_path: Path to the image file.
    :param target_size: Tuple representing the target size for image datasets.
    :return: Preprocessed image suitable for CNN input.
    """
    try:
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array_expanded_dims = np.expand_dims(img_array, axis=0)
        return tf.keras.applications.vgg16.preprocess_input(img_array_expanded_dims)
    except Exception as e:
        logger.error(f"Failed to preprocess image {image_path}: {e}")
        raise

def get_regression_model(input_shape=(13,), num_classes=1, **kwargs):

    model = Sequential([
        Dense(64, input_shape=input_shape, activation='relu'),
        Dropout(0.1),
        Dense(64, activation='relu'),
        Dropout(0.1),
        Dense(num_classes, activation='linear')
    ])

    optimizer = kwargs.get('optimizer', Adam(lr=0.001))
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
    return model

def get_random_forest_model(**kwargs):
    n_estimators = kwargs.get('n_estimators', 100)
    max_depth = kwargs.get('max_depth', None)
    min_samples_split = kwargs.get('min_samples_split', 2)
    min_samples_leaf = kwargs.get('min_samples_leaf', 1)
    
    model = RandomForestRegressor(n_estimators=n_estimators,
                                  max_depth=max_depth,
                                  min_samples_split=min_samples_split,
                                  min_samples_leaf=min_samples_leaf,
                                  random_state=42)
    return model

def get_xgboost_model(**kwargs):
    n_estimators = kwargs.get('n_estimators', 100)
    max_depth = kwargs.get('max_depth', 3)
    learning_rate = kwargs.get('learning_rate', 0.1)
    subsample = kwargs.get('subsample', 1)
    
    model = XGBRegressor(n_estimators=n_estimators,
                         max_depth=max_depth,
                         learning_rate=learning_rate,
                         subsample=subsample,
                         random_state=42)
    return model

def train_model_with_grid_search_cv(model, param_grid, x_train, y_train, x_test, y_test, cv=5):
    """
    Train a model using GridSearchCV for hyperparameter tuning and cross-validation.
    
    :param model: The model to train (RandomForestRegressor or XGBRegressor).
    :param param_grid: Dictionary with parameters names (str) as keys and lists of parameter settings to try as values.
    :param x_train, y_train: Training data.
    :param x_test, y_test: Test data.
    :param cv: Number of cross-validation folds.
    :return: The best estimator from GridSearchCV.
    """
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
    grid_search.fit(x_train, y_train)

    best_estimator = grid_search.best_estimator_
    predictions = best_estimator.predict(x_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"MSE: {mse}, MAE: {mae}")

    # Log to MLflow
    log_parameters_and_metrics(best_estimator, grid_search.best_params_, {"mse": mse, "mae": mae})

    return best_estimator

def log_parameters_and_metrics(model, params, metrics):
    with mlflow.start_run():
        mlflow.log_params(params)
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        log_model(model, model.__class__.__name__)

async def train_boston_housing_model(x_train, y_train, x_val, y_val, **kwargs):
    model = get_regression_model(input_shape=x_train.shape[1:], **kwargs)
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=kwargs.get('epochs', 100), batch_size=kwargs.get('batch_size', 32))
    return model, history
