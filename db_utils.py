import pyodbc
import json
from config import DATABASE_CONNECTION_STRING

def save_shap_values(model_type, dataset_name, shap_values):
    connection = pyodbc.connect(DATABASE_CONNECTION_STRING)
    cursor = connection.cursor()
    shap_values_json = json.dumps(shap_values)  # Assuming shap_values is a list or dict
    insert_query = """INSERT INTO ShapValues (model_type, dataset_name, shap_values)
                      VALUES (?, ?, ?)"""
    cursor.execute(insert_query, (model_type, dataset_name, shap_values_json))
    connection.commit()
    connection.close()

def get_shap_values(model_type, dataset_name):
    connection = pyodbc.connect(DATABASE_CONNECTION_STRING)
    cursor = connection.cursor()
    select_query = """SELECT shap_values FROM ShapValues
                      WHERE model_type = ? AND dataset_name = ?"""
    cursor.execute(select_query, (model_type, dataset_name))
    row = cursor.fetchone()
    connection.close()
    if row:
        return json.loads(row[0])  # Convert JSON back to Python list or dict
    return None
