CREATE TABLE ShapValues (
    id INT PRIMARY KEY IDENTITY,
    model_type VARCHAR(255),
    dataset_name VARCHAR(255),
    shap_values TEXT,
    created_at DATETIME DEFAULT GETDATE()
);
