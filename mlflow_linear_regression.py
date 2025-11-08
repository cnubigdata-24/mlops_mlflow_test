# mlflow --version
# pip install mlflow==3.4.0
# pip install scikit-learn pandas numpy
# pip show mlflow
# mkdir mlflow_test
# cd mlflow_test
# python mlflow_linear_regression.py
# mlflow ui

import mlflow
import mlflow.sklearn
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd

# Experiment setup
experiment_name = "LinearRegression-Experiment"
existing_exp = mlflow.get_experiment_by_name(experiment_name)
if existing_exp is None:
    experiment_id = mlflow.create_experiment(experiment_name)
    print(f"Created new experiment: '{experiment_name}'")
else:
    experiment_id = existing_exp.experiment_id
    print(f"Using existing experiment: '{experiment_name}'")

mlflow.set_experiment(experiment_name)

# Data generation
X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
with mlflow.start_run():
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Prediction and evaluation
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    
    # Log metrics
    mlflow.log_metric("mse", mse)
    
    # Create input_example
    input_example = pd.DataFrame(X_train[:5])
    
    # Log model
    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        input_example=input_example,
        registered_model_name="LinearRegressionModel"
    )
    
    print(f"Model logged with MSE: {mse:.4f}")
