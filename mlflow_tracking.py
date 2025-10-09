import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# Create MLflow client
client = MlflowClient()

# Experiment name to use
experiment_name = "classification_model_comparison"

# Restore deleted experiment if it exists
# deleted_experiments = client.search_experiments(view_type=mlflow.entities.ViewType.DELETED_ONLY)
# for exp in deleted_experiments:
#     if exp.name == experiment_name:
#         client.restore_experiment(exp.experiment_id)
#         print(f"Restored deleted experiment '{experiment_name}'.")
#         break

# Set experiment (now it should work since we've restored it)
mlflow.set_experiment(experiment_name)

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model 1: RandomForest
with mlflow.start_run(run_name="RandomForest_Model"):
    model1 = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model1.fit(X_train, y_train)
    predictions1 = model1.predict(X_test)
    acc1 = accuracy_score(y_test, predictions1)
    f1_1 = f1_score(y_test, predictions1)

    # Log parameters and metrics
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 3)
    mlflow.log_metric("accuracy", acc1)
    mlflow.log_metric("f1_score", f1_1)

    # Save model
    mlflow.sklearn.log_model(model1, "model1")

# Model 2: XGBoost
with mlflow.start_run(run_name="XGBoost_Model"):
    model2 = XGBClassifier(learning_rate=0.1, max_depth=3, n_estimators=100, random_state=42)
    model2.fit(X_train, y_train)
    predictions2 = model2.predict(X_test)
    acc2 = accuracy_score(y_test, predictions2)
    f1_2 = f1_score(y_test, predictions2)

    # Log parameters and metrics
    mlflow.log_param("model_type", "XGBoost")
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_param("max_depth", 4)
    mlflow.log_param("n_estimators", 200)
    mlflow.log_metric("accuracy", acc2)
    mlflow.log_metric("f1_score", f1_2)

    # Save model
    mlflow.sklearn.log_model(model2, "model2")

print(f"RandomForest Accuracy: {acc1}, F1-Score: {f1_1}")
print(f"XGBoost Accuracy: {acc2}, F1-Score: {f1_2}")
