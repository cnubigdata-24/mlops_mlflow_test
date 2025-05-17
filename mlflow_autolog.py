#pip install tensorflow
import mlflow
import mlflow.tensorflow
from mlflow.tracking import MlflowClient
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import os

# Create MLflow client
client = MlflowClient()

# Experiment name
experiment_name = "Sample_TF_Regression"

# Restore deleted experiment if it exists
deleted_experiments = client.search_experiments(view_type=mlflow.entities.ViewType.DELETED_ONLY)
for exp in deleted_experiments:
    if exp.name == experiment_name:
        client.restore_experiment(exp.experiment_id)
        print(f"Restored deleted experiment '{experiment_name}'.")
        break

# Set experiment name
mlflow.set_experiment(experiment_name)

# Enable autologging
print("Enabling MLflow autologging...")
mlflow.tensorflow.autolog()

# Load and subset the dataset
print("Loading sample data...")
X, y = load_diabetes(return_X_y=True)

# Use only a subset for faster training (e.g., 30 samples)
X = X[:1000]
y = y[:1000]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save input data to a DataFrame
df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
df["target"] = y

# Define a simple Keras model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Create output directory for artifacts
os.makedirs("outputs", exist_ok=True)

with mlflow.start_run(run_name="Sample_TF_Model"):
    print("Training started...")
    # Train the model (with reduced epochs for speed)
    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), verbose=0)
    print("Training completed.")
    
    # Save training loss plot
    plt.figure()
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Loss Curve")
    plt.savefig("outputs/loss_curve.png")
    mlflow.log_artifact("outputs/loss_curve.png")
    print("Saved and logged loss curve image.")
    
    # Save input data CSV
    df.to_csv("outputs/input_data.csv", index=False)
    mlflow.log_artifact("outputs/input_data.csv")
    print("Saved and logged input data CSV.")
