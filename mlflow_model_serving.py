import mlflow
import mlflow.tensorflow
import tensorflow as tf
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# Enable autologging
mlflow.tensorflow.autolog()

# Prepare data (sample only 100 rows for quick run)
X, y = load_diabetes(return_X_y=True)
X, y = X[:100], y[:100]
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Save training data snapshot
df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
df["target"] = y
os.makedirs("outputs", exist_ok=True)
df.to_csv("outputs/input_data.csv", index=False)

# Define model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# MLflow run
with mlflow.start_run() as run:
    print(">> Training started")
    history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
    print(">> Training completed")

    # Save loss curve
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Loss Curve")
    plt.savefig("outputs/loss_curve.png")
    mlflow.log_artifact("outputs/loss_curve.png")
    mlflow.log_artifact("outputs/input_data.csv")

    # Register model from logged run
    model_uri = f"runs:/{run.info.run_id}/model"
    model_details = mlflow.register_model(model_uri=model_uri, name="DiabetesTFModel")
    print(">> Model registered:", model_details.name)


# mlflow models serve -m "models:/DiabetesTFModel/1" --port 5001

# curl -X POST http://127.0.0.1:5001/invocations \
#      -H "Content-Type: application/json" \
#      -d '{"inputs": [[0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13]]}'

