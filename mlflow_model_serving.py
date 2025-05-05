import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import os

# Set experiment name
mlflow.set_experiment("PyTorch_Diabetes_Regression")

# Enable MLflow autologging for PyTorch
mlflow.pytorch.autolog()

# Load real diabetes dataset (442 samples, 10 features)
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Define simple PyTorch model
class DiabetesRegressor(nn.Module):
    def __init__(self, input_dim):
        super(DiabetesRegressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

model = DiabetesRegressor(X.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Output directory for logs
os.makedirs("outputs", exist_ok=True)

# Save dataset as CSV
df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
df["target"] = y
df.to_csv("outputs/input_data.csv", index=False)

# Start MLflow run
with mlflow.start_run(run_name="Diabetes_PyTorch_Model") as run:
    print(">> Training started...")

    train_losses, val_losses = [], []

    for epoch in range(15):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        model.eval()
        val_outputs = model(X_test_tensor)
        val_loss = criterion(val_outputs, y_test_tensor)

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

        print(f"Epoch {epoch+1}: Train Loss={loss.item():.4f}, Val Loss={val_loss.item():.4f}")

    print(">> Training complete.")

    # Save and log loss curve
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Loss Curve (Diabetes)")
    plt.savefig("outputs/loss_curve.png")
    mlflow.log_artifact("outputs/loss_curve.png")
    mlflow.log_artifact("outputs/input_data.csv")

    # Register the model
    model_uri = f"runs:/{run.info.run_id}/model"
    result = mlflow.register_model(model_uri, "MyDiabetesModel")
    print(f">> Model registered as: {result.name}, version: {result.version}")


# mlflow models serve -m "models:/MyDiabetesModel/1" --port 5001

# curl -X POST http://127.0.0.1:5001/invocations \
#      -H "Content-Type: application/json" \
#      -d '{"inputs": [[0.038, 0.050, 0.061, 0.021, -0.044, -0.034, -0.043, -0.002, 0.019, -0.017]]}'
