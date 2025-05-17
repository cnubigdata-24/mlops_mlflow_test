import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import os
import shutil

# Create MLflow client
client = MlflowClient()

# Experiment name
experiment_name = "Diabetes_Experiment"

# Restore deleted experiment if it exists
deleted_experiments = client.search_experiments(view_type=mlflow.entities.ViewType.DELETED_ONLY)
for exp in deleted_experiments:
    if exp.name == experiment_name:
        client.restore_experiment(exp.experiment_id)
        print(f"Restored deleted experiment '{experiment_name}'.")
        break

# Set up MLflow experiment
mlflow.set_experiment(experiment_name)

print(">> Enabling MLflow PyTorch autologging...")
mlflow.pytorch.autolog()

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Define model
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

# Prepare local model directory
model_dir = "diabetes_model_direct"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Start MLflow run
with mlflow.start_run(run_name="Diabetes_Run") as run:
    print("🚀 Training started...")
    for epoch in range(10):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        model.eval()
        val_outputs = model(X_test_tensor)
        val_loss = criterion(val_outputs, y_test_tensor)
        
        print(
            f"Epoch {epoch+1}/10 — Train Loss: {loss.item():.4f}, "
            f"Val Loss: {val_loss.item():.4f}"
        )
    
    print("Training complete.")
    
    # Save model directly to local directory (overwrite if exists)
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    mlflow.pytorch.save_model(model, model_dir)
    print(f"> Model saved locally at: {os.path.abspath(model_dir)}")
    
    # Register the model in Model Registry
    registered_model_name = "MyDiabetesModel"
    model_uri = f"file://{os.path.abspath(model_dir)}"
    result = mlflow.register_model(model_uri, registered_model_name)
    print(f">> Model registered: {result.name}, version {result.version}")
    
    print("\n To serve the registered model via REST API:")
    print(f"> mlflow models serve -m models:/{registered_model_name}/{result.version} --port 5001 --no-conda")

# Uncomment to test the API
# import requests
# import json
# test_data = {
#     "dataframe_split": {
#         "columns": ["feature_0", "feature_1", "feature_2", "feature_3", "feature_4", 
#                    "feature_5", "feature_6", "feature_7", "feature_8", "feature_9"],
#         "data": [[0.03807591, 0.05068012, 0.06169621, 0.02187239, -0.0442235,
#                  -0.03482076, -0.04340085, -0.00259226, 0.01990842, -0.01764613]]
#     }
# }
# response = requests.post(
#     "http://127.0.0.1:5001/invocations",
#     headers={"Content-Type": "application/json"},
#     data=json.dumps(test_data)
# )
# print(response.status_code)
# print(response.json())
