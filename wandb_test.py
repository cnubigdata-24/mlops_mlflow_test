# !pip install transformers datasets
# !pip install wandb
# !pip install scikit-learn

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 1: Initialize a W&B project
wandb.init(
    project="regression_experiment",
    name="sweep_run",
    notes="Testing W&B with PyTorch regression",
    tags=["pytorch", "regression", "wandb"],
    config={
        "learning_rate": 0.01,
        "batch_size": 32,
        "epochs": 50,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "hidden_units": 16
    }
)

# Load configuration from W&B
config = wandb.config

# Step 2: Generate and preprocess data
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Convert to PyTorch tensors
X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_val, y_val = torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

# Define a simple regression model
class SimpleRegressionModel(nn.Module):
    def __init__(self, input_dim, hidden_units):
        super(SimpleRegressionModel, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_units)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_units, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

# Initialize model, loss function, and optimizer
model = SimpleRegressionModel(input_dim=10, hidden_units=config.hidden_units)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)

# Step 3: Train the model and log experiment data
for epoch in range(config.epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    # Validation step
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)

    # Log metrics to W&B
    wandb.log({
        "epoch": epoch,
        "train_loss": loss.item(),
        "val_loss": val_loss.item()
    })

# Step 4: Save and log the trained model
model_path = "model.pth"
torch.save(model.state_dict(), model_path)

artifact = wandb.Artifact("regression_model", type="model")
artifact.add_file(model_path)

wandb.log_artifact(artifact)

# Step 5: End the experiment
wandb.finish()
