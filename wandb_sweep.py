sweep_config = {
    "method": "grid",
    "metric": {"name": "val_loss", "goal": "minimize"},
    "parameters": {
        "learning_rate": {"values": [0.001, 0.01, 0.1]},
        "batch_size": {"values": [16, 32, 64]},
        "hidden_units": {"values": [8, 16, 32]}
    }
}

sweep_id = wandb.sweep(sweep_config, project="regression_experiment")

def train():
    wandb.init()
    config = wandb.config

    model = SimpleRegressionModel(input_dim=10, hidden_units=config.hidden_units)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)

    for epoch in range(20):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)

        wandb.log({"epoch": epoch, "val_loss": val_loss.item()})

    wandb.finish()

wandb.agent(sweep_id, function=train, count=27)
