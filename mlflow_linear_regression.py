# pip install mlflow scikit-learn pandas numpy

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

# 데이터 생성
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2.5 * X + np.random.randn(100, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 실험 이름
experiment_name = "LinearRegression-Experiment"
client = MlflowClient()

# 기존 실험 확인
existing = [e for e in client.list_experiments() if e.name == experiment_name]
if existing:
    exp = existing[0]
    if exp.lifecycle_stage == "deleted":
        client.restore_experiment(exp.experiment_id)
    else:
        client.delete_experiment(exp.experiment_id)

# 새로 실험 생성
experiment_id = client.create_experiment(experiment_name)
mlflow.set_experiment(experiment_name)

# 실험 실행
with mlflow.start_run():  
    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    # 매개변수 및 메트릭 기록
    mlflow.log_param("alpha", 0.01)
    mlflow.log_metric("mse", mse)

    # 모델 저장
    mlflow.sklearn.log_model(model, "linear_regression_model")
    print(f"Model logged with MSE: {mse:.4f}")
