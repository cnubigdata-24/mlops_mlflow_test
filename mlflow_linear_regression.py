#pip install mlflow scikit-learn pandas numpy

import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2.5 * X + np.random.randn(100, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 실험 설정
mlflow.set_experiment("LinearRegression-Experiment")

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
    print(f"Model logged with MSE: {mse}")
