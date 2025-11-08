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

# 실험 설정
experiment_name = "LinearRegression-Experiment"
existing_exp = mlflow.get_experiment_by_name(experiment_name)
if existing_exp is None:
    experiment_id = mlflow.create_experiment(experiment_name)
    print(f"새 실험 '{experiment_name}' 생성")
else:
    experiment_id = existing_exp.experiment_id
    print(f"기존 실험 '{experiment_name}' 사용")

mlflow.set_experiment(experiment_name)

# 데이터 생성
X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 학습
with mlflow.start_run():
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 예측 및 평가
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    
    # 메트릭 로깅
    mlflow.log_metric("mse", mse)
    
    # input_example 생성
    input_example = pd.DataFrame(X_train[:5])
    
    # 모델 로깅
    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",             
        input_example=input_example,
        registered_model_name="LinearRegressionModel"
    )
    
    print(f"MSE: {mse:.4f}로 모델 기록")
