import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

experiment_name = "LinearRegression-Experiment"
client = MlflowClient()

for exp in client.search_experiments(view_type=mlflow.entities.ViewType.DELETED_ONLY):
    if exp.name == experiment_name:
        client.restore_experiment(exp.experiment_id)
        print(f"실험 '{experiment_name}' 복원")

try:
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment:
        experiment_id = experiment.experiment_id
        print(f"기존 실험 '{experiment_name}' 사용.")
    else:
        experiment_id = client.create_experiment(experiment_name)
        print(f"새 실험 '{experiment_name}' 생성")
except:
    experiment_id = client.create_experiment(experiment_name)
    print(f"새 실험 '{experiment_name}' 생성")

mlflow.set_experiment(experiment_name)

# 데이터 생성
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2.5 * X + np.random.randn(100, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# MLflow 실험 시작
with mlflow.start_run():
    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    # 매개변수 및 성능 기록
    mlflow.log_param("alpha", 0.01)
    mlflow.log_metric("mse", mse)

    # 모델 저장
    mlflow.sklearn.log_model(model, "linear_regression_model")

    print(f"MSE: {mse:.4f}로 모델 기록")
