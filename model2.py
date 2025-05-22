from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from config import config
from data import get_data

from clearml import Task

task = Task.init(
    project_name="ClearML",
    task_name='LogRegTraining'
)


def train(model, x_train, y_train) -> None:
    task.connect(model.get_params())
    model.fit(x_train, y_train)


def test(model, x_test, y_test) -> None:
    y_pred = model.predict(x_test)
    # Здесь необходимо получить метрики и логировать их в трекер
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    task.get_logger().report_scalar("accuracy", 'test', value=accuracy, iteration=0)
    print(accuracy)


if __name__ == "__main__":
    logistic_regression_model = LogisticRegression(
        max_iter=config["logistic_regression"]["max_iter"],
    )

    data = get_data()
    train(logistic_regression_model, data["x_train"], data["y_train"])
    test(logistic_regression_model, data["x_test"], data["y_test"])
