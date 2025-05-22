from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from config import config
from data import get_data

from clearml import Task

task = Task.init(
    project_name='ClearML',
    task_name='LogRegTraining2'
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
    decision_tree_model = DecisionTreeClassifier(
        random_state=config["random_state"],
        max_depth=config["decision_tree"]["max_depth"]
    )

    data = get_data()
    train(decision_tree_model, data["x_train"], data["y_train"])
    test(decision_tree_model, data["x_test"], data["y_test"])
