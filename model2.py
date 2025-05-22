from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
import numpy as np

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
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    f1 = f1_score(y_true=y_test, y_pred=y_pred, average='weighted')
    cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
    # ROC-AUC (поддержка мультикласса)
    if len(np.unique(y_test)) == 2:
        proba = model.predict_proba(x_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
    else:
        y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
        proba = model.predict_proba(x_test)
        auc = roc_auc_score(y_test_bin, proba, average='macro', multi_class='ovr')
    # Логирование метрик
    logger = task.get_logger()
    logger.report_scalar("accuracy", "test", value=accuracy, iteration=0)
    logger.report_scalar("f1_score", "test", value=f1, iteration=0)
    logger.report_scalar("roc_auc", "test", value=auc, iteration=0)
    # Логирование confusion matrix
    for i, row in enumerate(cm):
        for j, val in enumerate(row):
            logger.report_scalar("confusion_matrix", f"{i}_{j}", value=val, iteration=0)
    # Логирование параметров логистической регрессии
    task.connect({
        "coef_": model.coef_.tolist(),    # Преобразуем в список для логирования
        "intercept_": model.intercept_.tolist(),
        "regularization_C": model.C,
        "solver": model.solver
    })
    print(f'Accuracy: {accuracy:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}')
    print("Confusion matrix:\n", cm)
    print("Coef:", model.coef_, "Intercept:", model.intercept_, "C:", model.C, "Solver:", model.solver)


if __name__ == "__main__":
    logistic_regression_model = LogisticRegression(
        max_iter=config["logistic_regression"]["max_iter"],
    )
    data = get_data()
    train(logistic_regression_model, data["x_train"], data["y_train"])
    test(logistic_regression_model, data["x_test"], data["y_test"])