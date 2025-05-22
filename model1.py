from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
import numpy as np
import mlflow
import mlflow.sklearn
import pandas as pd

from config import config
from data import get_data


def train(model, x_train, y_train):
    model.fit(x_train, y_train)


def test(model, x_test, y_test):
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    f1 = f1_score(y_true=y_test, y_pred=y_pred, average='weighted')
    cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
    if len(np.unique(y_test)) == 2:
        proba = model.predict_proba(x_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
    else:
        y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
        proba = model.predict_proba(x_test)
        auc = roc_auc_score(y_test_bin, proba, average='macro', multi_class='ovr')
    # Логируем метрики
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_metric('f1_score', f1)
    mlflow.log_metric('roc_auc', auc)
    # Логируем confusion matrix как артефакт
    np.savetxt('confusion_matrix.csv', cm, delimiter=',', fmt='%d')
    mlflow.log_artifact('confusion_matrix.csv')
    # Логируем параметры модели
    mlflow.log_param('regularization_C', model.C)
    mlflow.log_param('solver', model.solver)
    # Логируем коэффициенты (как артефакт)
    np.savetxt('coef.csv', model.coef_, delimiter=',')
    mlflow.log_artifact('coef.csv')
    # Логируем свободный член
    np.savetxt('intercept.csv', model.intercept_, delimiter=',')
    mlflow.log_artifact('intercept.csv')
    print(f'Accuracy: {accuracy:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}')
    print("Confusion matrix:\n", cm)
    print("Coefficients:", model.coef_, "\nIntercept:", model.intercept_, "\nC:", model.C, "\nSolver:", model.solver)


if __name__ == "__main__":
    with mlflow.start_run(run_name="LogisticRegressionExperiment"):
        logistic_regression_model = LogisticRegression(
            max_iter=config["logistic_regression"]["max_iter"],
        )
        data = get_data()
        train(logistic_regression_model, data["x_train"], data["y_train"])
        test(logistic_regression_model, data["x_test"], data["y_test"])
        # Сохраняем саму модель
        mlflow.sklearn.log_model(logistic_regression_model, "model")
        input_example = data["x_test"][:1]
        mlflow.sklearn.log_model(
            logistic_regression_model,
            "model",
            input_example=input_example
        )
