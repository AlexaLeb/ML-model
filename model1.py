from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
import numpy as np

from config import config
from data import get_data

import wandb

wandb.init(
    project='wandb',
    name='LogReg'
)


def train(model, x_train, y_train) -> None:
    wandb.config.update(model.get_params())
    model.fit(x_train, y_train)


def test(model, x_test, y_test) -> None:
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

    # Логируем параметры и метрики
    wandb.log({
        'accuracy': accuracy,
        'f1_score': f1,
        'roc_auc': auc,
        'coef_': wandb.Histogram(model.coef_),
        'intercept_': wandb.Histogram(model.intercept_),
        'regularization_C': model.C,
        'solver': model.solver,
        'confusion_matrix': wandb.plot.confusion_matrix(
            probs=None,
            y_true=y_test,
            preds=y_pred,
            class_names=[str(i) for i in np.unique(y_test)]
        ),
    })
    print(f'Accuracy: {accuracy:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}')
    print("Confusion matrix:\n", cm)
    print("Coefficients:", model.coef_, "\nIntercept:", model.intercept_, "\nC:", model.C, "\nSolver:", model.solver)


if __name__ == "__main__":
    logistic_regression_model = LogisticRegression(
        max_iter=config["logistic_regression"]["max_iter"],
    )

    data = get_data()
    train(logistic_regression_model, data["x_train"], data["y_train"])
    test(logistic_regression_model, data["x_test"], data["y_test"])
