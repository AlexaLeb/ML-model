from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
import numpy as np
import mlflow
import mlflow.sklearn

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
    # Логируем параметры дерева
    mlflow.log_param('max_depth', model.get_depth())
    mlflow.log_param('n_leaves', model.get_n_leaves())
    mlflow.log_param('criterion', model.criterion)
    print(f'Accuracy: {accuracy:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}')
    print("Confusion matrix:\n", cm)
    print("Depth:", model.get_depth(), "Leaves:", model.get_n_leaves(), "Criterion:", model.criterion)


if __name__ == "__main__":
    with mlflow.start_run(run_name="DecisionTreeExperiment"):
        decision_tree_model = DecisionTreeClassifier(
            random_state=config["random_state"],
            max_depth=config["decision_tree"]["max_depth"]
        )
        data = get_data()
        train(decision_tree_model, data["x_train"], data["y_train"])
        test(decision_tree_model, data["x_test"], data["y_test"])
        mlflow.sklearn.log_model(decision_tree_model, "model")