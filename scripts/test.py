import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report
import json

def test_model():
    """
    Загружает модель из model.pkl и тестовые данные из dataset/iris_test.csv,
    вычисляет accuracy и classification_report, сохраняет оба в model_metrics.json.
    """
    df = pd.read_csv('dataset/iris_test.csv')
    X = df.drop(columns=['target'])
    y = df['target']
    model = joblib.load('model.pkl')
    preds = model.predict(X)
    accuracy = accuracy_score(y, preds)
    report = classification_report(y, preds, output_dict=True)
    metrics = {'accuracy': accuracy, 'report': report}
    with open('model_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Test completed. Accuracy={accuracy:.4f}, saved metrics to model_metrics.json")