import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib


def train_model():
    """
    Обучает LogisticRegression на dataset/iris_train.csv
    и сохраняет готовую модель в model.pkl
    """
    df = pd.read_csv('dataset/iris_train.csv')
    X = df.drop(columns=['target'])
    y = df['target']
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)
    joblib.dump(model, 'model.pkl')
    print("Model trained and saved to model.pkl")

