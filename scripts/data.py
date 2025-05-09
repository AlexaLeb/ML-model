import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import os


def load_data():
    """
    Загружает набор 'Ирисы Фишера' из sklearn и сохраняет его в dataset/iris.csv
    """
    data = load_iris(as_frame=True)
    df = pd.concat([data.data, data.target.rename('target')], axis=1)
    os.makedirs('dataset', exist_ok=True)
    df.to_csv('/app/dataset/iris.csv', index=False)
    print("Saved raw data to dataset/iris.csv")


def prepare_data():
    """
    Читает dataset/iris.csv, разбивает в соотношении 0.8:0.2 на train/test
    и сохраняет файлы iris_train.csv и iris_test.csv
    """
    df = pd.read_csv('dataset/iris.csv')
    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['target'])
    train.to_csv('/app/dataset/iris_train.csv', index=False)
    test.to_csv('/app/dataset/iris_test.csv', index=False)
    print("Saved train data to dataset/iris_train.csv")
    print("Saved test data to dataset/iris_test.csv")
