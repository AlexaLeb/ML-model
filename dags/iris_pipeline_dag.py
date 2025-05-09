# dags/iris_pipeline_dag.py
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

# импортируем наши функции
from scripts.data import load_data, prepare_data
from scripts.train import train_model
from scripts.test import test_model

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

with DAG(
    dag_id='iris_classification_pipeline',
    default_args=default_args,
    description='ETL + train + test pipeline for Iris dataset',
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,       # запуск вручную или @daily, если нужно
    catchup=False,
) as dag:

    # Задачи DAG-а
    t1 = PythonOperator(
        task_id='load_data',
        python_callable=load_data,
    )

    t2 = PythonOperator(
        task_id='prepare_data',
        python_callable=prepare_data,
    )

    t3 = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
    )

    t4 = PythonOperator(
        task_id='test_model',
        python_callable=test_model,
    )

    # Последовательность выполнения
    t1 >> t2 >> t3 >> t4