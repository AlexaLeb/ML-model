from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from scripts.weather import fetch_weather, save_weather

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2025, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
    "email_on_failure": False,
}

with DAG(
    dag_id="weather_moscow",
    default_args=default_args,
    schedule_interval="* * * * *",  # каждую минуту
    catchup=False,
    max_active_runs=1
) as dag:

    fetch = PythonOperator(
        task_id="fetch_weather",
        python_callable=fetch_weather
    )

    save = PythonOperator(
        task_id="save_weather",
        python_callable=lambda **ctx: save_weather(
            fetch.execute(context=ctx)
        )
    )

    fetch >> save