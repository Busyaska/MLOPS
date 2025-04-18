from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import timedelta, datetime

from download import download_dataset
from preprocess import preprocess_dataset
from train_model import train_model
from test_model import test_model


train_and_test_model_dag = DAG(
    dag_id="train_and_test_model_pipeline",
    start_date=datetime(2025, 4, 17),
    concurrency=4,
    schedule_interval=timedelta(minutes=5),
    max_active_runs=1,
    catchup=False,
)

download_dataset_task = PythonOperator(python_callable=download_dataset, task_id = "download_dataset", dag=train_and_test_model_dag)
preprocess_dataset_task = PythonOperator(python_callable=preprocess_dataset, task_id = "preprocess_dataset", dag=train_and_test_model_dag)
train_model_task = PythonOperator(python_callable=train_model, task_id = "train_model", dag=train_and_test_model_dag)
test_model_task = PythonOperator(python_callable=test_model, task_id="test_model", dag=train_and_test_model_dag)

download_dataset_task >> preprocess_dataset_task >> train_model_task >> test_model_task
