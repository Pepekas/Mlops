import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import kagglehub
import os
from train_model import train


def download_data():
    path = kagglehub.dataset_download("muhammetvarl/laptop-price")
    csv_path = os.path.join(path, "laptop_price.csv")

    df = pd.read_csv(csv_path, encoding='latin-1')
    df.to_csv("laptops.csv", index=False)
    print("Downloaded dataset shape: ", df.shape)
    return True


def clear_data():
    df = pd.read_csv("laptops.csv")

    df['Ram'] = df['Ram'].str.replace('GB', '').astype(int)
    df['Weight'] = df['Weight'].str.replace('kg', '').astype(float)

    cat_columns = ['Company', 'TypeName', 'OpSys']
    num_columns = ['Inches', 'Ram', 'Weight', 'Price_euros']

    df = df[cat_columns + num_columns].dropna()

    df = df[(df['Weight'] > 0) & (df['Price_euros'] > 0)]
    df = df.reset_index(drop=True)

    ordinal = OrdinalEncoder()
    df[cat_columns] = ordinal.fit_transform(df[cat_columns])

    df.to_csv('laptops_clear.csv', index=False)
    return True


dag_cars = DAG(
    dag_id="train_pipe",
    start_date=datetime(2025, 2, 3),
    max_active_tasks=4,
    schedule=timedelta(minutes=5),
    max_active_runs=1,
    catchup=False,
)

download_task = PythonOperator(python_callable=download_data, task_id="download_data", dag=dag_cars)
clear_task = PythonOperator(python_callable=clear_data, task_id="clear_data", dag=dag_cars)
train_task = PythonOperator(python_callable=train, task_id="train_model", dag=dag_cars)

download_task >> clear_task >> train_task