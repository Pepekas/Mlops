from datetime import datetime
import json
import logging
import os
import pandas as pd

from airflow import DAG
from airflow.operators.python import PythonOperator
from hooks import CarsHook   # убедитесь, что hooks.py существует

def _fetch_cars(conn_id: str, templates_dict: dict, batch_size: int = 1000, **_):
    logger = logging.getLogger(__name__)
    output_path = templates_dict["output_path"]

    logger.info("Fetching all cars from the API...")
    hook = CarsHook(conn_id=conn_id)
    cars = list(hook.get_cars(batch_size=batch_size))
    logger.info(f"Fetched {len(cars)} car records")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(cars, f)
    logger.info(f"Saved cars to {output_path}")

def _clean_cars_data(templates_dict: dict, **context):
    input_path = templates_dict["input_path"]
    output_path = templates_dict["output_path"]

    df = pd.read_json(input_path)
    df = df.drop_duplicates()
    df = df.dropna()
    cat_cols = ['Fuel_type', 'Transmission']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype('category').cat.codes
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_json(output_path, orient='records', indent=2)

with DAG(
    dag_id="02_hook",
    description="Fetches car data using a custom hook and cleans it.",
    start_date=datetime(2026, 2, 3),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
) as dag:
    fetch = PythonOperator(
        task_id="fetch_cars",
        python_callable=_fetch_cars,
        op_kwargs={"conn_id": "carsapi"},
        templates_dict={"output_path": "/data/raw/cars.json"},
    )
    clean = PythonOperator(
        task_id="clean_cars_data",
        python_callable=_clean_cars_data,
        templates_dict={
            "input_path": "/data/raw/cars.json",
            "output_path": "/data/cleaned/cars_cleaned.json",
        },
    )
    fetch >> clean