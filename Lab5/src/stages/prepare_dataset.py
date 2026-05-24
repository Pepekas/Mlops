import pandas as pd
import yaml
import sys
from pathlib import Path
from pandas.api.types import is_numeric_dtype

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.loggers import get_logger


def clean_data():
    with open('src/config.yaml', 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    df = pd.read_csv(config['data']['dataset_path'])

    df = df.drop_duplicates()

    for col in df.columns:
        if is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna('Unknown')

    prepared_path = config['data']['prepared_dataset_path']
    df.to_csv(prepared_path, index=False)


if __name__ == '__main__':
    clean_data()