import sys
import os
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.loggers import get_logger


def data_split():
    logger = get_logger('DATA_SPLIT')

    with open('src/config.yaml', 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    data_frame = pd.read_csv(config['data']['prepared_dataset_path'])

    train_dataset, test_dataset = train_test_split(
        data_frame,
        test_size=config['train']['test_size'],
        random_state=config['train']['random_state']
    )

    train_dataset.to_csv(config['data']['train_path'], index=False)
    test_dataset.to_csv(config['data']['test_path'], index=False)


if __name__ == "__main__":
    data_split()