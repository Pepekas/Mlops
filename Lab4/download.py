import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import os

def download_and_prepare():
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    os.makedirs('data', exist_ok=True)
    train.to_csv('data/train.csv', index=False)
    test.to_csv('data/test.csv', index=False)

if __name__ == "__main__":
    download_and_prepare()