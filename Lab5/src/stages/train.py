import pandas as pd
import yaml
import joblib
import json
import sys
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.model_scripts.train import get_model
from src.loggers import get_logger

def train_model():
    with open('src/config.yaml', 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    train = pd.read_csv(config['data']['train_path'])
    test = pd.read_csv(config['data']['test_path'])

    target = config['train']['target_column']
    X_train = train.drop(columns=[target])
    y_train = train[target]
    X_test = test.drop(columns=[target])
    y_test = test[target]

    cat_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    model = get_model()
    model.fit(X_train, y_train, cat_features=cat_features)

    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, 'models/model.pkl')

    preds = model.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, preds),
        'f1_score_weighted': f1_score(y_test, preds, average='weighted')
    }

    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

if __name__ == '__main__':
    train_model()