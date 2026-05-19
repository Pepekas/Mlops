import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import mlflow
import mlflow.sklearn

def train():
    train_df = pd.read_csv('data/train.csv')
    X_train = train_df.drop('MedHouseVal', axis=1)
    y_train = train_df['MedHouseVal']
    
    model = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
    
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("lab4_experiment")
    
    with mlflow.start_run() as run:
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, "model")
        model_uri = f"runs:/{run.info.run_id}/model"
        print(model_uri)

if __name__ == "__main__":
    train()