from catboost import CatBoostClassifier
import yaml

def get_model():
    with open('src/config.yaml', 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    model = CatBoostClassifier(
        iterations=config['train']['iterations'],
        depth=config['train']['depth'],
        learning_rate=config['train']['learning_rate'],
        loss_function=config['train']['loss_function'],
        verbose=0
    )
    return model