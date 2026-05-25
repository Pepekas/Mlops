from fastapi import FastAPI, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
import pickle
import pandas as pd
import logging
import uvicorn
from typing import Union
from database import get_db, StressLog

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    logger.info("CatBoost Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")

app = FastAPI(title="Teen Stress Level API")


class TeenFeatures(BaseModel):
    age: int
    gender: str
    daily_social_media_hours: float
    platform_usage: str
    sleep_hours: float
    screen_time_before_sleep: float
    academic_performance: float
    physical_activity: float
    social_interaction_level: str
    anxiety_level: float
    addiction_level: float
    depression_label: Union[str, int]


@app.post("/predict", summary="Predict stress level")
async def predict(teen: TeenFeatures, db: Session = Depends(get_db)):
    try:
        input_data = pd.DataFrame([teen.model_dump()])

        prediction = int(model.predict(input_data)[0])

        db_log = StressLog(
            gender=teen.gender,
            sleep_hours=teen.sleep_hours,
            social_media_hours=teen.daily_social_media_hours,
            predicted_stress=prediction
        )
        db.add(db_log)
        db.commit()

        return {"predicted_stress_level": prediction}

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005)