from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_predict_stress():
    payload = {
        "age": 16,
        "gender": "female",
        "daily_social_media_hours": 4.5,
        "platform_usage": "Instagram",
        "sleep_hours": 6.0,
        "screen_time_before_sleep": 2.0,
        "academic_performance": 4.0,
        "physical_activity": 2.0,
        "social_interaction_level": "medium",

        "anxiety_level": 5.0,
        "addiction_level": 3.0,
        "depression_label": 0
    }
    response = client.post("/predict", json=payload)


    assert response.status_code == 200
    assert "predicted_stress_level" in response.json()