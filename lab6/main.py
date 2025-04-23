import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Union
from process_data import process_user_data


class UserData(BaseModel):
    age: int = Field(alias='Age')
    gender: str = Field(alias='Gender')
    occupation: str = Field(alias='Occupation')
    sleep_hours: float = Field(alias='Sleep Hours')
    physical_activity: float = Field(alias='Physical Activity (hrs/week)')
    caffeine_intake: float = Field(alias='Caffeine Intake (mg/day)')
    alcohol_consumption: float = Field(alias='Alcohol Consumption (drinks/week)')
    smoking: str = Field(alias='Smoking')
    family_history_of_anxiety: str = Field(alias='Family History of Anxiety')
    stress_level: int = Field(alias='Stress Level (1-10)')
    heart_rate: int = Field(alias='Heart Rate (bpm)')
    breathing_rate: int = Field(alias='Breathing Rate (breaths/min)')
    sweating_level: int = Field(alias='Sweating Level (1-5)')
    dizziness: str = Field(alias='Dizziness')
    medication: str = Field(alias='Medication')
    therapy_sessions: int = Field(alias='Therapy Sessions (per month)')
    recent_major_life_event: str = Field(alias='Recent Major Life Event')
    diet_quality: int = Field(alias='Diet Quality (1-10)')


app = FastAPI()

@app.post('/predict/')
async def predict(user_data: UserData) -> dict[str, Union[float, str]]:

    try:
        model = joblib.load('models/model.pkl')
        data = user_data.model_dump(by_alias=True)
        processed_data = process_user_data(data)
        prediction = model.predict(processed_data)
        return {'prediction': prediction[0]}
    
    except Exception as e:
        return {'error': e}
    
