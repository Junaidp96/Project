# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = joblib.load('titanic_model.pkl')
scaler = StandardScaler()

app = FastAPI()

# Define the input data model
class TitanicInput(BaseModel):
    Pclass: int
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Sex_male: int  # One-hot encoding (Sex_male = 1 means male, 0 means female)
    Embarked_Q: int  # One-hot encoding (Embarked_Q = 1 means Q, 0 means not Q)
    Embarked_S: int  # One-hot encoding (Embarked_S = 1 means S, 0 means not S)

# Prediction endpoint
@app.post("/predict/")
def predict(data: TitanicInput):
    input_data = np.array([[data.Pclass, data.Age, data.SibSp, data.Parch, data.Fare,
                            data.Sex_male, data.Embarked_Q, data.Embarked_S]])
    input_data = scaler.transform(input_data)  # Apply the same scaling as in training
    prediction = model.predict(input_data)
    return {"survived": int(prediction[0])}

