from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from time import time
from threading import Lock

# Load the trained model and scaler
model = joblib.load('titanic_model.pkl')
scaler = joblib.load('scaler.pkl')  # Load the saved scaler

app = FastAPI()

# Global variable to track the number of requests
request_counter = 0
spike_counter = 0
last_spike_time = time()

# Lock to prevent race conditions when incrementing the counters
counter_lock = Lock()

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
    global request_counter, spike_counter, last_spike_time

    # Increment the total request count
    with counter_lock:
        request_counter += 1

        # Track spikes (e.g., if more than 100 requests in 60 seconds)
        current_time = time()
        if current_time - last_spike_time <= 60:  # Spike within the last minute
            spike_counter += 1
        else:
            last_spike_time = current_time
            spike_counter = 1  # Reset spike counter for the new minute

    try:
        # Create input array for prediction
        input_data = np.array([[data.Pclass, data.Age, data.SibSp, data.Parch, data.Fare,
                                data.Sex_male, data.Embarked_Q, data.Embarked_S]])

        # Apply the same scaling as in training
        input_data = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_data)
        
        return {"survived": int(prediction[0])}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in prediction: {str(e)}")

# Endpoint to retrieve the number of requests and spikes
@app.get("/status/")
def get_status():
    global request_counter, spike_counter
    return {
        "total_requests": request_counter,
        "spikes_last_minute": spike_counter
    }

