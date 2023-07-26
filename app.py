# import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from datetime import datetime
import numpy as np


app = FastAPI()
## Using Pydantic lib, defining the data type for all the inputs
class model_input(BaseModel):
    
    ph : float
    temperature : float 
    turbidity : float

model = joblib.load('modelss5.pkl')

@app.get('/')
async def home():
    return {'Hallo Manis, Fast Api nya sudah'}

@app.post('/weekly/')
async def week1(item:model_input):
    datetime_week = datetime.now().strftime("%Y-%m-%d")
    input = np.array([[item.ph,item.temperature,item.turbidity]])
    prediction = model.predict(input)[0]
    if prediction == 0 : 
        result = 'Bersih'
    else:
        result = 'Keruh'
    return {
        'input':{'week':item},
        'result':result,
        'tanggal':datetime_week
    }
@app.post('/month/')
async def month(item:model_input):
    datetime_month = datetime.now().strftime("%Y-%m-%d")
    input = np.array([[item.ph,item.temperature,item.turbidity]])
    prediction = model.predict(input)[0]
    if prediction == 0 : 
        result = 'Bersih'
    else:
        result = 'Keruh'
    return {
        'input':{'month':item},
        'result':result,
        'tanggal':datetime_month
    }
@app.post('/predict/{ph}/{temperature}/{turbidity}')
def predict_water_quality(ph: float, temperature: float, turbidity: float):
    # Ubah input data menjadi array numpy
    # input_data = [[data.ph, data.temperature, data.turbidity]]
    input_data = [[ph,temperature,turbidity]]
    
    # Lakukan prediksi dengan model
    prediction = model.predict(input_data)[0]
    
    # Ubah hasil prediksi menjadi label
    if prediction == 0:
        result = "Bersih"
    else:
        result = "Keruh"
    
    current_datetime = datetime.now().strftime("%Y-%m-%d")

    # Kembalikan hasil prediksi
    return {
        "input": {
            "ph": ph,
            "temperature": temperature,
            "turbidity": turbidity
        },
        "prediction": result,
        "datetime": current_datetime}

@app.get('/predict/{ph}/{temperature}/{turbidity}')
def get_prediction(ph: float, temperature: float, turbidity: float):
    # Ubah input data menjadi array numpy
    input_data = [[ph, temperature, turbidity]]

    # Lakukan prediksi dengan model
    prediction = model.predict(input_data)[0]

    # Ubah hasil prediksi menjadi label
    if prediction == 0:
        result = "Bersih"
    else:
        result = "Keruh"
        
    current_datetime = datetime.now().strftime("%Y-%m-%d")
    
    # Kembalikan hasil prediksi
    return {
        "input": {
            "ph": ph,
            "temperature": temperature,
            "turbidity": turbidity
        },
        "prediction": result,
        "datetime": current_datetime
    }
if __name__  == '__main__':
  uvicorn.run("app:app", host= "localhost", port=5000,reload=True)