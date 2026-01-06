from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()

lr_model = joblib.load("src/logistic_pipeline.pkl")
rf_model = joblib.load("src/rf_pipeline.pkl")

@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API Running"}

@app.post("/predict/logistic")
def predict_logistic(data: dict):
    df = pd.DataFrame([data])
    df = df.apply(pd.to_numeric)
    prediction = lr_model.predict(df)[0]
    return {"prediction": int(prediction)}

@app.post("/predict/rf")
def predict_rf(data: dict):
    df = pd.DataFrame([data])
    df = df.apply(pd.to_numeric)
    prediction = rf_model.predict(df)[0]
    return {"prediction": int(prediction)}
