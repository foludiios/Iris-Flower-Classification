from fastapi import FastAPI
import uvicorn
from fastapi.responses import JSONResponse
from src.utils import features
import joblib
import os

app = FastAPI()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "Model1.pkl")

MODEL = joblib.load(MODEL_PATH)

@app.get("/")
def greet():
    return {"Hello, welcome to the Iris prediction API!"}

@app.post("/predict")
def pred(Input: features):
    sep_len = Input.sepal_length_cm
    sep_wid = Input.sepal_width_cm
    pet_len = Input.petal_length_cm
    pet_wid = Input.petal_width_cm

    vals = [[sep_len, sep_wid, pet_len, pet_wid]]
    predict = MODEL.predict(vals)
    return {"prediction": predict.item()} 