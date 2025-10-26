import joblib
from data.dataset import X_test, y_test, df
import os
from sklearn.metrics import classification_report as cr
 

def predict(df, model_name):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(BASE_DIR, "models", model_name)
    model = joblib.load(MODEL_PATH)
    RX, Ry = df.randtest()
    prediction = model.predict(RX)
    print(f"Prediction: {prediction}")
    print(cr(Ry, prediction))

predict(df, 'Model1.pkl')