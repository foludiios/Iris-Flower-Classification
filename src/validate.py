from src.train import train, TopModels
from data.dataset import X_val, y_val, X_test, y_test
from sklearn.metrics import classification_report as cr
import joblib
from src.utils import Auto_metr
import os 

def choose():
    choice = input("Select 1 to use 'Default Choice' (i.e auto model), or model key to select specific model: ")
    if choice == '1':
        return TopModels.Auto(met=Auto_metr[0], avg=Auto_metr[1])
    else:
        return TopModels.Use(choice)
chosen = choose()

def validate(choice):
    preds = choice.predict(X_val)
    report = cr(y_val, preds)
    print(f"Validation Report: \n{report}")
    satisfied = input("Press 1 to proceed to test set (only if validation is satisfactory): ")
    if satisfied == '1':
        preds = choice.predict(X_test)
        report = cr (y_test, preds)
        print(f"Test Report: \n{report}")
        satisfactory = input("If Classification Report (on test set) is satisfactory, press 1 to output the model: ")
        if satisfactory == '1':
            name = input("Name model (include '.pkl'): ")
            save_path = "models"
            os.makedirs(save_path, exist_ok=True)
            model_file = os.path.join(save_path, name)
            joblib.dump(choice, model_file)
            print("Model.pkl saved!")
        else:
            print("Model not saved!")
    else:
        print("Try more hyperparameter tuning or model selection")


validate(chosen)