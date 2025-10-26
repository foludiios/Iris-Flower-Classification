from src.models import models 
from src.utils import param1 as p, Auto_metr
import joblib
 

def train():
    TModels = models(p)
    return TModels
TopModels = train()

def evaluate(Auto_met=Auto_metr[0], Auto_avg=Auto_metr[1]):
    """Auto_met specifies what metric (precision, recall, f1-score, or accuracy) the auto_model
    should optimize in model selection. Auto_avg specifies which averaging method the auto_model is to 
    optimize in model selection."""

    auto_model = TopModels.Auto(met=Auto_met, avg=Auto_avg)
    
    print(f"MODELS PASSED IN CURRENT PARAM (from utils.py): {list(TopModels.map.values())}")
    print(f"\n\nBest Hyperparameter Combinations: ")
    TopModels.BestModels() 
    print(f"\n\nClassification Reports: ")
    TopModels.Val_reports()
    print(f"\nDefault Choice: \n{auto_model}")

if __name__ == "__main__":
    evaluate()
