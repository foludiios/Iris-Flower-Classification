from data.dataset import df, dataset, X_train, X_test, X_val, y_train, y_test, y_val
from data.make_dataset import data 
from src.models import models
from src.utils import param1
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.train import train

def test_load():
    df = data(dataset)
    assert dataset is not None, "dataset is none"
    assert df is not None
    assert len(X_train) + len(y_train) + len(X_val) + len(y_val) + len(X_test) + len(y_test) == len(dataset) 

def test_models():
    mod = models(param1)
    assert mod is not None
    assert hasattr(mod, "predict")

def test_metric(met=accuracy_score):
    choice = train()
    pred = choice.predict(X_val)
    if met == accuracy_score:
        value = met(y_val, pred)
    else:
        value = met(y_val, pred, average='macro')
    assert value >= 0.95, "Metric value is too low"
