from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from data.dataset import y_train
from pydantic import BaseModel

class features(BaseModel):
    sepal_length_cm: float
    sepal_width_cm: float
    petal_length_cm: float
    petal_width_cm: float

param1=[      # list where item at index 0 is a dictionary that provides parameters for
    {        # each different model to be used,
            'RandomForest':{
                'n_estimators':[50,60,70,80,90],
                'max_depth':[3,4,5,6],
                'min_samples_split':[6,7,8],
                'min_samples_leaf':[3,4,5],
                'max_features':[2]},
            
            'DecisionTree':{
                'max_depth':[3,4,5],
                'min_samples_split':[3,5,7,9],
                'min_samples_leaf':[3,4,5],
                'max_features':["sqrt"]}
        },
    
    {     # and item at index 1 maps model parameter keys in item index 0 to actual sklearn models
            'RandomForest':RandomForestClassifier(random_state=42),
            'DecisionTree':DecisionTreeClassifier(random_state=42),
            'Default_DecisionTree':DecisionTreeClassifier()
        }
]


def gen_mlflow_metrics(reportd, metr='recall'):
    out = {
        'Accuracy' : reportd['accuracy'],
        'macro avg' : reportd['macro avg'][metr],
        'weighted avg' : reportd['weighted avg'][metr]
    }
    for i in y_train.unique():
        key = str(i)
        out[key] = reportd[key][metr]
    return out 

Auto_metr=['accuracy', 'weighted avg']