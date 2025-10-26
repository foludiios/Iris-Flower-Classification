from data.make_dataset import data, load_dataset
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(BASE_DIR, "IrisData.csv")
dataset = load_dataset(FILE_PATH)
df = data(dataset)
X_train, X_test, X_val, y_train, y_test, y_val =df.split(0.2,0.2,rand_st=42)