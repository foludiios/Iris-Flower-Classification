import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def load_dataset(data):
    if 'Unnamed: 0' in pd.read_csv(data).columns:
        data = pd.read_csv(data).drop(['Unnamed: 0'],axis=1)
    else:
        data=pd.read_csv(data)
    return data 

class data():
    def __init__(self,dataset):
        self.data = dataset

    def split(self, test_size, val_size=None, rand_st=5):
        X=self.data[self.data.columns[:-1]]
        y=self.data[self.data.columns[-1]]
        self.test_size, self.val_size, self.rand_st, self.X, self.y = test_size, val_size, rand_st, X, y
        if val_size == None:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rand_st)
            print("WARNING!! This only has training and test sets, we highly recommend including validation a set")
            return X_train, X_test, y_train, y_test
        else:
            X_train_temp, X_test, y_train_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=rand_st)
            val_size = (val_size / (1-test_size))
            X_train, X_val, y_train, y_val = train_test_split(X_train_temp, y_train_temp, test_size=val_size,
                                                              random_state=rand_st)
            return X_train, X_test, X_val, y_train, y_test, y_val

    def randtest(self):
        try:
            RX_train, RX_test, Ry_train, Ry_test = train_test_split(self.X,self.y,test_size=self.test_size,
                                                                   random_state=np.random.randint(1,2**31))
            return RX_test, Ry_test
        except AttributeError:
            print("Please use .split() and it's parameters to split data into fixed sets first")