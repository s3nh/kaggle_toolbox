import pandas as pd
import os 
import numpy as np 
import matplotlib.pyplot as plt
import lightgbm 
from sklearn import * 
def listfiles():
    datalist = os.listdir('data/')
    return datalist

def load_data():
    time = time.time()
    
    train_data = pd.read_csv('data/train.csv')
    test_data =  pd.read_csv('data/test.csv')
    print("Rozmiar treningowego zestawu danych : \n {}".format(train_data.shape))
    print("\n Rozmiar testowego zestaw danych: \n {}".format(test_data.shape))

    
    return train_data, test_data

def unique_val(data):
    for col in data.columns:
        print("Liczba unikalnych wartosci dla zmiennej \n {} wynosi \n {}".format(col, len(np.unique(data.col))))
        

if __name__ == '__main__':
    print("What files are in data folder")
    data=listfiles()
    print(data)
    train_data, test_data = load_data()
    
    print("Kolumny treningowego zestawu danych {}".format(train_data.shape))
    train_data.columns

    unique_val(train_data)

















































