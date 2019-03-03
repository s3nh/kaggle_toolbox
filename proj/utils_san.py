from __future__ import print_function


import lightgbm
from sklearn import * 
import pandas as pd 
import os 
import re 
import time 
from scipy import interp

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import lightgbm as lgb

# What to do 

class  SantanderGreed(object):

    def __init__(self, trainpath ,testpath, train_ratio):
        self.trainpath = trainpath
        self.testpath  = testpath
        self.train_ratio = train_ratio
        print('testpath : {}'.format(self.testpath))
        print('\n trainpath: {}'.format(self.trainpath))
        
    def load_data(self):
        train_df = pd.read_csv(self.trainpath)
        test_df  = pd.read_csv(self.testpath)
        return train_df, test_df

    def data_target(self, data, target_name = 'target'):
        if target_name in data.columns:
            y = data.loc[:, target_name]
            X = data.drop(target_name, axis = 1)
            train_X, test_X , train_y, test_y = train_test_split(X, y, test_size =self.train_ratio)
      
        else:
            AssertionError("There is no {} variable").format(target_name)
        return train_X, test_X, train_y, test_y

    def sample_train_data(dataset, class_len ):
        """
        task is the simples
        undersample to minority class length
        """
        np.random.seed(222)
        ixes = np.random.choice(dataset.index, class_len)
        under_df = data.ix[ixes]
        return under_df


        
def main():
    sangreed = SantanderGreed(trainpath='data/train.csv', 
            testpath='data/test.csv', train_ratio = 0.2)
    train_df, test_df = sangreed.load_data()
    print("Return valid dataset")
    train_X, test_X, train_y,  test_y = sangreed.data_target(train_df, target_name = 'target')
    print("Treningowy zestaw danych {}".format(train_X.shape))
    print("Testowy zestaw danych {}".format(test_X.shape))
        
if __name__ == '__main__':
    main()




