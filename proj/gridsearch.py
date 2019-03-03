import numpy as np 
import pandas as pd
from vm_gridsearch import * 
import matplotlib.pyplot as plt
import scipy
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold, train_test_split
from sklearn.metrics  import accuracy_score, roc_curve, auc

gradient_params = {
    "loss": ["deviance"], 
    "learning_rate": [ 0.01, 0.025, 0.05, 0.1, 0.2], 
    # Minimum number of samples required to split an internal node:
    "min_samples_split": [0.005, 0.05, 0.1, 0.4], 
    # Minimum number of samples required to be at a leaf node
    #"min_samples_leaf": np.linspace(0, 24, 4, dtype = 'int32'), 
    "max_depth": [3, 5, 7, 11, 13], 
    # Omitted min/max impurity split
    "max_features": np.linspace(0.1, 1.0,  10), 
    "verbose": [1],
    # Number of boosting estimators
    "n_estimators": np.linspace(10, 1000 , 10, dtype='int32'),
    #subsample of max_features
    "subsample": np.linspace(0.1, 1, 10) 
    
    }

def read_data():
    train_df = pd.read_csv('data/train.csv')
    test_df  = pd.read_csv('data/test.csv')
    return train_df, test_df

if __name__ == "__main__":
    print("Import worked fine")
    train_df, test_df=read_data()
    print("params for gradient boosting")
    print(gradient_params)
    
    y=train_df.loc[:, 'target']
    X=train_df.drop(['target', 'ID_code'], axis=1)
    X_train, X_valid, y_train, y_valid = train_test_split(
            X,y,test_size=0.2)
    scores= ['precision, recall']
    for score in scores:
        clf=GridSearchCV(GradientBoostingClassifier(),gradient_params, 
                cv=10 )
        clf.fit(X_train, y_train)
        clf.best_params_

        means=clf.cv_results_['mean_test_score']
        stds=clf.cv_results_['std_test_score']
    print(means)
    print(results)

      
