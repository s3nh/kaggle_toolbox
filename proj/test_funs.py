import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

def sample_train_data(dataset ,target,data_len, resp = True ):
    """
    task is the simples
    undersample to minority class length
    """
    np.random.seed(222)
    ixes = np.random.choice(dataset.index, data_len, replace = False)
    print(ixes)
    under_df = dataset.iloc[ixes]
    if resp==True:
        under_target = target.iloc[ixes]
        return under_df, under_target
    else:
        return under_df

if __name__ == "__main__":
    print("*"*30)
    print("Zaczytywanie zestawu danych")
    train_df=pd.read_csv("data/train.csv")
    print("train shape \n {}".format(train_df.shape))
    test_df=pd.read_csv("data/test.csv")
    print("test shape \n {}".format(test_df.shape))
    y = train_df.loc[:,"target"]
    x = train_df.drop("target", axis=1)

    under_df, under_target = sample_train_data(x , y, int(x.shape[0]/2))
    print("Under df shape".format(under_df.shape))
    print(under_df)
