import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def create_df(path : str):
    npload = np.load(path, allow_pickle=True)
    final_data_frame = pd.DataFrame(npload, columns=['feature', 'class', 'classID', 'fold'])
    # cast to int
    final_data_frame["fold"] = pd.to_numeric(final_data_frame["fold"])
    final_data_frame["classID"] = pd.to_numeric(final_data_frame["classID"])
    # delete class column
    del final_data_frame["class"]
    return final_data_frame
 
def train_val_test_split(dataframe):

    """
    Subsample data such that: 
     a training set consists of folds: 1, 2, 3, 4, 6
     a test set consists of folds: 5, 7, 8, 9, 10
    Return: X_train, y_train, X_val, y_val, X_test, y_test
    """

    train = dataframe[(dataframe["fold"] < 5) | (dataframe["fold"] == 6)] 
    test = dataframe[(dataframe["fold"] == 5) | (dataframe["fold"] >= 7)]
    print("Training folds: " + str([i for i in sorted(train["fold"].unique())]))
    print("Test folds: " + str([i for i in sorted(test["fold"].unique())]))
    
    # Creating a test set
    X_test = []
    y_test = []
    for fold in [5, 7, 8, 9, 10]:
        X_test.append(np.array(test['feature'][(test["fold"] == fold)].tolist()))
        y_test.append(to_categorical(np.array(test['classID'][(test["fold"] == fold)].tolist()), num_classes=10))
        
    # Creating a training and a validation set
    x = np.array(train['feature'].tolist())
    y = to_categorical(np.array(train['classID'].tolist()))
    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.20, random_state = 42)
    print("Size of training samples = ", X_train.shape[0])
    print("Size of validation samples = ", X_val.shape[0])
    print("Size of test samples = ", sum([len(i) for i in X_test]))
    
    return X_train, y_train, X_val, y_val, X_test, y_test
    