import pandas as pd
import numpy as np
from transforms import *
from pipeline import *


if __name__ == '__main__':
    #load and label the data
    df = pd.read_csv('../data/city.csv')
    df['assessor_id'] = df['assessor_id'].str[1:]
    df = transforms.add_labels(df)

    # Clean, drop, and engineer features. Impute missing values.
    # Impute missing values.
    clean = transforms.Preprocessing()
    df = clean.transform(df)

    # Handle class imbalance
    # pos_percent = 0.30 #add functionality and tinker
    balance = transforms.BalanceClasses()
    # balance = preprocessing.BalanceClasses(method=downsample, \
    # pos_class_percent=pos_percent)
    data = balance.transform(df)

    # split the data
    y = data.pop('labels')
    X = data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    # search for the best classifier
    # print results

    # tune the best classifier
    # print hyperparameters

    # fit the final model
    # print status update, "model is fitted"
    pipeline.pipe.fit(X_train, y_train)

    # score final model
    # print score
    y_pred = pipe.transform(X_test) #check this...might need to be.fit_transform


    # pickle and save final model
