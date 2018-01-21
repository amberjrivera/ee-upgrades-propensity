import pandas as pd
import numpy as np
from preprocessing import *
from pipeline import *


if __name__ == '__main__':
    #load and label the data
    df = pd.read_csv('../data/city.csv')
    df['assessor_id'] = df['assessor_id'].str[1:]
    df = preprocessing.add_labels(df)

    # Clean with Preprocessing class here, outside of pipeline.
    clean = preprocessing.Preprocessing()
    df = clean.fit_transform(df)

    # split the data
    y = df.pop('labels')
    X = df
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
