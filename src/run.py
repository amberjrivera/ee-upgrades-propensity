import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from transforms import add_labels, Preprocessing, BalanceClasses, DFselector
from pipeline import pipe


if __name__ == '__main__':
    # Load and label the data
    df = pd.read_csv('../data/city.csv', low_memory=False)
    df['assessor_id'] = df['assessor_id'].str[1:]
    df = add_labels(df)

    # Clean, drop, and engineer features. Impute missing values.
    # Impute missing values.
    clean = Preprocessing()
    df = clean.transform(df)

    # Handle class imbalance
    # pos_percent = 0.30 #add functionality and tinker
    balance = BalanceClasses()
    # balance = preprocessing.BalanceClasses(method=downsample, \
    # pos_class_percent=pos_percent)
    data = balance.transform(df)

    # Split the data
    y = data.pop('labels')
    X = data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    # Grid Search for the best classifier
    # TODO [insert gridsearchCV and param_grid]
    # print results
    # print("Best estimator found in search:")
    # print(grid_search.best_estimator_)
    #
    # print("Best score found in search was {}.".format(grid_search.best_score_))
    #
    # print("Best parameters found on training set:")
    # print(grid_search.best_params_)



    # Fit and Score any (interim) model
    model = pipe.fit(X_train, y_train)

    cv_folds = 3
    f1_weighted = round(cross_val_score(model, X_train, y_train, cv=cv_folds, \
    scoring='f1_weighted').mean(), 2)

    print("Average score found in CV: {}.".format(f1_weighted))
    print("Most important features found on training set:")
    print(pipe.steps[1][1].feature_importances_)



    # # Fit the (FINAL) model
    # clf = pipeline.pipe.fit(X_train, y_train)
    # print("Model is fit and ready to predict.")
    #
    # # Score the (FINAL) model
    # y_pred = pipe.predict(X_test)
    # print("Final results:)
    # print(classification_report(y_test, y_pred))









    # pickle and save final model
