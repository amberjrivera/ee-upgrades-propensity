import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import make_scorer, f1_score
from sklearn.decomposition import PCA, NMF
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from transforms import add_labels, Preprocessing, BalanceClasses, save_and_drop_ids, DFselector
from pipeline import pipe


if __name__ == '__main__':
    # Load and label the data
    df = pd.read_csv('../data/city.csv', low_memory=False)
    df['assessor_id'] = df['assessor_id'].str[1:]
    df = add_labels(df)

    # Clean, drop, and engineer features. Impute missing values.
    clean = Preprocessing()
    df = clean.transform(df)

    # Handle class imbalance
    pos_percent = 0.45
    balance = BalanceClasses(method='downsample', pos_percent=pos_percent)
    df = balance.transform(df)

    # Save and drop identifying info
    data, identity_df = save_and_drop_ids(df)

    # Split the data
    y = data.pop('labels')
    X = data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# -----------------------------------------------
    # Search for the best classifier
    pg =[
        {
            'classify': [RandomForestClassifier()],
            # 'classify__bootstrap': [True], #True
            # 'classify__class_weight': [None],
            'classify__criterion': ['entropy'], #'gini'
            'classify__max_depth': [5, 15, 25, 35], #5 (6, 8)
            'classify__max_features': ['auto', 25, 50], #50
            # 'classify__max_leaf_nodes': [None],
            # 'classify__min_impurity_decrease': [0.0],
            # 'classify__min_impurity_split': [None],
            'classify__min_samples_leaf': [10, 20, 30],
            'classify__min_samples_split': [5, 10, 15], #10
            # 'classify__min_weight_fraction_leaf': 0.0,
            'classify__n_estimators': [25, 50, 100, 200], #15, 20
            'classify__n_jobs': [-1],
            # 'classify__oob_score': [False], #False
            'classify__random_state': [42] #None
            # 'classify__verbose': [0],
            # 'classify__warm_start': [False]
        },
        {
            'classify': [GradientBoostingClassifier()],
            # 'classify__criterion': ['friedman_mse'],
            # 'classify__init': [None],
            'classify__learning_rate': [0.05], #0.1
            # 'classify__loss': ['deviance'],
            'classify__max_depth': [12],
            'classify__max_features': [15],
            'classify__max_leaf_nodes': [None, 10, 15], #None
            'classify__min_impurity_decrease': [0.01], #0.0
            # 'classify__min_impurity_split': [None],
            'classify__min_samples_leaf':[30],
            'classify__min_samples_split': [15],
            'classify__min_weight_fraction_leaf': [0.01],
            'classify__n_estimators': [80, 200, 600], #80;
            # 'classify__presort': ['auto'],
            # 'classify__random_state': [None],
            'classify__subsample': [0.9, 0.95, 0.975], #1.0
            # 'classify__verbose': [0],
            # 'classify__warm_start': [False]
        },
        ]

    # GridSearch for the best estimator (stratified built-in)
    # my_f1_scorer = make_scorer(f1_score, pos_label=1, average='binary')
    grid_search = GridSearchCV(pipe, param_grid=pg, cv=4, scoring='recall') #'f1'
    grid_search.fit(X_train, y_train)

    # print results:
    print("Best estimator found in search:")
    print(grid_search.best_estimator_)

    print("Best recall score found in search was {}.".format(grid_search.best_score_))

    print("Best parameters found on training set:")
    print(grid_search.best_params_)
# ------------------------------------------------------

    # Fit and score a training model
    # model = pipe.fit(X_train, y_train)
    #
    # cv_folds = StratifiedKFold(n_splits=4, random_state=42, shuffle=False) #so I can set a seed
    #
    # precision = round(cross_val_score(model, X_train, y_train, cv=cv_folds, \
    # scoring='precision').mean(), 2)
    #
    # recall = round(cross_val_score(model, X_train, y_train, cv=cv_folds, \
    # scoring='recall').mean(), 2)
    #
    # f1_weighted = round(cross_val_score(model, X_train, y_train, cv=cv_folds, \
    # scoring='f1_weighted').mean(), 2)
    #
    # print("Average scores found in CV were Precision: {0}, Recall: {1}, f1_weighted: {2} .".format(precision, recall, f1_weighted))
    # print("Most important features found on training set:")
    # # print(pipe.steps[2][1].feature_importances_)
    #
    # importances = [(score, name) for name, score in zip(X_train.columns, pipe.steps[1][1].feature_importances_)]
    #
    #
    # importances.sort(key=lambda tup: tup[0])
    # importances.reverse()
    # print(list(importances)[0:12])
    #
    # # Score the FINAL model
    # clf = pipe.fit(X_train, y_train)
    # print("Model is fit and ready to predict.")
    #
    # # Score the (FINAL) model
    # y_pred = clf.predict(X_test)
    # print("Final results:)
    # print(classification_report(y_test, y_pred))
    #
    #
    # Pickle and save FINAL model
