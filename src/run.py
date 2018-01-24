import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import make_scorer, f1_score
from sklearn.decomposition import PCA, NMF
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from transforms import add_labels, Preprocessing, BalanceClasses, DFselector
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
    # pos_percent = 0.25 #add functionality and tinker
    balance = BalanceClasses()
    # balance = preprocessing.BalanceClasses(method=downsample, \
    # pos_class_percent=pos_percent)
    data = balance.transform(df)

    # Split the data
    y = data.pop('labels')
    X = data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# ---------------------------------------------------------
    # Search for the best classifier and dim reduction
    pg =[
        {
            'classify': [RandomForestClassifier()],
            'classify__bootstrap': [True], #True
            # 'classify__class_weight': [None],
            'classify__criterion': ['entropy'], #'gini'
            'classify__max_depth': [5], #5 (6, 8)
            'classify__max_features': ['auto'], #50
            # 'classify__max_leaf_nodes': [None],
            # 'classify__min_impurity_decrease': [0.0],
            # 'classify__min_impurity_split': [None],
            'classify__min_samples_leaf': [20],
            'classify__min_samples_split': [10], #10
            # 'classify__min_weight_fraction_leaf': 0.0,
            'classify__n_estimators': [20], #15, 20
            'classify__n_jobs': [2] #1
            # 'classify__oob_score': [False], #False
            # 'classify__random_state': None,
            # 'classify__verbose': [0],
            # 'classify__warm_start': [False]
        },
        # {
        #     'reduce_dim': [PCA(), NMF()],
        #     'reduce_dim__n_components': [2, 10, 20],
        # },
        # {
        #     'reduce_dim': [SelectKBest()],
        #     'reduce_dim__k': [2, 10, 20],
        # }
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
            'classify__n_estimators': [600, 400], #80; try 200
            # 'classify__presort': ['auto'],
            # 'classify__random_state': [None],
            'classify__subsample': [0.85, 0.6], #1.0
            # 'classify__verbose': [0],
            # 'classify__warm_start': [False]
        },

        ]

    # GridSearch for for the best estimator
    my_f1_scorer = make_scorer(f1_score, pos_label=1, average='binary')
    grid_search = GridSearchCV(pipe, param_grid=pg, cv=3, scoring='f1')
    grid_search.fit(X_train, y_train)

    # print results:
    print("Best estimator found in search:")
    print(grid_search.best_estimator_)

    print("Best f1 score found in search was {}.".format(grid_search.best_score_))

    print("Best parameters found on training set:")
    print(grid_search.best_params_)
# ------------------------------------------------------

    # Fit and score a training model
    # model = pipe.fit(X_train, y_train)
    #
    # cv_folds = 4
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

    # # Score the FINAL model
    # clf = pipe.fit(X_train, y_train)
    # print("Model is fit and ready to predict.")
    #
    # # Score the (FINAL) model
    # y_pred = clf.predict(X_test)
    # print("Final results:)
    # print(classification_report(y_test, y_pred))


    # Pickle and save FINAL model
