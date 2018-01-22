#############################################
# Seeking best classifier on full feature matrix.
############################################

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import f1_score, classification_report
# from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
# from xgboost import XGBClassifier
# from catboost import CatBoostClassifier
from transforms import *
from attributes import Attributes


#CLEAN AND GET TO X
#load raw features data
df = pd.read_csv('../data/city.csv', low_memory=False)

#Clean up unique identifier
df['assessor_id'] = df['assessor_id'].str[1:]

# add labels
df = add_labels(df);

# quick clean
df = clean_and_drop(df);

# feature engineer
df = feature_engineer(df)

# impute numerical
df = impute(df)

# categorical columns to impute with mode
cols = ['exterior_wall_type', 'frame_type', 'heating_type', 'interior_wall_type', 'land_use']

df = cat_impute(df, cols)

# get dummies
dummy_cols = Attributes().get_cat_attribs()

df = pd.get_dummies(df, columns=dummy_cols, drop_first=True)

# undersample majority class to handle class imbalance
df = balance(df)

# Get to X
y = df.pop('labels')
X = df

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

#Scale continuous features
cols_to_scale = Attributes().get_num_attribs()

df = scale(df, cols_to_scale)

#COMPARE MODELS
model_names  = [
           # "Logistic Regression",
           # "K-nearest Neighbors",
           # # "Naive Bayes",
           # "Linear SVM",       #freezes up
           # "RBF SVM",          #low performing
           # "Gaussian Process", #freezes up
           # "Decision Tree",
           # "Random Forest",
           # "Extra Trees",
           "GradientBoosting",
           # "AdaBoost",
           # "XGBoost",
           # "CatBoost",
           # "LDA",
           # "QDA",
           # "Neural Net"
           ]

classifiers = [
    # LogisticRegression(penalty='l1', tol=0.001),
    # KNeighborsClassifier(n_neighbors=3),
    # GaussianNB(),
    # SVC(kernel='linear', C=0.25),
    # SVC(kernel='rbf', gamma=2, C=1),
    # GaussianProcessClassifier(1.0 * RBF(1.0)),
    # DecisionTreeClassifier(max_depth=5, random_state=None),
    # RandomForestClassifier(max_depth=5, n_estimators=10, max_features=20, random_state=None),
    # ExtraTreesClassifier(n_estimators=10, criterion='entropy', max_features=10),
    GradientBoostingClassifier(
        subsample=0.85, #try 0.6
        n_estimators=600, #try 200, 400
        min_weight_fraction_leaf=0.01,
        min_samples_split=15,
        min_samples_leaf=30,
        min_impurity_decrease=0.01,
        max_leaf_nodes=None, #try 10, 15
        max_features=15,
        max_depth=12,
        learning_rate=0.05
        )
    # AdaBoostClassifier(random_state=None),
    # XGBoost()
    # CatBoostClassifier(random_seed=rs),
    # LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto', n_components=10),
    # QuadraticDiscriminantAnalysis(reg_param=0.5, store_covariance=True),
    # MLPClassifier(alpha=1, tol=0.001, random_state=None)
    ]

cv_folds = 4

# Create summary dataframe "score board" to compare scores
scores = pd.DataFrame(columns=['Models', 'accuracy', 'f1_weighted', 'combined'])
scores['Models'] = model_names

for model_name, clf in zip(model_names, classifiers):
    # For each cv_fold, train, predict, score the selected classifier
    # Compare models based on mean accuracy and mean f1_score
    accuracy = round(cross_val_score(clf, X_train, y_train, cv=cv_folds, scoring='accuracy').mean(), 2)

    f1_weighted = round(cross_val_score(clf, X_train, y_train, cv=cv_folds, scoring='f1_weighted').mean(), 2)

    # Update the score board
    scores.loc[scores['Models'] == model_name, 'accuracy'] = accuracy
    scores.loc[scores['Models'] == model_name, 'f1_weighted'] = f1_weighted

    # # Get feature importances where possible
    # # Random Forest
    # if model_name == "Random Forest":
    #     forest = clf.fit(X_train, y_train) #TODO this is cv agnostic right now...
    #     importances = forest.feature_importances_
    #
    #     std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    #
    #     indices = np.argsort(importances)[::-1]
    #     print("Feature ranking: ")
    #     # for f in range(X_train.shape[1]):
    #     #     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    #
    #     for f in indices:
    #         print("Feature", f, ":", importances[f], X_train.columns[f])
    #
    # # Linear Discriminat Analysis
    # if model_name == "LDA":
    #     m = clf.fit(X_train, y_train)
    #     print("Covariances:")
    #     print(m.covariance_)
    #
    # # Quadratic Discriminat Analysis
    # if model_name == "QDA":
    #     m = clf.fit(X_train, y_train)
    #     print("Covariances:")
    #     print(m.covariance_)
    #
    # # Logistic regression
    # if model_name == 'Logistic Regression":
    #     m = clf.fit(X_train, y_train) #TODO this is cv agnostic right now... Also not interpretable
    #     print(m.coef

scores['combined'] = (scores['accuracy'] + scores['f1_weighted']) / 2

print('\nComparing Classifiers...\n')
print(scores)


# TUNE BEST MODEL
# gb = GradientBoostingClassifier() #out of bag estimate
# parameters = {
#     'learning_rate': [0.025, 0.05, 0.075],
#     'n_estimators': [200, 400, 600, 800],
#     'max_depth': [3, 6, 9, 12, 15],
#     'min_samples_split': [2, 6, 12, 18, 24],
#     'min_samples_leaf': [25, 30, 35],
#     'min_weight_fraction_leaf': [0.0, 0.005, 0.01, 0.025],
#     'subsample': [0.6, 0.75, 0.85, 0.95],
#     'max_features': [15, 20, 25],
#     'max_leaf_nodes': [10, 15, 20, None],
#     'min_impurity_decrease': [0.0, 0.001, 0.01],
# }

# clf = GridSearchCV(gb, param_grid=parameters, scoring='f1_weighted', n_jobs=2, cv=None, refit=True)
# clf.fit(X_train, y_train)
#
# print("Best parameters set found on training set:")
# print(clf.best_params_)
#
# print("Most important features found on training set:")
# print(clf.feature_importances_)

# --------------------
# clf_two = RandomizedSearchCV(gb, param_distributions=parameters, n_iter=100, scoring='f1', n_jobs=2, cv=None, refit=True)
#
# clf_two.fit(X_train, y_train)
# best_score = clf_two.best_score_
# print("Best score was {}.".format(best_score))
# print("Best parameters found on training set:")
# print(clf_two.best_params_)
