#############################################
# Seeking best classifier on full feature matrix.
############################################

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
# from xgboost import XGBClassifier
# from catboost import CatBoostClassifier
from transforms import add_labels, Preprocessing, extract_bt_df
from attributes import Attributes


# Load and label the data
df = pd.read_csv('../data/city.csv', low_memory=False)
df['assessor_id'] = df['assessor_id'].str[1:]
df = add_labels(df)

# Clean, drop, and engineer features. Impute missing values.
clean = Preprocessing()
df = clean.transform(df)

# Extract subset for backtesting map
data = extract_bt_df(df)

# Scale numerical features
cols_to_scale = Attributes().get_num_attribs()
scaler = RobustScaler()
data[cols_to_scale] = scaler.fit_transform(data[cols_to_scale])

# Split the data
y = data.pop('labels')
X = data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# Balance classes before training
# sampler = RandomOverSampler(random_state=42, ratio='auto') #50/50
sampler = RandomUnderSampler(random_state=42, ratio='auto') #50/50
# sampler = SMOTE(random_state=42, ratio='auto', n_jobs=-1)  #50/50
X_train_res, y_train_res = sampler.fit_sample(X_train, y_train)

#COMPARE MODELS
names  = [
           # "Logistic Regression",
           # "K-nearest Neighbors",
           # "Decision Tree",
           "Random Forest",
           "Extra Trees",
           "GradientBoosting",
           # "AdaBoost",
           # "LDA",
           # "Neural Net"
           ]

classifiers = [
    # LogisticRegression(penalty='l1', tol=0.001),
    # KNeighborsClassifier(
    #     n_neighbors=3,
    #     algorithm='auto',
    #     weights='distance',
    #     leaf_size=30,
    #     p=2,
    #     n_jobs=-1),
    # DecisionTreeClassifier(max_depth=5, random_state=42),
    RandomForestClassifier(
        bootstrap=True,
        criterion='entropy',
        max_depth=5,
        n_estimators=200,
        max_features='auto',
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1,
        oob_score=False),
    ExtraTreesClassifier(
        n_estimators=10,
        criterion='entropy',
        max_features=10,
        n_jobs=-1,
        class_weight='balanced_subsample',
        oob_score=False),
    GradientBoostingClassifier(
        subsample=0.95,
        n_estimators=200,
        min_weight_fraction_leaf=0.01,
        min_samples_split=15,
        min_samples_leaf=30,
        min_impurity_decrease=0.01,
        max_leaf_nodes=None,
        max_features=15,
        max_depth=12,
        learning_rate=0.05,
        random_state=42
        ),
    # AdaBoostClassifier(
    #     random_state=42,
    #     algorithm='SAMME.R',
    #     n_estimators=50,
    #     learning_rate=1.0),
    # LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto', n_components=10),
    # MLPClassifier(alpha=1, tol=0.001, random_state=None)
    ]

cv_folds = StratifiedKFold(n_splits=4, random_state=42, shuffle=False) #so I can set a seed

# Create summary dataframe "score board" to compare scores
scores = pd.DataFrame(columns=['Classifier', 'accuracy', 'precision', 'recall', 'f1'])
scores['Classifier'] = names

for name, clf in zip(names, classifiers):
    # For each cv_fold, train, predict, score the selected classifier

    accuracy = round(cross_val_score(clf, X_train_res, y_train_res, cv=cv_folds, \
    scoring='accuracy').mean(), 2)

    precision = round(cross_val_score(clf, X_train_res, y_train_res, cv=cv_folds, \
    scoring='precision').mean(), 2)

    recall = round(cross_val_score(clf, X_train_res, y_train_res, cv=cv_folds, \
    scoring='recall').mean(), 2)

    f1_score = round(cross_val_score(clf, X_train_res, y_train_res, cv=cv_folds, \
    scoring='f1').mean(), 2)

    # Update the score board
    scores.loc[scores['Classifier'] == name, 'accuracy'] = accuracy
    scores.loc[scores['Classifier'] == name, 'precision'] = precision
    scores.loc[scores['Classifier'] == name, 'recall'] = recall
    scores.loc[scores['Classifier'] == name, 'f1'] = f1_score


print('\nComparing Classifiers...\n')
print(scores)
