#############################################
# Quick and dirty to get a working model,
# and get a feel for strength of PRIZM data.
############################################

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import RobustScaler
# from sklearn.metrics import f1_score, classification_report
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



#CLEAN
# Predict using only PRIZM fields and a few others
df = pd.read_csv('../data/city.csv', usecols=[
'assessor_id',
'prizm_premier',
# 'prizm_social_group',
# 'prizm_lifestage',
'nrel_attached_garage',
'nrel_foundation',
'nrel_heating_fuel',
'nrel_size',
'nrel_stories',
'nrel_vintage',
'primary_building_type',
'year_built',
'last_sale_price'
])

df['assessor_id'] = df['assessor_id'].str[1:]
df.drop(df[df.year_built == 2017].index, inplace=True)
df['last_sale_price'].fillna(df['last_sale_price'].median(), inplace=True);

#MAKE DUMMIES
dummies = [
'prizm_premier',
# 'prizm_social_group',
# 'prizm_lifestage',
'nrel_attached_garage',
'nrel_foundation',
'nrel_heating_fuel',
'nrel_size',
'nrel_stories',
'nrel_vintage',
'primary_building_type',
]

df = pd.get_dummies(df, columns=dummies, drop_first=True) #splits 9 --> 57 cols


#ADD LABELS
#make list of addresses that have done any upgrade
positives = pd.read_csv('../data/Upgrade_Data.csv', usecols=[1], squeeze=True).str[1:].tolist()

positives = set(positives) #4864 homes in county

#make labels col based on membership in positives set
df['labels'] = df.apply(lambda row: 1 if row['assessor_id'] in positives else 0, axis=1)


#PREPARE TO MODEL
# undersample majority class to handle class imbalance (9/91 pos/neg)
num_pos = len(df[df['labels'] == 1])
num_neg = df.shape[0] - num_pos
num_to_drop = num_neg - num_pos #gets to 50/50 balance

df.drop(df.query('labels == 0').sample(n=num_to_drop).index, inplace=True)

y = df.pop('labels')
X = df

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

#scale continuous
cols_to_scale = ['year_built', 'last_sale_price']
scaler = RobustScaler()
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])




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
baseline = pd.DataFrame(columns=['Models', 'accuracy', 'f1_weighted', 'combined'])
baseline['Models'] = model_names

for model_name, clf in zip(model_names, classifiers):
    # For each cv_fold, train, predict, score the selected classifier
    # Compare models based on mean accuracy and mean f1_score
    accuracy = round(cross_val_score(clf, X_train, y_train, cv=cv_folds, scoring='accuracy').mean(), 2)

    f1_weighted = round(cross_val_score(clf, X_train, y_train, cv=cv_folds, scoring='f1_weighted').mean(), 2)

    # Update the score board
    baseline.loc[baseline['Models'] == model_name, 'accuracy'] = accuracy
    baseline.loc[baseline['Models'] == model_name, 'f1_weighted'] = f1_weighted

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

baseline['combined'] = (baseline['accuracy'] + baseline['f1_weighted']) / 2

print('\nComparing Classifiers...\n')
print(baseline)


# TUNE FINAL MODEL -- none are peforming very well; moving on to modeling full feature matrix.
# RepeatedStratifiedKFold can be used to repeat Stratified K-Fold n times with different randomization in each repetition.
# skf = StratifiedKFold(n_splits=3, shuffle=False, random_state=22)
# sfk.split
