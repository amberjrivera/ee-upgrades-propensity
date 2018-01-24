'''
WORK IN PROGRESS

Create and score a classification model.

Use StratifiedKfold cross-validation within Sklearn's GridSearchCV to
explore and compare potential models, using Sklearn's Pipeline constructor to
efficiently handle mixed numerical and categorical data and many transforms
for dimensionality reduction.

The final model is then saved as a pickled file for future use.
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer, StandardScaler, OneHotEncoder, LabelBinarizer, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from transforms import add_labels, Preprocessing, BalanceClasses, DFselector
from attributes import Attributes


#CREATE PIPELINE TO RECEIVE CLEAN, BALANCED FEATURE MATRIX
num_attribs, cat_attribs = Attributes().get_attribs()

num_pipeline = Pipeline([
        ('selector', DFselector(num_attribs)),
        ('std_scaler', StandardScaler())
    ])

cat_pipeline = Pipeline([
        ('selector', DFselector(cat_attribs)),
    ])

transform_pipeline = FeatureUnion(transformer_list=[
        ('num_pipeline', num_pipeline),
        ('cat_pipeline', cat_pipeline),
    ])


pipe = Pipeline([
        ('transform', transform_pipeline),
        # ('reduce_dim', SelectKBest(f_classif)),
        # ('reduce_dim', LinearDiscriminantAnalysis(solver='svd', n_components=20, store_covariance=True)),
        ('classify', RandomForestClassifier(
            bootstrap=True,
            criterion='entropy',
            max_depth=5,
            max_features=50,
            min_samples_leaf=20,
            min_samples_split=10,
            n_estimators=200,
            n_jobs=2
        ))
        # ('classify', GradientBoostingClassifier(
        #     subsample=0.85, #try 0.6
        #     n_estimators=600, #try 200, 400
        #     min_weight_fraction_leaf=0.01,
        #     min_samples_split=15,
        #     min_samples_leaf=30,
        #     min_impurity_decrease=0.01,
        #     max_leaf_nodes=None, #try 10, 15
        #     max_features=15,
        #     max_depth=12,
        #     learning_rate=0.05
        # )
        # )
     ])


# --------------------------------------
# #CREATE MODEL
# #load raw features data
# df = pd.read_csv('../data/city.csv')
#
# #Clean up unique identifier
# df['assessor_id'] = df['assessor_id'].str[1:]
#
# # add labels
# transforms.add_labels(df)
#
# #split out test data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
#
# #instantiate model - will swap out with final model I choose
# model = GradientBoostingClassifier()
#
# #placeholder for any algorithmic dimensionality reduction I'll do
# pca = PCA()
#
# #tune model - update once I choose best model and dim reduction
# param_grid = {}
# classifier = GridSearchCV(pipe, param_grid, scoring='f1_weighted', n_jobs=4, refit=True, cv=3,
#
# classifier.fit(X_train, y_train)
#
# # print out final score
#
# #pickle best model
# with open('ee-upgrade-propensity-model.pkl', 'wb') as f:
#     pickle.dump(classifier, f)
