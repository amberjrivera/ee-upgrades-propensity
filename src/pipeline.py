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
from sklearn.preprocessing import Imputer, StandardScaler, CategoricalEncoder, FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from transforms import *
from attributes import Attributes


#CREATE PIPELINE TO RECEIVE CLEAN, BALANCED FEATURE MATRIX
num_attribs, cat_attribs, other_attribs = Attributes().get_attribs()

num_pipeline = Pipieline([
        ('selector', DFselector(num_attribs)),
        ('std_scaler', StandardScalar())
    ])

cat_pipeline = Pipeline([
        ('selector', DFselector(cat_attribs)),
        ('encoder', CategoricalEncoder(encoding='onehot'))
    ])

other_pipeline = Pipeline([
        ('selector', DFselector(other_attribs))
    ])



transform_pipeline = FeatureUnion(transformer_list=[
        ('num_pipeline', num_pipeline),
        ('cat_pipeline', cat_pipeline),
        ('others_pipeline', others_pipeline)
    ])


pipe = Pipeline([
        ('transform', transform_pipeline),
        ('pca', PCA()), #placeholder for dim reduction
        ('classify', estimator)
     ])




# --------------------------------------
#CREATE MODEL
#load raw features data
df = pd.read_csv('../data/city.csv')

#Clean up unique identifier
df['assessor_id'] = df['assessor_id'].str[1:]

# add labels
transforms.add_labels(df)

#split out test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

#instantiate model - will swap out with final model I choose
model = GradientBoostingClassifier()

#placeholder for any algorithmic dimensionality reduction I'll do
pca = PCA()

#tune model - update once I choose best model and dim reduction
param_grid = {}
classifier = GridSearchCV(pipe, param_grid, scoring='f1_weighted', n_jobs=4, refit=True, cv=3,

classifier.fit(X_train, y_train)

# print out final score

#pickle best model
with open('ee-upgrade-propensity-model.pkl', 'wb') as f:
    pickle.dump(classifier, f)
