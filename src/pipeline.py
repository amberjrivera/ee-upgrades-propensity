'''
WORK IN PROGRESS

Create and score a classification model.

Use StratifiedKfold cross-validation within Sklearn's GridSearchCV to
explore and compare potential models, using Sklearn's Pipeline constructor to
efficiently handle mixed numerical and categorical data and many transforms
for dimensionality reduction.

The final model is then saved as a pickled file for future use.
'''

import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer, StandardScaler, CategoricalEncoder, FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from preprocessing import *


#CREATE PIPELINE
preprocess = [('cleaning', Cleaning()),
            ('imputation', CustomImpute()),
            ('permits', Permits()),
            ('engineering', Engineering())
            ])

numerical = [('selector', DFselector()),
            ('imputer', Imputer(strategy='median')),
            ('std_scaler', StandardScalar())
            ]

categorical = [('selector', DFselector()),
              ('cat_encoder', CategoricalEncoder(encoding='ordinal')) #tinker
              ]

#TODO: Add in sub-pipeline for all other attributes, then fold into transform_pipeline
# others = [
#          ]

preprocess_pipeline = Pipeline(preprocess)
num_pipeline = Pipeline(numerical)
cat_pipeline = Pipeline(categorical)
others_pipeline = Pipeline(others)
transform_pipeline = FeatureUnion(transformer_list=[
                    ('num_pipeline', num_pipeline),
                    ('cat_pipeline', cat_pipeline),
                    ('others_pipeline', others_pipeline)
                    ])


pipe = [('preprocess', preprocess_pipeline),
             ('transform', transform_pipeline),
             ('pca', pca),  #placeholder
             ('model', model)
             ]


#CREATE MODEL
#load raw features data
df = pd.read_csv('../data/city.csv')

#Clean up unique identifier
df['assessor_id'] = df['assessor_id'].str[1:]

# add labels
preprocessing.add_labels(df)

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
