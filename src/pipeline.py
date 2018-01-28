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
from sklearn.preprocessing import Imputer, StandardScaler, OneHotEncoder, LabelBinarizer, LabelEncoder, RobustScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from transforms import add_labels, Preprocessing, BalanceClasses, DFselector
from imblearn.over_sampling import SMOTE
from attributes import Attributes


#CREATE PIPELINE TO RECEIVE CLEAN, BALANCED FEATURE MATRIX
#Iterate: fold cleaning and balancing into the pipeline

num_attribs, cat_attribs = Attributes().get_attribs()

num_pipeline = Pipeline([
        ('select', DFselector(num_attribs)),
        ('scale', RobustScaler())
    ])

cat_pipeline = Pipeline([
        ('select', DFselector(cat_attribs)),
    ])

transform_pipeline = FeatureUnion(transformer_list=[
        ('num_pipeline', num_pipeline),
        ('cat_pipeline', cat_pipeline),
    ])


pipe = Pipeline([
        ('transform', transform_pipeline),
        ('balance', SMOTE(
            random_state=42,
            ratio={1:6972},
            n_jobs=-1)
        ),
        # ('classify', RandomForestClassifier(
        #     random_state = 42,
        #     bootstrap=True,
        #     criterion='entropy',
        #     max_depth=5,
        #     max_features='auto',
        #     min_samples_leaf=20,
        #     min_samples_split=10,
        #     n_estimators=800,
        #     n_jobs=-1,
        #     oob_score=False
        # ))
        ('classify', GradientBoostingClassifier(
            subsample=0.95,
            n_estimators=200, #tried 400, 600
            min_weight_fraction_leaf=0.01,
            min_samples_split=15,
            min_samples_leaf=30,
            min_impurity_decrease=0.01,
            max_leaf_nodes=10, #try 10, 15
            max_features=15,
            max_depth=12,
            learning_rate=0.05
        ))
        # ('classify', QuadraticDiscriminantAnalysis(
        #     reg_param=0.1,
        #     store_covariance=True,
        #     store_covariances=None,
        #     tol=0.0001,
        #     priors=np.array([0.91, 0.09])
        # ))
     ])
