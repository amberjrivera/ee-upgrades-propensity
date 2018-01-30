'''
Use Sklearn's Pipeline constructor to explore and compare various classifiers.
'''

import pandas as pd
import numpy as np
from transforms import DFselector
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, \
ExtraTreesClassifier


#USE PIPELINE TO GRIDSEARCH ON MULTIPLE ESTIMATORS
# num_attribs, cat_attribs = Attributes().get_attribs()
#
# num_pipeline = Pipeline([
#         ('select', DFselector(num_attribs)),
#         ('scale', RobustScaler())
#     ])
#
# cat_pipeline = Pipeline([
#         ('select', DFselector(cat_attribs)),
#     ])
#
# transform_pipeline = FeatureUnion(transformer_list=[
#         ('num_pipeline', num_pipeline),
#         ('cat_pipeline', cat_pipeline),
#     ])
#

full_pipeline = Pipeline([
        # ('transform', transform_pipeline),
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
     ])
