'''
A set of transformations to handle classifier preprocessing tasks.
Clean, impute, engineer the raw data to prepare it for modeling.
'''
import pandas as pd
import numpy as np
import pickle
from attributes import Attributes
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import TransformerMixin, BaseEstimator


def add_labels(df):
    '''
    Add target labels (y) to feature matrix.
    The label is conditional on membership in set of homes that have
    done any upgrade (separate csv file).

    Parameters
    ----------
    df :       Pandas dataframe of non-labeled features

    Returns
    ----------
    df :       Same Pandas dataframe, now with labels
    '''

    # Make list of addresses that have done any upgrade
    positives = set(pd.read_csv('../data/Upgrade_Data.csv', usecols=[1], \
    squeeze=True).str[1:].tolist()) #4864 homes in county

    # Make labels col based on membership in positives set
    df['labels'] = df.apply(lambda row: 1 if row['assessor_id'] in positives \
    else 0, axis=1) #9% of data in positive class.

    # # # List of 2016 upgrades, for hindcasting
    # sixteens = set(pd.read_csv('../data/upgrades_2016.csv', usecols=[1], \
    # squeeze=True).str[1:].tolist())
    #
    # df['labels_backtest'] = df.apply(lambda row: 1 if row['assessor_id'] in \
    # sixteens else 0, axis=1) #92 homes with full data upgraded in 2016

    # drop unique ID
    df.drop(columns='assessor_id', inplace=True)

    return df



class Preprocessing(object):
    '''
    Take in a feature matrix including target labels.
    Clean, drop, and engineer features.
    Impute missing values.
    Return processed Pandas dataframe.
    '''
    def __init__(self):
        pass

    def transform(self, df):
        ### CLEAN AND DROP
        attribs = Attributes()

        # drop customer segmentation info (3) #tinker
        segment_cols = attribs.get_segment_cols()
        df.drop(columns=segment_cols, inplace=True)

        # drop cols with data leakage (2)
        leak_cols = attribs.get_leak_cols()
        df.drop(columns=leak_cols, inplace=True)

        # drop rows with leakage
        df.drop(df[df.year_built == 2017].index, inplace=True)
        df.drop(df[df.effective_year_built == 2017].index, inplace=True)

        # drop cols with too many nulls (28)
        null_cols = attribs.get_null_cols()
        df.drop(columns=null_cols, inplace=True)

        # drop redundant features (74)
        redundant_cols = attribs.get_redundant_cols()
        df.drop(columns=redundant_cols, inplace=True)

        # drop irrelevant features (18)
        irrelevant_cols = attribs.get_irrelevant_cols()
        df.drop(columns=irrelevant_cols, inplace=True)

        # drop 1050 rows without sale_date or sale_price (same set)
        df.dropna(subset=['last_sale_price', 'last_sale_date'], inplace=True)

        # remap buidling_condition (misspelling intentional)
        df.replace({'buidling_condition':{
            'LOW':1,
            'FAIR':2,
            'AVERAGE':3,
            'AVERAGE +':4,
            'AVERAGE ++':5,
            'GOOD':6,
            'GOOD +':7,
            'GOOD ++':8,
            'VERY GOOD':9,
            'VERY GOOD +':10,
            'VERY GOOD ++':11,
            'EXCELLENT':12,
            'EXCELLENT +':13,
            'EXCELLENT++':14,
            'EXCEPTIONAL 1':15}
            }, inplace=True)

        # convert true/false to 1/0
        df['nrel_attached_garage'].astype(int, copy=False)

        # combine full and half baths
        df['num_baths'] = df['full_bath_count'] + (0.5 * df['half_bath_count'])
        df.drop(columns=['full_bath_count', 'half_bath_count'], inplace=True)


        ### FEATURE ENGINEER
        # Spatial clustering
        #TODO won't work in production b/c engineering off of labels.
        df['num_upgrades_parcel'] = \
        df['labels'].groupby(df['parcel_id']).transform('sum')

        df.drop(columns=['parcel_id', 'subdivision', 'zip'], inplace=True)

        # Days since last sale
        df['update_date'] = pd.to_datetime(df['update_date'])
        df['last_sale_date'] = pd.to_datetime(df['last_sale_date'])

        df['time_since_sale'] = \
        (df['update_date'] - df['last_sale_date']).dt.days

        df.drop(columns=['update_date', 'last_sale_date'], inplace=True)

        # Handle sparse permits data #TODO improve method
        #Quick: total permits ever
        permit_cols = attribs.get_permit_cols()

        df['num_permits_since_purchase'] = (df[permit_cols].notnull()).sum(1)
        df.drop(columns=permit_cols, inplace=True)


        ### IMPUTATION
        # Fill median (numerical)
        df['acres'].fillna(df['acres'].median(), inplace=True)

        df['census_income_median'].fillna(df['census_income_median'].median(),\
        inplace=True)

        # Fill mode (numerical)
        df['pv_potential_kwhr_yr'].fillna(df['pv_potential_kwhr_yr'].mode()[0],\
         inplace=True)

        # Fill 'Unknown'
        df.replace({'zillow_neighborhood': np.nan}, \
        {'zillow_neighborhood': 'Unknown'}, inplace=True)

        # Fill mode (categorical)
        cols = ['ac_type', 'exterior_wall_type', 'frame_type', 'heating_type', \
        'interior_wall_type', 'land_use', 'roof_cover_type']
        for col in cols:
            mode = df[col].mode()[0]
            df[col].fillna(mode, inplace=True)

        # DUMMYTIZE
        dummy_cols = Attributes().get_dummy_cols()
        df = pd.get_dummies(df, columns=dummy_cols, drop_first=True)

        processed = df
        return processed


def balance(X_train, y_train, method):
    if method == 'downsample':
        # Random downsample to balance classes
        sampler = RandomUnderSampler(random_state=42, ratio='auto', replacement=False, return_indices=True)

        X_train_res, y_train_res, idx_res = sampler.fit_sample(X_train, y_train)

    if method == 'ros':
        # Random oversample to balance classes
        sampler = RandomOverSampler(random_state=42, ratio='auto')

        X_train_res, y_train_res = sampler.fit_sample(X_train, y_train)

    if method == 'SMOTE':
        # Use SMOTE to over sample to balance classes
        sampler = SMOTE(random_state=42, ratio='auto', n_jobs=-1)

        X_train_res, y_train_res = sampler.fit_sample(X_train, y_train)

        idx_res = None

    return X_train_res, y_train_res, idx_res


def expected_value(y_test, y_pred, y_probs, num_jobs=500):
    # everything in NumPy
    y_test = np.array(y_test)

    #sort the indices
    idx_s = np.argsort(y_probs)[::-1]
    y_probs = y_probs[idx_s]
    y_pred = y_pred[idx_s]
    y_test = y_test[idx_s]

    # take the top n probable, based on numbers of jobs expected
    n = num_jobs
    y_probs_n = y_probs[:n]
    y_pred_n = y_pred[:n]
    y_test_n = y_test[:n]
    print("After taking the top n...")
    print(y_probs_n.shape, y_pred_n.shape, y_test_n.shape)

    # Get number of TP and FP in top n probabilities
    numPP = y_pred_n.sum()
    TP = y_test_n * y_pred_n
    numTP = TP.sum()
    numFP = numPP - numTP
    print("\nTP: {0}, FP: {1}.".format(numTP, numFP))
    print("There are {0} positive predictions.".format(numPP))
    print("There are {0} TP in the predictions".format(numTP))
    print("There are {0} FP in the predictions".format(numFP))

    return numTP, numFP


class DFselector(TransformerMixin, BaseEstimator):
    '''
    Custom class to select specific columns out of a Pandas dataframe,
    in preparation for Sklean's FeatureUnion and Pipeline.

    Returns Numpy array of selected features.
    '''
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values
