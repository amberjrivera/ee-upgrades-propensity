'''
A set of transform classes to handle classifier preprocessing tasks.
Clean, impute, engineer the raw data to prepare it for modeling.
*IN PROGRESS*
All functions will eventually be written as classes in order to feed
into Sklearn's Pipeline constructor.
'''
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer, StandardScaler, RobustScaler, LabelBinarizer
from sklearn.base import TransformerMixin, BaseEstimator
from attributes import Attributes


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
    positives = pd.read_csv('../data/Upgrade_Data.csv', usecols=[1], \
    squeeze=True).str[1:].tolist()

    positives = set(positives) #4864 homes in county

    #make labels col based on membership in positives set
    df['labels'] = df.apply(lambda row: 1 if row['assessor_id'] in positives \
    else 0, axis=1)  #9% of data is in positive class

    return df


# class Preprocessing(object, pd.DataFrame):
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

        #drop unique identiier (1)
        df.drop(columns='assessor_id', inplace=True)

        # drop PRIZM segments (3)
        df.drop(columns=[
            'prizm_premier',
            'prizm_social_group',
            'prizm_lifestage'
            ], inplace=True)

        # drop cols with data leakage (2)
        leak_cols = attribs.get_leak_cols()
        df.drop(columns=leak_cols, inplace=True)

        # drop rows with leakage
        df.drop(df[df.year_built == 2017].index, inplace=True)
        df.drop(df[df.effective_year_built == 2017].index, inplace=True)

        # drop cols with too many nulls (28)
        null_cols = attribs.get_null_cols()
        df.drop(columns=null_cols, inplace=True)

        # drop redundant features (72)
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


        ### FEATURE ENGINEER
        # Spatial clustering
        #TODO won't work in production bc engineering off of labels.
        df['num_upgrades_parcel'] = \
        df['labels'].groupby(df['parcel_id']).transform('sum')

        df['num_upgrades_subdivision'] = \
        df['labels'].groupby(df['subdivision']).transform('sum')

        df['num_upgrades_zip'] = \
        df['labels'].groupby(df['zip']).transform('sum')

        df.drop(columns=['parcel_id', 'subdivision', 'zip'], inplace=True)

        # Days since last sale
        df['update_date'] = pd.to_datetime(df['update_date'])
        df['last_sale_date'] = pd.to_datetime(df['last_sale_date'])

        df['time_since_sale'] = \
        (df['update_date'] - df['last_sale_date']).dt.days

        df.drop(columns=['update_date', 'last_sale_date'], inplace=True)

        # Handle sparse permits data #TODO improve method
            #Quick: total permits ever
            #Good: num_permits_since_purchase
            #Better: By category, num_permits_since_purchase
            #Best:
        permit_cols = attribs.get_permit_cols()

        #Quick: total permits ever
        df['permits'] = (df[permit_cols].notnull()).sum(1)
        df.drop(columns=permit_cols, inplace=True)


        ### IMPUTATION
        # Fill median (numerical)
        df['acres'].fillna(df['acres'].median(), inplace=True)

        df['census_income_median'].fillna(df['census_income_median'].median(),\
        inplace=True)

        # Fill mode (numerical)
        df['pv_potential_kwhr_yr'].fillna(df['pv_potential_kwhr_yr'].mode()[0],\
         inplace=True)

        df['pv_potential_watts'].fillna(df['pv_potential_watts'].mode()[0], \
        inplace=True)

        # Fill 'Unknown' (categorical)
        df.replace({'ac_type': np.nan}, {'ac_type': 'UNKNOWN'}, inplace=True)

        df.replace({'zillow_neighborhood': np.nan}, \
        {'zillow_neighborhood': 'Unknown'}, inplace=True)

        df.replace({'roof_cover_type': np.nan}, {'roof_cover_type': 'UNKNOWN'},\
         inplace=True)

        # Fill mode (categorical)
        cols = ['exterior_wall_type', 'frame_type', 'heating_type', \
        'interior_wall_type', 'land_use']
        for col in cols:
            mode = df[col].mode()[0]
            df[col].fillna(mode, inplace=True)

        # DUMMYTIZE
        dummy_cols = Attributes().get_dummy_cols()
        df = pd.get_dummies(df, columns=dummy_cols, drop_first=True)

        processed = df
        return processed


class BalanceClasses(object):
    '''
    Take in non-null feature matrix (Pandas dataframe) including target labels.
    Balance positive and negative classes.

    Possible methods are...TK

    Return processed Pandas dataframe.
    '''
    def __init__(self, pos_class_percent=0.25):
    # def __init__(self, method='downsample', pos_class_percent=0.25):
    # TODO add ability to specify method, and change desired % positive class
        # self.pos_percent = pos_class_percent
        # self.method = method
        pass

    def transform(self, df):
        # method='downsample'
        num_pos = len(df[df['labels'] == 1])
        num_neg = df.shape[0] - num_pos
        num_to_drop = int((num_neg*.79) - num_pos) #25% pos class - tinker
        df.drop(df.query('labels == 0').sample(n=num_to_drop).index, \
        inplace=True)

        balanced = df
        return balanced


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



# -----------------------------------------
# class CustomBinarizer(BaseEstimator, TransformerMixin):
#     def __init__(self):
#         pass
#
#     def fit(self, X, y=None):
#         return self
#
#     def transform(self, X):
#         return LabelBinarizer().fit(X).transform(X)

class CustomImpute(object):
    '''TK
    Custom class to handle missing vals in AC_type and roof_cover_type.
    '''
    pass


def clean_and_drop(df): # 18406 x 239-->17351 x 116 (nested cols)
    '''
    Clean up the raw data (Pandas dataframe).

    Parameters
    ----------
    df :  Loaded and labeled df

    Returns
    ----------
    df :  cleaned up feature matrix
    '''
    #drop unique identiier
    df.drop(columns='assessor_id', inplace=True)

    #drop PRIZM segments (3)
    df.drop(columns=[
        'prizm_premier',
        'prizm_social_group',
        'prizm_lifestage'
        ], inplace=True)

    # drop cols with data leakage (2)
    leak_cols = Attributes().get_leak_cols()
    df.drop(columns=leak_cols, inplace=True)

    # drop rows with leakage
    df.drop(df[df.year_built == 2017].index, inplace=True)
    df.drop(df[df.effective_year_built == 2017].index, inplace=True)

    # drop cols with too many nulls (28)
    null_cols = Attributes().get_null_cols()
    df.drop(columns=null_cols, inplace=True)

    # drop redundant features (72)
    redundant_cols = Attributes().get_redundant_cols()
    df.drop(columns=redundant_cols, inplace=True)

    # drop irrelevant features (18)
    irrelevant_cols = Attributes().get_irrelevant_cols()
    df.drop(columns=irrelevant_cols, inplace=True)

    # Drop 1050 rows without sale_date or sale_price (same set)
    df.dropna(subset=['last_sale_price', 'last_sale_date'], inplace=True)

    #Remap buidling_condition (misspelling intentional)
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

    return df


def feature_engineer(df): #17351 x 116 --> 17351 x 81
    # Spatial clustering
    df['num_upgrades_parcel'] = \
    df['labels'].groupby(df['parcel_id']).transform('sum')

    df['num_upgrades_subdivision'] = \
    df['labels'].groupby(df['subdivision']).transform('sum')

    df['num_upgrades_zip'] = \
    df['labels'].groupby(df['zip']).transform('sum')

    df.drop(columns=['parcel_id', 'subdivision', 'zip'], inplace=True)

    # Days since last sale
    df['update_date'] = pd.to_datetime(df['update_date'])
    df['last_sale_date'] = pd.to_datetime(df['last_sale_date'])

    df['time_since_sale'] = (df['update_date'] - df['last_sale_date']).dt.days

    df.drop(columns=['update_date', 'last_sale_date'], inplace=True)


    # Handle sparse permits data
        #Quick: total permits ever
        #Good: num_permits_since_purchase
        #Better: By category, num_permits_since_purchase
        #Best:
    permit_cols = Attributes().get_permit_cols()

    df['permits'] = (df[permit_cols].notnull()).sum(1)
    df.drop(columns=permit_cols, inplace=True)

    return df


def impute(df):
    # Fill median
    df['acres'].fillna(df['acres'].median(), inplace=True)

    df['census_income_median'].fillna(df['census_income_median'].median(), \
    inplace=True)

    # Fill mode
    df['pv_potential_kwhr_yr'].fillna(df['pv_potential_kwhr_yr'].mode()[0], \
    inplace=True)

    df['pv_potential_watts'].fillna(df['pv_potential_watts'].mode()[0], \
    inplace=True)

    # Fill 'Unknown'
    df.replace({'ac_type': np.nan}, {'ac_type': 'UNKNOWN'}, inplace=True)

    df.replace({'zillow_neighborhood': np.nan}, \
    {'zillow_neighborhood': 'Unknown'}, inplace=True)

    df.replace({'roof_cover_type': np.nan}, {'roof_cover_type': 'UNKNOWN'}, \
    inplace=True)

    return df


def cat_impute(df, cols):
    '''
    Fill mode on specified, categorical, columns.
    '''
    # cols = ['exterior_wall_type', 'frame_type', 'heating_type',
    # 'interior_wall_type', 'land_use']
    for col in cols:
        mode = df[col].mode()[0]
        df[col].fillna(mode, inplace=True)
    return df


def balance(df):
    # undersample majority class to handle class imbalance
    num_pos = len(df[df['labels'] == 1])
    num_neg = df.shape[0] - num_pos
    num_to_drop = int((num_neg*.79) - num_pos) #25% pos - tinker
    df.drop(df.query('labels == 0').sample(n=num_to_drop).index, inplace=True)
    return df


def scale(df, cols):
    scaler = RobustScaler(copy=False)
    df[cols] = scaler.fit_transform(df[cols])
    return df
