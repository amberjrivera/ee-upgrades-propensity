'''
A set of transform classes to handle classifier preprocessing tasks.
Clean, impute, engineer the raw data to prepare it for modeling.
*IN PROGRESS*
All functions will eventually be written as classes in order to feed
into Sklearn's Pipeline constructor.
'''
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import RobustScaler
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
    positives = set(pd.read_csv('../data/Upgrade_Data.csv', usecols=[1], \
    squeeze=True).str[1:].tolist()) #4864 homes in county

    #make labels col based on membership in positives set
    df['labels'] = df.apply(lambda row: 1 if row['assessor_id'] in positives \
    else 0, axis=1)
    #9% of data in positive class. Homes data only captures 1/3 of upgrades data.

    # List of 2016 upgrades, for hindcasting
    sixteens = set(pd.read_csv('../data/upgrades_2016.csv', usecols=[1], \
    squeeze=True).str[1:].tolist())

    df['2016_upgrade'] = df.apply(lambda row: 1 if row['assessor_id'] in \
    sixteens else 0, axis=1) #92 homes with full data upgraded in 2016

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

        # drop redundant features (73)
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

        # df['num_upgrades_subdivision'] = \
        # df['labels'].groupby(df['subdivision']).transform('sum')
        #
        # df['num_upgrades_zip'] = \
        # df['labels'].groupby(df['zip']).transform('sum')

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

        df['permits'] = (df[permit_cols].notnull()).sum(1)
        df.drop(columns=permit_cols, inplace=True)

        #Good: num_permits_since_purchase
        #Better: By category, num_permits_since_purchase
        # p = Permits()
        # permits_dict = p.permits_dict
        # meta_dict = p.meta_dict()
        #
        # for key, values in permits_dict.items():
        #     for val in values:
        #         if (df[val]) > df['last_sale_date'].dt.year:
        #             meta_dict[key] += 1
        #     turn key into col header & populate with count (val from meta_dict)
        # df.drop(columns=permit_cols, inplace=True)


        ### IMPUTATION
        # Fill median (numerical)
        df['acres'].fillna(df['acres'].median(), inplace=True)

        df['census_income_median'].fillna(df['census_income_median'].median(),\
        inplace=True)

        # Fill mode (numerical)
        df['pv_potential_kwhr_yr'].fillna(df['pv_potential_kwhr_yr'].mode()[0],\
         inplace=True)

        # df['pv_potential_watts'].fillna(df['pv_potential_watts'].mode()[0], \
        # inplace=True)

        df.replace({'zillow_neighborhood': np.nan}, \
        {'zillow_neighborhood': 'Unknown'}, inplace=True)

        df.replace({'roof_cover_type': np.nan}, {'roof_cover_type': 'UNKNOWN'},\
         inplace=True)

        # Fill mode (categorical)
        cols = ['ac_type', 'exterior_wall_type', 'frame_type', 'heating_type', \
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
    Take in non-null feature matrix, X_train (Pandas dataframe), and target labels, y_train (Pandas dataframe).

    Balance positive and negative classes.

    Possible methods are 'downsample', 'bootstrap', and 'SMOTE'.

    Return balanced data as a numpy array.
    '''
    def __init__(self, pos_percent=0.45, method='bootstrap'):
        self.pos_percent = pos_percent
        self.method = method

    def transform(self, X_train, y_train=None):
        if y_train:
            self.pos_num = y_train.value_counts()[1] #1631 / 1093
        else:
            self.pos_num = X_train['labels'].value_counts()[1]

        self.pos_target = int(X_train.shape[0] * self.pos_percent)

        if self.method =='downsample':
            self.neg_num = X_train.shape[0] - self.pos_num
            self.num_to_drop = int((self.neg_num - (self.pos_num * (1-self.pos_percent) / self.pos_percent)))

            X_train.drop(X_train.query('labels == 0').sample(n=self.num_to_drop).index, \
            inplace=True)

        elif self.method =='bootstrap':
            samples_needed = int(self.pos_target - self.pos_num)
            pos_idx = y_train.index[y_train == 1].tolist()

            # pos_rows = df[df['labels'] == 1]

            new_data = pos_rows.sample(samples_needed, replace=True, random_state=42, axis=0)

            df.append(new_data)  #append new data into dataset
            df = df.sample(frac=1).reset_index(drop=True) #shuffle it

        # elif: method == 'SMOTE':

        #else: error

        balanced = X_train
        return balanced



# def split(df):
#     '''
#     Custom function to split X and y into training and test sets.


def save_and_drop_ids(df):
    '''
    Function to save unique identifier, and latitude and longitude info for
    geographic visualization later.
    Then drop before trimming down the df for modeling.
    '''
    identify_df = df[['lat', 'lon', 'assessor_id', 'labels', '2016_upgrade']]

    df.drop(columns=['lat', 'lon', 'assessor_id', '2016_upgrade'], inplace=True) #keep labels for y

    return df, identify_df


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


def hindcast(df):
    '''
    Pass in identify_df matrix to prepare it for the demo map.
    '''
    # get rid of any homes that did NOT upgrade in 2016
    df.drop(df[df['2016_upgrade'] != 1].index, inplace=True)

    #TODO add corresponding y_pred labels by assessor_id

    #write to csv to input into Google's My Maps for presentation
    df.to_csv('../data/2016_hindcast.csv')
    return df



# -----------------------------------------
class CustomBinarizer(BaseEstimator, TransformerMixin):
    '''
    In order to fold cleaning steps into the Pipeline, I'll need to write my own class to dummytize the categorical columns. Sklearn will release CategoricalEncoder in the next version, but in the meantime there isn't a great way to handle dummies inside of Pipeline when there is more than one categorical feature mixed in with numerical features in the dataset.
    '''
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return LabelBinarizer().fit(X).transform(X)

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
