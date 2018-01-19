'''
PREPROCESSING
Clean, impute, engineer the raw data to prepare it for modeling.
*IN PROGRESS*
All functions will eventually be written as classes in order to feed into Sklearn's Pipeline constructor.
'''
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.base import TransformerMixin, BaseEstimator


# Signature template for any custom class to pass to Sklearn's Pipeline.
class CustomTransformer(TransformerMixin, BaseEstimator):
    def __init__(self):
        '''If you need to parameterize your transformer, set the args here. Don't need if there are zero params for this transformer.

        Inheriting from BaseEstimator introduces the constraint that the args all be named as keyword args, so no position args or **kwargs.
        '''
    pass


    def fit(self, X, y):
        '''Recommended signature for custom transformer's fit method.

        Set state here with whatever information is needed to transform later.

        In some cases fit may do nothing. For example transforming degrees F to Kelvin, requires no state.

        You can use y here, but won't have access to it in transform.
        '''
        #You have to return self, so we can chain!
        return self


    def transform(self, X):
        '''Transform some X data, optionally using state set in fit. This X may be the same X passed to fit, but it may also be new data, as in the case of a CV dataset. Both are treated the same.
        '''
        #Do transforms.
        #transformed = foo(X)
        # return transformed


class Clean:
    '''TK
    Fold clean_and_drop into a class.
    '''
    pass


#If imputing with either mean, median, or mode, will use sklearn.preprocessing.Imputer

class CustomImpute():
    '''TK
    Custom transformer to handle missing vals in AC_type and roof_cover_type.
    '''
    pass


class Permits():
    ''' TK
    Handle sparse permits data.
    Extract or engineer to get 35 variables into a few richer/denser variables.
    '''
    pass


class Engineering():
    ''' TK
    Custom transformer to do feature selection and extraction after exploring dimensionality reduction, and to engineer a few new features.
    '''
    pass


class Balance():
    ''' TK
    Custom transformer to balance the classes.
    '''
    pass





def add_labels(df):
    '''
    The label is conditional on membership in set of homes that have done any upgrade (separate csv file).

    Parameters
    ----------
    df :        Pandas dataframe of non-labeled features

    Returns
    ----------
    df :        Same Pandas dataframe, now with labels
    '''

    # Make list of addresses that have done any upgrade
    positives = pd.read_csv('../data/Upgrade_Data.csv', usecols=[1], squeeze=True).str[1:].tolist()

    positives = set(positives) #4864 homes in county

    #make labels col based on membership in positives set
    df['labels'] = df.apply(lambda row: 1 if row['assessor_id'] in positives else 0, axis=1)  #9% of data is in positive class

    return df


def clean_and_drop(filepath): #238 --> 50-60 attributes (many nested)
    '''
    Load the (test or new) data and quick-pass clean it up.

    Parameters
    ----------
    filepath :  path to csv of the data.

    Returns
    ----------
    df :        cleaned up feature matrix
    '''
    # #load raw features data
    # df = pd.read_csv('../data/city.csv')

    #drop PRIZM segments (3)
    df.drop(columns=[
        'prizm_premier',
        'prizm_social_group',
        'prizm_lifestage'
        ], inplace=True)

    #drop cols with data leakage (2)
    df.drop(columns=[
        'energy_program_upgrade_count', 'energy_program_upgrade_distinct'
        ], inplace=True)

    # drop rows with leakage
    df.drop(df[df.year_built == 2017].index, inplace=True)
    df.drop(df[df.effective_year_built == 2017].index, inplace=True)

    #drop cols with too many nulls (28)
    df.drop(columns=[
        'attached_garage_area',
        'building_count',
        'census_block_group',
        'census_block_id',
        'census_county_id',
        'census_state_id',
        'census_tract_id',
        'detached_garage_count',
        'fireplace_count',
        'floor_type',
        'geocode_missing',
        'gross_building_area',
        'has_pool',
        'has_solar_hot_water',
        'mail_to',
        'mailing_address_2',
        'mailing_country',
        'owner_2',
        'owner_3',
        'percent_air_conditioned',
        'rental_living_unit_count',
        'roof_type',
        'stories',
        'street_orientation',
        'unit_count',
        'utility_services',
        'wood_stove_flue_count',
        'zoning_code',
        ], inplace=True)

    #drop redundant features (72)
    df.drop(columns=[
        'adjusted_job_cost_2',
        'adjusted_job_cost_3',
        'carbon_base_2',
        'carbon_base_3',
        'carbon_improved_2',
        'carbon_improved_3',
        'carbon_savings_2',
        'carbon_savings_3',
        'electricity_usage_kwh_base_2',
        'electricity_usage_kwh_base_3',
        'electricity_usage_kwh_improved_2',
        'electricity_usage_kwh_improved_3',
        'electricity_usage_kwh_savings_2',
        'electricity_usage_kwh_savings_3',
        'energy_cost_base_2',
        'energy_cost_base_3',
        'energy_cost_improved_2',
        'energy_cost_improved_3',
        'energy_cost_savings_2',
        'energy_cost_savings_3',
        'gas_usage_therm_base_2',
        'gas_usage_therm_base_3',
        'gas_usage_therm_improved_2',
        'gas_usage_therm_improved_3',
        'gas_usage_therm_savings_2',
        'gas_usage_therm_savings_3',
        'geography_label',
        'heating_fuel',
        'incremental_job_cost_2',
        'incremental_job_cost_3',
        'lat',
        'lon',
        'mailing_zip',
        'monthly_cash_flow_2',
        'monthly_cash_flow_3',
        'monthly_payments_2',
        'monthly_payments_3',
        'monthly_savings_2',
        'monthly_savings_3',
        'neighborhood',
        'pv_heat_only_carbon_savings_2',
        'pv_heat_only_carbon_savings_3',
        'pv_heat_only_costs_2',
        'pv_heat_only_costs_3',
        'pv_heat_only_dollar_savings_2',
        'pv_heat_only_dollar_savings_3',
        'pv_heat_only_monthly_cash_flow_2',
        'pv_heat_only_monthly_cash_flow_3',
        'pv_heat_only_monthly_payments_2',
        'pv_heat_only_monthly_payments_3',
        'pv_heat_only_monthly_savings_2',
        'pv_heat_only_monthly_savings_3',
        'pv_net_zero_carbon_savings_2',
        'pv_net_zero_carbon_savings_3',
        'pv_net_zero_costs_2',
        'pv_net_zero_costs_3',
        'pv_net_zero_dollar_savings_2',
        'pv_net_zero_dollar_savings_3',
        'pv_net_zero_monthly_cash_flow_2',
        'pv_net_zero_monthly_cash_flow_3',
        'pv_net_zero_monthly_payments_2',
        'pv_net_zero_monthly_payments_3',
        'pv_net_zero_monthly_savings_2',
        'pv_net_zero_monthly_savings_3',
        'pv_w_required_heat_only_2',
        'pv_w_required_heat_only_3',
        'pv_w_required_net_zero_2',
        'pv_w_required_net_zero_3',
        'retrofit_cost_2',
        'retrofit_cost_3',
        'simple_payback_2',
        'simple_payback_3'
        ], inplace=True)

    #drop irrelevant features (18)
    df.drop(columns=[
        'address_id',
        'building_uid',
        'city',
        'county',
        'formatted_address',
        'geocode_partial_match',
        'geocode_place_id',
        'geocode_query_address',
        'geoid',
        'geoid_complete',
        'mailing_address',
        'mailing_city',
        'mailing_state',
        'owner_1',
        'route',
        'state',
        'street_number',
        'unit'
        ], inplace=True)

    # Drop rows without sale_date
    city['last_sale_date'].dropna(inplace=True)



    #TODO fold categories
    # land_use, site_type, frame_type, secondary_building_type, owner_occupied, roof_cover_type, and all permits fields

    return df


def impute(df):
    # Fill median
    df['acres'].fillna(df['acres'].median(), inplace=True)

    df['census_income_median'].fillna(df['census_income_median'].median(), inplace=True)

    df['last_sale_price'].fillna(df['last_sale_price'].median(), inplace=True)

    # Fill mode
    df['pv_potential_kwhr_yr'].fillna(df['pv_potential_kwhr_yr'].mode(), inplace=True)

    df['pv_potential_watts'].fillna(df['pv_potential_watts'].mode(), inplace=True)


def feature_engineer(df):
    pass
    # subdivision has 785 categories.
    # parcel_id has 34 categories
    # zip has 5 categories
    # Spatial clustering: create num_upgrades_subdivision, num_upgrades_parcel, num_upgrades_zip, then drop the original fields.

    # time_since_last_sale' --> use last_sale_date and update_date (date the data were pulled; uniform), then drop them.


def balance(): # either undersample before cv, or oversample within cv
    #TODO: Compare performance undersampling majority vs. oversampling minority. Try SMOTE.
    #TODO: If oversampling (within cv): Compare ADASYN, SMOTE, RandomOverSampler http://contrib.scikit-learn.org/imbalanced-learn/stable/over_sampling.html#smote-adasyn
    pass




if __name__ == '__main__':
    #load raw features data
    df = pd.read_csv('../data/city.csv')

    #Clean up unique identifier
    df['assessor_id'] = df['assessor_id'].str[1:]
