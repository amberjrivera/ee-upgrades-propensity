'''
PREPROCESSING
Clean, impute, engineer the raw data to prepare it for modeling.
*IN PROGRESS*
All functions will eventually be written as classes in order to feed into Sklearn's Pipeline constructor.
'''
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer, StandardScaler, RobustScaler, robust_scale
from sklearn.base import TransformerMixin, BaseEstimator


# Signature template for any custom class to pass to Sklearn's Pipeline.
class CustomTransformer(TransformerMixin, BaseEstimator):
    def __init__(self):
        '''If you need to parameterize your transformer, set the args here.
        Don't need it if there are zero params for this transformer.

        Inheriting from BaseEstimator introduces the constraint that the args
        all be named as keyword args, so no position args or **kwargs.
        '''
        pass


    def fit(self, X, y=None):
        '''Recommended signature for custom transformer's fit method.

        Set state here with whatever information is needed to transform later.

        In some cases fit may do nothing. For example transforming degrees F
        to Kelvin, requires no state.

        You can use y here, but won't have access to it in transform.
        '''
        #You have to return self, so we can chain!
        return self


    def transform(self, X):
        '''
        Transform some X data, optionally using the state set in fit.
        This X may be the same X passed to fit, but it may also be new data,
        as in the case of a CV dataset. Both are treated the same.
        '''
        #Do transforms.
        #transformed = foo(X)
        # return transformed


class Preprocessing(TransformerMixin, BaseEstimator, pd.DataFrame): #done
    '''
    Clean, drop, and engineer features.
    '''
    def __init__(self):
        pass

    # def fit(self, X, y=None):
    def fit(self, df):
        return self

    # def transform(self, X):
    def transform(self, df):
        ### CLEAN AND DROP
        # drop PRIZM segments (3)
        df.drop(columns=[
            'prizm_premier',
            'prizm_social_group',
            'prizm_lifestage'
            ], inplace=True)

        # drop cols with data leakage (2)
        df.drop(columns=[
            'energy_program_upgrade_count',
            'energy_program_upgrade_distinct'
            ], inplace=True)

        # drop rows with leakage
        df.drop(df[df.year_built == 2017].index, inplace=True)
        df.drop(df[df.effective_year_built == 2017].index, inplace=True)

        # drop cols with too many nulls (28)
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

        # drop redundant features (72)
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

        # drop irrelevant features (18)
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


        ### FEATURE ENGINEER
        # Spatial clustering
        #TODO won't work in production bc engineering off of labels.
        df['num_upgrades_parcel'] = df['labels'].groupby(df['parcel_id']).transform('sum')

        df['num_upgrades_subdivision'] = df['labels'].groupby(df['subdivision']).transform('sum')

        df['num_upgrades_zip'] = df['labels'].groupby(df['zip']).transform('sum')

        df.drop(columns=['parcel_id', 'subdivision', 'zip'], inplace=True)

        # Days since last sale
        df['update_date'] = pd.to_datetime(df['update_date'])
        df['last_sale_date'] = pd.to_datetime(df['last_sale_date'])

        df['time_since_sale'] = (df['update_date'] - df['last_sale_date']).dt.days

        df.drop(columns=['update_date', 'last_sale_date'], inplace=True)

        # Handle sparse permits data #TODO improve method
            #Quick: total permits ever
            #Good: num_permits_since_purchase
            #Better: By category, num_permits_since_purchase
            #Best:
        permit_cols = [
            'permit_electrical',
            'permit_plumbing',
            'permit_roof',
            'permit_gas_furnace',
            'permit_ac',
            'permit_water_heater',
            'permit_boiler',
            'permit_existing_boiler',
            'permit_direct_heater',
            'permit_misc',
            'permit_siding',
            'permit_evaporative_cooler',
            'permit_addition_or_remodel',
            'permit_gas_range',
            'permit_pool_or_hot_tub',
            'permit_gas_dryer',
            'permit_solar_thermal',
            'permit_gas_fireplace_or_stove',
            'permit_wood_fireplace_or_stove',
            'permit_heat_pump',
            'permit_generator',
            'permit_basement_finish',
            'permit_accessory_building',
            'permit_rooftop_pv',
            'permit_window_replacement',
            'permit_pool_heater',
            'permit_geothermal_system',
            'permit_electric_water_heater',
            'permit_wind_turbine',
            'permit_whole_house_fan',
            'permit_ev_charger',
            'permit_mechanical_ventilation',
            'permit_high_efficiency_heating',
            'permit_oil_heat',
            'permit_new_electrical_service'
                ]

        #Quick: total permits ever
        df['permits'] = (df[permit_cols].notnull()).sum(1)
        df.drop(columns=permit_cols, inplace=True)


        ### IMPUTATION
        # Fill median (numerical)
        df['acres'].fillna(df['acres'].median(), inplace=True)

        df['census_income_median'].fillna(df['census_income_median'].median(), inplace=True)

        # Fill mode (numerical)
        df['pv_potential_kwhr_yr'].fillna(df['pv_potential_kwhr_yr'].mode()[0], inplace=True)

        df['pv_potential_watts'].fillna(df['pv_potential_watts'].mode()[0], inplace=True)

        # Fill 'Unknown' (categorical)
        df.replace({'ac_type': np.nan}, {'ac_type': 'UNKNOWN'}, inplace=True)

        df.replace({'zillow_neighborhood': np.nan}, {'zillow_neighborhood': 'Unknown'}, inplace=True)

        df.replace({'roof_cover_type': np.nan}, {'roof_cover_type': 'UNKNOWN'}, inplace=True)

        # Fill mode (categorical)
        cols = ['exterior_wall_type', 'frame_type', 'heating_type', 'interior_wall_type', 'land_use']
        for col in cols:
            mode = df[col].mode()[0]
            df[col].fillna(mode, inplace=True)


        ### HANDLE CLASS IMBALANCE
        # undersample majority class to
        num_pos = len(df[df['labels'] == 1])
        num_neg = df.shape[0] - num_pos
        num_to_drop = int((num_neg*.79) - num_pos) #25% pos class - tinker, grid search later
        df.drop(df.query('labels == 0').sample(n=num_to_drop).index, inplace=True)

        processed = df
        return processed


class CatImpute(TransformerMixin, BaseEstimator):
    def __init__(self):
        """Impute missing values on dataframe of only categorical columns.

        Columns of dtype object are imputed with the most frequent value in column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0] for c in X], index=X.columns)

        return self

    def transform(self, X):
        return X.fillna(self.fill)


class NumericalImpute(TransformerMixin, BaseEstimator):
    '''TK
    Custom transformer to handle missing vals in AC_type and roof_cover_type.
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


def clean_and_drop(df): # 18406 x 239-->17351 x 116 (nested cols)
    '''
    Quick-pass clean up.

    Parameters
    ----------
    df :  Loaded and labeled df

    Returns
    ----------
    df :        cleaned up feature matrix
    '''

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
    permit_cols = [
        'permit_electrical',
        'permit_plumbing',
        'permit_roof',
        'permit_gas_furnace',
        'permit_ac',
        'permit_water_heater',
        'permit_boiler',
        'permit_existing_boiler',
        'permit_direct_heater',
        'permit_misc',
        'permit_siding',
        'permit_evaporative_cooler',
        'permit_addition_or_remodel',
        'permit_gas_range',
        'permit_pool_or_hot_tub',
        'permit_gas_dryer',
        'permit_solar_thermal',
        'permit_gas_fireplace_or_stove',
        'permit_wood_fireplace_or_stove',
        'permit_heat_pump',
        'permit_generator',
        'permit_basement_finish',
        'permit_accessory_building',
        'permit_rooftop_pv',
        'permit_window_replacement',
        'permit_pool_heater',
        'permit_geothermal_system',
        'permit_electric_water_heater',
        'permit_wind_turbine',
        'permit_whole_house_fan',
        'permit_ev_charger',
        'permit_mechanical_ventilation',
        'permit_high_efficiency_heating',
        'permit_oil_heat',
        'permit_new_electrical_service'
            ]

    df['permits'] = (df[permit_cols].notnull()).sum(1)
    df.drop(columns=permit_cols, inplace=True)

    return df


def impute(df):
    # Fill median
    df['acres'].fillna(df['acres'].median(), inplace=True)

    df['census_income_median'].fillna(df['census_income_median'].median(), inplace=True)

    # Fill mode
    df['pv_potential_kwhr_yr'].fillna(df['pv_potential_kwhr_yr'].mode()[0], inplace=True)

    df['pv_potential_watts'].fillna(df['pv_potential_watts'].mode()[0], inplace=True)

    # Fill 'Unknown'
    df.replace({'ac_type': np.nan}, {'ac_type': 'UNKNOWN'}, inplace=True)

    df.replace({'zillow_neighborhood': np.nan}, {'zillow_neighborhood': 'Unknown'}, inplace=True)

    df.replace({'roof_cover_type': np.nan}, {'roof_cover_type': 'UNKNOWN'}, inplace=True)

    return df


def cat_impute(df, cols):
    '''
    Fill mode on specified, categorical, columns.
    '''
    # cols = ['exterior_wall_type', 'frame_type', 'heating_type', 'interior_wall_type', 'land_use']
    for col in cols:
        mode = df[col].mode()[0]
        df[col].fillna(mode, inplace=True)
    return df


def balance(df):
    # undersample majority class to handle class imbalance
    num_pos = len(df[df['labels'] == 1])
    num_neg = df.shape[0] - num_pos
    num_to_drop = int((num_neg*.79) - num_pos) #25% pos - tinker, grid search later
    df.drop(df.query('labels == 0').sample(n=num_to_drop).index, inplace=True)
    return df


def scale(df, cols):
    scaler = RobustScaler(copy=False)
    df[cols] = scaler.fit_transform(df[cols])
    return df

# def scale(df, cols):
#     nums = df[cols]
#     df.drop(columns=cols, inplace=True)
#
#     scaler = RobustScaler(copy=False)
#     scaled_nums = scaler.fit_transform(nums)
#
#     scaled_df = pd.concat([df, scaled_nums], axis=1)
#     return scaled_df

# def scale(df, cols):
#     scaler = RobustScaler()
#     indices = [df.columns.get_loc(c) for c in df.columns if c in cols]
#     df[indices] = scaler.fit_transform(df[cols])
#     return df
