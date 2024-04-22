import sys
import os
import pandas as pd
from pathlib import Path
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json
import re

root_folder = '../../'

sys.path.append(root_folder)
# from src.data import make_dataset
from src import utils
from src.features import build_features, feature_stats

source_data_path = Path(root_folder, 'data', 'raw', 'data.csv')
first_export_path = Path(
    root_folder, 'data', 'interim', '1.0_first_process.csv'
)
preprocessed_data_path = Path(
    root_folder, 'data', 'interim', '2.1_preprocessed.csv'
)

def make_first_dataset(
    import_data_path:str=source_data_path,
    export:bool=True,
    export_data_path:str=first_export_path,
    verbose:bool=True,
)->pd.DataFrame:
    raw_df = pd.read_csv(
        import_data_path
    )
    print(f'Init shape: {raw_df.shape}')
    if verbose:
        raw_df.info()
    df = raw_df.dropna(subset=['target'])
    # Drop month-payed
    mo_mask = df['target'].str.contains('mo')
    df = df[~mo_mask]
    # Target
    df['target'] = build_features.get_numerical_target(df['target'])
    # Sqft and story
    df = build_features.get_df_numerical_sqft(df)
    df = build_features.get_df_numerical_story(df)
    # Beds and Baths
    df['baths'] = build_features.get_numerical_feature(
        df['baths'],
        to_none=('--', '~', ''),
        to_replace=((',', '.'),)
    )
    df = build_features.get_df_beds_from_baths(df)
    df['beds'] = build_features.get_numerical_feature(
        df['beds'],
        to_none=('--', '~', '', 'sqft', 'acre'),
    )
    # fill with True if mark exists
    df = build_features.get_df_private_pool(df)
    # MLS
    df = build_features.get_df_mls(df)
    # Prepare fireplace
    no_fireplace_labels = ('not applicable', 'no', 'no fireplace', '0')
    df['fireplace'] = build_features.get_bool_feature(
        df['fireplace'], no_fireplace_labels
    )
    # Status
    df['status'] = build_features.get_status_feature(df['status'])
    # Property type
    df = build_features.get_df_proptype(df)
    # Street
    df['street'] = build_features.get_street_feature(df['street'])
    # CLEAN INDEX FOR CONVENIENT WORK.
    # Then get home facts and schools
    df = df.reset_index(drop=True)
    df = build_features.get_df_home_facts(df)
    df = build_features.get_df_schools(df)
    # Year built
    df['year_built'] = build_features.get_num_year(df['year_built'])
    # Lotsize
    df['lotsize'] = build_features.get_numerical_lotsize(df['lotsize'])
    #Parking
    df = build_features.get_df_parking(df)
    # Heating and Cooling
    df = build_features.get_df_heat_cool(df)
    
    if export:
        df.to_csv(export_data_path)
    
    print(f'Result shape: {df.shape}')
    if verbose:
        df.info()
    
    return df


def make_preprocessed_dataset(
    import_data_path:str=first_export_path,
    export:bool=True,
    export_data_path:str=preprocessed_data_path,
    verbose:bool=True,
)->pd.DataFrame:
    """See detailed explanation in notebooks/2.1.-mv-eda.ipynb
    """
    df = pd.read_csv(
        import_data_path,
        index_col=0
    )
    df = build_features.get_df_with_lists(df)
    
    if verbose:
        df.info()

    service_cols = [
        'marked_interior_area',
        'price/sqft'
    ]

    advert_cols = [
        'status',
        'mls',
    ]

    address_cols = [
        'zipcode',
        'state',
        'city',
        'street'
    ]

    home_property_cols = [
        'property_type',
        'sqft',
        'lotsize',
        'stories',
        'baths',
        'beds',
        'fireplace',
        'private_pool',
        'year_built',
        'remodeled_year',
        'parking_type',
        'parking_count',
        'heating_type',
        'cooling_type',
        'central_heating',
        'central_cooling',
    ]

    school_data = [
        'school_rating',
        'school_distance',
        'school_grades',
        'school_name'
    ]

    check_duplicates_cols = [
        'target',
        *advert_cols,
        *address_cols,
        *home_property_cols,
    ]
    
    duplicated_mask = df[check_duplicates_cols].duplicated()
    duplicates_cnt = duplicated_mask.sum()
    print('Количество дубликатов:', duplicates_cnt)
    df = df.drop_duplicates(subset=check_duplicates_cols)
    print(f'Размер датасета: {df.shape}')
    
    df = df.drop(service_cols, axis=1)
    print(f'Размер датасета: {df.shape}')
    
    check_nans = [
        *advert_cols,
        *address_cols,
        *home_property_cols,
    ]
    row_nans = df[check_nans].isna().sum(axis=1)
    df = df[row_nans < 10]
    print(f'Размер датасета: {df.shape}')
    
    df = df[df['target'].between(1e4, 1e7)]
    print(f'Размер датасета: {df.shape}')
    
    df['log_target'] = np.log10(df['target'])
    
    clean, outliers = feature_stats.get_df_no_outliers(
        data=df,
        feature='log_target',
        method='z-score'
    )
    df = clean
    print(f'Размер датасета: {df.shape}')
    
    isin_dict = {
        'contract/new': ['new', 'under_contract'],
        'active/sale': ['active', 'auction', 'sale'],
        'foreclosure/pending': [
            'foreclosure', 'pre-foreclosure', 
            'pending', 'backup', 'contingency'
        ],
    }

    df['status'] = build_features.get_categorical_feature(
        df['status'],
        isin_dict=isin_dict
    )
    df['status'] = df['status'].fillna(df['status'].mode().iloc[0])
    
    df = df.drop(['street', 'zipcode', 'city'], axis=1)
    print(f'Размер датасета: {df.shape}')
    
    states_count = df['state'].value_counts(normalize=True)
    other_states = states_count.iloc[8:].index
    df['state'] = build_features.get_categorical_feature(
        df['state'],
        isin_dict={
            "other": other_states,
        },
        first_launch=False,
        lower_strip=False,
    )
    
    df = df[df['property_type'] != 'apt']
    print(f'Размер датасета: {df.shape}')
    df.loc[df['property_type'] == 'cape', 'property_type'] = np.nan

    # Fill in advance land properties as zero:
    df.loc[df['property_type'] == 'land', ['stories', 'beds', 'baths']] = 0

    df['property_type'] = build_features.get_categorical_feature(
        df['property_type'],
        isin_dict={
            "styled": ["log", "spanish", "europe", "modern", "transitional"],
            "condo": ["colonial", "condo"],
            "ranch/traditional": ["ranch", "traditional", "florida"],
            "land/mobile": ["mobile", "land"]
        },
        first_launch=False,
        lower_strip=False,
    )
    df['property_type'] = df['property_type'].fillna(
        df['property_type'].mode().iloc[0]
    )
    
    for feature in ('beds', 'baths'):
        df.loc[df[feature]>20, feature] = np.nan
    df = df[~(df['stories'] > 100)]
    print(f'Размер датасета: {df.shape}')

    for feature in ('stories', 'beds', 'baths'):
        df[feature] = np.round(df[feature])
        
    # FILL NANS according to property_type
    features = ('stories', 'beds', 'baths')
    reference = 'property_type'
    df = feature_stats.fillna_by_reference(
        df,
        reference=reference,
        features=features
    )

    # CROP ACCORDING TO THRESHOLD
    feature_thresholds = {
        'stories': (0, 4),
        'baths': (0, 6),
        'beds': (0, 7)
    }
    df = feature_stats.clip_by_thresholds(
        df, 
        feature_thresholds=feature_thresholds
    )
    
    features = ('sqft', 'lotsize')
    for feature in features:
        new_feature = 'log_' + feature
        df[new_feature] = np.log10(df[feature]+1)
    df = df.drop(list(features), axis=1)
    
    log_features = ('log_sqft', 'log_lotsize')

    # outliers_idx = {}
    for feature in log_features:
        print(feature)
        clean, outliers = feature_stats.get_df_no_outliers(
            data=df,
            feature=feature,
            method='z-score'
        )
        df.loc[outliers.index, feature] = np.nan
        # Fill forward to make less distribution disturbance
        df[feature] = df[feature].ffill()
        # Fill others
        df[feature] = df[feature].fillna(df[feature].median())
        
        true_outliers_cnt = outliers[outliers[feature].notna()].shape[0]
        print('Истинные выбросы (не NaNs):', true_outliers_cnt)
        print()
        
    df['parking_count'] = df['parking_count'].clip(0, 3)
    df['parking_count'] = df['parking_count'].fillna(0)

    df['parking_type'] = build_features.get_categorical_feature(
        df['parking_type'],
        isin_dict={
            'street': ['on street', 'off street'],
            'other': ['other', 'carport']
        },
        first_launch=False
    )
    df['parking_type'] = df['parking_type'].fillna('street')

    df['heating_type'] = build_features.get_categorical_feature(
        df['heating_type'],
        isin_dict={
            'other': ['other', 'furnace', 'radiant', 'wall', 'baseboard'],
            'gas/pump': ['gas', 'pump'],
        },
        first_launch=False
    )

    df['cooling_type'] = build_features.get_categorical_feature(
        df['cooling_type'],
        isin_dict={
            'other': ['other', 'wall'],
            'refrigeration/fan': [
                'electric', 'refrigeration', 'evaporative', 'fan'
            ],
        },
        first_launch=False
    )

    df = feature_stats.fillna_by_reference(
        df,
        reference='property_type',
        features=('cooling_type', 'heating_type')
    )
    
    df = build_features.get_df_updated_years(df)

    features = ('updated_years',)
    df = feature_stats.fill_outliers(df, features=features, method='iqr')

    # Fill forward to make less distribution disturbance
    feature = features[0]
    df[feature] = df[feature].ffill()

    # Drop original columns
    df = df.drop(['year_built'], axis=1)
    
    df = df.drop(['school_grades', 'school_name'], axis=1)

    df['school_count'] = df['school_distance'].apply(len)
    df['school_distance'] = build_features.get_numerical_distance(
        df['school_distance']
    )
    df['min_log_school_distance'] = (
        np.log10(df['school_distance'].apply(np.min) + 1e-3)
    )
    df['median_log_school_distance'] = (
        np.log10(df['school_distance'].apply(np.median) + 1e-3)
    )

    df['school_rating'] = build_features.get_numerical_rating(
        df['school_rating']
    )
    df['max_school_rating'] = df['school_rating'].apply(np.max)
    df['median_school_rating'] = df['school_rating'].apply(np.median)

    df = df.drop(['school_distance', 'school_rating'], axis=1)

    df['school_count'] = df['school_count'].clip(1, 8)

    features = ('min_log_school_distance', 'median_log_school_distance')

    df = feature_stats.fill_outliers(df, features=features)

    # Fill NaNs
    features = (
        'min_log_school_distance', 'median_log_school_distance',
        'max_school_rating', 'median_school_rating'
    )
    for feature in features:
        df[feature] = df[feature].ffill()
        df[feature] = df[feature].fillna(df[feature].median())

    if verbose:
        df.info()
    if export:
        df.to_csv(export_data_path)

if __name__ == '__main__':
    # make_first_dataset()
    make_preprocessed_dataset()