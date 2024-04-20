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
from src.features import build_features

source_data_path = Path(root_folder, 'data', 'raw', 'data.csv')
first_export_path = Path(root_folder, 'data', 'interim', '1.0_first_process.csv')

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
    feature_name = 'year_built'
    df[feature_name] = build_features.get_num_year(df[feature_name])
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


if __name__ == '__main__':
    make_first_dataset()