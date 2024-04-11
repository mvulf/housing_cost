import pandas as pd
import numpy as np

num_regex = r'[0-9]+\.{0,1}[0-9]*'

def get_max(value):
    """Get maximum value in the case of list and just value, if it single

    Args:
        value: list or scalar

    Returns:
        scalar (max in the case of list)
    """
    if isinstance(value, list):
        if len(value) > 0:
            return max(value)
        else:
            return np.nan
    return value


def replace(
    series:pd.Series, 
    to_replace:tuple,
)->pd.Series:
    """Replace by to_replace

    Args:
        stories: Series with stories
        to_replace: tuple(str, str)

    Returns:
        Series with numbered replacement (stories by default)
    """
    series = (
        series.str.lower()
        .str.strip()
    )
    
    for item in to_replace:
        series = series.str.replace(
            *item
        )
    
    return series


def convert_to_none(
    series:pd.Series,
    to_none:tuple,
    contains:bool=True,
    
)->pd.Series:
    """If series contains sub-strings from "to_none" - explicitly set None

    Args:
        series: Series to get Nones
        to_none: Tuple of sub-strings to mark rows as Nones
        contains: if True - check that series contains string. 
        Compare to full string if False

    Returns:
        Series with Nones instead of sub-string (or whole string) from "to_none"
    """
    series = series.copy()
    series = (
        series.str.lower()
        .str.strip()
    )
    
    for sub_str in to_none:
        if len(sub_str) == 0:
            series = series.replace(r'^s*$', np.nan, regex=True)
        else:
            if contains:
                series.loc[
                    series.str.contains(sub_str, na=False, regex=True)
                ] = np.nan
            else:
                series.loc[
                    series == sub_str
                ] = np.nan
    
    return series


def get_numerical_target(target:pd.Series)->pd.Series:
    """Get numerical feature from raw dataset. 
    Can contain only "$", "+", "," to drop.

    Args:
        target: Series to transform

    Returns:
        Transformed Series
    """
    if target.dtype == 'O':
        target = replace(
            target,
            to_replace=(
                ('$', ''),
                ('+', ''),
                (',', '')
            )
        )
        return pd.to_numeric(target)
    return target


# SQFT

def get_df_numerical_sqft(df:pd.DataFrame)->pd.DataFrame:
    """Mark rows with explicitly checked interior area.
    Make Nones for missing sqft.
    Get numerical sqft

    Args:
        df: dataframe to mark

    Returns:
        Copy of dataframe with "marked_interior_area" and numerical sqft
    """
    if df['sqft'].dtype == 'O':
        df = df.copy()
        
        # Mark Interior area if exists
        interior_area_mask = (
            df['sqft']
            .str.lower()
            .str.strip()
            .str.contains('total interior livable area', na=False)
        )
        df['marked_interior_area'] = False
        df.loc[interior_area_mask, 'marked_interior_area'] = True
        
        # Get Implicit Nans
        na_mask = (
            df['sqft'].str.contains('--', na=False)
        )
        df.loc[na_mask, 'sqft'] = np.nan
        
        # Drop symbols:
        df['sqft'] = df['sqft'].str.replace(
            r'[^0-9]+', 
            '', 
            regex=True
        )
        df['sqft'] = pd.to_numeric(df['sqft'])

    return df


#STORIES

def get_df_numerical_story(df:pd.DataFrame)->pd.DataFrame:
    """ Fill numerical story data and replace to numbers values in stories

    Args:
        df (pd.DataFrame): Dataframe to change

    Returns:
        pd.DataFrame: Corrected DF
    """
    if df['stories'].dtype == 'O':
        df = df.copy()
        df['stories'] = df['stories'].str.lower().str.strip()
        df['stories'] = replace(
            df['stories'],
            to_replace=(
                ('ground', '1'),
                ('one', '1'),
                ('two', '2'),
                ('three', '3'),
            )
        )
        
        df['stories_num'] = np.nan
        # Retrieve nums from stories
        series_with_nums = df['stories'].str.findall(num_regex)
        df['stories_num'] = pd.to_numeric(
            series_with_nums.apply(get_max)
        )
        df.loc[df['stories_num'] == 0, 'stories_num'] = np.nan
        # Lands has no stories
        lot_mask = (
            df['stories'].isin(['lot', 'acreage']) &
            df['stories_num'].isna()
        )
        df.loc[lot_mask, 'stories_num'] = 0
        
        # Consider types with one story
        ranch_mask = (
            df['stories'].isin(['ranch', 'traditional']) &
            df['stories_num'].isna()
        )
        df.loc[ranch_mask, 'stories_num'] = 1
        
        # Let us assume, that split houses has more than 1 story
        split_mask = (
            (
                df['stories'].str.contains('multi', na=False) |
                df['stories'].str.contains('split', na=False) |
                df['stories'].str.contains('tri', na=False) |
                df['stories'].str.contains('stories/levels', na=False)
            ) &
            df['stories_num'].isna()
        )
        df.loc[split_mask, 'stories_num'] = 1.5
        
        townhouse_mask = (
            df['stories'].isin(['townhouse', 'bi-level']) &
            df['stories_num'].isna()
        )
        df.loc[townhouse_mask, 'stories_num'] = 2
        
        condominium_mask = (
            df['stories'].isin(['condominium', 'mid-rise']) &
            df['stories_num'].isna()
        )
        df.loc[condominium_mask, 'stories_num'] = 3
        
        df['stories'] = df['stories_num']
        df = df.drop('stories_num', axis=1)
        
    return df


# BATHS and BEDS

def get_numerical_feature(
    series:pd.Series,
    to_none:tuple=(),
    to_replace:tuple=(),
)->pd.Series:
    """Get numerical values if they are not ambiguous 
    (only one number contains). 
    But before getting numerical feature - refine series

    Args:
        series: series to get numerical feature
        to_none: tuple to get none value. Defaults to ().
        to_replace: tuple to replace. Defaults to ().

    Returns:
        Series with numerical type
    """
    series = series.copy()
    
    if series.dtype == 'O':
        series = convert_to_none(series, to_none)
        series = replace(series, to_replace)
        
        nums = series.str.findall(num_regex)
        nums = nums.apply(lambda x: x if isinstance(x, list) else [])
        
        nums = nums.apply(lambda x: x[0] if len(x) == 1 else np.nan)
        
        return pd.to_numeric(nums)
    
    return series


def get_df_beds_from_baths(
    df:pd.DataFrame
)->pd.DataFrame:
    """ Function considers cases when word "Baths" included instead of number.
    Then we assume that number of baths is the same as the number of beds,
    so there are bedrooms with baths.

    Args:
        df: Source dataframe

    Returns:
        Dataframe with updated beds-count
    """
    df = df.copy()
    df.loc[df['beds'] == 'Baths', 'beds'] =\
        df.loc[df['beds'] == 'Baths', 'baths'].apply(str)
    
    return df


# Private Pool

def get_df_private_pool(df:pd.DataFrame)->pd.DataFrame:
    """Create bool private_pool feature and drop origins

    Args:
        df: Source df

    Returns:
        df with bool private_pool
    """
    df = df.copy()
    if 'private pool' in df.columns and 'PrivatePool' in df.columns:
        # fill with "yes" if mark exists
        df['private_pool'] = False
        df.loc[
            df['private pool'].notna() | df['PrivatePool'].notna(), 
            'private_pool'
        ] = True
        # Drop old columns
        df = df.drop(['private pool', 'PrivatePool'], axis=1)
    return df


# Fireplace

def get_bool_feature(
    series:pd.Series,
    to_none:tuple=(),    
)->pd.Series:
    """Create bool feature by None - False

    Args:
        series: series to get numerical feature
        to_none: tuple to get none value. Defaults to ().

    Returns:
        Series with boolean type
    """
    
    series = series.copy()
    
    if series.dtype == 'O':
        # Convert to none 
        series = convert_to_none(series, to_none, contains=False)
        # True, if values are keeped
        series = series.notna()
    
    return series


# MLS

def get_df_mls(df:pd.DataFrame)->pd.DataFrame:
    """Get "mls" which is True, when any information exists, and False, 
    when None or implicit None

    Args:
        df: Source df

    Returns:
        df with "mls"-feature and dropped origin features
    """
    if 'mls-id' in df.columns and 'MlsId' in df.columns:
        df = df.copy()
        # Get lowercase and strip
        df['mls-id'] = (
            df['mls-id'].str.lower()
            .str.strip()
        )
        df['MlsId'] = (
            df['MlsId'].str.lower()
            .str.strip()
        )
        # Drop implicit Nones
        for mls_column in ['mls-id', 'MlsId']:
            df[mls_column] = (
                df[mls_column].str.lower()
                .str.strip()
            )
            df.loc[
                df[mls_column].str.contains('no', na=False) &
                ~df[mls_column].str.contains(',', na=False),
                mls_column
            ] = np.nan
            df[mls_column] = convert_to_none(df[mls_column], ('',))
        # Fill with True, if any information exists
        df['mls'] = False
        df.loc[
            df['mls-id'].notna() | df['MlsId'].notna(), 
            'mls'
        ] = True
        # Drop old columns
        df = df.drop(['mls-id', 'MlsId'], axis=1)
        
    return df