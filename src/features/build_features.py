import pandas as pd
import numpy as np


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
            return None
    return value


def replace(
    series:pd.Series, 
    replace_map:tuple,
)->pd.Series:
    """Replace by replace_map

    Args:
        stories: Series with stories
        replace_map: tuple(str, str)

    Returns:
        Series with numbered replacement (stories by default)
    """
    series = series.str.lower()
    
    for to_replace in replace_map:
        series = series.str.replace(
            *to_replace
        )
    
    return series


def convert_to_none(
    series:pd.Series,
    to_none:tuple,
    
):
    series = series.copy()
    series = (
        series.str.lower()
        .str.strip()
    )
    
    for sub_str in to_none:
        if len(sub_str) == 0:
            series = series.replace(r'^s*$', np.nan, regex=True)
        else:
            series.loc[series.str.contains(sub_str, na=False, regex=True)] = None
    
    return series


def get_numerical_target(target:pd.Series):
    """Get numerical feature from raw dataset. Can contain only "$", "+", "," to drop.

    Args:
        target: Series to transform

    Returns:
        Transformed Series
    """
    if target.dtype == 'O':
        target = replace(
            target,
            replace_map=(
                ('$', ''),
                ('+', ''),
                (',', '')
            )
        )
        return pd.to_numeric(target)
    return target


# SQFT

def get_df_with_numerical_sqft(df:pd.DataFrame):
    """Mark rows with explicitly checked interior area.
    Make Nones for missing sqft.
    Get numerical sqft

    Args:
        df: dataframe to mark

    Returns:
        Copy of dataframe with "marked_interior_area" and numerical sqft
    """
    if df['sqft'].dtype == 'O':
        df_copy = df.copy()
        
        # Mark Interior
        interior_area_mask = (
            df['sqft']
            .str.lower()
            .str.strip()
            .str.contains('total interior livable area', na=False)
        )
        df_copy['marked_interior_area'] = False
        df_copy.loc[interior_area_mask, 'marked_interior_area'] = True
        
        # Get Nans
        na_mask = (
            df['sqft'].str.contains('--', na=False)
        )
        df_copy.loc[na_mask, 'sqft'] = None
        
        # Drop symbols:
        df_copy['sqft'] = df_copy['sqft'].str.replace(
            r'[^0-9]+', 
            '', 
            regex=True
        )
        df_copy['sqft'] = pd.to_numeric(df_copy['sqft'])
        
        return df_copy
    return df


#STORIES

def get_df_with_numerical_story(df:pd.DataFrame):
    """ Fill numerical story data and replace to numbers values in stories

    Args:
        df (pd.DataFrame): Dataframe to change

    Returns:
        pd.DataFrame: Corrected DF
    """
    df_copy = df.copy()
    df_copy['stories'] = df_copy['stories'].str.lower().str.strip()
    df_copy['stories'] = replace(
        df_copy['stories'],
        replace_map=(
            ('ground', '1'),
            ('one', '1'),
            ('two', '2'),
            ('three', '3'),
        )
    )
    
    df_copy['stories_num'] = None
    # Retrieve nums from stories
    series_with_nums = df_copy['stories'].str.findall(r'[0-9]+[\.]{0,1}[0-9]*')
    df_copy['stories_num'] = pd.to_numeric(
        series_with_nums.apply(get_max)
    )
    df_copy.loc[df_copy['stories_num'] == 0, 'stories_num'] = None
    # Lands has no stories
    lot_mask = (
        df_copy['stories'].isin(['lot', 'acreage']) &
        df_copy['stories_num'].isna()
    )
    df_copy.loc[lot_mask, 'stories_num'] = 0
    
    ranch_mask = (
        df_copy['stories'].isin(['ranch', 'traditional']) &
        df_copy['stories_num'].isna()
    )
    df_copy.loc[ranch_mask, 'stories_num'] = 1
    
    # Let us assume, that split houses has more than 1 story
    split_mask = (
        (
            df['stories'].str.contains('multi', na=False) |
            df['stories'].str.contains('split', na=False) |
            df['stories'].str.contains('tri', na=False) |
            df['stories'].str.contains('stories/levels', na=False)
        ) &
        df_copy['stories_num'].isna()
    )
    df_copy.loc[split_mask, 'stories_num'] = 1.5
    
    townhouse_mask = (
        df_copy['stories'].isin(['townhouse', 'bi-level']) &
        df_copy['stories_num'].isna()
    )
    df_copy.loc[townhouse_mask, 'stories_num'] = 2
    
    condominium_mask = (
        df_copy['stories'].isin(['condominium', 'mid-rise']) &
        df_copy['stories_num'].isna()
    )
    df_copy.loc[condominium_mask, 'stories_num'] = 3
    
    return df_copy


# BATHS and BEDS

