import pandas as pd
import numpy as np
import json
import re

num_regex = r'[0-9]+\.{0,1}[0-9]*'


def replace_str(
    string:str,
    to_replace:tuple,
    lower_strip:bool=True,
    regexp:bool=False,
)->str:
    """Replace substrings in string

    Args:
        string: original string for replace
        to_replace: tuple of tuples to replace
        lower_strip: Make lower if true. Defaults to True.

    Returns:
        String with replaced substings
    """
    if lower_strip:
        string = string.lower().strip()
    for item in to_replace:
        if regexp:
            string = re.sub(*item, string)
        else:
            string = string.replace(*item)
    return string
        

def convert_str_to_none(
    string:str,
    to_none:tuple,
    contains:bool=True,
    lower_strip:bool=True,
)->str:
    """Convert string to none, if it contains substrings from to_none

    Args:
        string: original string
        to_none: tuple of substrings to check
        contains: just contains or full coincidence. Defaults to True.
        lower_strip: Make lower if true. Defaults to True.

    Returns:
        String or np.nan if original string contains any substrings from to_none
    """
    if lower_strip:
        string = string.lower().strip()
    
    for sub_str in to_none:
        if len(sub_str) == 0:
            if len(string) == 0:
                return np.nan
        else:
            if contains:
                if sub_str in string:
                    return np.nan
            else:
                if sub_str == string:
                    return np.nan
    return string
    

def get_array(
    str_list:list,
    to_replace=(),
    to_none=(),
    dtype='float32'
):
    """Get numerical array 

    Args:
        str_list: list of string for convertation to array
        to_replace: Tuple of tuples to replace (old_str, new_str). 
        Defaults to ().
        to_none: tuple for convertation to none. Defaults to ().
    """
    def prepare_string(
        string,
    )->str:
        """
        Returns:
            String with replacements or even none, if to_none condition satisfied
        """
        string = replace_str(
            string,
            to_replace=to_replace
        )
        string = convert_str_to_none(
            string,
            to_none=to_none
        )
        return string
        
    if isinstance(str_list, list):
        if len(str_list) > 0:
            array = np.array(
                list(map(
                    lambda x: prepare_string(x),
                    str_list
                )),
                dtype=dtype
            )
            return array
    return np.nan


# def get_min(value):
#     """Get minimum value in the case of list and just value, if it single

#     Args:
#         value: list or scalar

#     Returns:
#         scalar (min in the case of list)
#     """
#     if isinstance(value, list):
#         if len(value) > 0:
#             return min(value)
#         else:
#             return np.nan
#     return value

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
    lower_strip:bool=True,
)->pd.Series:
    """Replace by to_replace

    Args:
        stories: Series with stories
        to_replace: tuple of tuple(str, str)
        lower_strip: if True, make lower and strip

    Returns:
        Series with numbered replacement (stories by default)
    """
    if lower_strip:
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
    lower_strip:bool=True,
    
)->pd.Series:
    """If series contains sub-strings from "to_none" - explicitly set None

    Args:
        series: Series to get Nones
        to_none: Tuple of sub-strings to mark rows as Nones
        contains: if True - check that series contains string. 
        Compare to full string if False
        lower_strip: if True, make lower and strip

    Returns:
        Series with Nones instead of sub-string (or whole string) from "to_none"
    """
    series = series.copy()
    if lower_strip:
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


def get_numerical_feature(
    series:pd.Series,
    to_none:tuple=(),
    to_replace:tuple=(),
    # strategy:str='first', # Should be in ['first', 'last', 'max', 'min'] 
)->pd.Series:
    """Get numerical values if they are not ambiguous 
    (only one number contains). 
    But before getting numerical feature - refine series

    Args:
        series: series to get numerical feature
        to_none: tuple to get none value. Defaults to ().
        to_replace: tuple to replace. Defaults to ().
        strategy: what value get from list. 

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


def get_mask_dict(
    series:pd.Series,
    keywords:tuple=(),
)->dict:
    """Get dict of masks acording to keywords

    Args:
        series: series for mask obtaining
        keywords: tuple of words to get mask. Defaults to ().

    Returns:
        Dict {keyword: mask}
    """
    mask_dict = {}
    for keyword in keywords:
        mask_dict[keyword] = series.str.contains(keyword, na=False)
    return mask_dict


def get_categorical_feature(
    series:pd.Series,
    isin_dict:dict=None,
    mask_dict:dict=None,
    lower_strip:bool=True,
    first_launch:bool=True,
)->pd.Series:
    """Get categorical grouped categorical features according to dicts.

    Args:
        series: Source series
        isin_dict: Dict which keys - new values, and values are lists of 
        old values. They are used to check that value isin the list. 
        Defaults to None.
        mask_dict: Dict which keys - new values, and values are mask to which 
        necessary to apply new values. Defaults to None.
        lower_strip: if True, make lower and strip
        first_launch: init series with nans, if first_launch == True
        

    Returns:
        Series with new groups
    """
    if lower_strip:
        series = series.str.strip().str.lower()
    
    new_series = series.copy()
    if first_launch:
        new_series.loc[:] = np.nan
    # Prepare group values
    if isinstance(isin_dict, dict):
        for new_value, old_values in isin_dict.items():
            new_series.loc[series.isin(old_values)] = new_value
    # Fill by masks:
    if isinstance(mask_dict, dict):
        for new_value, mask in mask_dict.items():
            new_series.loc[mask] = new_value
    
    return new_series


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


def get_series_json(
    series:pd.Series,
    to_replace:tuple=(
        ("'", '"'),
        ('None', 'null'),
    ),
    complicated_quotes:bool=True,
)->pd.Series:
    """ Get deserialized series with string-transformation to python 
    structures, like lists or dict via json.loads

    Args:
        series: original series with str instead of 
        python-structure in the cells
        to_replace: tuple to replace
        complicated_quotes: process complicated quotes if True

    Returns:
        deserialized series
    """
    series = series.copy()

    if complicated_quotes:
        # Replace single quotes inside another quotes pair
        series = series.apply(
            lambda string:\
                re.sub(r" '([a-zA-Z]+)' ", r" \g<1> ", string)
        )
        # Replace single quote to "`"
        series = series.apply(
            lambda string:\
                re.sub(
                    r'([a-zA-Z]{1}\\*)\'(\s*[a-zA-Z]{1})', 
                    r'\g<1>`\g<2>', 
                    string
                )
        )
    # Replace single quotes to double, None to json-applicable null
    series = replace(
        series=series,
        to_replace=to_replace,
        lower_strip=False
    )
    # Load data via json and return series in python-structure
    series_json = series.apply(json.loads)
    return series_json


def get_df_with_lists(
    df:pd.DataFrame,
    column_names:tuple=(
        'school_rating',
        'school_distance',
        'school_grades',
        'school_name'
    ),
    complicated_quotes:bool=False,
)->pd.DataFrame:
    """Convert str to lists

    Args:
        df: Origin df
        column_names: Columns to convert str to lists. 
        Defaults to ( 'school_rating', 'school_distance', 'school_grades', 'school_name' ).
        complicated_quotes: process complicated quotes if True

    Returns:
        Dataframe with lists instead of str inside series
    """
    
    df = df.copy()
    for name in column_names:
        if name in df.columns:
            df[name] = get_series_json(
                df[name],
                complicated_quotes=complicated_quotes,
            )
        else:
            print(f'WARNING: {name} is not in the df.columns')
            
    return df


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


# Fireplace used bool_feature


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
            df[mls_column] = convert_to_none(df[mls_column], ('',), contains=False)
        # Fill with True, if any information exists
        df['mls'] = False
        df.loc[
            df['mls-id'].notna() | df['MlsId'].notna(), 
            'mls'
        ] = True
        # Drop old columns
        df = df.drop(['mls-id', 'MlsId'], axis=1)
        
    return df
    

# STATUS

def get_status_feature(status:pd.Series)->pd.Series:
    """ Prepare isin- and mask- dicts to set them to get_categorical_feature

    Args:
        status: Series for update

    Returns:
        Series with updated categories
    """
    status = status.str.strip().str.lower()
    
    isin_status_dict = {
        'sale': ['for sale'],
        'active': ['active', 'a active'],
    }
    
    foreclosure_mask = (
        status.str.contains('foreclos', na=False) & 
        ~status.str.contains('pre', na=False)
    )
    
    pre_foreclosure_mask = (
        status.str.contains('foreclos', na=False) & 
        status.str.contains('pre', na=False)
    )
    
    under_contract_mask = (
        status.str.contains('under contract', na=False) &
        ~status.str.contains('backup', na=False)
    )
    
    backup_mask = (
        status.str.contains('backup', na=False)
    )
    
    contingency_mask = (
        status.str.contains('contingen', na=False)
    )
    
    # Let us consider "p" as pending: https://support.mlslistings.com/s/article/Status-Definitions 
    pending_mask = (
        (
            status.str.contains('pending', na=False) |
            (status == 'p')
        ) &
        ~backup_mask &
        ~contingency_mask
    )
    
    auction_mask = (
        status.str.contains('auction', na=False) &
        ~pre_foreclosure_mask
    )
    
    new_mask = (
        status.str.contains('new', na=False)
    )
    
    mask_status_dict = {
        'foreclosure': foreclosure_mask,
        'pre-foreclosure': pre_foreclosure_mask,
        'under_contract': under_contract_mask,
        'backup': backup_mask,
        'contingency': contingency_mask,
        'pending': pending_mask,
        'auction': auction_mask,
        'new': new_mask,
    }
    
    new_status = get_categorical_feature(
        status,
        isin_dict=isin_status_dict,
        mask_dict=mask_status_dict,
    )
    
    return new_status


def get_df_proptype(df:pd.DataFrame)->pd.DataFrame:
    
    if 'propertyType' in df.columns:
        df = df.copy()
        df['propertyType'] = df['propertyType'].str.lower().str.strip()
        property_type = df['propertyType']
        
        # FILL TYPES
        land_mask = (
            property_type.isin(['lot/land', 'land'])
        )
        
        mobile_mask = (
            property_type.str.contains('manufactured', na=False) |
            property_type.str.contains('mobile', na=False) |
            property_type.str.contains(r'mo2\S*le', regex=True, na=False) |
            property_type.str.contains('prefab', na=False) |
            property_type.str.contains('modular', na=False)
        )
        
        ranch_mask = (
            (
                property_type.str.contains('ranch', na=False) |
                property_type.str.contains('bungalow', na=False) |
                property_type.str.contains('cabin', na=False) |
                property_type.str.contains(r'ca2\S*n', na=False) |
                property_type.str.contains(r'cottage', na=False) |
                property_type.str.contains(r'farm', na=False) |
                property_type.str.contains(r'garden home', na=False)
            ) &
            ~mobile_mask
        )
        
        single_mask = (
            (
                property_type.str.contains('single', na=False) |
                property_type.str.contains('detached', na=False) |
                property_type.str.contains('split', na=False) |
                property_type.str.contains('multi-level', na=False) |
                property_type.str.contains(r'low[-\s]*rise', na=False)
            ) &
            ~property_type.str.contains('condominium', na=False) &
            ~ranch_mask &
            ~mobile_mask
        )
        
        multi_mask = (
            (
                property_type.str.contains(
                    r'multi[-\s]+family', regex=True, na=False
                ) |
                property_type.str.contains(r'mid[-\s]*rise', na=False) |
                property_type.str.contains(r'high[-\s]*rise', na=False)
            ) &
            ~property_type.str.contains('condominium', na=False) &
            ~ranch_mask &
            ~mobile_mask &
            ~single_mask
        )
        
        town_mask = (
            property_type.str.contains('town', na=False)
        )
        
        condo_mask = (
            (
                property_type.str.contains(r'co-*op', regex=True, na=False) |
                property_type.str.contains('condo', na=False)
            ) &
            ~town_mask
        )
        
        apt_mask = (
            property_type.str.contains('apart', na=False) &
            ~town_mask &
            ~condo_mask &
            ~ranch_mask
        )
        
        mask_type_dict = {
            'land': land_mask,
            'mobile': mobile_mask,
            'ranch': ranch_mask,
            'single': single_mask,
            'multi': multi_mask,
            'town': town_mask,
            'condo': condo_mask,
            'apt': apt_mask,
        }
        
        new_property_type = get_categorical_feature(
            property_type,
            mask_dict=mask_type_dict,
        )
        df['property_type'] = new_property_type
        
        # FILL STYLES, if is empty
        modern_mask = (
            (
                (
                    property_type.str.contains(r'modern', na=False) |
                    property_type.str.contains('contemp', na=False)
                ) &
                ~property_type.str.contains(r'mid[-\s]*century', na=False)
            ) &
            df['property_type'].isna()
        )
        
        traditional_mask = (
            (
                property_type.str.contains('traditional', na=False) |
                property_type.str.contains(r'mid[-\s]*century', na=False)
            ) &
            ~modern_mask &
            df['property_type'].isna()
        )
        
        colonial_mask = (
            property_type.str.contains('colonial', na=False) &
            ~modern_mask &
            ~traditional_mask &
            df['property_type'].isna()
        )
        
        spanish_mask = (
            (
                property_type.str.contains('mediterranean', na=False) |
                property_type.str.contains('spanish', na=False)
            ) &
            ~modern_mask &
            ~traditional_mask &
            ~colonial_mask &
            df['property_type'].isna()
        )
        
        florida_mask = (
            (
                property_type.str.contains('florida', na=False)
            ) &
            ~modern_mask &
            ~traditional_mask &
            ~colonial_mask &
            ~spanish_mask &
            df['property_type'].isna()
        )
        
        transitional_mask = (
            (
                property_type.str.contains('transitional', na=False)
            ) &
            ~modern_mask &
            ~traditional_mask &
            ~colonial_mask &
            ~spanish_mask &
            ~florida_mask &
            df['property_type'].isna()
        )
        
        europe_mask = (
            (
                property_type.str.contains('victorian', na=False) |
                property_type.str.contains('queen', na=False) |
                property_type.str.contains('european', na=False) |
                property_type.str.contains('tudor', na=False) |
                property_type.str.contains('french', na=False)
            ) &
            ~modern_mask &
            ~traditional_mask &
            ~colonial_mask &
            ~spanish_mask &
            ~florida_mask &
            ~transitional_mask &
            df['property_type'].isna()
        )
        
        cape_mask = (
            (
                property_type.str.contains('cape', na=False)
            ) &
            ~modern_mask &
            ~traditional_mask &
            ~colonial_mask &
            ~spanish_mask &
            ~florida_mask &
            ~transitional_mask &
            ~europe_mask &
            df['property_type'].isna()
        )
        
        log_mask = (
            (
                property_type.str.contains('craftsman', na=False) |
                property_type.str.contains('log', na=False)
            ) &
            ~modern_mask &
            ~traditional_mask &
            ~colonial_mask &
            ~spanish_mask &
            ~florida_mask &
            ~transitional_mask &
            ~europe_mask &
            ~cape_mask &
            df['property_type'].isna()
        )
        
        mask_style_dict = {
            'modern': modern_mask,
            'traditional': traditional_mask,
            'colonial': colonial_mask,
            'spanish': spanish_mask,
            'florida': florida_mask,
            'transitional': transitional_mask,
            'europe': europe_mask,
            'cape': cape_mask,
            'log': log_mask,
        }
        
        new_property_type = get_categorical_feature(
            property_type,
            mask_dict=mask_style_dict,
            # first_launch=False
        )
        df.loc[df['property_type'].isna(), 'property_type'] = new_property_type
        # df['property_type'] = new_property_type
        
        # FILL STORIES by values from propertyType
        property_type = replace(
            property_type,
            to_replace=(
                ('ground', '1'),
                ('one', '1'),
                ('single', '1'),
                ('multi', '1.5'),
                ('split', '1.5'),
                ('tri', '1.5'),
                ('two', '2'),
                ('bi', '2'),
                ('three', '3'),
            )
        )
        
        stor_property_type = property_type[
            (
                property_type.str.contains('stor', na=False) |
                property_type.str.contains('level', na=False)
            ) &
            df['stories'].isna()
        ]
        
        series_with_nums = stor_property_type.str.findall(num_regex)

        df.loc[df['stories'].isna(), 'stories'] = pd.to_numeric(
            series_with_nums.apply(get_max)
        )
        
        # Drop old columns
        df = df.drop(['propertyType'], axis=1)
    
    return df


def get_street_feature(
    street:pd.Series,   
)->pd.Series:
    """Create street feature with true Nones

    Args:
        series: street series origin

    Returns:
        Series with real Nans
    """
    
    to_none = (
        'disclosed',
        'available',
        'unknown',
    )
    
    street = street.copy()

    street_with_nans = convert_to_none(street, to_none, contains=True)
    street.loc[street_with_nans.isna()] = np.nan
    
    return street


# HOMEFACTS AND SCHOOL
def get_df_home_facts(df:pd.DataFrame)->pd.DataFrame:
    """ Retrieve features from homeFacts columns if the latter exists.

    Args:
        df: Dataframe with homeFacts column

    Returns:
        DataFrame with new features retrieved from homeFacts
    """

    df = df.copy()
    if 'homeFacts' in df.columns:
        facts_df = pd.DataFrame(
            get_series_json(
                df['homeFacts'],
                to_replace=(
                    ('"closet"', 'closet'),
                    ("'", '"'),
                    ('None', 'null'),
                ),
            )
        )
        facts_lt = []

        def get_fact_lt(row:pd.Series):
            """Function to get dict of facts which appends to the facts_lt

            Args:
                row: current row
            """
            row_dict = {}
            
            def get_dict(fact_pair):
                """Function for the list-mapping inside the Series 
                to put facts from current row to the row_dict and then append 
                this dict to the facts_lt

                Args:
                    fact_pair: pair from list to which map applies
                """
                label = (
                    fact_pair['factLabel'].lower().strip().replace(' ', '_')
                )
                value = fact_pair['factValue']
                row_dict[label] = value
            
            list(map(get_dict, row['homeFacts']['atAGlanceFacts']));
            facts_lt.append(row_dict)
        
        # Apply list-mapping function and concat result to df 
        facts_df.apply(get_fact_lt, axis=1)
        df = pd.concat((df, pd.DataFrame(facts_lt)), axis=1)
    
        # DROP old column
        df = df.drop('homeFacts', axis=1)
    
    return df


def get_df_schools(df:pd.DataFrame)->pd.DataFrame:
    """ Retrieve features from schools columns if the latter exists.

    Args:
        df: Dataframe with schools column

    Returns:
        DataFrame with new features retrieved from schools
    """

    df = df.copy()
    if 'schools' in df.columns:
        school_series = get_series_json(df['schools'])
        school_series = school_series.apply(lambda x: x[0])
        schools_df = pd.DataFrame(
            school_series
        )
        schools_lt = []

        def get_school_lt(row:pd.Series):
            """Function to get dict of school columns 
            which appends to the schools_lt

            Args:
                row: current row
            """
            root_dict = row.values[0]
            
            feature_dict = {}
            
            def get_dict(sub_dict):
                """Function for the list-mapping inside the Series 
                to put school features from current row to the row_dict 
                and then append this dict to the schools_lt

                Args:
                    sub_dict: pair from list to which map applies
                """
                
                for key in sub_dict:
                    # Recursive call for sub-dict with key "data"
                    if key == 'data':
                        get_dict(sub_dict[key])
                        continue
                    # Get lower-case key and apply sub-dict
                    modified_key = 'school_'+key.lower().strip() 
                    feature_dict[modified_key] = sub_dict[key]
                    
                return feature_dict

            schools_lt.append(get_dict(root_dict))
        
        # Apply list-mapping function and concat result to df 
        schools_df.apply(get_school_lt, axis=1)
        df = pd.concat((df, pd.DataFrame(schools_lt)), axis=1)
    
        # DROP old column
        df = df.drop('schools', axis=1)
    
    return df


# Year build
def get_num_year(
    series:pd.Series, 
    between_years:tuple=(1700, 2024)
)->pd.Series:
    """Convert year to numerical value with dropping future years

    Args:
        series: original year description
        between_years: Threshold of min-max year. Defaults to (1700, 2024).

    Returns:
        Cropped year series (numerical)
    """
    
    series = series.copy()
    
    series = get_numerical_feature(
        series,
        to_none=('no data',)
    )
    
    series.loc[~series.between(*between_years)] = np.nan
    
    return series


# Lotsize
def get_numerical_lotsize(lotsize:pd.Series)->pd.Series:
    """Get numerical lotsize with acres convertation to sqft

    Args:
        lotsize: series for the transformation

    Returns:
        Series with numerical lotsize in sqft
    """
    lotsize = lotsize.copy()
    sqft_in_acre = 43560
    
    if lotsize.dtype == 'O':
        lotsize = lotsize.copy()
    
        to_none = ('-', '—', '', 'no')
        to_replace = (
            (',', ''),
        )
        
        lotsize_num = get_numerical_feature(
            lotsize,
            to_none=to_none,
            to_replace=to_replace,
        )
        
        # Prepare acre mask
        acre_mask = lotsize.str.contains('acre', na=False)
        lotsize = lotsize_num
        # Convert implicit acres to sqft with acre_mask values
        full_acre_mask = acre_mask | (lotsize < 1)
        lotsize.loc[full_acre_mask] = (
            lotsize.loc[full_acre_mask] * sqft_in_acre
        )

    return lotsize


# Parking
def get_df_parking(
    df:pd.DataFrame,
)->pd.DataFrame:
    
    df = df.copy()
    
    if 'parking' in df.columns:
        # Convert to Nans
        df['parking'] = convert_to_none(
            df['parking'],
            to_none=(
                'no data',
                'null',
                ''
            ),
            contains=False
        )
        # Get categorical feature with four main values
        # on street
        on_mask = (
            df['parking'].str.contains('on street', na=False) |
            df['parking'].str.contains('driveway', na=False)
        )
        # off street
        off_mask = (
            df['parking'].str.contains('off street', na=False) |
            df['parking'].str.contains('parking', na=False)
        )
        # carport
        port_mask = (
            df['parking'].str.contains('carport', na=False)
        )
        # garage (attached or detached)
        garage_mask = (
            df['parking'].str.contains('garage', na=False) |
            df['parking'].str.contains('attached', na=False) |
            df['parking'].str.contains('detached', na=False)
        )
        mask_dict = {
            'on street': on_mask,
            'off street': off_mask,
            'carport': port_mask,
            'garage': garage_mask
        }
        df['parking_type'] = get_categorical_feature(
            df['parking'],
            mask_dict=mask_dict,
        )
        
        # Get number of parking lots
        def get_sum(find_list):
            result = np.nan
            if isinstance(find_list, list):
                if len(find_list)>0:
                    find_list = map(int, find_list)
                    result = sum(find_list)
            return result
        df['parking_count'] = df['parking'].str.findall(r'[0-9]+')
        df['parking_count'] = df['parking_count'].apply(get_sum)
        # Fill with 1, if any parking was found
        nan_count_mask = df['parking_count'].isna()
        notna_parking_type_mask = df['parking_type'].notna()
        df.loc[nan_count_mask & notna_parking_type_mask, 'parking_count'] = 1
        
        # Fill with space if parking_count not na
        notna_parking_cnt = (
            df['parking_count'].notna() &
            (df['parking_count']>0)
        )
        space_mask = notna_parking_cnt & ~notna_parking_type_mask  
        # Fill with "other"
        other_mask = (
            df['parking_type'].isna() &
            df['parking'].notna()
        )
        mask_dict = {
            'space': space_mask,
            'other': other_mask,
        }
        df['parking_type'] = get_categorical_feature(
            df['parking_type'],
            mask_dict=mask_dict,
            first_launch=False,
        )
        df = df.drop('parking', axis=1)
        
    return df


# HEATING and COOLING

def get_df_heat_cool(
    df:pd.DataFrame
)->pd.DataFrame:
    
    df = df.copy()
    
    # features = ['heating', 'cooling']
    keywords_dict = {
        'heating': (
            'electric', 'gas', 'pump', 'furnace',
            'baseboard', 'radiant', 'wall',
        ),
        'cooling': (
            'electric', 'wall', 'refrigeration', 'evaporative', 'fan', 
        )
    }
    features = list(keywords_dict.keys())
    
    if ('heating' in df.columns) and ('cooling' in df.columns):
        for feature in features:
            df[feature] = df[feature].str.lower().str.strip()
            df[feature] = convert_to_none(
                df[feature],
                to_none=('no data', ''),
                contains=False
            )
            
            central_mask = (
                df[feature].str.contains('central', na=False) |
                df[feature].str.contains('forced', na=False)
            )
            df['central_' + feature] = central_mask
            
            mask_dict = get_mask_dict(
                df[feature],
                keywords_dict[feature]
            )
            df[feature + '_type'] = get_categorical_feature(
                df[feature],
                mask_dict=mask_dict
            )
            
            # Fill nans with "central" if True
            na_mask = df[feature + '_type'].isna()
            mask = na_mask & central_mask
            df.loc[mask, feature + '_type'] = 'central'
            
            # Fill with "other"
            na_mask = df[feature + '_type'].isna()
            notna_feature_mask = df[feature].notna()
            mask = na_mask & notna_feature_mask
            df.loc[mask, feature + '_type'] = 'other'
            
        df = df.drop(features, axis=1)
    else:
        print('WARNING: no "heating" or "cooling" in columns')

    return df

# SCHOOL

def get_numerical_distance(
    series:pd.Series
)->pd.Series:
    """Get array with numerical distance

    Args:
        series: original series

    Returns:
        Series with arrays instead of lists
    """
    series = series.copy()

    def get_configured_array(source_list:list):
        to_replace=(
            ('mi', ''),
        )
        to_none=()
        array = get_array(
            source_list,
            to_replace=to_replace,
            to_none=to_none
        )
        return array
    
    series = series.apply(get_configured_array)
    
    return series
    

def get_numerical_rating(
    series:pd.Series
)->pd.Series:
    """Get array with numerical rating

    Args:
        series: original series

    Returns:
        Series with arrays instead of lists
    """
    series = series.copy()

    def get_configured_array(source_list:list):
        to_replace=(
            ('/10', ''),
        )
        to_none=('nr', 'null', 'na', '')
        array = get_array(
            source_list,
            to_replace=to_replace,
            to_none=to_none
        )
        return array
    
    series = series.apply(get_configured_array)
    
    return series


def get_df_updated_years(
    df:pd.DataFrame,
    last_year:float=None,
)->pd.DataFrame:
    """Get df with years count from last remodeling/building
    """
    df = df.copy()
    if 'remodeled_year' in df.columns:
        year_columns = ['remodeled_year', 'year_built']
        if last_year is None:
            last_year = df[year_columns].max().max()
            print(f'Последний упоминаемый год: {last_year}')
        df['remodeled_year'] = get_num_year(df['remodeled_year'])
        df['updated_years'] = last_year - df[year_columns].max(axis=1)
        df = df.drop('remodeled_year', axis=1)
    return df