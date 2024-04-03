import pandas as pd

def get_numerical_target(target:pd.Series):
    """Get numerical feature from raw dataset. Can contain only "$", "+", "," to drop.

    Args:
        target: Series to transform

    Returns:
        Transformed Series
    """
    if target.dtype == 'O':
        target = target.str.replace('$', '')
        target = target.str.replace('+', '')
        target = target.str.replace(',', '')
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

def replace_to_numbers(stories:pd.Series):
    """Replace stories series to numbers

    Args:
        stories: Series with stories

    Returns:
        Series with numbered stories replacement
    """
    stories = stories.str.lower()
    # replace
    replace_dt = {
        'ground': '1',
        'one': '1',
        'two': '2',
        'three': '3',
    }
    
    for word in replace_dt:
        stories = stories.str.replace(
            word, replace_dt[word]
        )
    
    return stories


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


def get_df_with_numerical_story(df:pd.DataFrame):
    df_copy = df.copy()
    df_copy['stories'] = replace_to_numbers(df_copy['stories'])
    
    df_copy['stories_num'] = None
    
    series_with_nums = df_copy['stories'].str.findall(r'[0-9]+[\.]{0,1}[0-9]*')
    
    df_copy['stories_num'] = pd.to_numeric(
        series_with_nums.apply(get_max)
    )
    df_copy.loc[df_copy['stories_num'] == 0, 'stories_num'] = None
    
    # TODO: continue
    
    return df_copy