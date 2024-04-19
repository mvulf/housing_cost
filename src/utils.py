import pandas as pd

def get_percentage(count, df_size, fmt='.3f'):
    """Return formatted percents

    Args:
        count: number of feature, to which necessary to get percentage
        df_size: size of the reference dataset, to calculate percentage
        fmt: Print formatter. Defaults to '.3f'.

    Returns:
        String with formated percentage
    """
    return f'{count/df_size * 100:{fmt}}%'


# def get_dict(**kwargs)->dict:
#     return vars()['kwargs']
def get_dict(**kwargs)->dict:
    """
    Returns:
        Dict of kwargs
    """
    return {**kwargs}