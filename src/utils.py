import pandas as pd

def get_percentage(count, df_size, fmt='.3f'):
    """_summary_

    Args:
        count: number of feature, to which necessary to get percentage
        df_size: size of the reference dataset, to calculate percentage
        fmt: Print formatter. Defaults to '.3f'.

    Returns:
        String with formated percentage
    """
    return f'{count/df_size * 100:{fmt}}%'