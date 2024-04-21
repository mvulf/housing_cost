from scipy.stats import normaltest
import pandas as pd
import numpy as np


def test_normal(
    data,
    alpha:float = 0.01,
    verbose:bool=True,
    fmt:str="p = {:.3e}"
)->float:
    
    H0 = 'Данные распределены нормально'
    Ha = 'Данные не распределены нормально (мы отвергаем H0)'
    
    _ , p = normaltest(data)
    
    if verbose:
        print(fmt.format(p))

    if p > alpha:
        print(H0)
    else:
        print(Ha)
        

def get_outliers_iqr(
    data:pd.DataFrame,
    feature:str,
    left:float=1.5,
    right:float=1.5,
    verbose:bool=True
)->tuple:
    """ Get outliers and cleaned from outliers dataframe by Tukey`s method

    Args:
        data: source data
        feature: feature to find outliers
        left: left IQR-coef. Defaults to 1.5.
        right: right IQR-coef. Defaults to 1.5.
        verbose: print additional info. Defults to True

    Returns:
        Cleaned dataframe and dataframe with outliers
    """
    x = data[feature]
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    iqr = q3 - q1
    bounds = (q1 - left*iqr, q3 + right*iqr)
    
    cleaned = data[x.between(*bounds)]
    outliers = data[~x.between(*bounds)]
    
    print(
        f'Количество выбросов в {feature} по методу Тьюки: {outliers.shape[0]}'
    )
    
    return cleaned, outliers, bounds


def get_outliers_z_score(
    data:pd.DataFrame,
    feature:str,
    left:float=3.0,
    right:float=3.0,
    verbose:bool=True
)->tuple:
    """ Get outliers and cleaned from outliers dataframe by z-deviation

    Args:
        data: source data
        feature: feature to find outliers
        left: left z. Defaults to 3.0.
        right: right z. Defaults to 3.0.
        verbose: print additional info. Defults to True

    Returns:
        Cleaned dataframe and dataframe with outliers
    """
    x = data[feature]
    mu = x.mean()
    sigma = x.std()
    bounds = (mu - left*sigma, mu + right*sigma)
    
    cleaned = data[x.between(*bounds)]
    outliers = data[~x.between(*bounds)]
    
    shape = outliers.shape[0]
    print(
        f'Количество выбросов в {feature} по методу z-отклонений: {shape}'
    )
    
    return cleaned, outliers, bounds


def get_df_no_outliers(
    data:pd.DataFrame,
    feature:str,
    method:str='iqr',
    eps:int=0,
    **kwargs
)->pd.DataFrame:
    """Get df with no outliers according to selected method

    Args:
        data: source dataset
        feature: feature
        method: Method for outliers detection. Defaults to 'iqr'.
        eps: Outliers count threshold. Defaults to 0.

    Raises:
        ValueError: raize if method not in ["iqr", "z-score"]

    Returns:
        Cleaned dataset and Outliers dataset
    """
    clean = data.copy()
    outliers = data.copy()
    all_outliers_list = []
    
    if method == 'iqr':
        get_outliers = get_outliers_iqr
    elif method == 'z-score':
        get_outliers = get_outliers_z_score
    else:
        raise ValueError('method can be "iqr" or "z-score"')
    
    i = 0
    while outliers.shape[0] > eps:
        i += 1
        print(f'ITERATION {i}')
        clean, outliers, _ = get_outliers(
            clean,
            feature,
            **kwargs
        )
        all_outliers_list.append(outliers.copy())
    
    all_outliers = pd.concat(all_outliers_list)
    # all_outliers = all_outliers.drop_duplicates()
    
    return clean, all_outliers