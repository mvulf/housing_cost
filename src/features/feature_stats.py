from scipy.stats import normaltest
from scipy.stats import wilcoxon

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


def test_wilcoxon(
    df:pd.DataFrame,
    bool_features:tuple,
    target:str='log_target',
    alpha:float=0.01
)->dict:
    """ Conduct Wilcoxon sign-ranked test to check statistical significance
    of difference between two distributions

    Args:
        df: Source df
        bool_features: features to check statistical significance
        target: Feature to check distribution. Defaults to 'log_target'.
        alpha: Significance level. Defaults to 0.01.

    Returns:
        Dict of features and test results. True if statistical significance
        was founded
    """
    
    stat_importance = {}
    
    for feature in bool_features:
        print('Рассматриваемый признак:', feature)
        data_true = df[df[feature]][target]
        data_false = df[~df[feature]][target]
        min_length = min(data_true.shape[0], data_false.shape[0])
        print('Размер выборок:', min_length)
        stat, p = wilcoxon(
            data_true.iloc[:min_length], 
            data_false.iloc[:min_length]
        )
        if p > alpha:
            print(f'p-value={p:.2e} > alpha={alpha:.2e}')
            print(f'НЕТ статистически значимого различия по {target}')
            stat_importance[feature] = False
        else:
            print(f'p-value={p:.2e} <= alpha={alpha:.2e}')
            print(f'ЕСТЬ статистически значимое различие по {target}')
            stat_importance[feature] = True
        print()
    
    return stat_importance
        

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


def fillna_by_reference(
    df:pd.DataFrame,
    reference:str,
    features:tuple=(),
)->pd.DataFrame:
    """Fill nans by reference feature

    Args:
        df: data
        reference: name of the reference feature
        features: Features which is necessary to fill. Defaults to ().

    Returns:
        df with filled nans
    """
    df = df.copy()
    
    for feature in features:
        for ref_value in df[reference].unique():
            ref_value_mask = df[reference] == ref_value
            df.loc[
                (df[feature].isna()) & (ref_value_mask), 
                feature
            ] = df[ref_value_mask][feature].mode().iloc[0]
    
    return df


def clip_by_thresholds(
    df:pd.DataFrame,
    feature_thresholds:dict,
)->pd.DataFrame:
    """Clip features by thresholds

    Args:
        df: source data
        feature_thresholds: dict of feature thresholds

    Returns:
        Dataframe with clipped features
    """
    for feature in feature_thresholds:
        thresholds = feature_thresholds[feature]
        df[feature] = df[feature].clip(*thresholds)
    
    return df


def fill_outliers(
    df:pd.DataFrame,
    features:tuple=(),
    verbose:bool=True,
    method:str='z-score'
)->pd.DataFrame:
    
    df = df.copy()
    for feature in features:
        if verbose:
            print(feature)
        clean, outliers = get_df_no_outliers(
            data=df,
            feature=feature,
            method=method,
            verbose=verbose,
        )
        if verbose:
            true_outliers_cnt = outliers[outliers[feature].notna()].shape[0]
            print('Истинные выбросы (не NaNs):', true_outliers_cnt)
        # Get thresholds for clean dataset and clip original dataset by them
        thresholds = clean[feature].min(), clean[feature].max()
        df[feature] = df[feature].clip(*thresholds)
    
    return df


def get_correlated_df(
    df:pd.DataFrame,
    threshold:float = 0.7,
    method:str='spearman',
)->pd.DataFrame:
    """Get correlation dataframe with cols/rows contains values more than threshold

    Args:
        df: source df
        threshold: Correlation threshold. Defaults to 0.7 - refers to multicorrelation.
        method: Method to compute correlation. Defaults to 'spearman'.

    Returns:
        Dataframe with correlated features
    """
    corr_df = df.corr(method=method)
    # Drop ones on diagonals
    real_corr_df = corr_df.copy()
    for feature in real_corr_df.columns:
        real_corr_df.loc[feature, feature] = np.nan
    
    correlated_cols = (
        real_corr_df[np.abs(real_corr_df) >= threshold].notna().any()
    )
    
    corr_df = corr_df.loc[correlated_cols, correlated_cols]
    
    return corr_df
        
        