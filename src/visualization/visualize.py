import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from IPython.display import display

import sys
root_folder = '../../'
sys.path.append(root_folder)
from src.utils import get_percentage


def plot_countplot(
    data:pd.DataFrame, 
    categorical_feature:str=None,
    hue_feature:str=None,
    log_scale:bool=False,
    categorical_label:str=None,
    title:str=None,
    orient:str='v',
    figsize:tuple=(),
    ordered:bool=True,
    xrotation:int=90,
):
    """Plot countplot

    Args:
        data: data for plotting
        categorical_feature: main feature to count. Defaults to None.
        hue_feature: if color labeling is required (optional). Defaults to None.
        log_scale: scale for count-axis. Defaults to False.
        categorical_label: special label for categorical axis, if required. Defaults to None.
        title: plot title. Defaults to None.
        orient: plot orientation. Defaults to 'v'.
        figsize: size of the figure. Defaults to None.
        ordered: Order by count. Defaults to None.

    Raises:
        ValueError: If orientation differ from ['h', 'v']

    Returns:
        figure with plot
    """
    orient = orient.strip().lower()
    if orient not in ['h', 'v']:
        raise ValueError('incorrect "orient"')
    
    order = None
    if ordered:
        order = (
            data[categorical_feature].value_counts().index
        )
    if orient == 'h':
        x, y = None, categorical_feature
        x_label, y_label = None, categorical_label
    elif orient == 'v':
        y, x = None, categorical_feature
        y_label, x_label = None, categorical_label
        
    ax = sns.countplot(
        data=data,
        x=x,
        y=y,
        hue=hue_feature,
        orient=orient,
        log_scale=log_scale,
        order=order
    )
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    
    ax.tick_params(axis='x', labelrotation=xrotation)

    ax.set_title(title)
    fig = ax.get_figure()
    
    if len(figsize) == 2:
        fig.set_size_inches(*figsize)
    
    fig.tight_layout()
    
    return fig


def plot_boxplot(
    data:pd.DataFrame, 
    numeric_feature:str,
    categorical_feature:str=None,
    hue_feature:str=None,
    log_scale:bool=False,
    numeric_label:str=None,
    categorical_label:str=None,
    title:str=None,
    orient:str='h',
    figsize:tuple=(),
    ordered:bool=False,
    xrotation=None,
):
    """Plot boxplot

    Args:
        data: data for plotting
        numeric_feature: name of the feature to plot
        categorical_feature: if several categories necessary to analyse (optional). Defaults to None.
        hue_feature: if color labeling is required (optional). Defaults to None.
        log_scale: scale for numeric-axis. Defaults to False.
        numeric_label: special label for numeric axis, if required. Defaults to None.
        categorical_label: special label for categorical axis, if required. Defaults to None.
        title: plot title. Defaults to None.
        orient: boxplots orientation. Defaults to 'h'.
        figsize: size of the figure. Defaults to None.
        ordered: Order by numerical value. Defaults to None.

    Raises:
        ValueError: If orientation differ from ['h', 'v']

    Returns:
        figure with plot
    """
    orient = orient.strip().lower()
    if orient not in ['h', 'v']:
        raise ValueError('incorrect "orient"')
    if orient == 'h':
        x, y = numeric_feature, categorical_feature
        x_label, y_label = numeric_label, categorical_label
    elif orient == 'v':
        y, x = numeric_feature, categorical_feature
        y_label, x_label = numeric_label, categorical_label
    
    order = None
    if ordered:
        order = (
            data.groupby(categorical_feature)[numeric_feature]
            .median().sort_values().index
        )
        
    ax = sns.boxplot(
        data=data,
        x=x,
        y=y,
        hue=hue_feature,
        log_scale=log_scale,
        order=order
    )
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)

    ax.set_title(title)
    fig = ax.get_figure()
    
    if len(figsize) == 2:
        fig.set_size_inches(*figsize)
    
    if xrotation:
        ax.tick_params(axis='x', labelrotation=xrotation)
    
    fig.tight_layout()
    
    return fig


def plot_box_hist_plot(
    data:pd.DataFrame, 
    numeric_feature:str,
    log_scale:bool=False,
    numeric_label:str=None,
    title:str=None,
    std_lines:bool=False,
):
    """Plot boxplot above histplot

    Args:
        data: data for plotting
        numeric_feature: name of the feature to plot
        log_scale: scale for numeric-axis. Defaults to False.
        numeric_label: Special label for numeric axis, if required. Defaults to None.
        title: Plot title. Defaults to None.

    Returns:
        Plot figure with two axes
    """
    fig, (ax_box, ax_hist) = plt.subplots(
        nrows=2, 
        ncols=1,
        sharex=True,
        gridspec_kw={
            'height_ratios': [0.2, 0.8]
        },
    )

    sns.boxplot(
        data=data,
        x=numeric_feature,
        ax=ax_box,
    )
    ax_box.set_xlabel('')
    
    sns.histplot(
        data=data,
        x=numeric_feature,
        ax=ax_hist,
        log_scale=log_scale,
    )
    if numeric_label:
        ax_hist.set_xlabel(numeric_label)
    if std_lines:
        x = data[numeric_feature]
        ax_hist.axvline(x.mean(), color='k')
        ax_hist.axvline(x.mean() - 3*x.std(), color='k', ls='--')
        ax_hist.axvline(x.mean() + 3*x.std(), color='k', ls='--')
    
    fig.suptitle(title)
    fig.tight_layout()
    
    return fig


def plot_scatter(
    data:pd.DataFrame, 
    x:str,
    y:str,
    logx:bool=False,
    logy:bool=False,
    x_label:str=None,
    y_label:str=None,
    regr:bool=False,
    title:str=None,
    **kwargs,
):
    """Plot scatter plot

    Args:
        data: data for plotting
        x: numerical feature
        y: numerical feature
        logx: scale for x-axis. Defaults to False.
        logy: scale for y-axis. Defaults to False.
        x_label: x-axis label. Defaults to None.
        y_label: y-axis label. Defaults to None.
        regr: plot regresssion line. Defaults to False.
        title: Plot title. Defaults to None.

    Returns:
        _description_
    """
    fig, ax = plt.subplots()
    
    params = {
        'data': data,
        'x': x,
        'y': y,
        'ax': ax,
    }
    
    if regr:
        ax = sns.regplot(
            **params,
            **kwargs,
        )
    else:
        ax = sns.scatterplot(
            **params,
            **kwargs,
        )
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')

    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)

    ax.set_title(title)
    
    return fig


def plot_numerical_stats(
    df:pd.DataFrame,
    feature_names:tuple = (),
    target:str='target',
    numeric_label='price [$]', # Can be 'log price [10^x $]'
    logx:bool=True,
    logy:bool=True,
    verbose:bool=True,
    std_lines:bool=False
):
    
    for feature_name in feature_names:
        print(feature_name)
        if verbose:
            display(df[feature_name].describe())
            print(
                f'Доля объявлений с пропуском в {feature_name}:', 
                get_percentage(df[feature_name].isna().sum(), df.shape[0])
            )

        plot_box_hist_plot(
            df, 
            feature_name, 
            log_scale=logx, 
            title=f'{feature_name} count',
            std_lines=std_lines
        );
        
        plot_scatter(
            data=df,
            x=feature_name,
            y=target,
            logx=logx,
            logy=logy,
            y_label=numeric_label,
            title=f'Price - {feature_name} plot',
            linewidth=0,
        );


def plot_categorical_stats(
    df:pd.DataFrame,
    feature_names:tuple = (),
    target:str='target',
    numeric_label:str='price [$]', # Can be 'log price [10^x $]'
    title:str=None,
    log_count:bool=False,
    log_target:bool=False,
    verbose:bool=True,
):
    
    for feature_name in feature_names:
        print(feature_name)
        if verbose:
            display(df[feature_name].value_counts(normalize=True))
            print(
                f'Доля объявлений с пропуском в {feature_name}:', 
                get_percentage(df[feature_name].isna().sum(), df.shape[0])
            )
        
        plot_countplot(
            df,
            categorical_feature=feature_name,
            log_scale=log_count,
            title=f'{feature_name} count',
        );
        plt.show()
        
        plot_boxplot(
            df,
            numeric_feature=target,
            categorical_feature=feature_name,
            orient='v',
            log_scale=log_target,
            numeric_label=numeric_label,
            title=f'Price dependency on {feature_name}',
            xrotation=90,
            ordered=True
        );
        plt.show()


def plot_heatmap(
    heat_data,
    title:str='',
    vmin:float=-1.,
    vmax:float=1.,
    cmap:str='coolwarm',
    annot:bool=True,
    fmt:str='.2f',
    **kwargs,
):
    """Plot heatmap

    Args:
        heat_data: Data for the heatmap
        title: Title of the plot. Defaults to ''.
        vmin: Min threshold for colormap. Defaults to -1..
        vmax: Max threshold for colormap. Defaults to 1..
        cmap: Colormap style. Defaults to 'coolwarm'.
        annot: Annotation in plot. Defaults to True.
        fmt: Annotation format. Defaults to '.2f'.

    Returns:
        Figure with heatmap
    """
    ax = sns.heatmap(
        data=heat_data,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        annot=annot,
        fmt=fmt,
        **kwargs,   
    )
    
    ax.set_title(title)
    
    fig = ax.get_figure()
    
    return fig


def plot_nans_stat(
    df:pd.DataFrame,
    check_nans:list,
    display_row_stat:bool=True,
    verbose:bool=True
)->tuple:
    """ Display nans statistic

    Args:
        df: source dataframe
        check_nans: columns list to check nans
        display_row_stat: display count of missings if True
        verbose: write numerical values if true

    Returns:
        Tuple: figure, sorted
    """
    # Columns of nans-checking sorted by number of nans
    sorted_check_nans = (
        df[check_nans].isna().sum().sort_values(ascending=False).index
    )
    if verbose:
        print('Доля пропусков по колонкам:')
        display(
            df[sorted_check_nans].isna().mean()
        )
    if display_row_stat:
        row_nans = pd.DataFrame(df.isna().sum(axis=1))
        row_nans.columns = ['row_nans']
        if verbose:
            print(
                'Нормир. распределение количества пропусков по строкам'
            )
            display(
                (
                    row_nans['row_nans']
                    .value_counts(normalize=True)
                    .sort_index()
                )
            )
        plot_box_hist_plot(
            row_nans,
            numeric_feature='row_nans',
            # numeric_label='Nans row count',
            title='Распределение количества строк по числу пропусков'
        )

    # Nans visualisation
    fig, axes = plt.subplots(
        2, 1, 
        figsize=(8, 12), 
    )
    sns.heatmap(df[sorted_check_nans].isna(), ax=axes[0])
    sns.barplot(df[sorted_check_nans].isna().sum(), ax=axes[1])
    axes[1].tick_params(axis='x', labelrotation=90)
    fig.suptitle('Пропуски в колонках')
    fig.tight_layout()
    
    return fig, sorted_check_nans