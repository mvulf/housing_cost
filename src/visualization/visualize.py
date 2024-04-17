import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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
    
    return fig


def plot_box_hist_plot(
    data:pd.DataFrame, 
    numeric_feature:str,
    log_scale:bool=False,
    numeric_label:str=None,
    title:str=None,
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