import pandas as pd
from IPython.display import display

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score

import mlflow
from mlflow.models import infer_signature
from mlflow.models.model import ModelInfo


def get_X_y(
    df:pd.DataFrame, 
    target_name:str='target',
    verbose:bool=True,
)->tuple:
    """Get features and target

    Args:
        df: Origin df
        target_name: Name of the targer column. Defaults to 'target'.
        verbose: Write info about dataset shapes, if True. 
        Defaults to True.

    Returns:
        features (X), and target(y)
    """
    X = df.drop(target_name, axis=1)
    y = df[target_name]
    
    if verbose:
        print(f'X shape: {X.shape}')
        print(f'y shape: {y.shape}')
    
    return X, y


def make_pipeline(
    pipe_elements:list,
)->tuple:
    """Get Pipeline and its params set explicitly

    Args:
        pipe_elements: list of elements in the form (name, class, params),
        or in the form (name, class)

    Raises:
        ValueError: Error if pipe_elements form is incorrect

    Returns:
        Pipeline obj, params of the pipe
    """
    pipe_list = []
    pipe_params = {}
    
    for element in pipe_elements:
        # Depending on number of sub elements, unpack
        if len(element) == 3:
            name, elem_class, params = element
        elif len(element) == 2:
            name, elem_class = element
            params = None
        else:
            raise ValueError(
                'pipe_elements should contain tuples with size 2 or 3'
            )
        # Create class instance with predefined params
        if isinstance(params, dict):
            obj = elem_class(**params)
            # Get pipe-format of the params name and keep them
            for param_name in params:
                pipe_params[f'{name}__{param_name}'] = params[param_name]
        elif params is None:
            obj = elem_class()
        pipe_list.append(
            (name, obj)
        )
        
    pipe = Pipeline(pipe_list)
    return pipe, pipe_params


def replace_metric(
    string, 
    to_replace:tuple=(
            ('neg_', ''),
            ('mean_absolute_percentage_error', 'mape'),
    ),
):
    """Replace substrings in the string

    Args:
        string: string to modify
        to_replace: tuple of replace-tuples. Defaults to 
        ( ('neg_', ''), ('mean_absolute_percentage_error', 'mape'), ).

    Returns:
        string with replaced sub-strings
    """
    for replace_pair in to_replace:
        string = string.replace(*replace_pair)
    return string


def get_avg_cv_metric(
    cv_dict:dict,
    averaging:str='mean',
    verbose:bool=True,
    fmt:str='{:,.3f}',
)->dict:
    """Return mean/median value of metric after CV in dict-format 
    and display it if verbose

    Args:
        cv_dict: Source cross-validation metric.
        averaging: Method of averaging ("mean" or "median"). 
        Defaults to 'mean'.
        verbose: Display metrics or not. Defaults to True.
        fmt: Format of displaying. Defaults to '{:,.3f}'.

    Raises:
        ValueError: if averaging is not in ("mean" or "median")

    Returns:
        Dict of averages metric
    """
    cv_df = pd.DataFrame(cv_dict)
    # Change back to positive metrics
    neg_mask = cv_df.columns.str.contains('neg_')
    cv_df.loc[:,neg_mask] = cv_df.loc[:,neg_mask]*(-1)
    # Replace column names and save them
    cv_df.columns = list(
        map(
            lambda x: replace_metric(x), 
            cv_df.columns
        )
    )
    # Drop columns with time
    cv_df = cv_df.drop(
        cv_df.columns[cv_df.columns.str.contains('time')], 
        axis=1
    )
    if averaging == 'mean':
        cv_series = cv_df.mean()
    elif averaging == 'median':
        cv_series = cv_df.median()
    else:
        raise ValueError('Averaging not in the list ["mean", "median"]')
    
    if verbose:
        train_columns_mask = cv_series.index.str.contains('train')
        # Prepare dataframe, if it contains train_metrics
        if len(train_columns_mask) > 0: 
            train_vals = cv_series.loc[train_columns_mask]
            # Drop excess text
            train_vals.index = list(
                map(
                    lambda x: x.replace('train_', ''),
                    train_vals.index 
                )
            )
            # Get in format (metric,'train')
            train_vals = pd.DataFrame(train_vals, columns=['train'])
            
            test_columns_mask = cv_series.index.str.contains('test')
            test_vals = cv_series.loc[test_columns_mask]
            # Drop excess text
            test_vals.index = list(
                map(
                    lambda x: x.replace('test_', ''),
                    test_vals.index 
                )
            )
            # Get in format (metric,'test')
            test_vals = pd.DataFrame(test_vals, columns=['test'])
            # Combine two metric columns
            df_to_display = pd.concat((train_vals, test_vals), axis=1)
            # Display in selected format
            display(df_to_display.map(fmt.format))
        # Display series as it is, if train is absent
        else:
            display(cv_series.map(fmt.format))
    
    return cv_series.to_dict()
        
        
def cross_validate_pipe(
    pipe:Pipeline,
    X,
    y,
    cv=5,
    scoring=(
        'neg_mean_absolute_percentage_error',
        'r2'
    ),
    return_train_score=True,
    averaging:bool=True,
    **kwargs,
)->dict:
    """Cross-validate pipe and get averaged metrics

    Args:
        pipe: pipeline to validate
        X: feature dataframe
        y: target
        cv: CV-rule. Defaults to number of splits = 5.
        scoring: metrics for scoring. 
        Defaults to ( 'neg_mean_absolute_percentage_error', 'r2' ).
        return_train_score: If true train_score also returns. Defaults to True.
        averaging: Average metrics. Defaults to True.

    Returns:
        metrics
    """
    metrics = cross_validate(
        estimator=pipe,
        X=X,
        y=y,
        cv=cv,
        scoring=scoring,
        return_train_score=return_train_score,
    )
    
    if averaging:
        metrics = get_avg_cv_metric(
            metrics,
            **kwargs
        )
    
    return metrics

        
def log_pipe_mlflow(
    X, 
    pipe:Pipeline,
    pipe_name:str,
    pipe_params:dict,
    metrics:dict,
    training_info:str,
    experiment_name:str='Housing cost',
    artifact_path:str='housing_model',
    tracking_uri:str='http://127.0.0.1:8080',
)->ModelInfo:
    """Log pipeline in mlflow

    Args:
        X: data for model signature creation
        pipe: pipe or model to log
        pipe_name: name of the pipe
        pipe_params: parameters of the pipe
        metrics: obtained metrics
        training_info: description of the run
        experiment_name: Name of the experiment. Defaults to 'Housing cost'.
        artifact_path: path to artefacts. Defaults to 'housing_model'.
        tracking_uri: uri of the server. Defaults to 'http://127.0.0.1:8080'.

    Returns:
        Model info after logging
    """
    
    # Set our tracking server uri for logging
    mlflow.set_tracking_uri(uri=tracking_uri)

    # Create a new MLflow Experiment
    mlflow.set_experiment(experiment_name)
    
    # Start an MLflow run
    with mlflow.start_run():
        # Log the hyperparameters
        mlflow.log_params(pipe_params)
        
        # Log metrics
        for metric_name in metrics:
            mlflow.log_metric(metric_name, metrics[metric_name])
        
        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag("Training Info", training_info)
        
        # Infer the model signature
        signature = infer_signature(X, pipe.predict(X))
        
        # Log the model
        model_info = mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path=artifact_path,
            signature=signature,
            input_example=X,
            registered_model_name=pipe_name,
        )
    
    return model_info