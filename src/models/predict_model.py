import pandas as pd
from IPython.display import display

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score

import mlflow
from mlflow.models import infer_signature
from mlflow.models.model import ModelInfo


def replace_metric(
    string, 
    log_target,
    to_replace:tuple=(
            ('neg_', ''),
            ('mean_absolute_percentage_error', 'mape'),
            ('train', 'cv_train'),
            ('test', 'cv_validation')
    ),
):
    """Replace substrings in the string

    Args:
        string: string to modify
        to_replace: tuple of replace-tuples.
        log_target: does target in log or not

    Returns:
        string with replaced sub-strings
    """
    
    if log_target:
        to_replace = list(to_replace)
        to_replace.extend(
            [
                ('mape', 'mape_log'),
                ('r2', 'r2_log'),
            ]
        )
    
    for replace_pair in to_replace:
        string = string.replace(*replace_pair)
    return string


def display_metric_dataset(
    series:pd.Series,
    prefixes:tuple=('cv_train', 'cv_validation'),
    fmt:str='{:,.3f}',
)->pd.DataFrame:
    """ Display metric dataset in more convenient way

    Args:
        series: Metric series
        prefixes: Prefixes of the metrixs. Defaults to ('cv_train', 'cv_validation').
        fmt: Display format. Defaults to '{:,.3f}'.

    Returns:
        Displayed dataframe
    """
    values_list = []
    for name in prefixes:
        mask = series.index.str.contains(name)
        values = series.loc[mask]
        # Drop excess text
        values.index = list(
            map(
                lambda x: x.replace(f'{name}_', ''),
                values.index 
            )
        )
        # Get in format (metric,'train')
        values = pd.DataFrame(values, columns=[name])
        values_list.append(values)
    # Combine metric columns
    df_to_display = pd.concat(values_list, axis=1)
    display(df_to_display.map(fmt.format))
    
    return df_to_display


def get_avg_cv_metric(
    cv_dict:dict,
    log_target:bool,
    averaging:str='mean',
    verbose:bool=True,
)->dict:
    """Return mean/median value of metric after CV in dict-format 
    and display it if verbose

    Args:
        cv_dict: Source cross-validation metric.
        averaging: Method of averaging ("mean" or "median"). 
        Defaults to 'mean'.
        verbose: Display metrics or not. Defaults to True.
        log_target: does target in log or not

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
            lambda x: replace_metric(x, log_target=log_target), 
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
        train_columns_mask = cv_series.index.str.contains('cv_train')
        # Prepare dataframe, if it contains train_metrics
        if len(train_columns_mask) > 0: 
            display_metric_dataset(cv_series);
        # Display series as it is, if train is absent
        else:
            display_metric_dataset(
                cv_series,
                prefixes=('cv_validation',)
            );
    
    return cv_series.to_dict()
        
        
def cross_validate_pipe(
    pipe:Pipeline,
    X,
    y,
    log_target:bool=True,
    cv=5,
    scoring=(
        'neg_mean_absolute_percentage_error',
        'r2'
    ),
    return_train_score=True,
    averaging:bool=True,
    njobs:int=-1,
    **kwargs,
)->dict:
    """Cross-validate pipe and get averaged metrics

    Args:
        pipe: pipeline to validate
        X: feature dataframe
        y: target
        log_target: does target in log or not. Defaults to True
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
        n_jobs=njobs,
    )
    
    if averaging:
        metrics = get_avg_cv_metric(
            metrics,
            log_target=log_target,
            **kwargs
        )
    
    return metrics

# RETURN VALUE BY ITS EXPONENT
exp10 = lambda x: 10 ** x 

def predict(
    model:Pipeline, 
    X:pd.DataFrame
):
    """ Make model prediction and exponentiate it

    Args:
        model: prediction model
        X: dataset for prediction

    Returns:
        real prediction
    """
    log_prediction = model.predict(X)
    prediction = exp10(log_prediction)
    
    return prediction


def get_metrics(
    model:Pipeline,
    X:pd.DataFrame,
    y_true:pd.Series,
    prefix:str=None,
    metrics:dict=None
)->dict:
    if metrics is None:
        metrics = {}
    
    # Get real target
    y_true = y_true.copy()
    y_true = exp10(y_true)
    # Predict real target
    y_pred = predict(model, X)
    
    # Prepare metric lables
    mape = 'mape'
    r2 = 'r2'
    if not(prefix is None):
        mape = f'{prefix}_{mape}'
        r2 = f'{prefix}_{r2}'
    
    # Estimate real metrics
    metrics[mape] = mean_absolute_percentage_error(y_true, y_pred)
    metrics[r2] = r2_score(y_true, y_pred)
    
    return metrics


def get_train_test_metrics(
    model:Pipeline,
    X_train:pd.DataFrame,
    X_test:pd.DataFrame,
    y_train:pd.Series,
    y_test:pd.Series,
    verbose:bool=True,
)->dict:
    
    metrics = {}
    
    # Get train metrics
    metrics = get_metrics(
        model=model,
        X=X_train,
        y_true=y_train,
        prefix='train',
        metrics=metrics
    )
    # Get test metrics
    metrics = get_metrics(
        model=model,
        X=X_test,
        y_true=y_test,
        prefix='test',
        metrics=metrics
    )
    
    if verbose:
        series = pd.Series(metrics)
        display_metric_dataset(
            series,
            prefixes=('train', 'test')
        );
    
    return metrics


def log_pipe_mlflow(
    X, 
    pipe:Pipeline,
    pipe_name:str,
    pipe_params:dict,
    metrics:dict,
    training_info:str,
    experiment_name:str,
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
        experiment_name: Name of the experiment.
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