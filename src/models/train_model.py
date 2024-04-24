import pandas as pd
from IPython.display import display

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


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
        elif elem_class == ColumnTransformer:
            obj = elem_class(params)
            pipe_params[f'{name}__transformers'] = params
        elif params is None:
            obj = elem_class()
        pipe_list.append(
            (name, obj)
        )
        
    pipe = Pipeline(pipe_list)
    return pipe, pipe_params