import pandas as pd
import numpy as np
import requests
from sklearn.metrics import mean_absolute_percentage_error

fmt = '.3f'

request_size = 30

test_path = './data/2.0_test.csv'

exp10 = lambda x: 10 ** x 

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


if __name__ == '__main__':
    # Load model
    test = pd.read_csv(test_path, index_col=0)
    # test.info()
    X_test, y_test = get_X_y(test, target_name='log_target')
    
    random_index = np.random.randint(
        0,
        len(y_test),
        size=request_size
    )
    
    test_dict = X_test.iloc[random_index].to_dict()
    # print(test_dict)
    
    # Make POST-requiest
    r = requests.post(
        'http://localhost:4000/predict',
        json=test_dict
    )
    print(f'Status code: {r.status_code}')
    if r.status_code == 200:
        y_pred = r.json()["prediction"]
        y_true = y_test[random_index].values
        y_true = exp10(y_true)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        
        print(f'Predicted price [$]: {np.round(y_pred)}')
        print(f'True price [$]: {y_true}')
        print(f'MAPE = {mape*100:{fmt}}%')
    else:
        print(r.text)