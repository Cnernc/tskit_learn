import multiprocessing as mp
import pandas as pd
import numpy as np

from typing import Generator, Tuple

from sklearn.base import BaseEstimator
from sklearn import clone as sklearn_clone

def _custom_clone_model(model: BaseEstimator | object) -> BaseEstimator | object: 
    #Est ce que c'est vraiment nécessaire ?
    try:
        cloned_model = sklearn_clone(model)
        return cloned_model
    except Exception as e:
        if hasattr(model, "copy"):
            return model.copy()
        elif hasattr(model, "get_params") and hasattr(model, "set_params"):
            cloned_model = model.__class__()
            cloned_model.set_params(**model.get_params())
            return cloned_model
        else:
            return model
            e

def _clean_and_reindex(X:pd.DataFrame, y:pd.Series = None) -> pd.DataFrame:
    if y is None:
        y = X
    return (
        X
        .replace([np.inf, -np.inf], np.nan)
        .dropna(how='all', axis=1)
        .dropna(how='all', axis=0)
        .reindex(y.index, method='ffill')
        .ffill()
        .fillna(0)
    )

def _window_grouper(
    X: np.ndarray, freq_retraining: int, min_train_steps: int, rolling_window_size: int, lookahead_steps:int, 
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    
    assert 2 < min_train_steps <= len(X), ("min_train_steps should be less than or equal to the length of X and greater than 2")
    assert freq_retraining <= len(X), ("freq_retraining should be less than or equal to the length of X")
    assert lookahead_steps < min_train_steps, ("lookahead_steps should be less than min_train_steps")

    def _get_slices() -> tuple[slice, slice]:
        if rolling_window_size:
            training_slice = slice(max(0, training_date - 1 - rolling_window_size ), training_date - 1 - lookahead_steps)
            test_slice = slice(training_date, min(training_date + freq_retraining, len(X)))
        else:
            training_slice = slice(0, training_date - 1 - lookahead_steps)
            test_slice = slice(training_date, min(training_date + freq_retraining, len(X)))
        return training_slice, test_slice

    for training_date in range(min_train_steps, len(X), freq_retraining):
        training_slice, test_slice = _get_slices()
        X_train, X_test = X[training_slice], X[test_slice]
        yield X_train, X_test

def _fit_predict_static(
        model: BaseEstimator | object, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
    ) -> np.ndarray:
    """Static fit and predict for the multiprocessing"""
    if (X_train.size == 0) or (y_train.size == 0) or (X_test.size == 0):
        Warning("Empty training or test data fed into the model. Returning nan values")
        return np.full((X_test.shape[0], y_train.shape[1]), np.nan)
    y_hat = model.fit(X_train, y_train).predict(X_test)
    # del X_train, y_train, X_test
    return y_hat

def _fit_predict_ndarray(
        model: BaseEstimator | object, X: np.ndarray, y: np.ndarray, 
        freq_retraining: int, min_train_steps: int, rolling_window_size: int, lookahead_steps:int, 
        n_jobs: int
    ) -> np.ndarray:
    
    assert X.shape[0] == y.shape[0], ("X and y should have the same number of rows")
    assert np.all(np.isfinite(X)) and np.all(np.isfinite(y)), ("X and y should not have any missing/infinite values")

    X_generator = _window_grouper(X, freq_retraining, min_train_steps, rolling_window_size, lookahead_steps)
    y_generator = _window_grouper(y, freq_retraining, min_train_steps, rolling_window_size, lookahead_steps)

    tasks = (
        ( _custom_clone_model(model), X_train.copy(), y_train.copy(), X_test.copy() )
        for (X_train, X_test), (y_train, _) in zip(X_generator, y_generator)
    )

    with mp.Pool(n_jobs) as pool:
        results = pool.starmap(_fit_predict_static, tasks)

    y_hat = np.concatenate(results)
    return np.concatenate([np.full(len(y) - len(y_hat), np.nan), y_hat])

def _fit_predict_ds(
        model: BaseEstimator | object, X: pd.DataFrame, y: pd.Series,
        freq_retraining: int, min_train_steps: int, 
        rolling_window_size: int, lookahead_steps:int, 
        skipna: bool, n_jobs: int
    ) -> pd.Series:
    assert not X.empty, f"No features for {y.name}"

    X = _clean_and_reindex(X, y)    
    y_ = y if not skipna else y.dropna()
    try:
        y_hat_values = _fit_predict_ndarray(
            model, X.values, y_.values, 
            freq_retraining, min_train_steps, 
            rolling_window_size, lookahead_steps, 
            n_jobs
        )
    except AssertionError as e:
        print(f'An error occurred during the fit of {y.name}. Returning NaN values. {e}')
        y_hat_values = np.nan

    return pd.Series(data=y_hat_values, name=y.name, index=y.index)


##### Unidimensional fit and predict: #####
##### train each asset separately     #####

def _fit_predict_unidimensional(
        model: BaseEstimator | object, X: pd.DataFrame, y: pd.DataFrame, 
        freq_retraining: int, min_train_steps: int, 
        rolling_window_size: int, lookahead_steps:int, 
        n_jobs: int, skipna: bool
    ) -> pd.DataFrame:

    def _fit_helper(col:str) -> pd.Series:
        X_ = X.loc[:, col] if isinstance(X.columns, pd.MultiIndex) else X
        y_ = y.loc[:, col] if not skipna else y.loc[:, col].dropna()
        return _fit_predict_ds(
            model, X_, y_, 
            freq_retraining, min_train_steps, rolling_window_size, lookahead_steps, 
            skipna, n_jobs
        )

    return pd.concat({col: _fit_helper(col) for col in y.columns}, axis=1)


##### Multidimensional fit and predict:             #####
##### When all the assets have the same features    #####
##### and the training is made on the whole dataset #####

def _window_splitter(
        df:pd.DataFrame, freq_retraining:int, min_train_steps:int, rolling_window_size:int, lookahead_steps:int
    ) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:

    assert 'date' in df.columns

    dates = df['date'].sort_values().unique()

    for i in range(min_train_steps, len(dates), freq_retraining):
        start_training = max(0, i-rolling_window_size) if rolling_window_size else 0
        end_training = i - lookahead_steps
        start_test = i
        end_test = min(i+freq_retraining, len(dates) - 1)

        training_slice = ( dates[start_training] <= df['date']) & (df['date'] < dates[end_training]) 
        test_slice = (dates[start_test] <= df['date']) & (df['date'] < dates[end_test])

        yield df[training_slice], df[test_slice]

def _reshaper(X:pd.DataFrame, y:pd.DataFrame) -> pd.DataFrame:
    X, y = X.sort_index(axis=1).sort_index(axis=0), y.sort_index(axis=1).sort_index(axis=0)
    assert isinstance(X.columns, pd.MultiIndex), "can't handle non-multidimensional data for multidim fit"
    assert y.columns.equals(X.columns.get_level_values(0).unique()), "X and y should have the same assets"

    X = _clean_and_reindex(X, y)
    y.columns = y.columns.map(lambda x: (x, 'target'))
    df = (
        pd.concat([X, y], axis=1)
        .reorder_levels([1, 0], axis=1)
        .sort_index(axis=1)
        .stack()
    )
    df = df[df['target'].notna()]
    df.index = df.index.set_names(['date', 'asset'])
    df = df.reset_index().sort_values(['date', 'asset'])
    return df 

def _fit_predict_shaped(
        model: BaseEstimator | object, df_train:pd.DataFrame, df_test:pd.DataFrame
    ) -> pd.DataFrame:
    X_train = df_train.drop(columns=['date', 'asset', 'target'])
    y_train = df_train['target']
    X_test  = df_test.drop(columns=['date', 'asset', 'target'])
    df_test['pred'] = model.fit(X_train, y_train).predict(X_test)
    df_test = df_test.set_index(['date', 'asset']).unstack()
    return df_test['pred']

def _fit_predict_multidimensional(
        model:BaseEstimator | object, X: pd.DataFrame, y: pd.DataFrame, 
        freq_retraining: int, min_train_steps: int, rolling_window_size: int, lookahead_steps:int, 
        n_jobs: int
    ) -> pd.DataFrame:

    df = _reshaper(X, y)

    tasks = ( 
        (_custom_clone_model(model), df_train, df_test) 
        for df_train, df_test in _window_splitter(df, freq_retraining, min_train_steps, rolling_window_size, lookahead_steps)
    )
    with mp.Pool(n_jobs) as pool:
        results = pool.starmap(_fit_predict_shaped, tasks)

    y_hat = (
        pd.concat(results, axis=0)
        .sort_index(axis=1)
        .sort_index(axis=0)
        .reindex(y.index, method='ffill')
        .reindex(y.columns, axis=1)
    )
    if any(y_hat.isna().all()):
        print(f"Warning: {y_hat[y_hat.isna().all()].columns} models returned NaN values")

    return y_hat

def _fit_predict_df(
        model:BaseEstimator | object, X: pd.DataFrame, y: pd.DataFrame, 
        freq_retraining: int, min_train_steps: int, rolling_window_size: int, lookahead_steps:int, 
        independant_fit:bool, skipna: bool, n_jobs: int
    ) -> pd.DataFrame:

    if independant_fit:
        return _fit_predict_unidimensional(model, X, y, skipna, 
            freq_retraining, min_train_steps, rolling_window_size, lookahead_steps, 
            n_jobs
        )
    else:
        return _fit_predict_multidimensional(model, X, y, 
            freq_retraining, min_train_steps, rolling_window_size, lookahead_steps, 
            n_jobs
        )
