from sklearn.base import BaseEstimator
from sklearn import clone as sklearn_clone
from typing import Callable, Any
import multiprocessing as mp
import pandas as pd
import numpy as np

def _custom_clone_model(model: BaseEstimator | object) -> BaseEstimator | object:
    if isinstance(model, BaseEstimator):
        return sklearn_clone(model)
    elif hasattr(model, "get_params"):
        return model.__class__(**model.get_params())
    else:
        return model
    
def _get_cpu_count() -> int:
    return max(1, mp.cpu_count() - 2)

def _clean_and_reindex(X:pd.DataFrame, y:pd.Series = None) -> pd.DataFrame:
    if y is None:
        y = X
    return (
        X
        .dropna(how='all', axis=1)
        .dropna(how='all', axis=0)
        .reindex(y.index)
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
        .fillna(0)
    )

def expanding_decorator(
        f:Callable[[pd.DataFrame, Any], pd.Series],
        n_jobs:int = _get_cpu_count(),
    ) -> Callable[[pd.DataFrame, Any], pd.DataFrame]:

    def f_expanding(df:pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        ans = pd.DataFrame(index = df.index)

        tasks = ( df[df.index < i].copy() for i in df.index )
        with mp.Pool(n_jobs) as pool:
            results = pool.map(f, tasks, args, kwargs)
    
        for i, res in zip(df.index, results):
            ans.loc[i, :] = res

        return ans

    return f_expanding

def expanding_apply(
        df:pd.DataFrame, 
        f:Callable[[pd.DataFrame, Any], pd.Series],
        args:tuple = (),
        kwargs:dict = {},
        n_jobs:int = _get_cpu_count()
    ) -> pd.DataFrame:

    return expanding_decorator(f, n_jobs)(df, *args, **kwargs)