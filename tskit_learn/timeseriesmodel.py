import numpy as np
import pandas as pd
import multiprocessing as mp
from sklearn.base import BaseEstimator
from .utilitaires import (
    _custom_clone_model, _fit_predict_ndarray, _fit_predict_ds,  _fit_predict_df
)

class BaseTimeSeriesModel:

    n_jobs = max(1, mp.cpu_count() - 2)
        
    def __init__(
        self, model: BaseEstimator | object, 
        freq_retraining: int, rolling_window_size: int, 
        min_train_steps: int, lookahead_steps: int,
    ) -> None:
        
        self.model = model        
        self.window_params = {
            "rolling_window_size": rolling_window_size,
            "freq_retraining": freq_retraining,
            "min_train_steps": max(min_train_steps if min_train_steps else freq_retraining, lookahead_steps),
            "lookahead_steps": lookahead_steps,
        }

        assert all(isinstance(val, int) for val in self.window_params.values())
        assert rolling_window_size is None or rolling_window_size > lookahead_steps, (
            "rolling_window_size should be greater than lookahead_steps"
            )
        
        if not isinstance(model,  BaseEstimator):
            assert hasattr(model, "fit") and hasattr(model, "predict"), (
                "model should have fit and predict methods"
            )
            Warning(f'TimeSeriesModel is optimised for sklearn.base.BaseEstimator not {model.__class__.__name__}')
        
        if not hasattr(model, "get_params"):
            Warning(f"model can't be cloned: {model.__class__.__name__}, multiprocessing won't be used. You can force the use of multiprocessing by setting n_jobs")
            BaseTimeSeriesModel.n_jobs = 1
            
    def fit(
        self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series | pd.DataFrame, 
        independant_fit:bool = True, skipna: bool = True, 
    ) -> 'BaseTimeSeriesModel':
        
        kwargs = {
            "model": self.model, "X": X.copy(), "y": y.copy(),
            **self.window_params,
            "n_jobs": BaseTimeSeriesModel.n_jobs,
        }
        print({
            "n folds": (len(y.index) - self.window_params["min_train_steps"]) // self.window_params["freq_retraining"],
            "n features": len(X.columns) / (len(y.columns) if isinstance(y, pd.DataFrame) else 1),
            "n columns": len(y.columns) if isinstance(y, pd.DataFrame) else 1,
            "n datapoints": len(y.index) * ( 1 if isinstance(y, pd.DataFrame) and independant_fit else len(y.columns) ),
            "n trainings": (1 if independant_fit else len(y.columns)) * (len(y.index) - self.window_params["min_train_steps"]) // self.window_params["freq_retraining"],
            "model": self.model.__class__.__name__,
        })

        match (type(X), type(y)):
            case (np.ndarray, np.ndarray):
                y_hat = _fit_predict_ndarray(**kwargs)
            case (pd.DataFrame, pd.Series):
                y_hat = _fit_predict_ds(**kwargs, skipna=skipna)
            case (pd.DataFrame, pd.DataFrame):
                y_hat = _fit_predict_df(**kwargs, independant_fit=independant_fit, skipna=skipna)                
            case _:
                raise ValueError(
                    f"""Unsupported types: X type {type(X)}, y type {type(y)}. X,y should be in :
                    (np.ndarray, np.ndarray) ; (pd.DataFrame, pd.Series) ; (pd.DataFrame, pd.DataFrame)
                    """
                )
            
        self.y_hat = y_hat.copy()
        return self 
    
    def predict(self, _: None= None) -> np.ndarray | pd.Series | pd.DataFrame:
        assert hasattr(self, "y_hat"), "Model should be fit before predict"
        return self.y_hat.copy()
    
    def fit_predict(
            self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series | pd.DataFrame, 
            independant_fit:bool = True, skipna: bool = True, 
        ) -> np.ndarray | pd.Series | pd.DataFrame:
        return self.fit(X, y, independant_fit, skipna).predict()
    
    def get_params(self) -> dict:
        return {
            **self.window_params,
            "n_jobs": BaseTimeSeriesModel.n_jobs,
        }
    
    def copy(self) -> 'BaseTimeSeriesModel':
        return self.__class__(
            _custom_clone_model(self.model), **self.get_params()
        )

class RollingModel(BaseTimeSeriesModel):
    def __init__(
        self, model: BaseEstimator | object, 
        window_size: int, freq_retraining: int = 1, 
        lookahead_steps:int = 0,
    ) -> None:
        super().__init__(
            model = model, freq_retraining=freq_retraining, rolling_window_size=window_size, 
            min_train_steps=window_size, lookahead_steps=lookahead_steps, 
        )

class ExpandingModel(BaseTimeSeriesModel):
    def __init__(
        self, model: BaseEstimator | object, freq_retraining: int, 
        min_train_steps: int = None, lookahead_steps:int = 0,
    ) -> None:
        super().__init__(
            model = model, freq_retraining=freq_retraining, rolling_window_size=None, 
            min_train_steps=min_train_steps, lookahead_steps=lookahead_steps, 
        )
