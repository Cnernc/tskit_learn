import numpy as np
import pandas as pd
import multiprocessing as mp
from typing import Generator, Tuple
from sklearn.base import BaseEstimator
from .utilitaires import _custom_clone_model, _clean_and_reindex

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
            "min_train_steps": min_train_steps if min_train_steps else freq_retraining,
            "lookahead_steps": lookahead_steps,
        }

        if not isinstance(model,  BaseEstimator):
            assert hasattr(model, "fit") and hasattr(model, "predict") and hasattr(model, "get_params"), "model should have fit and predict methods"
            Warning(f'TimeSeriesModel is optimised for sklearn.base.BaseEstimator not {model.__class__.__name__}')
        
        if not hasattr(model, "get_params"):
            Warning(f"model can't be cloned: {model.__class__.__name__}, multiprocessing won't be used. You can force the use of multiprocessing by setting n_jobs")
            BaseTimeSeriesModel.n_jobs = 1

    @staticmethod
    def window_grouper(
        X: np.ndarray, rolling_window_size: int, freq_retraining: int, min_train_steps: int, lookahead_steps:int
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        
        assert 2 < min_train_steps <= len(X), ("min_train_steps should be less than or equal to the length of X and greater than 2")
        assert freq_retraining <= len(X), ("freq_retraining should be less than or equal to the length of X")
        
        training_date = min_train_steps

        def _get_slices(
                training_date:int, freq_retraining:int, 
                rolling_window_size:int, lenght:int
            ) -> tuple[slice, slice]:
            if rolling_window_size:
                training_slice = slice(max(0, training_date - rolling_window_size - 1), training_date - 1 - lookahead_steps)
                test_slice = slice(training_date, min(training_date + freq_retraining, lenght))
            else:
                training_slice = slice(0, training_date - 1 - lookahead_steps)
                test_slice = slice(training_date, min(training_date + freq_retraining, lenght))
            return training_slice, test_slice

        while training_date < len(X):
            training_slice, test_slice = _get_slices(training_date, freq_retraining, rolling_window_size, len(X))
            X_train, X_test = X[training_slice], X[test_slice]
            training_date += freq_retraining
            yield X_train, X_test

    @staticmethod
    def _fit_predict_static(
        model: BaseEstimator | object, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray
    ) -> np.ndarray:
        """Static fit and predict for the multiprocessing"""
        return model.fit(X_train, y_train).predict(X_test)

    def _fit_predict_ndarray(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        assert X.shape[0] == y.shape[0], ("X and y should have the same number of rows")
        assert np.all(np.isfinite(X)) and np.all(np.isfinite(y)), ("X and y should not have any missing/infinite values")

        X_generator = BaseTimeSeriesModel.window_grouper(X, **self.window_params)
        y_generator = BaseTimeSeriesModel.window_grouper(y, **self.window_params)
        
        tasks = (
            (_custom_clone_model(self.model), X_train, y_train, X_test)
            for (X_train, X_test), (y_train, _) in zip(X_generator, y_generator)
        )

        with mp.Pool(BaseTimeSeriesModel.n_jobs) as pool:
            results = pool.starmap(
                BaseTimeSeriesModel._fit_predict_static, tasks
                )

        y_hat = np.concatenate(results)
        return np.concatenate([np.full(len(y) - len(y_hat), np.nan), y_hat])

    def _fit_predict_ds(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        assert not X.empty, f"No features for {y.name}"
        X = _clean_and_reindex(X, y)
        y_hat_values = self._fit_predict_ndarray(X.values, y.values)
        return pd.Series(y_hat_values, y.index)

    def _fit_predict_df(self, X: pd.DataFrame, y: pd.DataFrame, skipna: bool) -> pd.DataFrame:

        y_hat = pd.DataFrame(index = y.index)
        for col in y.columns:
            X_ = X.loc[:, col] if isinstance(X.columns, pd.MultiIndex) else X
            y_ = y.loc[:, col] if not skipna else y.loc[:, col].dropna()
            y_hat[col] = self._fit_predict_ds(X_, y_)
        return y_hat
    
        #Can't use multiprocessing because there are child multi process called in the loop
        X_tasks = (X.loc[:, col] if isinstance(X.columns, pd.MultiIndex) else X for col in y.columns)
        y_tasks = (y.loc[:, col].dropna() if skipna else y.loc[:, col] for col in y.columns)
        tasks = zip(X_tasks, y_tasks)
        with mp.Pool(BaseTimeSeriesModel.n_jobs) as pool:
            results = pool.starmap(self._fit_predict_ds, tasks)
        return pd.DataFrame(results, index = y.index, columns = y.columns)
 
    def fit(
        self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series | pd.DataFrame, skipna: bool = True, 
    ) -> 'BaseTimeSeriesModel':
        
        match (type(X), type(y)):
            case (np.ndarray, np.ndarray):
                y_hat = self._fit_predict_ndarray(X, y)
            case (pd.DataFrame, pd.Series):
                y_hat = self._fit_predict_ds(X, y if not skipna else y.dropna()).reindex(y.index)
            case (pd.DataFrame, pd.DataFrame):
                y_hat = self._fit_predict_df(X, y, skipna).reindex(y.index)
            case _:
                raise ValueError(
                    f"""Unsupported types: X type {type(X)}, y type {type(y)}. X,y should be in :
                    (np.ndarray, np.ndarray) ; (pd.DataFrame, pd.Series) ; (pd.DataFrame, pd.DataFrame)
                    """
                )
        self.y, self.X, self.y_hat = y.copy(), X.copy(), y_hat

        return self 
    
    def predict(self, _: None= None) -> np.ndarray | pd.Series | pd.DataFrame:
        assert hasattr(self, "y_hat"), "Model should be fit before predict"
        return self.y_hat.copy()
    
    def fit_predict(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series | pd.DataFrame, skipna: bool = True) -> np.ndarray | pd.Series | pd.DataFrame:
        return self.fit(X, y, skipna).predict()
    
    @property
    def residual(self) -> np.ndarray | pd.Series | pd.DataFrame:
        assert hasattr(self, "y_hat"), "Model should be fit before computing the residual"
        return self.y - self.predict()
    
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
        window_size: int, lookahead_steps:int = 0,
    ) -> None:
        super().__init__(
            model = model, freq_retraining=1, rolling_window_size=window_size, 
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
