import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from .timeseriesmodel import BaseTimeSeriesModel
from .utilitaires import _get_cpu_count

class AutoRegressiveModel(BaseTimeSeriesModel):

    def __init__(
        self, model: BaseEstimator | object, freq_retraining: int, lookahead_steps: int = 0,
        autoregressive_order: int = 0, integration_order: int = 0, moving_average_order: int = 0, 
        min_train_steps: int = None, n_jobs: int = _get_cpu_count(), 
    ) -> None:
        super().__init__(
            model=model, freq_retraining=freq_retraining, rolling_window_size=None, 
            min_train_steps=min_train_steps, lookahead_steps=lookahead_steps, 
            n_jobs=n_jobs
        )
        self.autoregressive_order = autoregressive_order
        self.integration_order = integration_order
        self.moving_average_order = moving_average_order

    @staticmethod
    def _differentiate(series: pd.Series, integration_order: int, is_geometric: bool = False) -> pd.Series:
        series = series.dropna()
        for _ in range(integration_order):
            series = series.diff() if not is_geometric else series.pct_change()
        return series.fillna(0)
    
    @staticmethod
    def _integrate(series: pd.Series, integration_order: int, is_geometric: bool = False) -> pd.Series:
        series = series.dropna()
        for _ in range(integration_order):
            series = series.cumsum() if not is_geometric else (1 + series).cumprod()
        return series.replace([np.inf, -np.inf], np.nan).ffill()
    
    @staticmethod
    def _autoregressive_features(y: pd.Series, autoregressive_order: int) -> pd.DataFrame:
        return pd.concat({
            f'ar_{i}': y.shift(i)
            for i in range(1, autoregressive_order + 1)
        }, axis=1)
    
    @staticmethod
    def _moving_average_features(residual: pd.Series, moving_average_order: int) -> pd.DataFrame:
        return pd.concat({
            f'ma_{i}': residual.rolling(i).mean()
            for i in range(1, moving_average_order + 1)
        }, axis=1)
    
    def _fit_predict_ar_ds(self, X: pd.DataFrame, y: pd.Series, is_geometric: bool) -> pd.Series:
        
        y = self._differentiate(y, self.integration_order, is_geometric)
        ar_features = self._autoregressive_features(y, self.autoregressive_order)
        X = pd.concat([X, ar_features], axis=1) if not X.empty else ar_features
        y_hat = super()._fit_predict_ds(X, y)

        if self.moving_average_order > 0:
            residual = (y - y_hat).shift(1)
            ma_features = AutoRegressiveModel._moving_average_features(residual, self.moving_average_order)
            X = pd.concat([X, ma_features], axis=1)
            y_hat = super()._fit_predict_ds(X, y)
        
        y_hat = AutoRegressiveModel._integrate(y_hat, self.integration_order, is_geometric)

        return y_hat
    
    def _fit_predict_ar_df(self, X: pd.DataFrame, y: pd.DataFrame, is_geometric: bool, skipna: bool) -> pd.DataFrame:

        y_hat = pd.DataFrame(index = y.index)
        for col in y.columns:
            X_ = X.loc[:, col] if isinstance(X.columns, pd.MultiIndex) else X
            y_ = y.loc[:, col] if not skipna else y.loc[:, col].dropna()
            y_hat[col] = self._fit_predict_ar_ds(X_, y_, is_geometric)
            
        return y_hat
    
    def fit(self, X: pd.DataFrame, y: pd.Series | pd.DataFrame, is_geometric: bool = False, skipna: bool = True) -> pd.Series | pd.DataFrame:
        if is_geometric:
            raise NotImplementedError("Geometric integration is not yet supported")
        match (type(X), type(y)):
            case (pd.DataFrame, pd.Series):
                self.y_hat = self._fit_predict_ar_ds(X, y.dropna() if skipna else y, is_geometric)
            case (pd.DataFrame, pd.DataFrame):
                self.y_hat = self._fit_predict_ar_df(X, y, is_geometric, skipna)
            case _:
                raise ValueError(f"Unsupported types: X type {type(X)}, y type {type(y)}")
        return self
    
    def fit_predict(self, X: pd.DataFrame, y: pd.Series | pd.DataFrame, is_geometric: bool = False, skipna: bool = True) -> pd.Series | pd.DataFrame:
        self.fit(X, y, is_geometric, skipna)
        return self.y_hat
    
    def get_params(self) -> dict:
        return {
            **super().get_params(),
            "autoregressive_order": self.autoregressive_order,
            "integration_order": self.integration_order,
            "moving_average_order": self.moving_average_order,
        }