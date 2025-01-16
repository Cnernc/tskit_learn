# tskit-learn

A Python package for time series modeling that transforms any scikit-learn estimator into a time series compatible model. It offers different modeling approaches such as Rolling, Expanding, and AutoRegressive methods.

This package allows you to:
- Convert any scikit-learn model into a time series model
- Work with non scikit-learn models (as long as they implement .fit() and .predict() methods)
- Automatically handle pandas temporal indexes and MultiIndex
- Choose between different time series approaches:
  - Rolling window: Train your model on a moving fixed-size window
  - Expanding window: Train your model on an increasing window of historical data
  - AutoRegressive: Include lagged values as features for time series forecasting

## Usage Examples

### Basic Usage with Different Models

```python
from sklearn.ensemble import RandomForestRegressor
from tskit_learn import ExpandingModel, RollingModel

# ExpandingModel: retrain the model every 30 days using all past data
tsmodel = ExpandingModel(
    model=RandomForestRegressor(), 
    freq_retraining=30, 
    min_train_steps=252 # The first 'min_train_steps' values will be NaN
)
tsmodel.fit(X, y)
y_hat = tsmodel.predict()

# RollingModel: retrain daily using a 30-day rolling window
tsmodel = RollingModel(
    model=RandomForestRegressor(), 
    rolling_window=30
)
tsmodel.fit(X, y)
y_hat = tsmodel.predict()

```
### Compatible with classification or regression
```python

from sklearn.linear_model import LogisticRegression

tsmodel = RollingModel(
    tsmodel=LogisticRegression(), 
    rolling_window=30
)
tsmodel.fit(X, y > 0)
p = tsmodel.predict()

```

### Forecast with no lookahead
```python

tsmodel = ExpandingModel(
    tsmodel=RandomForestRegressor(), 
    freq_retraining=252, 
    lookahead_steps = 1 # Will adapt its training window not to train on look ahead
) 
tsmodel.fit(X, y.shift(-1))
y_hat = tsmodel.predict()
```



### Integration with Scikit-learn Pipeline

```python
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

# Works with sklearn pipelines
pipe = make_pipeline(PCA(n_components=3), LinearRegression())
tsmodel = ExpandingModel(model=pipe, freq_retraining=252)
tsmodel.fit(X, X)
X_hat = tsmodel.predict()
```

### Support for Non Scikit-learn Models

```python
from xgboost import XGBRegressor

# Compatible with any model having fit() and predict() methods
xgb = XGBRegressor()
tsmodel = ExpandingModel(model=xgb, freq_retraining=252)
tsmodel.fit(X, y)
y_hat = tsmodel.predict()
```

### MultiIndex Support

```python
# Handles pd.MultiIndex for column-specific predictions
zscore = (y - y.rolling(252).mean()) / y.rolling(252).std()
X.columns = pd.MultiIndex.from_product([y.columns, X.columns])
for col in y.columns:
    X.loc[:, (col, 'zscore')] = zscore[col].shift(1)
tsmodel.fit(X, y) 
y_hat = tsmodel.predict()
```

### AutoRegressive Features

```python
from tskit_learn import AutoRegressiveModel

# AutoRegressiveModel with ARIMA features
tsmodel = AutoRegressiveModel(
    model=RandomForestRegressor(), 
    freq_retraining=252, 
    autoregressive_order=10,  # Use past 10 days as features
    integration_order=1,      # Differentiate target once
    moving_average_order=5,   # Use 5-day MA of residuals
)
tsmodel.fit(X, y)
y_hat = tsmodel.predict()

# Geometric integration for price data
tsmodel.fit(X, price, is_geometric=True) 
price_hat = tsmodel.predict()
```

## Parameters

### RollingModel
- `model`: Base estimator (sklearn compatible)
- `rolling_window`: Size of the rolling window
- `freq_retraining`: Frequency of model retraining (in nb of steps)

### ExpandingModel
- `model`: Base estimator (sklearn compatible)
- `min_train_steps`: Minimum number of days before starting predictions
- `freq_retraining`: Frequency of model retraining (in nb of steps)

### AutoRegressiveModel
- `model`: Base estimator (sklearn compatible)
- `autoregressive_order`: Number of lagged values to use
- `integration_order`: Number of differencing operations
- `moving_average_order`: Order of moving average features
- `freq_retraining`: Frequency of model retraining (in nb of steps)

## Contributing

Contributions are welcome! Feel free to:
1. Fork the project
2. Create a feature branch
3. Submit a Pull Request

## License

This project is licensed under the MIT License.
```