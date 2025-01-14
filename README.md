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
model = ExpandingModel(
    model=RandomForestRegressor(), 
    freq_retraining=30, 
    min_train_steps=252 # The first 'min_train_steps' values will be NaN
)
model.fit(X, y)
y_hat = model.predict()

# RollingModel: retrain daily using a 30-day rolling window
model = RollingModel(
    model=RandomForestRegressor(), 
    rolling_window=30
)
model.fit(X, y)
y_hat = model.predict()
```

### Integration with Scikit-learn Pipeline

```python
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

# Works with sklearn pipelines
model = make_pipeline(PCA(n_components=3), LinearRegression())
model = ExpandingModel(model=model, freq_retraining=252)
model.fit(X, X)
X_hat = model.predict()
```

### Support for Non Scikit-learn Models

```python
from xgboost import XGBRegressor

# Compatible with any model having fit() and predict() methods
xgb = XGBRegressor()
model = ExpandingModel(model=xgb, freq_retraining=252)
model.fit(X, y)
y_hat = model.predict()
```

### MultiIndex Support

```python
# Handles pd.MultiIndex for column-specific predictions
zscore = (y - y.rolling(252).mean()) / y.rolling(252).std()
X.columns = pd.MultiIndex.from_product([y.columns, X.columns])
for col in y.columns:
    X.loc[:, (col, 'zscore')] = zscore[col].shift(1)
model.fit(X, y) 
y_hat = model.predict()
```

### AutoRegressive Features

```python
from tskit_learn import AutoRegressiveModel

# AutoRegressiveModel with ARIMA features
model = AutoRegressiveModel(
    model=RandomForestRegressor(), 
    freq_retraining=252, 
    autoregressive_order=10,  # Use past 10 days as features
    integration_order=1,      # Differentiate target once
    moving_average_order=5,   # Use 5-day MA of residuals
)
model.fit(X, y)
y_hat = model.predict()

# Geometric integration for price data
model.fit(X, price, is_geometric=True) 
price_hat = model.predict()
```

## Parameters

### RollingModel
- `model`: Base estimator (sklearn compatible)
- `rolling_window`: Size of the rolling window
- `freq_retraining`: Frequency of model retraining (in days)

### ExpandingModel
- `model`: Base estimator (sklearn compatible)
- `min_train_steps`: Minimum number of days before starting predictions
- `freq_retraining`: Frequency of model retraining (in days)

### AutoRegressiveModel
- `model`: Base estimator (sklearn compatible)
- `autoregressive_order`: Number of lagged values to use
- `integration_order`: Number of differencing operations
- `moving_average_order`: Order of moving average features
- `freq_retraining`: Frequency of model retraining (in days)

## Contributing

Contributions are welcome! Feel free to:
1. Fork the project
2. Create a feature branch
3. Submit a Pull Request

## License

This project is licensed under the MIT License.
```