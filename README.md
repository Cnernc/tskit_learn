# TimeseriesModel

A Python package for time series modeling that transforms any scikit-learn estimator into a time series compatible model. It offers different modeling approaches such as Rolling, Expanding, and AutoRegressive methods.

This package allows you to:
- Convert any scikit-learn model into a time series model
- Work with non scikit-learn models in degraded mode (as long as they implement .fit() and .predict() methods)
- Automatically handle pandas temporal indexes
- Choose between different time series approaches:
  - Rolling window: Train your model on a moving fixed-size window
  - Expanding window: Train your model on an increasing window of historical data
  - AutoRegressive: Include lagged values as features for time series forecasting

The package seamlessly integrates with pandas DatetimeIndex and automatically manages temporal data structures, making it easy to work with time series predictions.

## Installation

```bash
pip install -tsmodel .
```

## Usage

```python
from timeseriesModel import RollingModel, ExpandingModel, AutoRegressiveModel

model = RollingModel(window_size=30)
y_hat = model.fit_predict(X, y)

```

### Available Models

1. **RollingModel**
   - Uses a fixed-size rolling window
   - Ideal for data with short-term seasonality

2. **ExpandingModel**
   - Uses all available historical data
   - Suitable for long-term trends

3. **AutoRegressiveModel**
   - Autoregressive model for time series
   - Takes into account temporal dependencies

## Parameters

### RollingModel
- `window_size`: Size of the rolling window
- `base_estimator`: Base estimator (sklearn compatible)

### ExpandingModel
- `min_periods`: Minimum number of periods to start predictions
- `base_estimator`: Base estimator (sklearn compatible)

### AutoRegressiveModel
- `lags`: Number of time lags to use
- `base_estimator`: Base estimator (sklearn compatible)

## Contributing

Contributions are welcome! Feel free to:
1. Fork the project
2. Create a feature branch
3. Submit a Pull Request

## License

This project is licensed under the MIT License.
```