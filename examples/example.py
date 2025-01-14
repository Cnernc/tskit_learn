import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor

from tskit_learn.timeseriesmodel import ExpandingModel, RollingModel
from tskit_learn.autoregressivemodel import AutoRegressiveModel

X = pd.read_parquet("./sample_data/features.parquet")
y = pd.read_parquet("./sample_data/target.parquet")

#Build any scikit-learn model and train it using only data from the past to avoir look ahead bias.

# ExpandingModel will retrain the model every 30 days using all the data from the past.
model = ExpandingModel(
    model = RandomForestRegressor(), 
    freq_retraining = 30, 
    min_train_steps=252
)
model.fit(X, y)
y_hat = model.predict()

#Rolling model will retrain the model every days using data from a rolling window of 30 days
model = RollingModel(model = RandomForestRegressor(), rolling_window = 30)
model.fit(X, y)
y_hat = model.predict()

# Compatible with sklearn objects
model = make_pipeline(PCA(n_components=3), LinearRegression())
model = ExpandingModel(model = model, freq_retraining = 252)
model.fit(X, X)
X_hat = model.predict()

#Can also be used with non scikit-learn model if they have a fit and predict method
xgb = XGBRegressor()
assert hasattr(xgb, "fit") and hasattr(xgb, "predict")
model = ExpandingModel(model = xgb, freq_retraining = 252)
model.fit(X, y)
y_hat = model.predict()

#Compatible with pd.MultiIndex. If X.columns is a pd.MultiIndex, only X[col] will be used to predict y[col].
zscore = (y - y.rolling(252).mean()) / y.rolling(252).std()
X.columns = pd.MultiIndex.from_product([y.columns, X.columns])
for col in y.columns:
    X.loc[:, (col, 'zscore')] = zscore[col].shift(1)
model.fit(X, y) 
y_hat = model.predict()

# AutoRegressiveModel train the model with ARIMA features (autoregressive, integration and moving average)
model = AutoRegressiveModel(
    model = RandomForestRegressor(), 
    freq_retraining = 252, 
    autoregressive_order = 10, # Autoregressive features will use the past 'autoregressive_order' days as features to predict the next day
    integration_order = 1, # Integration will differentiate the target 'integration_order' times before predicting and then integrate the prediction 'integration_order' times
    moving_average_order = 5, # Moving average features will use a moving average of the residuals (y-y_hat) of the past 'moving_average_order' days to predict the next day
)
model.fit(y, X)
y_hat = model.predict()

# is_geometric = True use .pct_change() instead of .diff() to differentiate the data. 
# Useful to work with price data as target when 'integration_order' > 0
price = pd.read_parquet("sample_data/price.parquet")
model.fit(X, price, is_geometric=True) 
y_hat = model.predict()
