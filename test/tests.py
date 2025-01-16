# import unittest
# import numpy as np
# import multiprocessing as mp
# import pandas as pd
# from sklearn.linear_model import LinearRegression
# from tskit_learn.utilitaires import _custom_clone_model, _clean_and_reindex
# from tskit_learn.timeseriesmodel import RollingModel, AutoRegressiveModel, set_n_jobs

# class TestUtilitaires(unittest.TestCase):

#     def test_custom_clone_model_with_sklearn_model(self):
#         model = LinearRegression()
#         cloned = _custom_clone_model(model)
#         self.assertIsInstance(cloned, LinearRegression)
#         self.assertNotEqual(id(model), id(cloned))
#         self.assertEqual(cloned.get_params(), model.get_params())

#     def test_custom_clone_model_with_custom_object(self):
#         class CustomModel:
#             def fit(self, X, y):
#                 pass
#             def predict(self, X):
#                 return np.zeros(len(X))
#             def get_params(self):
#                 return {}
        
#         model = CustomModel()
#         cloned = _custom_clone_model(model)
#         self.assertIs(model.__class__, cloned.__class__)
#         self.assertNotEqual(id(model), id(cloned))

#     def test_clean_and_reindex_with_full_dataframe(self):
#         X = pd.DataFrame({
#             'A': [1, 2, 3, np.inf],
#             'B': [4, np.nan, 6, 7],
#             'C': [np.nan, np.nan, np.nan, np.nan]
#         }, index=[0, 1, 2, 3])
#         y = pd.Series([10, 20, 30, 40], index=[0, 1, 2, 3])
#         cleaned = _clean_and_reindex(X, y)
#         expected = pd.DataFrame({
#             'A': [1, 2, 3, 3],
#             'B': [4, 0, 6, 7]
#         }, index=[0, 1, 2, 3])
#         pd.testing.assert_frame_equal(cleaned, expected)

#     def test_clean_and_reindex_with_missing_y(self):
#         X = pd.DataFrame({
#             'A': [1, 2, np.nan],
#             'B': [4, 5, 6],
#         }, index=[0, 1, 2])
#         cleaned = _clean_and_reindex(X)
#         expected = pd.DataFrame({
#             'A': [1, 2, 2],
#             'B': [4, 5, 6],
#         }, index=[0, 1, 2])
#         pd.testing.assert_frame_equal(cleaned, expected)

# class TestBaseTimeSeriesModel(unittest.TestCase):

#     def setUp(self):
#         self.model = LinearRegression()
#         self.window_size = 5
#         self.freq_retraining = 2
#         self.min_train_steps = 5
#         self.lookahead_steps = 1
#         self.base_model = RollingModel(
#             model=self.model,
#             window_size=self.window_size,
#             lookahead_steps=self.lookahead_steps
#         )
#         self.X = pd.DataFrame({
#             'feature1': np.arange(10),
#             'feature2': np.arange(10, 20)
#         })
#         self.y = pd.Series(np.arange(100, 110))

#     def test_initialization(self):
#         self.assertEqual(self.base_model.freq_retraining, self.freq_retraining)
#         self.assertEqual(self.base_model.window_params['rolling_window_size'], self.window_size)
#         self.assertEqual(self.base_model.lookahead_steps, self.lookahead_steps)
#         self.assertIsInstance(self.base_model.model, LinearRegression)

#     def test_fit_predict(self):
#         self.base_model.fit(self.X, self.y)
#         y_hat = self.base_model.predict()
#         self.assertEqual(len(y_hat), len(self.y))
#         self.assertIsInstance(y_hat, pd.Series)

#     def test_residual(self):
#         self.base_model.fit(self.X, self.y)
#         residual = self.base_model.residual
#         self.assertEqual(len(residual), len(self.y))
#         self.assertIsInstance(residual, pd.Series)

#     def test_copy(self):
#         copied_model = self.base_model.copy()
#         self.assertIsInstance(copied_model, RollingModel)
#         self.assertNotEqual(id(self.base_model), id(copied_model))
#         self.assertEqual(copied_model.get_params(), self.base_model.get_params())

# class TestAutoRegressiveModel(unittest.TestCase):

#     def setUp(self):
#         self.model = LinearRegression()
#         self.freq_retraining = 2
#         self.autoregressive_order = 2
#         self.moving_average_order = 2
#         self.ar_model = AutoRegressiveModel(
#             model=self.model,
#             freq_retraining=self.freq_retraining,
#             lookahead_steps=1,
#             autoregressive_order=self.autoregressive_order,
#             moving_average_order=self.moving_average_order
#         )
#         self.X = pd.DataFrame({
#             'feature1': np.random.randn(100),
#             'feature2': np.random.randn(100)
#         })
#         self.y = pd.Series(np.random.randn(100))

#     def test_initialization(self):
#         self.assertEqual(self.ar_model.freq_retraining, self.freq_retraining)
#         self.assertEqual(self.ar_model.autoregressive_order, self.autoregressive_order)
#         self.assertEqual(self.ar_model.moving_average_order, self.moving_average_order)

#     def test_fit_predict(self):
#         y_hat = self.ar_model.fit_predict(self.X, self.y)
#         self.assertEqual(len(y_hat), len(self.y))
#         self.assertIsInstance(y_hat, pd.Series)

#     def test_get_params(self):
#         params = self.ar_model.get_params()
#         expected = {
#             'rolling_window_size': None,
#             'freq_retraining': self.freq_retraining,
#             'min_train_steps': self.freq_retraining,
#             'lookahead_steps': 1,
#             'n_jobs': set_n_jobs(None) # Assuming default n_jobs
#             ,
#             'autoregressive_order': self.autoregressive_order,
#             'integration_order': 0,
#             'moving_average_order': self.moving_average_order
#         }
#         for key in expected:
#             self.assertIn(key, params)

# class TestSetNJobs(unittest.TestCase):

#     def test_set_n_jobs(self):
#         original_n_jobs = set_n_jobs(2)
#         self.assertEqual(original_n_jobs, 2)
#         new_n_jobs = set_n_jobs(5)
#         self.assertEqual(new_n_jobs, min(5, mp.cpu_count() - 2))
#         self.assertGreaterEqual(new_n_jobs, 1)

# if __name__ == '__main__':
#     unittest.main()
