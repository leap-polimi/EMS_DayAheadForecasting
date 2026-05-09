"""Self-contained LEAR model used for day-ahead electricity price forecasting.

The LEAR model is a parameter-rich ARX structure estimated with L1
regularization. It trains 24 LASSO models, one for each hour of the next day.

This version is intentionally self-contained for repository publication: it does
not depend on project-specific data_wrangling modules from the original code.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LassoCV, LassoLarsIC
from sklearn.preprocessing import RobustScaler
from sklearn.utils._testing import ignore_warnings

logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")


class LEAR:
    """LEAR model for day-ahead electricity price forecasting.

    Args:
        calibration_window: Number of past days used for daily recalibration.
        small: If True, use LassoCV. If False, use LassoLarsIC.
    """

    def __init__(self, calibration_window: int = 364, small: bool = False):
        self.calibration_window = calibration_window
        self.small = small
        self.models: dict[int, Lasso] = {}
        self.scaler_x = RobustScaler()
        self.scaler_y = RobustScaler()

    @ignore_warnings(category=ConvergenceWarning)
    def recalibrate(
        self,
        Xtrain: np.ndarray,
        Ytrain: np.ndarray,
        normalization: str = "median",
        transformation: str = "invariant",
    ) -> None:
        """Fit 24 hourly LASSO models on the calibration window.

        The normalization/transformation arguments are retained for API
        compatibility with the original implementation. This self-contained
        version uses robust median/IQR scaling.
        """
        if Xtrain.ndim != 2 or Ytrain.ndim != 2:
            raise ValueError("Xtrain and Ytrain must be 2D arrays.")

        Xtrain = Xtrain.astype(float, copy=True)
        Ytrain = Ytrain.astype(float, copy=True)

        # Scale all non-dummy features. The last seven features are weekday dummies.
        if Xtrain.shape[1] > 7:
            X_no_dummies = self.scaler_x.fit_transform(Xtrain[:, :-7])
            Xtrain[:, :-7] = X_no_dummies

        Ytrain_scaled = self.scaler_y.fit_transform(Ytrain)

        self.models = {}
        for hour in range(24):
            if self.small:
                alpha_model = LassoCV(eps=1e-6, n_alphas=100, cv=7, max_iter=5000)
                alpha = alpha_model.fit(Xtrain, Ytrain_scaled[:, hour]).alpha_
            else:
                alpha_model = LassoLarsIC(criterion="aic", max_iter=2500)
                alpha = alpha_model.fit(Xtrain, Ytrain_scaled[:, hour]).alpha_

            model = Lasso(max_iter=5000, alpha=alpha)
            model.fit(Xtrain, Ytrain_scaled[:, hour])
            self.models[hour] = model

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the 24 hourly prices of the next day."""
        if not self.models:
            raise RuntimeError("Model has not been recalibrated yet.")

        X = X.astype(float, copy=True)
        if X.shape[1] > 7:
            X[:, :-7] = self.scaler_x.transform(X[:, :-7])

        y_pred_scaled = np.zeros(24)
        for hour in range(24):
            y_pred_scaled[hour] = self.models[hour].predict(X)[0]

        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(1, -1))
        return y_pred

    def recalibrate_predict(
        self,
        Xtrain: np.ndarray,
        Ytrain: np.ndarray,
        Xtest: np.ndarray,
        normalization: str = "median",
        transformation: str = "invariant",
    ) -> np.ndarray:
        """Recalibrate the model and forecast the next day."""
        self.recalibrate(
            Xtrain=Xtrain,
            Ytrain=Ytrain,
            normalization=normalization,
            transformation=transformation,
        )
        return self.predict(Xtest)

    def _build_and_split_xys(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        date_test: pd.Timestamp,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create LEAR input-output arrays from training and test dataframes."""
        if df_train.index[0].hour != 0 or df_test.index[0].hour != 0:
            print("Warning: LEAR input index does not start at 00:00.")

        n_exogenous_inputs = len(df_train.columns) - 1
        n_features = 96 + 7 + n_exogenous_inputs * 72

        index_train_values = df_train.loc[df_train.index[0] + pd.Timedelta(weeks=1):].index
        index_test_values = df_test.loc[date_test:date_test + pd.Timedelta(hours=23)].index

        pred_dates_train = index_train_values.round("1h")[::24]
        pred_dates_test = index_test_values.round("1h")[::24]

        index_train = pd.DataFrame(index=pred_dates_train, columns=[f"h{hour}" for hour in range(24)])
        index_test = pd.DataFrame(index=pred_dates_test, columns=[f"h{hour}" for hour in range(24)])

        for hour in range(24):
            index_train[f"h{hour}"] = index_train.index + pd.Timedelta(hours=hour)
            index_test[f"h{hour}"] = index_test.index + pd.Timedelta(hours=hour)

        x_train = np.zeros((index_train.shape[0], n_features))
        x_test = np.zeros((index_test.shape[0], n_features))
        y_train = np.zeros((index_train.shape[0], 24))

        feature_index = 0

        # Historical prices: D-1, D-2, D-3, D-7.
        for hour in range(24):
            for past_day in [1, 2, 3, 7]:
                past_train = pd.to_datetime(index_train[f"h{hour}"].values) - pd.Timedelta(hours=24 * past_day)
                past_test = pd.to_datetime(index_test[f"h{hour}"].values) - pd.Timedelta(hours=24 * past_day)
                x_train[:, feature_index] = df_train.loc[past_train, "Price"]
                x_test[:, feature_index] = df_test.loc[past_test, "Price"]
                feature_index += 1

        # Exogenous inputs: D-1, D-7, and D.
        for hour in range(24):
            for past_day in [1, 7]:
                for exog in range(1, n_exogenous_inputs + 1):
                    past_train = pd.to_datetime(index_train[f"h{hour}"].values) - pd.Timedelta(hours=24 * past_day)
                    past_test = pd.to_datetime(index_test[f"h{hour}"].values) - pd.Timedelta(hours=24 * past_day)
                    x_train[:, feature_index] = df_train.loc[past_train, f"Exogenous {exog}"]
                    x_test[:, feature_index] = df_test.loc[past_test, f"Exogenous {exog}"]
                    feature_index += 1

            for exog in range(1, n_exogenous_inputs + 1):
                future_train = pd.to_datetime(index_train[f"h{hour}"].values)
                future_test = pd.to_datetime(index_test[f"h{hour}"].values)
                x_train[:, feature_index] = df_train.loc[future_train, f"Exogenous {exog}"]
                x_test[:, feature_index] = df_test.loc[future_test, f"Exogenous {exog}"]
                feature_index += 1

        # Weekday dummies.
        for day_of_week in range(7):
            x_train[index_train.index.dayofweek == day_of_week, feature_index] = 1
            x_test[index_test.index.dayofweek == day_of_week, feature_index] = 1
            feature_index += 1

        for hour in range(24):
            future_train = pd.to_datetime(index_train[f"h{hour}"].values)
            y_train[:, hour] = df_train.loc[future_train, "Price"]

        return x_train, y_train, x_test

    def recalibrate_and_forecast_next_day(
        self,
        df: pd.DataFrame,
        calibration_window: int,
        next_day_date: pd.Timestamp,
        normalization: str = "median",
        transformation: str = "invariant",
    ) -> np.ndarray:
        """Recalibrate on the latest window and forecast a selected day."""
        next_day_date = pd.Timestamp(next_day_date)
        df_train = df.loc[: next_day_date - pd.Timedelta(hours=1)]
        df_train = df_train.iloc[-calibration_window * 24:]
        df_test = df.loc[next_day_date - pd.Timedelta(weeks=2):, :]

        x_train, y_train, x_test = self._build_and_split_xys(
            df_train=df_train,
            df_test=df_test,
            date_test=next_day_date,
        )

        return self.recalibrate_predict(
            Xtrain=x_train,
            Ytrain=y_train,
            Xtest=x_test,
            normalization=normalization,
            transformation=transformation,
        )
