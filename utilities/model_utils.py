# model_utils.py

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, train_test_split
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ===========================================
# Model Saving/Loading
# ===========================================
def save_model(model, filepath):
    """
    Save model to file via joblib.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath):
    """
    Load model from file.
    """
    model = joblib.load(filepath)
    return model

# ===========================================
# Regression Model Training & Evaluation
# ===========================================
def train_regression_model(model, X_train, y_train):
    """
    Train a generic regression model (e.g., Linear, Ridge, RandomForest).
    """
    print("Training regression model...")
    model.fit(X_train, y_train)
    return model

def evaluate_regression_model(model, X_test, y_test):
    """
    Evaluate regression model performance using MSE, MAE, R2.
    """
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"Evaluation:\n MSE={mse:.4f} | MAE={mae:.4f} | R2={r2:.4f}")
    return {"mse": mse, "mae": mae, "r2": r2}

# ===========================================
# SARIMAX for Time Series
# ===========================================
def train_sarimax(endog, exog=None, order=(1,1,1), seasonal_order=(0,0,0,0)):
    """
    Train a SARIMAX model for time series forecasting of credit spreads.
    """
    print("Fitting SARIMAX model...")
    model = SARIMAX(endog, exog=exog, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(disp=False)
    print(results.summary())
    return results

def forecast_sarimax(model_fit, steps=12, exog_future=None):
    """
    Forecast with the fitted SARIMAX model.
    """
    forecast = model_fit.forecast(steps=steps, exog=exog_future)
    return forecast

# ===========================================
# Hyperparameter Tuning (GridSearch) 
# (Example for scikit-learn regressors)
# ===========================================
def perform_hyperparameter_tuning(estimator, param_grid, X_train, y_train, cv=5, scoring='neg_mean_squared_error'):
    """
    Generic hyperparameter tuning using GridSearchCV for any scikit-learn estimator.
    """
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    print(f"Best Params: {grid_search.best_params_}")
    print(f"Best Score: {grid_search.best_score_}")
    return grid_search.best_estimator_

# ===========================================
# Visualization Helpers
# ===========================================
def plot_forecast(actual, predicted, title="Forecast vs Actual", steps_ahead=12):
    """
    Plot the forecasted values against the actual values.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(actual.index, actual, label='Actual')
    if hasattr(predicted, 'index'):
        plt.plot(predicted.index, predicted, label='Predicted', linestyle='--')
    else:
        # If predicted is not a Series with index, just plot the tail
        forecast_index = actual.index[-steps_ahead:]
        plt.plot(forecast_index, predicted, label='Predicted', linestyle='--')
    plt.title(title)
    plt.legend()
    plt.show()

def plot_feature_importance(model, feature_names):
    """
    Plot feature importances for tree-based models or coefficients for linear models.
    """
    if hasattr(model, "feature_importances_"):
        # Tree-based
        importances = model.feature_importances_
        importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    elif hasattr(model, "coef_"):
        # Linear or Ridge/Lasso
        importances = model.coef_
        importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances.flatten()})
    else:
        print("No importances or coefficients available for this model.")
        return

    importance_df = importance_df.sort_values("Importance", ascending=False)
    plt.figure(figsize=(8, 5))
    plt.bar(importance_df["Feature"], importance_df["Importance"])
    plt.xticks(rotation=45, ha="right")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.show()