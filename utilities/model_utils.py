"""
Model utilities for macroeconomic regime analysis and credit risk modeling.

This module provides comprehensive model training, evaluation, and management utilities:
- Data splitting strategies for time series data
- Model persistence (saving/loading)
- Regression model training and evaluation
- SARIMAX time series modeling
- Hyperparameter tuning with cross-validation
- Model visualization and interpretation
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split
from statsmodels.tsa.statespace.sarimax import SARIMAX

from config import RANDOM_STATE, CV_FOLDS, TIME_SERIES_CV_FOLDS
from utilities.logging_config import get_logger, log_data_info, log_model_info, LogContext, log_function_call

# Initialize module logger
logger = get_logger(__name__)

# Set up matplotlib for better visualization
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

@log_function_call
def chronological_split(
    df: pd.DataFrame,
    split_date: Union[str, pd.Timestamp],
    train_inclusive: bool = True,
    test_inclusive: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame chronologically based on a cutoff date.
    
    This function is specifically designed for time series data to maintain
    temporal order and prevent data leakage from future information.
    
    Args:
        df: Input DataFrame to split (must have datetime index or date column)
        split_date: Cutoff date for splitting
        train_inclusive: Whether to include split_date in training set
        test_inclusive: Whether to include split_date in test set
        
    Returns:
        Tuple of (train_df, test_df)
        
    Raises:
        ValueError: If df is empty or split_date is invalid
    """
    with LogContext("chronological_split", logger=logger, split_date=split_date):
        # Input validation
        if df.empty:
            raise ValueError("Input DataFrame is empty")
        
        # Convert split_date to pandas Timestamp
        if isinstance(split_date, str):
            split_date = pd.to_datetime(split_date)
        elif not isinstance(split_date, pd.Timestamp):
            raise ValueError("split_date must be string or pandas Timestamp")
        
        logger.info("Starting chronological split",
                   input_shape=df.shape,
                   split_date=str(split_date),
                   train_inclusive=train_inclusive,
                   test_inclusive=test_inclusive)
        
        # Handle different index types
        if isinstance(df.index, pd.DatetimeIndex):
            # Datetime index
            if train_inclusive:
                train_mask = df.index <= split_date
            else:
                train_mask = df.index < split_date
            
            if test_inclusive:
                test_mask = df.index >= split_date
            else:
                test_mask = df.index > split_date
                
        else:
            # Non-datetime index - try to find date column
            date_cols = df.select_dtypes(include=['datetime64']).columns
            
            if len(date_cols) == 0:
                logger.warning("No datetime index or columns found, using row numbers for split")
                # Fall back to row-based split if no date information
                split_point = int(len(df) * 0.8)  # Default 80/20 split
                train_df = df.iloc[:split_point].copy()
                test_df = df.iloc[split_point:].copy()
                return train_df, test_df
            
            # Use first datetime column
            date_col = date_cols[0]
            
            if train_inclusive:
                train_mask = df[date_col] <= split_date
            else:
                train_mask = df[date_col] < split_date
            
            if test_inclusive:
                test_mask = df[date_col] >= split_date
            else:
                test_mask = df[date_col] > split_date
        
        # Apply splits
        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()
        
        # Log split information
        logger.info("Chronological split completed",
                   train_shape=train_df.shape,
                   test_shape=test_df.shape,
                   train_percentage=(len(train_df) / len(df)) * 100,
                   test_percentage=(len(test_df) / len(df)) * 100,
                   date_range_train=(train_df.index.min(), train_df.index.max()) if hasattr(train_df.index, 'min') else None,
                   date_range_test=(test_df.index.min(), test_df.index.max()) if hasattr(test_df.index, 'min') else None)
        
        # Validate no overlap
        if len(train_df) > 0 and len(test_df) > 0:
            if hasattr(train_df.index, 'max') and hasattr(test_df.index, 'min'):
                if train_df.index.max() >= test_df.index.min():
                    logger.warning("Potential data leakage: train and test sets overlap")
        
        return train_df, test_df

# ===========================================
# Model Persistence Utilities
# ===========================================

@log_function_call
def save_model(
    model: Any,
    filepath: Union[str, Path],
    include_metadata: bool = True,
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Save a trained model to disk with comprehensive error handling and logging.
    
    This function saves models using joblib with optional metadata storage.
    It creates directories as needed and validates the save operation.
    
    Args:
        model: Trained model to save
        filepath: Path where to save the model
        include_metadata: Whether to save additional metadata
        metadata: Additional metadata to save (if include_metadata is True)
        
    Returns:
        True if save was successful, False otherwise
        
    Raises:
        ValueError: If filepath is invalid or model is None
    """
    with LogContext("model_saving", logger=logger, filepath=str(filepath)):
        # Input validation
        if model is None:
            raise ValueError("Model cannot be None")
        
        if not filepath:
            raise ValueError("Filepath cannot be empty")
        
        # Convert to Path object
        filepath = Path(filepath) if isinstance(filepath, str) else filepath
        
        logger.info("Saving model to disk",
                   model_type=type(model).__name__,
                   filepath=str(filepath),
                   include_metadata=include_metadata)
        
        try:
            # Create directory if it doesn't exist
            filepath.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Directory created/verified", directory=str(filepath.parent))
            
            # Prepare metadata if requested
            save_data = {"model": model}
            if include_metadata:
                if metadata is None:
                    metadata = {
                        "model_type": type(model).__name__,
                        "saved_at": pd.Timestamp.now(),
                        "python_version": os.sys.version,
                        "description": "Model saved via utilities.model_utils.save_model"
                    }
                save_data["metadata"] = metadata
                
                logger.info("Metadata prepared for saving", metadata_keys=list(metadata.keys()))
            
            # Save the model
            joblib.dump(save_data, filepath, compress=3)
            
            # Verify the file was created and is not empty
            if filepath.exists() and filepath.stat().st_size > 0:
                logger.info("Model saved successfully",
                           filepath=str(filepath),
                           file_size_bytes=filepath.stat().st_size)
                return True
            else:
                logger.error("Model save failed - file not created or empty")
                return False
                
        except Exception as e:
            logger.error("Failed to save model", error=str(e), filepath=str(filepath))
            return False


@log_function_call
def load_model(
    filepath: Union[str, Path],
    load_metadata: bool = True
) -> Union[Any, Tuple[Any, Dict[str, Any]]]:
    """
    Load a saved model from disk with validation and error handling.
    
    This function loads models saved with save_model and optionally returns
    any saved metadata for model provenance tracking.
    
    Args:
        filepath: Path to the saved model file
        load_metadata: Whether to load and return metadata
        
    Returns:
        If load_metadata is False: Model object
        If load_metadata is True: Tuple of (model, metadata_dict)
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        ValueError: If file is corrupted or invalid
    """
    with LogContext("model_loading", logger=logger, filepath=str(filepath)):
        # Input validation
        filepath = Path(filepath) if isinstance(filepath, str) else filepath
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        logger.info("Loading model from disk",
                   filepath=str(filepath),
                   file_size_bytes=filepath.stat().st_size,
                   load_metadata=load_metadata)
        
        try:
            # Load the saved data
            saved_data = joblib.load(filepath)
            
            # Handle different save formats
            if isinstance(saved_data, dict) and "model" in saved_data:
                model = saved_data["model"]
                metadata = saved_data.get("metadata", {})
                logger.info("Model loaded successfully with metadata format",
                           model_type=type(model).__name__,
                           has_metadata=bool(metadata))
            else:
                # Legacy format (direct model save)
                model = saved_data
                metadata = {
                    "model_type": type(model).__name__,
                    "loaded_at": pd.Timestamp.now(),
                    "note": "Legacy model format - metadata limited"
                }
                logger.info("Model loaded successfully (legacy format)",
                           model_type=type(model).__name__)
            
            # Log model information
            log_model_info(model, "loaded_model", logger)
            
            if load_metadata:
                logger.info("Returning model with metadata")
                return model, metadata
            else:
                logger.info("Returning model only")
                return model
                
        except Exception as e:
            logger.error("Failed to load model", error=str(e), filepath=str(filepath))
            raise ValueError(f"Failed to load model from {filepath}: {e}")

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