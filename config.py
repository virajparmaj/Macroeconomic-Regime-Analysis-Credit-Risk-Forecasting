"""
Configuration module for macroeconomic regime analysis and credit risk modeling.

This module contains all project-wide configuration settings including:
- File paths and directories
- Model hyperparameters
- Training and testing parameters
- Data processing settings
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Tuple

from utilities.logging_config import get_logger

# Initialize logger
logger = get_logger(__name__)

# =========================================================
# Project Structure and Paths
# =========================================================

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()
logger.info("Project root determined", project_root=str(PROJECT_ROOT))

# Data directory paths
DATA_DIR = PROJECT_ROOT / "data"
ORIGINAL_DATA_DIR = DATA_DIR / "original"
logger.info("Data directories configured", data_dir=str(DATA_DIR), original_data_dir=str(ORIGINAL_DATA_DIR))

# Use absolute paths to prevent directory traversal and ensure cross-platform compatibility
MACRO_CREDIT_DATA_PATH = DATA_DIR / "merged_macroeconomic_credit.csv"
MACRO_DATA_PATH = DATA_DIR / "macroeconomic_data_merged.csv"
CREDIT_DATA_PATH = DATA_DIR / "credit_spread_monthly_mean.csv"

# Model directory paths
MODEL_CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"
MODEL_FINAL_DIR = PROJECT_ROOT / "models" / "final"
DL_MODEL_DIR = PROJECT_ROOT / "models" / "deep"

# Log all configured paths
logger.info("File paths configured", 
           macro_credit_data=str(MACRO_CREDIT_DATA_PATH),
           macro_data=str(MACRO_DATA_PATH),
           credit_data=str(CREDIT_DATA_PATH),
           model_checkpoint_dir=str(MODEL_CHECKPOINT_DIR),
           model_final_dir=str(MODEL_FINAL_DIR),
           dl_model_dir=str(DL_MODEL_DIR))

# =========================================================
# Global Hyperparameters and Settings
# =========================================================

# Random state for reproducibility
RANDOM_STATE = 42
logger.info("Random state configured", random_state=RANDOM_STATE)

# Data split parameters
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1  # For models that use validation sets
logger.info("Data split parameters configured", test_size=TEST_SIZE, validation_size=VALIDATION_SIZE)

# Dimensionality reduction parameters
N_COMPONENTS = 4
logger.info("PCA components configured", n_components=N_COMPONENTS)

# Clustering parameters
N_CLUSTERS = 3
CLUSTERING_METHODS = ["kmeans", "gmm"]  # Available clustering methods
logger.info("Clustering parameters configured", n_clusters=N_CLUSTERS, methods=CLUSTERING_METHODS)

# =========================================================
# Time Series Configuration
# =========================================================

# Training and testing date ranges (Data covers 1996-12-01 to 2022-08-01)
TRAIN_START_DATE = "1996-12-01"
TRAIN_END_DATE = "2018-12-31"
TEST_START_DATE = "2019-01-01"
TEST_END_DATE = "2022-08-01"

# Validation date range (optional, for models that use validation)
VALIDATION_START_DATE = "2018-09-01"
VALIDATION_END_DATE = "2018-12-31"

logger.info("Time series date ranges configured",
           train_start=TRAIN_START_DATE,
           train_end=TRAIN_END_DATE,
           validation_start=VALIDATION_START_DATE,
           validation_end=VALIDATION_END_DATE,
           test_start=TEST_START_DATE,
           test_end=TEST_END_DATE)

# =========================================================
# Deep Learning Configuration
# =========================================================

# Sequence parameters for LSTM models
SEQ_LEN = 12  # Number of time steps (months) fed to LSTM
FORECAST_STEPS = 1  # Forecast horizon (t + 1 month)
logger.info("Sequence parameters configured", seq_len=SEQ_LEN, forecast_steps=FORECAST_STEPS)

# Training parameters
BATCH_SIZE = 32
EPOCHS = 150
EARLY_STOPPING_PATIENCE = 20
LEARNING_RATE = 0.001
logger.info("Training parameters configured", 
           batch_size=BATCH_SIZE,
           epochs=EPOCHS,
           early_stopping_patience=EARLY_STOPPING_PATIENCE,
           learning_rate=LEARNING_RATE)

# =========================================================
# Data Processing Configuration
# =========================================================

# Outlier detection parameters
OUTLIER_ZSCORE_THRESHOLD = 3.0
WINSORIZE_LIMITS = (0.01, 0.01)  # Lower and upper percentiles for winsorization
logger.info("Outlier processing configured",
           zscore_threshold=OUTLIER_ZSCORE_THRESHOLD,
           winsorize_limits=WINSORIZE_LIMITS)

# Feature engineering parameters
DEFAULT_LAGS = [1, 2, 3]  # Default lag periods for time series features
DEFAULT_ROLLING_WINDOWS = [3, 6, 12]  # Default rolling window sizes
logger.info("Feature engineering parameters configured",
           default_lags=DEFAULT_LAGS,
           default_rolling_windows=DEFAULT_ROLLING_WINDOWS)

# Missing value handling
DEFAULT_FILLNA_METHOD = "ffill"  # Forward fill as default
logger.info("Missing value handling configured", default_fillna_method=DEFAULT_FILLNA_METHOD)

# =========================================================
# Model Evaluation Configuration
# =========================================================

# Cross-validation parameters
CV_FOLDS = 5
TIME_SERIES_CV_FOLDS = 3  # Fewer folds for time series due to data constraints
logger.info("Cross-validation configured", cv_folds=CV_FOLDS, time_series_cv_folds=TIME_SERIES_CV_FOLDS)

# Evaluation metrics
REGRESSION_METRICS = ["mse", "mae", "r2", "rmse", "mape"]
CLASSIFICATION_METRICS = ["accuracy", "precision", "recall", "f1"]
logger.info("Evaluation metrics configured",
           regression_metrics=REGRESSION_METRICS,
           classification_metrics=CLASSIFICATION_METRICS)

# =========================================================
# Logging and Monitoring Configuration
# =========================================================

# Logging settings
LOG_LEVEL = "INFO"
ENABLE_FILE_LOGGING = True
ENABLE_CONSOLE_LOGGING = True
ENABLE_JSON_LOGGING = False  # Set to True for production environments
logger.info("Logging configuration completed",
           log_level=LOG_LEVEL,
           file_logging=ENABLE_FILE_LOGGING,
           console_logging=ENABLE_CONSOLE_LOGGING,
           json_logging=ENABLE_JSON_LOGGING)

# =========================================================
# Utility Functions
# =========================================================

def get_all_config() -> Dict[str, Any]:
    """
    Get all configuration parameters as a dictionary.
    
    Returns:
        Dictionary containing all configuration parameters
    """
    config = {
        "paths": {
            "project_root": str(PROJECT_ROOT),
            "data_dir": str(DATA_DIR),
            "original_data_dir": str(ORIGINAL_DATA_DIR),
            "macro_credit_data": str(MACRO_CREDIT_DATA_PATH),
            "macro_data": str(MACRO_DATA_PATH),
            "credit_data": str(CREDIT_DATA_PATH),
            "model_checkpoint_dir": str(MODEL_CHECKPOINT_DIR),
            "model_final_dir": str(MODEL_FINAL_DIR),
            "dl_model_dir": str(DL_MODEL_DIR),
        },
        "hyperparameters": {
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "validation_size": VALIDATION_SIZE,
            "n_components": N_COMPONENTS,
            "n_clusters": N_CLUSTERS,
            "clustering_methods": CLUSTERING_METHODS,
        },
        "time_series": {
            "train_start_date": TRAIN_START_DATE,
            "train_end_date": TRAIN_END_DATE,
            "validation_start_date": VALIDATION_START_DATE,
            "validation_end_date": VALIDATION_END_DATE,
            "test_start_date": TEST_START_DATE,
            "test_end_date": TEST_END_DATE,
        },
        "deep_learning": {
            "seq_len": SEQ_LEN,
            "forecast_steps": FORECAST_STEPS,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "learning_rate": LEARNING_RATE,
        },
        "data_processing": {
            "outlier_zscore_threshold": OUTLIER_ZSCORE_THRESHOLD,
            "winsorize_limits": WINSORIZE_LIMITS,
            "default_lags": DEFAULT_LAGS,
            "default_rolling_windows": DEFAULT_ROLLING_WINDOWS,
            "default_fillna_method": DEFAULT_FILLNA_METHOD,
        },
        "evaluation": {
            "cv_folds": CV_FOLDS,
            "time_series_cv_folds": TIME_SERIES_CV_FOLDS,
            "regression_metrics": REGRESSION_METRICS,
            "classification_metrics": CLASSIFICATION_METRICS,
        },
        "logging": {
            "log_level": LOG_LEVEL,
            "enable_file_logging": ENABLE_FILE_LOGGING,
            "enable_console_logging": ENABLE_CONSOLE_LOGGING,
            "enable_json_logging": ENABLE_JSON_LOGGING,
        }
    }
    
    logger.info("Configuration dictionary retrieved", config_keys=list(config.keys()))
    return config


def validate_paths() -> bool:
    """
    Validate that all required directories exist or can be created.
    
    Returns:
        True if all paths are valid, False otherwise
    """
    required_dirs = [
        DATA_DIR,
        ORIGINAL_DATA_DIR,
        MODEL_CHECKPOINT_DIR,
        MODEL_FINAL_DIR,
        DL_MODEL_DIR,
    ]
    
    all_valid = True
    for dir_path in required_dirs:
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info("Directory validated/created", directory=str(dir_path))
        except Exception as e:
            logger.error("Failed to create directory", directory=str(dir_path), error=str(e))
            all_valid = False
    
    return all_valid


def get_data_file_info() -> Dict[str, Dict[str, Any]]:
    """
    Get information about data files (existence, size, etc.).
    
    Returns:
        Dictionary with information about each data file
    """
    data_files = {
        "macro_credit": MACRO_CREDIT_DATA_PATH,
        "macro": MACRO_DATA_PATH,
        "credit": CREDIT_DATA_PATH,
    }
    
    file_info = {}
    for name, path in data_files.items():
        info = {
            "path": str(path),
            "exists": path.exists(),
        }
        
        if path.exists():
            stat = path.stat()
            info.update({
                "size_bytes": stat.st_size,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "modified_time": stat.st_mtime,
            })
        
        file_info[name] = info
    
    logger.info("Data file information retrieved", file_count=len(file_info))
    return file_info


# Initialize and validate configuration on import
logger.info("Configuration module loaded successfully")
if not validate_paths():
    logger.warning("Some path validations failed, but configuration loading continues")
