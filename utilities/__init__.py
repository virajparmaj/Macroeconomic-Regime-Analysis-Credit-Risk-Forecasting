"""
Utilities package for macroeconomic regime analysis and credit risk modeling.

This package provides core functionality for:
- Data processing and cleaning
- Feature engineering utilities  
- Model training and evaluation helpers
- Statistical analysis functions
- Comprehensive logging and monitoring

The package is designed with extensive logging for production monitoring
and debugging. All functions include proper error handling and validation.
"""

from .logging_config import (
    # Logging setup and configuration
    setup_logging,
    get_logger,
    log_function_call,
    log_data_info,
    log_model_info,
    LogContext,
    init_default_logging,
)

# Import data processing utilities with error handling
try:
    from .data_processing import (
        preprocess_data,
        create_features,
        run_eda,
        pipeline_data_preparation,
    )
    _DATA_PROCESSING_AVAILABLE = True
except ImportError as e:
    _DATA_PROCESSING_AVAILABLE = False
    print(f"Warning: Data processing module import failed: {e}")

# Import core functions with error handling
try:
    from .functions import (
        # Data cleaning
        remove_outliers_zscore,
        winsorize_series,
        clip_infinities,
        
        # Statistical utilities
        perform_ljung_box_test,
        plot_acf_pacf,
        
        # Feature engineering
        add_lag_features,
        add_rolling_features,
        add_growth_rates,
        add_interaction_terms,
        
        # Unsupervised learning
        perform_pca,
        cluster_data,
        evaluate_clustering,
        
        # EDA helpers
        plot_correlations,
        basic_eda,
        
        # Sequence utilities
        fit_scaler,
        make_sequences,
    )
    _FUNCTIONS_AVAILABLE = True
except ImportError as e:
    _FUNCTIONS_AVAILABLE = False
    print(f"Warning: Functions module import failed: {e}")

# Import model utilities with error handling
try:
    from .model_utils import (
        # Data splitting
        chronological_split,
        
        # Model persistence
        save_model,
        load_model,
        
        # Model training and evaluation
        train_regression_model,
        evaluate_regression_model,
        
        # Time series models
        train_sarimax,
        forecast_sarimax,
        
        # Hyperparameter tuning
        perform_hyperparameter_tuning,
        
        # Visualization
        plot_forecast,
        plot_feature_importance,
    )
    _MODEL_UTILS_AVAILABLE = True
except ImportError as e:
    _MODEL_UTILS_AVAILABLE = False
    print(f"Warning: Model utilities module import failed: {e}")

# Package metadata and version
__version__ = "1.0.0"
__author__ = "Macroeconomic Regime Analysis Team"
__email__ = "team@example.com"

# Package description
__description__ = """
Utilities for macroeconomic regime analysis and credit risk modeling.

This package provides comprehensive tools for:
- Time series data processing and feature engineering
- Unsupervised learning for regime detection
- Supervised learning for credit risk forecasting
- Model evaluation and visualization
- Production-ready logging and monitoring

All functions include extensive logging, error handling, and validation
suitable for both research and production environments.
"""

# Determine available functionality
AVAILABLE_MODULES = {
    "data_processing": _DATA_PROCESSING_AVAILABLE,
    "functions": _FUNCTIONS_AVAILABLE,
    "model_utils": _MODEL_UTILS_AVAILABLE,
}

# Build dynamic __all__ based on available modules
__all__ = [
    # Logging utilities (always available)
    "setup_logging",
    "get_logger",
    "log_function_call",
    "log_data_info",
    "log_model_info",
    "LogContext",
    "init_default_logging",
    "AVAILABLE_MODULES",
    "__version__",
    "__author__",
    "__description__",
]

# Add data processing exports if available
if _DATA_PROCESSING_AVAILABLE:
    __all__.extend([
        "preprocess_data",
        "create_features",
        "run_eda", 
        "pipeline_data_preparation",
    ])

# Add functions exports if available
if _FUNCTIONS_AVAILABLE:
    __all__.extend([
        "remove_outliers_zscore",
        "winsorize_series",
        "clip_infinities",
        "perform_ljung_box_test",
        "plot_acf_pacf",
        "add_lag_features",
        "add_rolling_features",
        "add_growth_rates",
        "add_interaction_terms",
        "perform_pca",
        "cluster_data",
        "evaluate_clustering",
        "plot_correlations",
        "basic_eda",
        "fit_scaler",
        "make_sequences",
    ])

# Add model utilities exports if available
if _MODEL_UTILS_AVAILABLE:
    __all__.extend([
        "chronological_split",
        "save_model",
        "load_model",
        "train_regression_model",
        "evaluate_regression_model",
        "train_sarimax",
        "forecast_sarimax",
        "perform_hyperparameter_tuning",
        "plot_forecast",
        "plot_feature_importance",
    ])

# Initialize logging when package is imported
def _init_package_logging():
    """Initialize default logging for the package."""
    try:
        # Initialize logger for this package
        logger = get_logger(__name__)
        logger.info(
            "Utilities package loaded",
            version=__version__,
            available_modules=AVAILABLE_MODULES,
            total_exports=len(__all__)
        )
    except Exception:
        # Silently fail logging initialization to avoid import issues
        pass

# Initialize logging
_init_package_logging()

# Package initialization complete
print(f"Utilities package v{__version__} loaded successfully")
print(f"Available modules: {sum(AVAILABLE_MODULES.values())}/{len(AVAILABLE_MODULES)}")
print(f"Total exports: {len(__all__)} functions")