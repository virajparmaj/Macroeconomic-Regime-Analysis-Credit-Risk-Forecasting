"""
Data processing and preprocessing utilities for macroeconomic regime analysis.

This module provides comprehensive data cleaning, preprocessing, and feature engineering
functionality specifically designed for time series macroeconomic and credit risk data.
"""

import os
from pathlib import Path
from typing import List, Optional, Union, Dict, Any, Tuple

import numpy as np
import pandas as pd

from config import (
    MACRO_CREDIT_DATA_PATH,
    DEFAULT_FILLNA_METHOD,
    OUTLIER_ZSCORE_THRESHOLD,
    WINSORIZE_LIMITS
)
from utilities.logging_config import get_logger, log_data_info, LogContext, log_function_call

# Initialize module logger
logger = get_logger(__name__)

# Import utility functions with error handling
try:
    from utilities.functions import (
        basic_eda,
        plot_correlations,
        remove_outliers_zscore,
        winsorize_series,
        clip_infinities,
        add_lag_features,
        add_rolling_features,
        add_growth_rates,
        add_interaction_terms
    )
    logger.info("Successfully imported utility functions from functions module")
except ImportError as e:
    logger.error("Failed to import utility functions", error=str(e))
    raise

@log_function_call
def preprocess_data(
    df: pd.DataFrame,
    outlier_cols: Optional[List[str]] = None,
    winsorize_cols: Optional[List[str]] = None,
    fillna_method: str = DEFAULT_FILLNA_METHOD,
    outlier_threshold: float = OUTLIER_ZSCORE_THRESHOLD,
    winsorize_limits: Tuple[float, float] = WINSORIZE_LIMITS,
    clip_infinity_value: float = 0.0
) -> pd.DataFrame:
    """
    Comprehensive data cleaning and preprocessing pipeline.
    
    This function performs multiple data cleaning steps including:
    - Sorting by date index
    - Handling missing values with various methods
    - Outlier detection and removal
    - Winsorization of extreme values
    - Clipping infinite values
    
    Args:
        df: Input DataFrame to preprocess
        outlier_cols: List of column names to apply outlier removal
        winsorize_cols: List of column names to apply winsorization
        fillna_method: Method for filling missing values ('ffill', 'bfill', 'zero')
        outlier_threshold: Z-score threshold for outlier detection
        winsorize_limits: Tuple of lower and upper percentiles for winsorization
        clip_infinity_value: Value to use when clipping infinities
        
    Returns:
        Preprocessed DataFrame
        
    Raises:
        ValueError: If df is empty or fillna_method is invalid
        TypeError: If df is not a pandas DataFrame
    """
    with LogContext("data_preprocessing", logger=logger):
        # Input validation
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(df)}")
        
        if df.empty:
            raise ValueError("Input DataFrame is empty")
        
        logger.info("Starting data preprocessing",
                   input_shape=df.shape,
                   input_columns=list(df.columns),
                   fillna_method=fillna_method,
                   outlier_cols=outlier_cols,
                   winsorize_cols=winsorize_cols)
        
        # Create a copy to avoid modifying the original DataFrame
        df_processed = df.copy()
        log_data_info(df_processed, "preprocessing_start", logger)
        
        # Step 1: Sort by date index (if index is datetime-like)
        if hasattr(df_processed.index, 'to_datetime'):
            df_processed = df_processed.sort_index()
            logger.info("DataFrame sorted by index")
        else:
            logger.warning("DataFrame index is not datetime-like, skipping sort")
        
        # Step 2: Handle missing values
        df_processed = _handle_missing_values(
            df_processed, 
            method=fillna_method,
            logger=logger
        )
        
        # Step 3: Outlier removal
        if outlier_cols:
            df_processed = _handle_outliers(
                df_processed,
                outlier_cols,
                threshold=outlier_threshold,
                logger=logger
            )
        
        # Step 4: Winsorization
        if winsorize_cols:
            df_processed = _handle_winsorization(
                df_processed,
                winsorize_cols,
                limits=winsorize_limits,
                logger=logger
            )
        
        # Step 5: Clip infinities
        df_processed = _handle_infinities(
            df_processed,
            fill_value=clip_infinity_value,
            logger=logger
        )
        
        # Log final state
        log_data_info(df_processed, "preprocessing_complete", logger)
        
        logger.info("Data preprocessing completed successfully",
                   output_shape=df_processed.shape,
                   output_columns=list(df_processed.columns))
        
        return df_processed


def _handle_missing_values(
    df: pd.DataFrame,
    method: str,
    logger: Any
) -> pd.DataFrame:
    """
    Handle missing values using various methods.
    
    Args:
        df: DataFrame to process
        method: Method for filling missing values
        logger: Logger instance
        
    Returns:
        DataFrame with missing values handled
    """
    initial_null_count = df.isnull().sum().sum()
    
    if method == 'ffill':
        df = df.ffill()
        logger.info("Applied forward fill for missing values")
    elif method == 'bfill':
        df = df.bfill()
        logger.info("Applied backward fill for missing values")
    elif method == 'zero':
        df = df.fillna(0)
        logger.info("Filled missing values with zeros")
    else:
        raise ValueError(f"Invalid fillna_method: {method}")
    
    final_null_count = df.isnull().sum().sum()
    logger.info("Missing values handled",
               initial_nulls=initial_null_count,
               final_nulls=final_null_count,
               nulls_filled=initial_null_count - final_null_count)
    
    return df


def _handle_outliers(
    df: pd.DataFrame,
    outlier_cols: List[str],
    threshold: float,
    logger: Any
) -> pd.DataFrame:
    """
    Handle outliers in specified columns using Z-score method.
    
    Args:
        df: DataFrame to process
        outlier_cols: Columns to process for outliers
        threshold: Z-score threshold
        logger: Logger instance
        
    Returns:
        DataFrame with outliers handled
    """
    outliers_removed = 0
    
    for col in outlier_cols:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found for outlier removal")
            continue
        
        initial_count = len(df)
        df[col] = remove_outliers_zscore(df[col], threshold=threshold)
        final_count = len(df)
        removed_count = initial_count - final_count
        outliers_removed += removed_count
        
        logger.info(f"Outliers removed from column '{col}'",
                   removed_count=removed_count,
                   threshold=threshold)
    
    logger.info("Outlier removal completed",
               total_outliers_removed=outliers_removed,
               final_shape=df.shape)
    
    return df


def _handle_winsorization(
    df: pd.DataFrame,
    winsorize_cols: List[str],
    limits: Tuple[float, float],
    logger: Any
) -> pd.DataFrame:
    """
    Apply winsorization to specified columns.
    
    Args:
        df: DataFrame to process
        winsorize_cols: Columns to winsorize
        limits: Lower and upper percentile limits
        logger: Logger instance
        
    Returns:
        DataFrame with winsorization applied
    """
    for col in winsorize_cols:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found for winsorization")
            continue
        
        df[col] = winsorize_series(df[col], limits=limits)
        logger.info(f"Winsorization applied to column '{col}'",
                   limits=limits)
    
    logger.info("Winsorization completed",
               winsorized_columns=winsorize_cols,
               limits=limits)
    
    return df


def _handle_infinities(
    df: pd.DataFrame,
    fill_value: float,
    logger: Any
) -> pd.DataFrame:
    """
    Handle infinite values in the DataFrame.
    
    Args:
        df: DataFrame to process
        fill_value: Value to use for infinities
        logger: Logger instance
        
    Returns:
        DataFrame with infinities handled
    """
    inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    
    if inf_count > 0:
        df = clip_infinities(df, fill_value=fill_value)
        logger.info("Infinite values clipped",
                   infinity_count=inf_count,
                   fill_value=fill_value)
    else:
        logger.info("No infinite values found")
    
    return df

@log_function_call
def create_features(
    df: pd.DataFrame,
    lag_columns: Optional[List[str]] = None,
    lag_periods: Optional[List[int]] = None,
    rolling_columns: Optional[List[str]] = None,
    rolling_windows: Optional[List[int]] = None,
    growth_columns: Optional[List[str]] = None,
    interaction_pairs: Optional[List[Tuple[str, str]]] = None,
    target_column: Optional[str] = 'Credit_Spread',
    forecast_horizon: int = 1
) -> pd.DataFrame:
    """
    Create comprehensive features for macroeconomic and credit risk modeling.
    
    This function generates various types of time series features including:
    - Lag features for temporal dependencies
    - Rolling window statistics for trend analysis
    - Growth rates for momentum indicators
    - Interaction terms for non-linear relationships
    - Target variables for supervised learning
    
    Args:
        df: Input DataFrame
        lag_columns: Columns to create lag features for
        lag_periods: List of lag periods to create
        rolling_columns: Columns to create rolling features for
        rolling_windows: List of rolling window sizes
        growth_columns: Columns to calculate growth rates for
        interaction_pairs: List of column pairs for interaction terms
        target_column: Column to use for target variable creation
        forecast_horizon: Number of periods ahead for target variable
        
    Returns:
        DataFrame with engineered features
        
    Raises:
        ValueError: If target_column is not found in DataFrame
    """
    with LogContext("feature_engineering", logger=logger):
        # Input validation
        if df.empty:
            raise ValueError("Input DataFrame is empty")
        
        # Set default values
        if lag_columns is None:
            lag_columns = ['GDP', 'CPI', 'Credit_Spread']
        if lag_periods is None:
            lag_periods = [1, 2, 3]
        if rolling_columns is None:
            rolling_columns = ['Unemployment_Rate', 'Credit_Spread']
        if rolling_windows is None:
            rolling_windows = [3, 6, 12]
        if growth_columns is None:
            growth_columns = ['GDP', 'CPI', 'Industrial_Production']
        if interaction_pairs is None:
            interaction_pairs = [('GDP_growth', 'CPI_growth'), ('Credit_Spread', 'FEDFUNDS')]
        
        logger.info("Starting feature engineering",
                   input_shape=df.shape,
                   lag_columns=lag_columns,
                   lag_periods=lag_periods,
                   rolling_columns=rolling_columns,
                   rolling_windows=rolling_windows,
                   growth_columns=growth_columns,
                   interaction_pairs=interaction_pairs)
        
        # Create a copy to avoid modifying the original
        df_features = df.copy()
        initial_columns = set(df_features.columns)
        
        # Step 1: Create lag features
        df_features = _create_lag_features(
            df_features,
            columns=lag_columns,
            lags=lag_periods,
            logger=logger
        )
        
        # Step 2: Create rolling features
        df_features = _create_rolling_features(
            df_features,
            columns=rolling_columns,
            windows=rolling_windows,
            logger=logger
        )
        
        # Step 3: Create growth rate features
        df_features = _create_growth_features(
            df_features,
            columns=growth_columns,
            logger=logger
        )
        
        # Step 4: Create interaction terms
        df_features = _create_interaction_features(
            df_features,
            col_pairs=interaction_pairs,
            logger=logger
        )
        
        # Step 5: Create target variable
        if target_column and target_column in df_features.columns:
            df_features = _create_target_variable(
                df_features,
                target_column=target_column,
                forecast_horizon=forecast_horizon,
                logger=logger
            )
        elif target_column:
            logger.warning(f"Target column '{target_column}' not found in DataFrame")
        
        # Log feature creation results
        final_columns = set(df_features.columns)
        new_features = final_columns - initial_columns
        
        logger.info("Feature engineering completed",
                   input_shape=df.shape,
                   output_shape=df_features.shape,
                   new_features_count=len(new_features),
                   new_features=list(new_features))
        
        return df_features


def _create_lag_features(
    df: pd.DataFrame,
    columns: List[str],
    lags: List[int],
    logger: Any
) -> pd.DataFrame:
    """Create lag features for specified columns."""
    available_columns = [col for col in columns if col in df.columns]
    missing_columns = set(columns) - set(available_columns)
    
    if missing_columns:
        logger.warning("Some columns not found for lag features", missing_columns=list(missing_columns))
    
    if available_columns:
        df = add_lag_features(df, columns=available_columns, lags=lags)
        logger.info("Lag features created", columns=available_columns, lags=lags)
    
    return df


def _create_rolling_features(
    df: pd.DataFrame,
    columns: List[str],
    windows: List[int],
    logger: Any
) -> pd.DataFrame:
    """Create rolling window features for specified columns."""
    available_columns = [col for col in columns if col in df.columns]
    missing_columns = set(columns) - set(available_columns)
    
    if missing_columns:
        logger.warning("Some columns not found for rolling features", missing_columns=list(missing_columns))
    
    if available_columns:
        for window in windows:
            df = add_rolling_features(df, columns=available_columns, window=window)
        logger.info("Rolling features created", columns=available_columns, windows=windows)
    
    return df


def _create_growth_features(
    df: pd.DataFrame,
    columns: List[str],
    logger: Any
) -> pd.DataFrame:
    """Create growth rate features for specified columns."""
    available_columns = [col for col in columns if col in df.columns]
    missing_columns = set(columns) - set(available_columns)
    
    if missing_columns:
        logger.warning("Some columns not found for growth features", missing_columns=list(missing_columns))
    
    if available_columns:
        df = add_growth_rates(df, columns=available_columns)
        logger.info("Growth rate features created", columns=available_columns)
    
    return df


def _create_interaction_features(
    df: pd.DataFrame,
    col_pairs: List[Tuple[str, str]],
    logger: Any
) -> pd.DataFrame:
    """Create interaction features for specified column pairs."""
    valid_pairs = []
    for col1, col2 in col_pairs:
        if col1 in df.columns and col2 in df.columns:
            valid_pairs.append((col1, col2))
        else:
            missing = []
            if col1 not in df.columns:
                missing.append(col1)
            if col2 not in df.columns:
                missing.append(col2)
            logger.warning(f"Columns not found for interaction: {missing}")
    
    if valid_pairs:
        df = add_interaction_terms(df, col_pairs=valid_pairs)
        logger.info("Interaction features created", valid_pairs=valid_pairs)
    
    return df


def _create_target_variable(
    df: pd.DataFrame,
    target_column: str,
    forecast_horizon: int,
    logger: Any
) -> pd.DataFrame:
    """Create target variable for supervised learning."""
    target_name = f"Target_{forecast_horizon}step_ahead"
    df[target_name] = df[target_column].shift(-forecast_horizon)
    
    logger.info("Target variable created",
               target_column=target_column,
               target_name=target_name,
               forecast_horizon=forecast_horizon)
    
    return df

@log_function_call
def run_eda(
    df: pd.DataFrame,
    correlation_columns: Optional[List[str]] = None,
    correlation_title: str = "Correlation Matrix",
    save_plots: bool = False,
    plot_dir: Optional[str] = None
) -> None:
    """
    Run comprehensive exploratory data analysis.
    
    This function performs various EDA tasks including:
    - Basic statistical analysis
    - Correlation analysis and visualization
    - Missing value analysis
    - Data quality checks
    
    Args:
        df: DataFrame to analyze
        correlation_columns: Columns to include in correlation analysis
        correlation_title: Title for correlation plot
        save_plots: Whether to save plots to disk
        plot_dir: Directory to save plots (if save_plots is True)
        
    Raises:
        ValueError: If df is empty
    """
    with LogContext("exploratory_data_analysis", logger=logger):
        # Input validation
        if df.empty:
            raise ValueError("Input DataFrame is empty")
        
        logger.info("Starting exploratory data analysis",
                   shape=df.shape,
                   columns=list(df.columns))
        
        # Step 1: Basic EDA
        logger.info("Running basic EDA")
        basic_eda(df)
        
        # Step 2: Correlation analysis
        if correlation_columns is None:
            correlation_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        available_columns = [col for col in correlation_columns if col in df.columns]
        
        if available_columns:
            logger.info("Running correlation analysis", columns=available_columns)
            plot_correlations(
                df,
                columns=available_columns,
                title=correlation_title,
                save=save_plots,
                filepath=os.path.join(plot_dir, "correlation_matrix.png") if save_plots and plot_dir else None
            )
        else:
            logger.warning("No numeric columns found for correlation analysis")
        
        # Step 3: Additional EDA insights
        _analyze_data_quality(df, logger)
        
        logger.info("Exploratory data analysis completed")


def _analyze_data_quality(df: pd.DataFrame, logger: Any) -> None:
    """Analyze data quality and log insights."""
    # Missing value analysis
    missing_analysis = df.isnull().sum()
    high_missing = missing_analysis[missing_analysis > len(df) * 0.1]
    
    if not high_missing.empty:
        logger.warning("Columns with high missing values detected",
                      columns=high_missing.to_dict(),
                      threshold="10%")
    
    # Data type analysis
    dtype_counts = df.dtypes.value_counts().to_dict()
    logger.info("Data types distribution", dtype_counts=dtype_counts)
    
    # Duplicate rows analysis
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        logger.warning("Duplicate rows detected", duplicate_count=duplicate_count)
    else:
        logger.info("No duplicate rows found")

@log_function_call
def pipeline_data_preparation(
    filepath: Union[str, Path] = MACRO_CREDIT_DATA_PATH,
    index_col: Union[int, str] = 0,
    parse_dates: Union[bool, List[str]] = True,
    outlier_cols: Optional[List[str]] = None,
    winsorize_cols: Optional[List[str]] = None,
    fillna_method: str = DEFAULT_FILLNA_METHOD,
    run_eda_flag: bool = True,
    save_processed_data: bool = False,
    output_path: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Complete data preparation pipeline from loading to final processing.
    
    This pipeline includes:
    - Data loading with proper parsing
    - Data preprocessing (cleaning, outlier handling, etc.)
    - Feature engineering
    - Exploratory data analysis
    - Final cleanup and validation
    
    Args:
        filepath: Path to the data file
        index_col: Column to use as index
        parse_dates: Whether to parse dates automatically
        outlier_cols: Columns to process for outliers
        winsorize_cols: Columns to winsorize
        fillna_method: Method for filling missing values
        run_eda_flag: Whether to run EDA
        save_processed_data: Whether to save the processed data
        output_path: Path to save processed data (if save_processed_data is True)
        
    Returns:
        Fully processed DataFrame ready for modeling
        
    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If data loading fails or DataFrame is empty
    """
    with LogContext("data_preparation_pipeline", logger=logger, filepath=str(filepath)):
        # Validate file existence
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        # Set default columns for processing
        if outlier_cols is None:
            outlier_cols = ['Credit_Spread']
        if winsorize_cols is None:
            winsorize_cols = ['GDP']
        
        logger.info("Starting data preparation pipeline",
                   filepath=str(filepath),
                   outlier_cols=outlier_cols,
                   winsorize_cols=winsorize_cols,
                   fillna_method=fillna_method)
        
        # Step 1: Load data
        logger.info("Loading data from file")
        try:
            df = pd.read_csv(filepath, index_col=index_col, parse_dates=parse_dates)
            logger.info("Data loaded successfully", shape=df.shape, columns=list(df.columns))
        except Exception as e:
            logger.error("Failed to load data", error=str(e))
            raise ValueError(f"Failed to load data from {filepath}: {e}")
        
        # Validate loaded data
        if df.empty:
            raise ValueError("Loaded DataFrame is empty")
        
        log_data_info(df, "data_loaded", logger)
        
        # Step 2: Preprocess data
        logger.info("Starting data preprocessing")
        df = preprocess_data(
            df,
            outlier_cols=outlier_cols,
            winsorize_cols=winsorize_cols,
            fillna_method=fillna_method
        )
        
        # Step 3: Feature engineering
        logger.info("Starting feature engineering")
        df = create_features(df)
        
        # Step 4: Exploratory Data Analysis (optional)
        if run_eda_flag:
            logger.info("Starting exploratory data analysis")
            run_eda(df)
        
        # Step 5: Final cleanup
        logger.info("Performing final cleanup")
        initial_shape = df.shape
        df = df.dropna()
        rows_removed = initial_shape[0] - df.shape[0]
        
        logger.info("Final cleanup completed",
                   initial_rows=initial_shape[0],
                   final_rows=df.shape[0],
                   rows_removed=rows_removed,
                   final_columns=list(df.columns))
        
        # Step 6: Save processed data (optional)
        if save_processed_data:
            if output_path is None:
                output_path = filepath.parent / f"processed_{filepath.name}"
            
            logger.info("Saving processed data", output_path=str(output_path))
            df.to_csv(output_path, index=True)
            logger.info("Processed data saved successfully")
        
        # Final validation
        _validate_processed_data(df, logger)
        
        logger.info("Data preparation pipeline completed successfully",
                   final_shape=df.shape,
                   final_columns=list(df.columns))
        
        return df


def _validate_processed_data(df: pd.DataFrame, logger: Any) -> None:
    """Validate the processed data meets quality standards."""
    validation_results = {
        "has_data": not df.empty,
        "has_numeric_columns": len(df.select_dtypes(include=[np.number]).columns) > 0,
        "no_infinite_values": not np.isinf(df.select_dtypes(include=[np.number])).any().any(),
        "minimal_missing": df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) < 0.05,
    }
    
    all_passed = all(validation_results.values())
    
    if all_passed:
        logger.info("Data validation passed", **validation_results)
    else:
        logger.warning("Data validation had issues", **validation_results)
    
    # Additional checks
    if df.index.dtype.kind in ['M', 'm']:  # Check if index is datetime-like
        logger.info("Datetime index detected", index_range=(df.index.min(), df.index.max()))
    else:
        logger.info("Non-datetime index detected", index_type=str(df.index.dtype))