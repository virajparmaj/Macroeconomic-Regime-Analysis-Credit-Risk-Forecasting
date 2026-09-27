"""
Comprehensive utility functions for macroeconomic regime analysis and credit risk modeling.

This module provides a wide range of functions for:
- Data cleaning and preprocessing
- Feature engineering for time series data
- Unsupervised learning (PCA, clustering)
- Statistical analysis and diagnostics
- Sequence data preparation for deep learning
- Visualization utilities
"""

import warnings
from pathlib import Path
from typing import List, Optional, Union, Dict, Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats.mstats import winsorize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

from config import N_COMPONENTS, N_CLUSTERS, RANDOM_STATE
from utilities.logging_config import get_logger, log_data_info, log_model_info, LogContext, log_function_call

# Initialize module logger
logger = get_logger(__name__)

# Configure matplotlib for better plotting
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
warnings.filterwarnings('ignore', category=FutureWarning)

# ===========================================
# Data Cleaning & Utilities
# ===========================================

@log_function_call
def remove_outliers_zscore(
    series: pd.Series,
    threshold: float = 3.0
) -> pd.Series:
    """
    Remove outliers from a pandas Series based on Z-score threshold.
    
    This function identifies and removes outliers using the Z-score method,
    which measures how many standard deviations a data point is from the mean.
    
    Args:
        series: Input pandas Series to process
        threshold: Z-score threshold for outlier detection (default: 3.0)
        
    Returns:
        Series with outliers removed
        
    Raises:
        ValueError: If series is empty or has no valid data
    """
    with LogContext("outlier_removal", logger=logger, threshold=threshold):
        if series.empty:
            raise ValueError("Input Series is empty")
        
        # Drop NaN values for Z-score calculation
        clean_series = series.dropna()
        if clean_series.empty:
            raise ValueError("Series contains only NaN values")
        
        initial_length = len(series)
        
        # Calculate Z-scores
        z_scores = np.abs(stats.zscore(clean_series))
        mask = z_scores < threshold
        
        # Apply mask to original series (preserves NaN positions)
        outlier_mask = series.dropna().index.isin(clean_series[mask].index)
        result = series[outlier_mask]
        
        outliers_removed = initial_length - len(result)
        
        logger.info("Outlier removal completed",
                   initial_length=initial_length,
                   final_length=len(result),
                   outliers_removed=outliers_removed,
                   threshold=threshold,
                   outlier_percentage=(outliers_removed / initial_length) * 100)
        
        return result

@log_function_call
def winsorize_series(
    series: pd.Series,
    limits: Tuple[float, float] = (0.01, 0.01)
) -> pd.Series:
    """
    Winsorize a pandas Series to limit extreme values.
    
    Winsorization caps extreme values at specified percentiles,
    reducing the impact of outliers without completely removing them.
    
    Args:
        series: Input pandas Series to winsorize
        limits: Tuple of lower and upper percentile limits (default: (0.01, 0.01))
        
    Returns:
        Winsorized Series
        
    Raises:
        ValueError: If series is empty or limits are invalid
    """
    with LogContext("winsorization", logger=logger, limits=limits):
        if series.empty:
            raise ValueError("Input Series is empty")
        
        if not (0 <= limits[0] < limits[1] <= 0.5):
            raise ValueError("Invalid limits: must be 0 <= lower < upper <= 0.5")
        
        initial_stats = {
            "mean": series.mean(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max()
        }
        
        # Apply winsorization
        result = winsorize(series, limits=limits)
        result = pd.Series(result, index=series.index, name=series.name)
        
        final_stats = {
            "mean": result.mean(),
            "std": result.std(),
            "min": result.min(),
            "max": result.max()
        }
        
        logger.info("Series winsorization completed",
                   initial_stats=initial_stats,
                   final_stats=final_stats,
                   limits=limits)
        
        return result

def clip_infinities(df, fill_value=0):
    """
    Replace +/- infinities with a fill value in a DataFrame.
    """
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(fill_value)
    return df

def perform_ljung_box_test(residuals, lags=10):
    """
    Perform the Ljung-Box test on residuals to test for autocorrelation.
    """
    lb_test = acorr_ljungbox(residuals, lags=[lags], return_df=True)
    print("Ljung-Box test result:\n", lb_test)

def plot_acf_pacf(series, lags=30):
    """
    Plot ACF and PACF for a given series.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(series.dropna(), lags=lags, ax=axes[0])
    plot_pacf(series.dropna(), lags=lags, ax=axes[1])
    plt.show()

# ===========================================
# Feature Engineering
# ===========================================
def add_lag_features(df, columns, lags=[1,2]):
    """
    Add lagged versions of specified columns to capture temporal dependency.
    e.g., for credit spreads, or macro variables like GDP growth, etc.
    """
    for col in columns:
        if col not in df.columns:
            warnings.warn(f"Column '{col}' not found in DataFrame.")
            continue
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df

def add_rolling_features(df, columns, window=3):
    """
    Add rolling mean or std to capture short-term trends.
    """
    for col in columns:
        if col in df.columns:
            df[f"{col}_rollmean{window}"] = df[col].rolling(window).mean()
            df[f"{col}_rollstd{window}"] = df[col].rolling(window).std()
    return df

def add_growth_rates(df, columns):
    """
    Add percentage growth rates to relevant macro columns, e.g. GDP, CPI.
    """
    for col in columns:
        if col in df.columns:
            df[f"{col}_growth"] = df[col].pct_change() * 100.0
    return df

def add_interaction_terms(df, col_pairs):
    """
    Add interaction terms (product or ratio) between pairs of columns.
    """
    for (col1, col2) in col_pairs:
        if col1 in df.columns and col2 in df.columns:
            df[f"{col1}_x_{col2}"] = df[col1] * df[col2]
            # Ratio could be added similarly
            # df[f"{col1}_div_{col2}"] = df[col1] / (df[col2] + 1e-6)
    return df

# ===========================================
# Unsupervised Learning: PCA & Clustering
# ===========================================
def perform_pca(df, columns, n_components=2, scale=True):
    """
    Perform PCA on specified columns, return principal components DataFrame 
    plus the fitted PCA object for explained variance, etc.
    """
    data_subset = df[columns].dropna().copy()
    if scale:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_subset)
    else:
        data_scaled = data_subset.values
    
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data_scaled)
    
    pc_df = pd.DataFrame(data=principal_components, 
                         columns=[f"PC{i+1}" for i in range(n_components)],
                         index=data_subset.index)
    return pc_df, pca

@log_function_call
def cluster_data(
    df: pd.DataFrame,
    columns: List[str],
    method: str = "kmeans",
    n_clusters: int = N_CLUSTERS,
    random_state: int = RANDOM_STATE,
    scale_data: bool = True,
    return_labels_only: bool = False
) -> Union[Tuple[np.ndarray, Any], np.ndarray]:
    """
    Perform clustering analysis using K-Means or Gaussian Mixture Models.
    
    This function clusters data using unsupervised learning methods to identify
    distinct regimes or patterns in the data, commonly used for macroeconomic
    regime detection.
    
    Args:
        df: Input DataFrame containing the data to cluster
        columns: List of column names to use for clustering
        method: Clustering method ('kmeans' or 'gmm')
        n_clusters: Number of clusters to create
        random_state: Random state for reproducibility
        scale_data: Whether to scale data before clustering
        return_labels_only: Whether to return only labels or both labels and model
        
    Returns:
        If return_labels_only is False: Tuple of (cluster_labels, fitted_model)
        If return_labels_only is True: Only cluster_labels
        
    Raises:
        ValueError: If method is invalid or columns not found
    """
    with LogContext("clustering_analysis", logger=logger,
                   method=method, n_clusters=n_clusters, columns=columns):
        
        # Input validation
        if method.lower() not in ["kmeans", "gmm"]:
            raise ValueError("method must be either 'kmeans' or 'gmm'")
        
        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Columns not found in DataFrame: {missing_columns}")
        
        logger.info("Starting clustering analysis",
                   input_shape=df.shape,
                   method=method,
                   n_clusters=n_clusters,
                   columns=columns,
                   scale_data=scale_data)
        
        # Prepare data for clustering
        data_subset = df[columns].dropna()
        
        if data_subset.empty:
            raise ValueError("No valid data available for clustering after dropping NaNs")
        
        logger.info("Data prepared for clustering",
                   original_shape=df.shape,
                   clustering_shape=data_subset.shape,
                   dropped_rows=df.shape[0] - data_subset.shape[0])
        
        # Scale data if requested
        if scale_data:
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data_subset)
            logger.info("Data scaled using StandardScaler")
        else:
            data_scaled = data_subset.values
            logger.info("Using unscaled data for clustering")
        
        # Initialize and fit clustering model
        if method.lower() == "kmeans":
            model = KMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                n_init=10,  # Explicitly set to avoid FutureWarning
                max_iter=300
            )
            logger.info("K-Means model initialized")
        elif method.lower() == "gmm":
            model = GaussianMixture(
                n_components=n_clusters,
                random_state=random_state,
                n_init=10,
                max_iter=300
            )
            logger.info("Gaussian Mixture Model initialized")
        
        # Fit the model and get labels
        labels = model.fit_predict(data_scaled)
        
        # Calculate clustering quality metrics
        silhouette_avg = silhouette_score(data_scaled, labels)
        
        # Calculate cluster sizes and statistics
        cluster_sizes = np.bincount(labels)
        cluster_stats = {
            "cluster_sizes": cluster_sizes.tolist(),
            "cluster_percentages": (cluster_sizes / len(labels) * 100).tolist()
        }
        
        # Add method-specific metrics
        if method.lower() == "gmm":
            cluster_stats["bic"] = model.bic(data_scaled)
            cluster_stats["aic"] = model.aic(data_scaled)
            cluster_stats["converged"] = model.converged_
        
        logger.info("Clustering completed successfully",
                   method=method,
                   n_clusters=n_clusters,
                   silhouette_score=silhouette_avg,
                   cluster_stats=cluster_stats)
        
        # Log model information
        log_model_info(model, f"{method}_clustering", logger)
        
        # Add labels to original DataFrame
        df_result = df.copy()  # Avoid SettingWithCopyWarning
        df_result.loc[data_subset.index, "Regime_Label"] = labels
        
        logger.info("Cluster labels added to DataFrame",
                   labeled_rows=len(labels),
                   unlabeled_rows=len(df) - len(labels))
        
        if return_labels_only:
            return labels
        else:
            return labels, model

def evaluate_clustering(X, model, labels, method):
    """
    Return silhouette and BIC (if GaussianMixture) for any clustering model.
    """
    sil = silhouette_score(X, labels)
    bic = model.bic(X) if method == "gmm" else None
    return {"silhouette": sil, "bic": bic}

# ===========================================
# EDA Helpers
# ===========================================
def plot_correlations(df, columns=None, title="Correlation Matrix"):
    """
    Plot correlation matrix for specified or all numeric columns.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    corr_matrix = df[columns].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title(title)
    plt.show()

def basic_eda(df):
    """
    Print basic EDA stats: shape, head, summary stats, missing values, etc.
    """
    print("DataFrame shape:", df.shape)
    print("Head:\n", df.head())
    print("Description:\n", df.describe())
    print("Missing values:\n", df.isnull().sum())
    
# ===========================================
#  Sequence-building utilities for DL models
# ===========================================

def fit_scaler(train_df, feature_cols):
    """Fit StandardScaler on training slice (2-D)."""
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])
    return scaler

def make_sequences(df, feature_cols, target_col,
                   seq_len=12, step_ahead=1):
    """
    Convert a 2-D dataframe into
      X: (samples, seq_len, n_features)
      y: (samples,  step_ahead)
    """
    X, y, dates = [], [], []
    values = df[feature_cols + [target_col]].values
    for i in range(len(df) - seq_len - step_ahead + 1):
        X.append(values[i:i+seq_len, :-1])
        y.append(values[i+seq_len+step_ahead-1, -1])
        dates.append(df.index[i+seq_len+step_ahead-1])
    return np.array(X), np.array(y), dates
