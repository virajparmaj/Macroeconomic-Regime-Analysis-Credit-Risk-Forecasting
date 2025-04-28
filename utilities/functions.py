# functions.py

import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score

from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# ===========================================
# Data Cleaning & Utilities
# ===========================================
def remove_outliers_zscore(series, threshold=3):
    """
    Remove outliers from a pandas Series based on a Z-score threshold.
    """
    mask = abs(stats.zscore(series.dropna())) < threshold
    return series[mask]

def winsorize_series(series, limits=(0.01, 0.01)):
    """
    Winsorize a series to limit extreme values.
    """
    from scipy.stats.mstats import winsorize
    return winsorize(series, limits=limits)

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

def cluster_data(df, columns, method="kmeans", n_clusters=3, random_state=42):
    """
    Cluster the data using K-Means or GaussianMixture on specified columns.
    Returns the cluster labels and the model object.
    """
    data_subset = df[columns].dropna()
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_subset)

    if method.lower() == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=random_state)
    elif method.lower() == "gmm":
        model = GaussianMixture(n_components=n_clusters, random_state=random_state)
    else:
        raise ValueError("method must be either 'kmeans' or 'gmm'")

    labels = model.fit_predict(data_scaled)
    score = silhouette_score(data_scaled, labels)
    print(f"Clustering method: {method}, Silhouette Score: {score:.3f}")
    
    # Merge labels back to the main df (only for rows used in clustering)
    df.loc[data_subset.index, "Regime_Label"] = labels
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
from sklearn.preprocessing import StandardScaler

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
