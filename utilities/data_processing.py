# data_preprocessing.py

import pandas as pd
import numpy as np
import os

from config import RAW_DATA_PATH
from functions import (
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

def load_raw_data(filepath=RAW_DATA_PATH):
    """
    Load raw macroeconomic + credit risk data from CSV or other sources.
    """
    print(f"Loading raw data from {filepath} ...")
    df = pd.read_csv(filepath, parse_dates=['Date'], index_col='Date')
    print("Data loaded successfully. Data shape:", df.shape)
    return df

def preprocess_data(df, outlier_cols=None, winsorize_cols=None, fillna_method='ffill'):
    """
    General data cleaning pipeline: fill missing, handle outliers, etc.
    """
    # Sort by date index
    df.sort_index(inplace=True)

    # Forward-fill or back-fill missing values
    df.fillna(method=fillna_method, inplace=True)
    
    # Optionally remove or winsorize outliers
    if outlier_cols:
        for col in outlier_cols:
            df[col] = remove_outliers_zscore(df[col], threshold=3)
    
    if winsorize_cols:
        for col in winsorize_cols:
            df[col] = winsorize_series(df[col], limits=(0.01,0.01))
    
    # Replace infinities, if any
    df = clip_infinities(df, fill_value=0)
    
    return df

def create_features(df):
    """
    Create new features such as growth rates, rolling stats, or lags 
    that are relevant to macroeconomic + credit risk modeling.
    """
    # Example: columns you'd want lagged
    df = add_lag_features(df, columns=['GDP', 'CPI', 'Credit_Spread'], lags=[1,2])
    
    # Rolling means for certain economic indicators
    df = add_rolling_features(df, columns=['Unemployment_Rate', 'Credit_Spread'], window=3)
    
    # Growth rates for certain macro columns
    df = add_growth_rates(df, columns=['GDP', 'CPI', 'Industrial_Production'])
    
    # Interaction terms between certain columns
    df = add_interaction_terms(df, col_pairs=[('GDP_growth','CPI_growth'), ('Credit_Spread','FEDFUNDS')])
    
    # You can also create a 'Target' variable, e.g. next-month credit spread
    df['Target'] = df['Credit_Spread'].shift(-1)  # 1-month ahead forecast
    return df

def run_eda(df):
    """
    Run basic EDA and correlation checks.
    """
    basic_eda(df)
    # Possibly restrict columns to numeric for correlation matrix
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    plot_correlations(df, columns=numeric_cols, title="Correlation Matrix")

def pipeline_data_preparation(filepath=RAW_DATA_PATH):
    """
    Complete pipeline: load, clean, feature engineering, EDA, finalize.
    """
    # 1) Load raw data
    df = load_raw_data(filepath)

    # 2) Preprocess (outlier removal, winsorization, etc.)
    df = preprocess_data(df, 
                         outlier_cols=['Credit_Spread'], 
                         winsorize_cols=['GDP'])

    # 3) Feature engineering
    df = create_features(df)

    # 4) EDA
    run_eda(df)

    # 5) Drop any new NaNs from shifting or rolling
    df.dropna(inplace=True)

    print("Final dataset shape after pipeline:", df.shape)
    return df