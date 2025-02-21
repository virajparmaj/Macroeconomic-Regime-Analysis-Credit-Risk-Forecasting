# Macroeconomic Regime Analysis & Credit Risk Forecasting

This project combines unsupervised learning and time series forecasting to detect economic regimes using macroeconomic indicators and predict credit risk (e.g., credit spreads). In simple terms, it tells us when the economy is expanding or contracting and forecasts how risky lending might become.

## Overview

- **What It Does:**  
  - Uses six macroeconomic factors (CPI, Fed Funds, Industrial Production, GDP, Unemployment, Consumer Sentiment) to detect hidden economic regimes.
  - Merges these monthly datasets (from January 1996 to August 2022) with credit risk data (e.g., BAMLH0A0HYM2 credit spreads) to forecast credit risk trends.
  
- **Industry Relevance:**  
  - Helps banks and investors adjust risk management strategies.
  - Provides early warnings for increasing credit risk during economic downturns.

## Data Preparation

1. **Data Collection:**  
   - Macro data from FRED for six key indicators.
   - Credit risk data (credit spread series) from FRED.
2. **Data Merging & Normalization:**  
   - All datasets are resampled to monthly frequency (standardized to the end-of-month).
   - The merged dataset covers all six macroeconomic factors plus the credit spread.
3. **Data Cleaning & Preprocessing:**  
   - Missing values are filled using forward fill or linear interpolation.
   - Additional feature engineering includes calculating growth rates, rolling averages, and lag features.

## Modeling Approach

- **Unsupervised Learning:**  
  - Techniques like PCA, K-Means, or Hidden Markov Models (HMM) are used to detect macroeconomic regimes.
- **Time Series Forecasting:**  
  - Models such as ARIMA/SARIMA and regression methods forecast credit risk.
- **Deep Learning:**  
  - LSTM networks are employed to capture non-linear patterns and long-term dependencies.
- **Workflow:**  
  - Data is split chronologically for training and testing.
  - Forecast accuracy is evaluated using metrics like RMSE, MAE, and MAPE.

## Environment Setup with Poetry

This project uses [Poetry](https://python-poetry.org/) for dependency management and virtual environment handling. To set up the environment, run the following commands:

1. **Initialize Poetry (if not already initialized):**

   ```bash
   poetry init
