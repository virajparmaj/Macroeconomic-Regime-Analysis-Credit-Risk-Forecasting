# config.py

# =========================================================
# File Paths (Adjust to match your folder structure)
# =========================================================
MACRO_CREDIT_DATA_PATH = "../../data/merged_macroeconomic_credit.csv"
MACRO_DATA_PATH = "../../data/macroeconomic_data_merged.csv"
CREDIT_DATA_PATH = "../../data/credit_spread_monthly_mean.csv"
MODEL_CHECKPOINT_DIR = "models/checkpoints/"
MODEL_FINAL_DIR = "models/final/"

# =========================================================
# Global Hyperparameters / Settings
# =========================================================
RANDOM_STATE = 42
TEST_SIZE = 0.2 
N_COMPONENTS = 4
N_CLUSTERS = 3 

# For Time Series (Dates cover 1996-12-01 to 2022-08-01)
TRAIN_START_DATE = "1996-12-01"
TRAIN_END_DATE   = "2018-12-31"
TEST_START_DATE  = "2019-01-01"
TEST_END_DATE    = "2022-08-01"

# =========================================================
# Observed Clusters from Unsupervised Learning
# =========================================================
# Preliminary analysis suggests 3 distinct clusters, typically:
#   - Cluster 0: Expansion
#   - Cluster 1: Mild/Moderate Phase
#   - Cluster 2: Recession