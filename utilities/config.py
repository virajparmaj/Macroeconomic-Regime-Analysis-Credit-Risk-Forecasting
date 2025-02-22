# config.py

# ===========================================
# File Paths
# ===========================================
RAW_DATA_PATH = "data/raw/macro_data.csv"
CLEAN_DATA_PATH = "data/processed/clean_macro_data.csv"
MODEL_CHECKPOINT_DIR = "models/checkpoints/"
MODEL_FINAL_DIR = "models/final/"

# ===========================================
# Global Hyperparameters / Settings
# ===========================================
RANDOM_STATE = 42
TEST_SIZE = 0.2

# For Clustering
N_CLUSTERS = 3

# For PCA
N_COMPONENTS = 2

# For Time Series
TRAIN_START_DATE = "2000-01-01"
TRAIN_END_DATE = "2018-12-31"
TEST_START_DATE = "2019-01-01"
TEST_END_DATE = "2022-12-31"