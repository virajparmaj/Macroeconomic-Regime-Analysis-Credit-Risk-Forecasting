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

# --------------  NEW LINES --------------
SEQ_LEN          = 12      # months fed to LSTM
FORECAST_STEPS   = 1       # t + 1 month horizon
BATCH_SIZE       = 32
EPOCHS           = 150
DL_MODEL_DIR     = "models/deep/"
# ---------------------------------------
