import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

PAIR = "EURUSD"                
DATA_DIR = "data"              
START_DATE = "2015-01-01"     
END_DATE = None                
TEST_SPLIT_DATE = "2023-01-01" 
WINDOW = 20                    
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3
MODEL_SAVE_PATH = f"outputs/{PAIR}_price_cnn.h5"
SEED = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)

csv_path = os.path.join(DATA_DIR, f"{PAIR}.csv")
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Data file {csv_path} not found.")

df = pd.read_csv(csv_path, parse_dates=['Date'])
df = df.set_index('Date').sort_index()

if START_DATE:
    df = df.loc[df.index >= pd.to_datetime(START_DATE)]
if END_DATE:
    df = df.loc[df.index <= pd.to_datetime(END_DATE)]

if "Price" not in df.columns:
    raise ValueError("CSV file must contain 'Price' column. Your columns: " + ", ".join(df.columns))

df["return_next"] = df["Price"].pct_change().shift(-1)

df = df.dropna(subset=["Price", "return_next"]).copy()

df["log_price"] = np.log(df["Price"])

# next, we create rolling window features
# then, train and test splits
# and finally, scaling and standardisation

# then, build and train the 1D CNN model
# then, evaluate and visualize results