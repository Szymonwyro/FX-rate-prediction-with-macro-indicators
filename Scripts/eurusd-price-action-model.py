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
WINDOW = 40                    
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-2
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

df["return"] = np.log(df["Price"])

# ROLLING WINDOWS -----------------------------------------------------------------

def create_windows(series, window):
    X = []
    for i in range(len(series) - window + 1):
        X.append(series[i : i + window])
    return np.array(X)

logp = df["return"].values
X_all = create_windows(logp, WINDOW)

y_all = df["return_next"].values[WINDOW - 1 :]

index_all = df.index[WINDOW - 1 :]

# check shapes and indexing alignment
print("X_all shape (samples, window):", X_all.shape)
print("y_all shape (samples,):", y_all.shape)
assert X_all.shape[0] == y_all.shape[0], "Mismatch between number of windows and number of targets"

# 4) TRAIN/TEST SPLIT --------------------------------------------------------------
if TEST_SPLIT_DATE:
    split_idx = np.searchsorted(index_all, pd.to_datetime(TEST_SPLIT_DATE))
else:
    # fallback: last 20% as test
    split_idx = int(0.8 * len(X_all))

X_train = X_all[:split_idx]
X_test  = X_all[split_idx:]
y_train = y_all[:split_idx]
y_test  = y_all[split_idx:]
index_test = index_all[split_idx:]

print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

# 5) DATA SCALING ----------------------------------------------------------------

scaler = StandardScaler()
X_train_flat = X_train.reshape(-1, 1)  # shape (n_train_windows * window, 1)
scaler.fit(X_train_flat)               # fit only on training data

def scale_windows(X, scaler):
    """Apply fitted scaler to a 2D array of windows."""
    Xf = X.reshape(-1, 1)
    Xf = scaler.transform(Xf)
    return Xf.reshape(X.shape)

X_train_scaled = scale_windows(X_train, scaler)
X_test_scaled  = scale_windows(X_test, scaler)

# Conv1D expects shape (samples, timesteps, channels)
X_train_scaled = X_train_scaled.reshape((-1, WINDOW, 1))
X_test_scaled  = X_test_scaled.reshape((-1, WINDOW, 1))

# 6) MODEL DEFINITION -------------------------------------------------------------

def build_cnn(window, channels=1, lr=1e-3):
    inp = layers.Input(shape=(window, channels))
    x = layers.Conv1D(filters=32, kernel_size=3, activation="relu", padding="causal")(inp)
    x = layers.Conv1D(filters=64, kernel_size=3, activation="relu", padding="causal")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(filters=128, kernel_size=3, activation="relu", padding="causal")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    # Single linear output for regression (next-day return)
    out = layers.Dense(1, activation="linear")(x)
    model = models.Model(inputs=inp, outputs=out)
    opt = optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss="mse", metrics=["mae"])
    return model

# TRAINING -------------------------------------------------------------------------
es = callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
model = build_cnn(WINDOW, channels=1, lr=LEARNING_RATE)
model.summary()

history = model.fit(
    X_train_scaled,
    y_train,
    validation_split=0.15,  # uses last 15% of X_train as validation (shuffle=False below preserves order)
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[es],
    shuffle=False  # do not shuffle time series data
)

# 8) EVALUATION ON TEST SET -------------------------------------------------------
# Predict on the held-out, strictly later test set.
y_pred = model.predict(X_test_scaled).flatten()

# Mean Squared Error (regression metric)
mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE: {mse:.6e}")

# Directional accuracy: percentage of days where sign(pred) == sign(actual)
# - Small returns near zero are noisy; we can ignore them with a threshold if desired.
sign_match = (np.sign(y_pred) == np.sign(y_test)).astype(int)
dir_acc = np.mean(sign_match)
print(f"Directional accuracy on test set: {dir_acc * 100:.2f}%")

# Directional accuracy ignoring tiny returns (threshold to avoid classifying noise)
THRESH = 1e-5
valid_idx = np.abs(y_test) > THRESH
if valid_idx.sum() > 0:
    dir_acc_nz = np.mean((np.sign(y_pred[valid_idx]) == np.sign(y_test[valid_idx])).astype(int))
    print(f"Directional accuracy (|actual| > {THRESH}): {dir_acc_nz * 100:.2f}%")

# 9) SAVE MODEL ---------------------------------------------------------------------
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
model.save(MODEL_SAVE_PATH)
print("Model saved to:", MODEL_SAVE_PATH)

# 10) PLOTS ---------------------------------------------------------------------------
plt.figure(figsize=(10,4))
plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="val loss")
plt.title("Training / Validation Loss (MSE)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,5))
nplot = min(200, len(y_test))
plt.plot(index_test[:nplot], y_test[:nplot], label="Actual", linewidth=1)
plt.plot(index_test[:nplot], y_pred[:nplot], label="Predicted", linewidth=1)
plt.title(f"{PAIR} Next-Day Return: Actual vs Predicted (first {nplot} test samples)")
plt.legend()
plt.tight_layout()
plt.show()

# Scatter predicted vs actual
plt.figure(figsize=(5,5))
plt.scatter(y_test, y_pred, alpha=0.4, s=8)
plt.xlabel("Actual return")
plt.ylabel("Predicted return")
plt.title("Predicted vs Actual scatter")
lims = np.percentile(np.concatenate([y_test, y_pred]), [1, 99])
plt.xlim(lims[0], lims[1])
plt.ylim(lims[0], lims[1])
plt.plot([lims[0], lims[1]], [lims[0], lims[1]], color="red", lw=1)
plt.tight_layout()
plt.show()

# Quick tabular sample
sample_df = pd.DataFrame({"actual": y_test, "predicted": y_pred}, index=index_test)
print(sample_df.head(10))