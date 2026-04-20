import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Input
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from keras.callbacks import EarlyStopping
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from rapidfuzz import fuzz


#=============================================================
# Basin name matching
#=============================================================
# ============================================================
# Basin name matching
# ============================================================
def is_match(basin, runoff_list, threshold=75):

    for r in runoff_list:
        candidate = (r["basin"])

        # 🔑 STEP 1: match diretto robusto (substring bidirezionale)
        if basin in candidate or candidate in basin:
            return candidate

        # 🔑 STEP 2: fuzzy matching
        score = fuzz.token_set_ratio(basin, candidate)
        if score >= threshold:
            return candidate

    return False

# ============================================================
# Plot distribution
# ============================================================
def plot_distribution(stats_df, variables, filename, power=0.2):
    fig, axes = plt.subplots(1, len(variables), figsize=(5 * len(variables), 5))
    if len(variables) == 1:
        axes = [axes]
    for ax, var in zip(axes, variables):
        values = stats_df[var] ** power
        sns.histplot(values, bins=20, kde=True, ax=ax)
        ax.set_title(f'Distribution of {var}')
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close()

# ============================================================
# Custom loss
# ============================================================
def custom_loss(y_true, y_pred):
    # RMSE
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    rmse = tf.sqrt(mse)

    # Entropy
    #residuals = y_true - y_pred
    #mean_res = tf.reduce_mean(residuals)
    #var_res = tf.reduce_mean(tf.square(residuals - mean_res)) + 1e-6
    #entropy = 0.5 * tf.math.log(2.0 * np.pi * np.e * var_res)

    # Combine
    #alpha = 0.9   # peso errore medio
    #beta  = 0.1   # peso distribuzione errori

    return rmse

# ============================================================
# Model
# ============================================================
def create_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(100, return_sequences=True),
        LSTM(50),
        Dense(1)
    ])

    # insert early stopping callback

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=custom_loss,
        metrics=['mae']
    )
    return model


# ============================================================
# Create sequences
# ============================================================

def create_sequences(df, seq_length=72):
    X, y = [], []
    features = df.columns.drop("runoff")
    data = df[features].values
    target = df["runoff"].values
    for i in range(len(df) - seq_length):
        seq_X = data[i:i+seq_length]
        seq_y = target[i+seq_length]
        if np.isnan(seq_X).any() or np.isnan(seq_y):
            continue
        X.append(seq_X)
        y.append(seq_y)
    return np.array(X), np.array(y)



# ============================================================
# Compute stats for imbalance score
# ============================================================

def compute_stats(basins, basin_data):
    if len(basins) == 0:
        return 0, 0

    vals = np.concatenate([basin_data[b] for b in basins if len(basin_data[b]) > 0])

    if len(vals) == 0:
        return 0, 0

    return np.mean(vals), np.std(vals)

# ============================================================
# Imbalance score
# ============================================================
def imbalance_score(splits):
    stats = [compute_stats(splits[k]) for k in splits]
    means = [s[0] for s in stats]
    stds = [s[1] for s in stats]

    return np.std(means) + np.std(stds)



# ============================================================
# Imbalance score based on month distribution
# ============================================================
def imbalance_score_month(split_dict, df):

    global_mean = df["runoff"].mean()
    global_std = df["runoff"].std()

    score = 0

    for split in split_dict:

        d = df[df["month"].isin(split_dict[split])]["runoff"]

        if len(d) == 0:
            return np.inf

        score += abs(d.mean() - global_mean) / global_std
        score += abs(d.std() - global_std) / global_std

    return score

# ============================================================
# Print stats
# ============================================================

def stats(name, d):
    print(f"\n{name}")
    print("n samples:", len(d))
    print("mean runoff:", d["runoff"].mean())
    print("std runoff:", d["runoff"].std())
    print("min/max:", d["runoff"].min(), d["runoff"].max())


#============================================================
# Train cluster model
#============================================================

def train_cluster(cluster_id, basins, basin_matrices):

    print(f"\n==============================")
    print(f"Cluster {cluster_id} | basins: {len(basins)}")
    print(f"==============================")

    # ----------------------------
    # Combine basin data
    # ----------------------------
    data = pd.concat([basin_matrices[b] for b in basins])

    features = ["precipitation", "Corr_time", "mean_H", "runoff"]

    # ----------------------------
    # Train/Val/Test split (time-aware simple split)
    # ----------------------------
    n = len(data)

    train_end = int(0.7 * n)
    val_end = int(0.85 * n)

    train_df = data.iloc[:train_end].copy()
    val_df = data.iloc[train_end:val_end].copy()
    test_df = data.iloc[val_end:].copy()

    # ----------------------------
    # Scaling (fit ONLY on train)
    # ----------------------------
    scaler = StandardScaler()

    train_df[features]= scaler.fit_transform(train_df[features])
    val_df[features]= scaler.transform(val_df[features])
    test_df[features]= scaler.transform(test_df[features])

    # ----------------------------
    # Sequences
    # ----------------------------
    X_train, y_train = create_sequences(train_df)
    X_val, y_val= create_sequences(val_df)
    X_test, y_test= create_sequences(test_df)

    # ----------------------------
    # Model
    # ----------------------------
    model = create_model((X_train.shape[1], X_train.shape[2]))

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=30,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    # ----------------------------
    # Save model
    # ----------------------------
    model_path = f"/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/output/lstm_cluster_{cluster_id}.h5"
    model.save(model_path)

    # ----------------------------
    # Evaluate
    # ----------------------------
    y_pred = model.predict(X_test)

    r_mean = scaler.mean_[3]
    r_std = scaler.scale_[3]

    y_pred_inv = y_pred * r_std + r_mean
    y_test_inv = y_test * r_std + r_mean

    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    bias = np.mean(y_pred_inv - y_test_inv)

    print(f"Cluster {cluster_id} RMSE: {rmse:.3f} | Bias: {bias:.3f}")

    return model, history, scaler