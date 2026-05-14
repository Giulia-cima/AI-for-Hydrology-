import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from helper_functions import is_match
from hy2dl.datasetzoo.basindataset import BasinDataset
import torch
from torch.utils.data import DataLoader
from hy2dl.modelzoo.mflstm import MFLSTM
import torch.nn as nn

def main():
    # ============================================================
    # LOAD DATA
    # ============================================================
    time_start = "2010-09-01 00:00:00"
    time_end = "2020-07-01 00:00:00"

    precipitation = pd.read_pickle(
        "/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/output/corected_precipitation/mean_h_precipitation_20040901_20200831_cumulative_by_basin.pkl"
    ).loc[time_start:time_end]

    area_data = pd.read_csv(
        "/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/output/basin_stats.csv"
    )

    runoff = pd.read_pickle(
        "/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/input/list_q.pkl"
    )

    # ============================================================
    # BUILD DATAFRAME
    # ============================================================
    basin_list = precipitation.columns
    basin_matrices = {}

    for basin in basin_list:
        basin_norm = basin.lower().replace(" ", "_").replace("à", "a").replace("è", "e").replace("é", "e")
        if basin_norm not in [r["basin"].replace(" ", "_").lower() for r in runoff]:
            basin_norm = is_match(basin_norm, runoff)
            if not basin_norm:
                print(f"Runoff data missing for basin: {basin}")
                continue
        print(f"Processing basin: {basin}")

        try:
            idx = area_data[area_data["basin"] == basin.lower()].index[0]
        except:
            print(f"{basin}")

        r_values = next(r["discharge"] for r in runoff if r["basin"].replace(" ", "_").lower() == basin_norm)
        r_time = next(r["date"] for r in runoff if r["basin"].replace(" ", "_").lower() == basin_norm)
        runoff_series = pd.Series(r_values, index=pd.to_datetime(r_time))
        runoff_aligned = runoff_series.reindex(precipitation.index)

        area_val = area_data.loc[idx, "area_km2"]
        mean_H = area_data.loc[idx, "mean_H"]

        df = pd.DataFrame({
            "precipitation": precipitation[basin].values,
            "Corr_time": 0.25 + 0.27 * np.sqrt(area_val),
            "mean_H": mean_H,
            "runoff": runoff_aligned.values
        }, index=precipitation.index)

        # Cleaning
        df.loc[df["precipitation"] < 0, "precipitation"] = 0
        df.loc[df["runoff"] < 0, "runoff"] = 0
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.loc[df["runoff"] > 10000, "runoff"] = np.nan
        df.loc[df["precipitation"] > 10000, "precipitation"] = np.nan
        zero_precip = df["precipitation"] == 0
        zero_groups = (zero_precip != zero_precip.shift()).cumsum()
        zero_durations = zero_precip.groupby(zero_groups).transform("sum")
        df.loc[(zero_precip) & (zero_durations >= 720), "precipitation"] = np.nan

        # Features
        df["precip_log"] = np.log1p(df["precipitation"])
        df["log_area"] = np.log1p(area_val)
        df["log_mean_H"] = np.log1p(mean_H)
        df["runoff_log"] = np.log1p(df["runoff"])
        df["log_corr_time"] = np.log1p(df["Corr_time"])
        df["basin"] = basin
        df["year"] = df.index.year
        df.dropna(inplace=True)
        basin_matrices[basin] = df

    df_all = pd.concat(basin_matrices.values())
    df_all.sort_index(inplace=True)

    # ============================================================
    # TRAIN / VAL / TEST SPLIT
    # ============================================================
    train_df = df_all[df_all["year"].between(2010, 2016)]
    val_df = df_all[df_all["year"].between(2017, 2018)]
    test_df = df_all[df_all["year"].between(2019, 2020)]

    seq_len = 24
    train_ds = BasinDataset(train_df, seq_len=seq_len)
    val_ds = BasinDataset(val_df, seq_len=seq_len)
    test_ds = BasinDataset(test_df, seq_len=seq_len)

    batch_size = 64
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # ============================================================
    # MODEL CONFIG
    # ============================================================
    model_config = {
        "input_size_lstm": 1,  # daily dynamic feature: precip_log
        "hidden_size": 64,
        "no_of_layers": 2,
        "predict_last_n": 1,
        "dropout_rate": 0.2,
        "custom_freq_processing": {"daily": None},
        "dynamic_embeddings": False,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MFLSTM(model_config).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # ============================================================
    # TRAINING LOOP
    # ============================================================
    n_epochs = 20
    train_history = []
    val_history = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            x_dyn = {k: v.to(device) for k, v in batch.items() if k.startswith("x_d")}
            x_stat = batch["x_s"].to(device)
            y = batch["y"].to(device)

            sample = x_dyn.copy()
            sample["x_s"] = x_stat

            y_hat = model(sample)["y_hat"].squeeze(-1)[:, -1]
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * y.size(0)

        train_loss /= len(train_loader.dataset)
        train_history.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x_dyn = {k: v.to(device) for k, v in batch.items() if k.startswith("x_d")}
                x_stat = batch["x_s"].to(device)
                y = batch["y"].to(device)

                sample = x_dyn.copy()
                sample["x_s"] = x_stat

                y_hat = model(sample)["y_hat"].squeeze(-1)[:, -1]
                loss = criterion(y_hat, y)
                val_loss += loss.item() * y.size(0)
        val_loss /= len(val_loader.dataset)
        val_history.append(val_loss)

        print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")

    # ============================================================
    # TEST EVALUATION
    # ============================================================
    model.eval()
    test_loss = 0
    y_true_all = []
    y_pred_all = []
    with torch.no_grad():
        for batch in test_loader:
            x_dyn = {k: v.to(device) for k, v in batch.items() if k.startswith("x_d")}
            x_stat = batch["x_s"].to(device)
            y = batch["y"].to(device)

            sample = x_dyn.copy()
            sample["x_s"] = x_stat

            y_hat = model(sample)["y_hat"].squeeze(-1)[:, -1]
            test_loss += criterion(y_hat, y).item() * y.size(0)
            y_true_all.append(y.cpu().numpy())
            y_pred_all.append(y_hat.cpu().numpy())

        # Complete test evaluation
        test_loss /= len(test_loader.dataset)
        print(f"Test MSE: {test_loss:.6f}")

        # Flatten all predictions and targets
        y_true_all = np.concatenate(y_true_all, axis=0)
        y_pred_all = np.concatenate(y_pred_all, axis=0)

        # ============================================================
        # PLOT TRAINING HISTORY
        # ============================================================
        plt.figure(figsize=(8, 5))
        plt.plot(train_history, label='Train Loss', marker='o')
        plt.plot(val_history, label='Val Loss', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/output/MFLSTM/training_history.png")
        plt.close()

        # ============================================================
        # SCATTER PLOT OBSERVED VS PREDICTED
        # ============================================================
        predictions_exp = np.expm1(y_pred_all)
        targets_exp = np.expm1(y_true_all)

        plt.figure(figsize=(8, 8))
        plt.scatter(targets_exp, predictions_exp, alpha=0.3)
        min_val = min(targets_exp.min(), predictions_exp.min())
        max_val = max(targets_exp.max(), predictions_exp.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        plt.xlabel("Observed runoff")
        plt.ylabel("Predicted runoff")
        plt.title("Predicted vs Observed")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/output/MFLSTM/evaluation.png")
        plt.close()

        # ============================================================
        # TIME SERIES PLOT
        # ============================================================
        plt.figure(figsize=(15, 5))
        plt.plot(targets_exp, label="Observed", color="blue", alpha=0.7)
        plt.plot(predictions_exp, label="Predicted", color="orange", alpha=0.7)
        plt.xlabel("Time index")
        plt.ylabel("Runoff (mm)")
        plt.title("Predicted vs Observed Runoff Over Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/output/MFLSTM/time_series.png")
        plt.close()

    if __name__ == "__main__":
        main()