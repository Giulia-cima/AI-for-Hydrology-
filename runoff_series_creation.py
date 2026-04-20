import os
import pickle
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from helper_functions import is_match

#==============================================================================================================================
# SETTINGS
#==============================================================================================================================

folder = "/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/WL_Q_nc/"
output_folder = "/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/plots_WLQ/"
points_csv = '/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/stations_coordinates_Q_PO.csv' # CSV file with station coordinates and names THIS CAN CHANGE

files = os.listdir(folder)
files.sort()
list_nan = []
list_file = []
list_q = []
points_df = pd.read_csv(points_csv)
Basins = points_df['Basin']
Sections = points_df['Section']
stations = []

#=============================================================================================================
# LIST OF STATIONS: create a list of station names from the Basins and Sections columns of the points_df dataframe
#=============================================================================================================
for basin, section in zip(Basins, Sections):
    station_name = f"{basin}_{section}"
    station_name = station_name.replace("_", "")
    # remove spaces from station_name
    station_name = station_name.replace(" ", "")
    station_name = station_name.lower()
    # remove any accents from station_name
    station_name = station_name.replace("à", "a")
    station_name = station_name.replace("è", "e")
    station_name = station_name.replace("é", "e")
    station_name = station_name.replace("ì", "i")
    station_name = station_name.replace("ò", "o")
    station_name = station_name.replace("ù", "u")
    stations.append(station_name)


#=============================================================================================================
stations.sort()

full_time_index = pd.date_range(
    start="2004-09-01 00:00:00",
    end="2023-08-31 00:00:00",
    freq='h')
#=============================================================================================================
# MAIN LOOP: iterate over files, extract Q and H, align with full_time_index, store in list_q, plot and save figures
#=============================================================================================================


for file in files:
    if file.endswith(".nc"):
        name = file[:-21]
        # put  a space instead of _ in the name
        name = name.replace("_", "")
        # put everything in lower case
        name = name.lower()
        old_name = name

        if name not in stations:
            name = is_match(name, stations)
            if not name:
                print(f"Runoff data missing for basin: {old_name}")
                continue

        print(f"Processing file: {file} with name: {name}")
        ds = xr.open_dataset(os.path.join(folder, file))

        Q = ds["Q"].where(ds["Q"] >= 0).where(ds["Q"] <= 10000)
        H = ds["H"].where(ds["H"] >= 0).where(ds["H"] <= 10000)

        time = full_time_index
        Q_values = Q.values
        H_values = H.values

        print(f"Length of time: {len(time)}")
        print(f"Length of Q: {len(Q_values)}")
        print(f"Length of H: {len(H_values)}")

        mean_Q = np.nanmean(Q_values)
        std_Q = np.nanstd(Q_values)

        print(f"File: {name}")
        print(f"Mean of Q: {mean_Q:.2f}")
        print(f"Standard deviation of Q: {std_Q:.2f}")

        nan_points = np.isnan(Q_values).sum()
        total_points = Q_values.size
        percentage_nan = (nan_points / total_points) * 100

        list_nan.append((file, percentage_nan, total_points - nan_points, mean_Q, std_Q))

        list_file.append(file)

        if name == '' or name == ' ':
            name = file[:-20].replace("_", " ")

        print(f"Length of Q: {len(Q_values)}")

        list_q.append({
            "date": time,
            "basin": name,
            "discharge": Q_values
        })

        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        axs[0].plot(time, Q, label='Q', color='blue')
        axs[0].set_ylabel('Q (m^3/s)')
        axs[0].legend()
        axs[1].plot(time, H, label='H', color='orange')
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('H (m)')
        axs[1].legend()
        # Add percentage of NaN values in Q as text in the top right corner of the first subplot
        axs[0].text(0.95, 0.95, f'NaN in Q: {percentage_nan:.2f}%', transform=axs[0].transAxes,
                    fontsize=10, verticalalignment='top', horizontalalignment='right')
        plt.tight_layout()
        output_file = os.path.join(output_folder, file.replace(".nc", ".png"))
        # SET PLOT TITLE AS THE NAME OF THE FILE WITHOUT THE LAST 18 CHARACTERS
        plt.suptitle(name, fontsize=12)
        plt.savefig(output_file)
        plt.close()

print(f"Number of Q time series with less than 25% of NaN values: {len(list_q)}")

#==============================================================================================
# SAVE RESULTS: save list_file and list_q as pkl files, save list_nan as a csv file in the output folder
#==============================================================================================

with open(os.path.join(output_folder, "list_file.pkl"), "wb") as f:
    pickle.dump(list_file, f)

output_folder_pkl = "/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/"

# save list_q as a pkl
with open(os.path.join(output_folder_pkl, "list_q.pkl"), "wb") as f:
    pickle.dump(list_q, f)

df_nan = pd.DataFrame(list_nan, columns=['file', 'percentage_nan', 'total_points_not_nan', 'mean_Q', 'std_Q'])
df_nan = df_nan.sort_values(by='percentage_nan', ascending=True)
df_nan.to_csv(os.path.join(output_folder, "nan_percentage.csv"), index=False)




