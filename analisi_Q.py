import os
import pickle
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

folder = "/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/WL_Q_nc/"
files = os.listdir(folder)
# sort files by name
files.sort()
list_nan = []
list_file = []
list_q = []
output_folder = "/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/plots_WLQ/"



for file in files:
    if file.endswith(".nc"):
        name = file.split(".")[0]
        # put  a space instead of _ in the name
        name = name.replace("_", " ")
        # REMOVE THE LAST 18 CHARACTERS OF THE NAME
        name = name[:-18]

        # Here you can add the code to read and process the .nc file
        # For example, you can use xarray to read the netCDF file:
        # import xarray as xr
        ds = xr.open_dataset(os.path.join(folder, file))
        Q = ds['Q']  # Assuming 'Q' is the variable name for river discharge
        H = ds['H']
        time = ds['time'].values
        # SET VALUES BELOW 0 TO np.nan
        Q = Q.where(Q >= 0)
        H = H.where(H >= 0)
        # set values of Q that are above 10000 to np.nan
        Q = Q.where(Q <= 10000)
        H = H.where(H <= 10000)
        # compute mean and std of Q , avoiding nan values
        mean_Q = Q.mean().item()
        std_Q = Q.std().item()
        print(f"Mean of Q: {mean_Q:.2f}")
        print(f"Standard deviation of Q: {std_Q:.2f}")
        # compute the number of data points in Q that is nan over the total number of data points in Q and print that percentage
        total_points = Q.size
        nan_points = Q.isnull().sum().item()
        total_points_not_nan = total_points - nan_points
        percentage_nan = (nan_points / total_points) * 100
        print (f"Percentage of NaN values in Q: {percentage_nan:.2f}%")
        print(f"Total points in Q: {total_points}")


        # store in a list the name of the file and the percentage of
        list_nan.append((file, percentage_nan, total_points_not_nan, mean_Q, std_Q))


        if percentage_nan <=25:
            list_file.append(file)
            # append Q to list_q and the name of the file to list_file
            list_q.append({
                "date": time,
                "basin": name,
                "discharge": Q
            })


        # make a plot of Q vs time AND H vs time in a single figure and save that plot in the output folder with the same name as the input file but with .png extension
        # add in the plot the percentage of NaN values in Q as a text in the top right corner of the plot.
        #MAKE TWO SUBPLOTS, ONE FOR Q AND ONE FOR H
        fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
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

# save list_file as a pkl

with open(os.path.join(output_folder, "list_file.pkl"), "wb") as f:
    pickle.dump(list_file, f)

output_folder_pkl = "/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/"

# save list_q as a pkl
with open(os.path.join(output_folder_pkl, "list_q.pkl"), "wb") as f:
    pickle.dump(list_q, f)




# convert list into a dataframe and save it as a csv file in the output folder with the name "nan_percentage.csv"
df_nan = pd.DataFrame(list_nan, columns=['file', 'percentage_nan', 'total_points_not_nan', 'mean_Q', 'std_Q'])
# sort df by percentage_nan in ascending order
df_nan = df_nan.sort_values(by='percentage_nan', ascending=True)
df_nan.to_csv(os.path.join(output_folder, "nan_percentage.csv"), index=False)




