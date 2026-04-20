import os
import pandas
import numpy as np
from scipy.io import loadmat
import xarray as xr
import pickle
import matplotlib.pyplot as plt



def analisi_coordinate():
    file_path= "/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/Q/"
    dataframe = []
    i = 0

    for file in os.listdir(file_path):
        if file.endswith(".mat"):

            mat_data = loadmat(file_path + file)
            bacin_name = mat_data['a1sNomeSezione'][0].item() if 'a1sNomeSezione' in mat_data else None
            section_name = mat_data['a1sNomeBacino'][0].item()if 'a1sNomeBacino' in mat_data else None
            latitude = float(mat_data['a1dLat'][0]) if 'a1dLat' in mat_data else None
            longitude = float(mat_data['a1dLon'][0]) if 'a1dLon' in mat_data else None
            station_ref = "STATION_" + str(i)
            dataframe.append([bacin_name, section_name, latitude, longitude, station_ref])
            i += 1

    dataframe = pandas.DataFrame(dataframe, columns=['Section','Basin','Latitude', 'Longitude', 'Station_Ref'])
    dataframe["Duplicate"] = dataframe.duplicated(subset=["Latitude", "Longitude"], keep=False)
    duplicates = dataframe[dataframe["Duplicate"]]
    dup_counts = duplicates.groupby(["Latitude", "Longitude"]).size().reset_index(name="Count")
    print(f"Number of duplicated stations: {len(duplicates)}")
    print(dup_counts)

    duplicates.to_csv("/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/output/duplicate_stations.csv", index=False)
    dataframe.to_csv("/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/output/stations_coordinates_Q.csv", index=False)

    return None

#===========================================================================================
#===========================================================================================

def correct_coordinates():
    # Load data
    df_portate = pandas.read_csv("/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/output/stations_coordinates_Q.csv")
    df_portate_po = pandas.read_csv("/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/stations_coordinates_Q_PO.csv")

    # merge sect and basin
    df_portate["Section_Basin"] = df_portate["Section"] + "_" + df_portate["Basin"]
    df_portate_po["Section_Basin"] = df_portate_po["Section"] + "_" + df_portate_po["Basin"]

    # Define key columns
    key_col = "Section_Basin"
    lat_col = "Latitude"
    lon_col = "Longitude"

    duplicates_mask = df_portate.duplicated(subset=[lat_col, lon_col], keep=False)
    n_duplicates = duplicates_mask.sum()
    print(f"Number of duplicated coordinate entries in original dataframe: {n_duplicates}")
    df_duplicates = df_portate[duplicates_mask]
    print(df_duplicates)
    lookup = df_portate_po.set_index(key_col)[[lat_col, lon_col]]
    df_corrected = df_portate.copy()
    df_corrected.update(lookup)
    duplicates_mask = df_corrected.duplicated(subset=[lat_col, lon_col], keep=False)
    n_duplicates = duplicates_mask.sum()
    print(f"Number of duplicated coordinate entries: {n_duplicates}")
    df_duplicates = df_corrected[duplicates_mask]
    print(df_duplicates)
    df_duplicates.to_csv("/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/output/duplicate_stations_corrected.csv", index=False)
    df_corrected["is_from_po"] = df_corrected[key_col].isin(df_portate_po[key_col])
    df_corrected["Duplicate"] = df_corrected.apply(lambda row: False if row["is_from_po"] else row["Duplicate"], axis=1)
    df_corrected["Section_Basin"] = df_corrected["Basin"].str.lower() + df_corrected["Section"].str.lower()
    df_corrected["Section_Basin"] = df_corrected["Section_Basin"].str.replace("_", "")
    df_corrected["Section_Basin"] = df_corrected["Section_Basin"].str.replace(" ", "")

    output_path = "/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/output/coordinate_corrette/stations_coordinates_Q_corrected.csv"
    df_corrected.to_csv(output_path, index=False)
    print(f"Saved corrected file to: {output_path}")
    return None

#===========================================================================================
#===========================================================================================
def pad_or_trim(da, full_time_index):
    n_full = len(full_time_index)
    n_data = da.shape[0]

    # create full NaN array
    padded = xr.DataArray(
        np.full(n_full, np.nan),
        coords=[full_time_index],
        dims=["time"]
    )

    # copy existing values into padded array (truncate if needed)
    n_copy = min(n_full, n_data)
    padded[:n_copy] = da.values[:n_copy]
    return padded
#===========================================================================================
#===========================================================================================

def portate():
    df_portate = pandas.read_csv(
        "/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/output/coordinate_corrette/stations_coordinates_Q_corrected.csv"
    )
    folder_path = "/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/WL_Q_nc/"
    files = os.listdir(folder_path)

    start_date = "2004-09-01 00:00:00"
    end_date = "2023-08-31 23:00:00"
    full_time_index = pandas.date_range(start=start_date, end=end_date, freq="h")

    df_dict = {}

    for file in files:
        if file.endswith(".nc"):
            # Extract standardized name
            name = file[:-21].replace("_", "").lower()

            if name in df_portate["Section_Basin"].values:
                print(f"Processing file: {file}")

                ds = xr.open_dataset(os.path.join(folder_path, file))
                Q = ds["Q"]
                H = ds["H"]
                lat = ds["lat"].values
                lon = ds["lon"].values

                # SET VALUES BELOW 0 TO np.nan
                Q = Q.where(Q >= 0)
                H = H.where(H >= 0)
                # REMOVE VALUES ABOVE 1E6 (considered outliers)
                Q = Q.where(Q <= 1e6)
                H = H.where(H <= 1e6)

                df_dict[name] = pandas.DataFrame({
                    "Q": pad_or_trim(Q, full_time_index).to_pandas(),
                    "H": pad_or_trim(H, full_time_index).to_pandas(),
                    "Latitude": lat,
                    "Longitude": lon
                }, index=full_time_index)

    # Save dictionary to pickle
    with open(
        "/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/output/coordinate_corrette/stations_data_dict.pkl",
        "wb"
    ) as f:
        pickle.dump(df_dict, f)
    return df_dict

#===========================================================================================
#===========================================================================================
def hydrological_year(date):
    if date.month >= 10:
        return date.year + 1
    else:
        return date.year

def get_longest_period(df, column, use_time=False):
    """
    Returns start and end of the longest non-NaN period.

    Parameters:
    - df: DataFrame with DatetimeIndex
    - column: column name (str)
    - use_time: if True -> longest by time duration
                if False -> longest by number of points
    """

    mask = df[column].notna()

    # identify consecutive blocks
    groups = (mask != mask.shift()).cumsum()

    # compute start, end, and size
    periods = df[mask].groupby(groups).agg(
        start=(column, lambda x: x.index.min()),
        end=(column, lambda x: x.index.max()),
        length=(column, "size")
    )

    # compute duration
    periods["duration"] = periods["end"] - periods["start"]

    if len(periods) == 0:
        return None, None

    # choose longest period
    if use_time:
        longest = periods.loc[periods["duration"].idxmax()]
    else:
        longest = periods.loc[periods["length"].idxmax()]

    return longest["start"], longest["end"]
def q_stats(x):
    return pandas.Series({
        "25th": x.quantile(0.25),
        "50th": x.quantile(0.5),
        "75th": x.quantile(0.75),
        "95th": x.quantile(0.95),
        "mean": x.mean(),
        "std": x.std()
    })

def plot_seasonal_cycle(df_statistics):
    for station in df_statistics["Station"].unique():
        df_s = df_statistics[df_statistics["Station"] == station].sort_values("Month")

        # --- Create 2 subplots (shared x-axis)
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

        # ======================
        # --- Q subplot (top)
        # ======================

        ax1.plot(df_s["Month"], df_s["50th_Q"], label="Median Q")
        ax1.fill_between(
            df_s["Month"],
            df_s["25th_Q"],
            df_s["75th_Q"],
            alpha=0.3
        )
        ax1.set_ylabel("Discharge (Q)")
        ax1.grid()
        ax1.legend(loc="best")

        # ======================
        # --- H subplot (bottom)
        # ======================

        ax2.plot(df_s["Month"], df_s["50th_H"], label="Median H", color="orange")
        ax2.fill_between(
            df_s["Month"],
            df_s["25th_H"],
            df_s["75th_H"],
            alpha=0.3,
            color="orange"
        )

        ax2.set_xlabel("Month")
        ax2.set_ylabel("Water Level (H)")
        ax2.grid()
        ax2.legend(loc="best")

        # --- Title
        fig.suptitle(f"Seasonal Cycle (Q & H) - {station}")

        # --- Layout fix
        plt.tight_layout()

        # --- Save
        plt.savefig(
            f"/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/output/coordinate_corrette/PLOTS/seasonal_cycle_QH_{station}.png")
        plt.close()

#===========================================================================================
#===========================================================================================
def summary():

    # --- Create summary of available data
    stations_summary = []
    df_dict_path = "/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/output/coordinate_corrette/stations_data_dict.pkl"
    df_duplicated = pandas.read_csv("/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/output/duplicate_stations_corrected.csv")

    df_percentiles = []
    stations_summary_updated = []
    df_daily = {}

    with open(df_dict_path, "rb") as f:
        df_dict = pickle.load(f)

    for station, df in df_dict.items():
        has_Q = not df["Q"].isna().all()
        has_H = not df["H"].isna().all()
        Latitude = df["Latitude"].iloc[0] if not df["Latitude"].isna().all() else np.nan
        Longitude = df["Longitude"].iloc[0] if not df["Longitude"].isna().all() else np.nan

        df = df.resample("D").agg(lambda x: x.mean() if x.count() >= 0.8 * 24 else np.nan)

        if df["Q"].isna().sum() / len(df) > 0.8:
            has_Q = False

        if df["H"].isna().sum() / len(df) > 0.8:
            has_H = False

        # store in df_daily  only if both has_Q and has_H are True
        if has_Q and has_H:
            df_daily[station] = df
            stations_summary.append({
                "Station": station,
                " Latitude": Latitude,
                "Longitude": Longitude
            })

    df_stations_summary = pandas.DataFrame(stations_summary)
    df_stations_summary["Duplicate"] = df_stations_summary.apply(lambda row: True if ((row[" Latitude"] in df_duplicated["Latitude"].values) & (row["Longitude"] in df_duplicated["Longitude"].values)) else False, axis=1)
    df_stations_summary = df_stations_summary[~((df_stations_summary["Duplicate"] == True))]

    count_station = 0

    for station, df in df_daily.items():
        if station in df_stations_summary["Station"].values:

            df["Month"] = df.index.month
            df.index = pandas.to_datetime(df.index)

            # insert the hydrological year in the dataframe
            df["Hydrological_Year"] = df.index.map(hydrological_year)
            df_hydro_year = (
                df.groupby("Hydrological_Year").apply(lambda x: (x["Q"].notna().sum() >= 0.8 * len(x)) and (x["H"].notna().sum() >= 0.8 * len(x)),include_groups=False).reset_index(name="has_data"))
            n_years_with_data = df_hydro_year["has_data"].sum()
            years = df_hydro_year[df_hydro_year["has_data"] == True]["Hydrological_Year"].tolist()
            if len(years) >= 5:
                count_station += 1
            elif n_years_with_data < 5:
                print(f"Station {station} has only {n_years_with_data} hydrological years with sufficient data. Skipping.")
                continue

            start_Q, end_Q = get_longest_period(df, "Q", use_time=True)
            start_H, end_H = get_longest_period(df, "H", use_time=True)
            Latitude = df["Latitude"].iloc[0] if not df["Latitude"].isna().all() else np.nan
            Longitude = df["Longitude"].iloc[0] if not df["Longitude"].isna().all() else np.nan

            stations_summary_updated.append({
                "Station": station,
                "Latitude": Latitude,
                "Longitude": Longitude,
                "start_Q": start_Q,
                "end_Q": end_Q,
                "start_H": start_H,
                "end_H": end_H,
                 "year": ", ".join(map(str, years))
            })

            # --- Plot full time series for Q and H (two stacked subplots)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

            # --- Q subplot
            ax1.plot(df.index, df["Q"], label="Discharge (Q)", color="blue")
            ax1.set_ylabel("Discharge (Q)")
            ax1.grid()
            ax1.legend(loc="best")

            # --- H subplot
            ax2.plot(df.index, df["H"], label="Water Level (H)", color="orange")
            ax2.set_ylabel("Water Level (H)")
            ax2.set_xlabel("Time")
            ax2.grid()
            ax2.legend(loc="best")

            # --- Title and layout
            fig.suptitle(f"Time Series - {station}")
            plt.tight_layout()

            # --- Save figure
            plt.savefig(f"/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/output/coordinate_corrette/PLOTS_time_series/time_series_QH_{station}.png")
            plt.close()

            # Group by month
            grouped = df.groupby("Month")

            for month, group in grouped:
                stats_Q = q_stats(group["Q"].dropna())
                stats_H = q_stats(group["H"].dropna())

                # create a list of dictionaries with the percentiles for Q and H for each month and station
                df_percentiles.append({
                    "Station" : station,
                    "Month": month,
                    "25th_Q": stats_Q["25th"],
                    "50th_Q": stats_Q["50th"],
                    "75th_Q": stats_Q["75th"],
                    "95th_Q": stats_Q["95th"],
                    "mean_Q": stats_Q["mean"],
                    "std_Q": stats_Q["std"],
                    "25th_H": stats_H["25th"],
                    "50th_H": stats_H["50th"],
                    "75th_H": stats_H["75th"],
                    "95th_H": stats_H["95th"],
                    "mean_H": stats_H["mean"],
                    "std_H": stats_H["std"]
                })
            print(f"Processed station: {station} - Q period: {start_Q} to {end_Q} - H period: {start_H} to {end_H}")
        else:
            continue

    # --- Save percentiles to  pkl
    with open( "/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/output/coordinate_corrette/stations_monthly_percentiles.pkl","wb") as f:
        pickle.dump(df_percentiles, f)

    df_stations_summary_updated = pandas.DataFrame(stations_summary_updated)
    df_stations_summary_updated.to_csv("/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/output/coordinate_corrette/stations_summary_updated.csv", index=False)
    df_statistics = pandas.DataFrame(df_percentiles)
    plot_seasonal_cycle(df_statistics)

    print(f"Monthly percentiles saved! Processed {count_station} stations with both Q and H available.")

    return None




if __name__ == "__main__":
    # Step 1: Analyze coordinates and identify duplicates
    analisi_coordinate()

    # Step 2: Correct coordinates using the reference file and count duplicates
    correct_coordinates()

    # Step 3: Load NetCDF files, extract data, and store in a dictionary
    portate()

    # Step 4: Create a summary of available data for each station
    summary()
