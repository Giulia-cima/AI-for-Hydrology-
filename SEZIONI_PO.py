import pandas
def format_station_name(name):
    name = name.replace(" ", "")
    name = name.replace("_", "")
    name = name.lower()
    return name


sezioni_po_lista = pandas.read_csv("/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/sezioni_po_lista.csv")
input_file = pandas.read_csv("/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/output/stations_coordinates_Q.csv")

input_file["uniformed_section"] = input_file["Section"].apply(format_station_name)
input_file["uniformed_basin"] = input_file["Basin"].apply(format_station_name)
sezioni_po_lista["uniformed_section"] = sezioni_po_lista["Section"].apply(format_station_name)
sezioni_po_lista["uniformed_basin"] = sezioni_po_lista["Basin"].apply(format_station_name)

# create a new file that contains the coordinates of the stations in the Po river basin. We need to merge the two dataframes based on the uniformed names of the stations and the basins
merged_df = pandas.merge(input_file, sezioni_po_lista, on=["uniformed_section", "uniformed_basin"], how="inner")
# take the line from the sezioni_po_lista that is not in the merged df and create a new line in the merged df with the coordinates of the sezioni_po_lista and the name of the station and the basin from the sezioni_po_lista
for index, row in sezioni_po_lista.iterrows():
    if not ((merged_df["uniformed_section"] == row["uniformed_section"]) & (merged_df["uniformed_basin"] == row["uniformed_basin"])).any():
        # Replace commas with dots in the Latitude and Longitude strings
        lat = row["Latitude"].replace(",", ".")
        lat = pandas.to_numeric(lat, errors="coerce")

        lon = row["Longitude"].replace(",", ".")
        lon = pandas.to_numeric(lon, errors="coerce")

        new_row = {
            "Section_x": row["Section"],
            "Basin_x": row["Basin"],
            "Latitude_x": lat,
            "Longitude_x": lon,
            "uniformed_section": row["uniformed_section"],
            "uniformed_basin": row["uniformed_basin"],
            "Section_y": row["Section"],
            "Basin_y": row["Basin"],
            "Latitude_y":lat,
            "Longitude_y": lon,
            "Drained_area" : row["Drained_area"]
        }
        # do not use append because it is deprecated, use concat instead
        merged_df = pandas.concat([merged_df, pandas.DataFrame([new_row])], ignore_index=True)

# same for the coordinates, where the column "Latitude_x" is null, fill it with the value of the column "Latitude_y" and where the column "Longitude_x" is null, fill it with the value of the column "Longitude_y"
merged_df["Latitude_x"] = merged_df["Latitude_x"].fillna(merged_df["Latitude_y"])
merged_df["Longitude_x"] = merged_df["Longitude_x"].fillna(merged_df["Longitude_y"])
merged_df = merged_df.drop(columns=["uniformed_section", "uniformed_basin", "Section_y", "Basin_y", "Duplicate", "Station_Ref"])
merged_df = merged_df.rename(columns={"Section_x": "Section", "Basin_x": "Basin"})

merged_df["Latitude_x"] = merged_df["Latitude_x"].astype(float)
merged_df["Longitude_x"] = merged_df["Longitude_x"].astype(float)
merged_df["Latitude_y"] = (merged_df["Latitude_y"] .astype(str) .str.replace(",", ".", regex=False))
merged_df["Latitude_y"] = pandas.to_numeric(merged_df["Latitude_y"], errors="coerce")
merged_df["Longitude_y"] = (merged_df["Longitude_y"] .astype(str) .str.replace(",", ".", regex=False))
merged_df["Longitude_y"] = pandas.to_numeric(merged_df["Longitude_y"], errors="coerce")
merged_df["Latitude_diff"] = (merged_df["Latitude_x"].round(2) != merged_df["Latitude_y"].round(2))
merged_df["Longitude_diff"] = (merged_df["Longitude_x"].round(2) != merged_df["Longitude_y"].round(2))

# where both differences are true , change the coordinates to the coordinates of the sezioni_po_lista
merged_df.loc[merged_df["Latitude_diff"] & merged_df["Longitude_diff"], ["Latitude_x", "Longitude_x"]] = merged_df.loc[merged_df["Latitude_diff"] & merged_df["Longitude_diff"], ["Latitude_y", "Longitude_y"]].values
merged_df = merged_df.drop(columns=["Latitude_y", "Longitude_y", "Latitude_diff", "Longitude_diff"])
merged_df = merged_df.rename(columns={"Latitude_x": "Latitude", "Longitude_x": "Longitude"})
print(merged_df)

# save the merged dataframe to a new csv file called "stations_coordinates_Q_PO.csv"
merged_df.to_csv("/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/output/stations_coordinates_Q_PO.csv", index=False)



