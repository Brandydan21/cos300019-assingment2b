import pandas as pd

'''
This script is used to combine the Scat volume flat with the Scat neighbour file 
'''

volume_file = "../processed_data/scats_volume_flat.csv"
neighbours_file = "../data_sets/scat_neighbours.csv"
output_file = "../processed_data/scat_volume_with_coords_and_neighbours.csv"

# Load input files 
volume_df = pd.read_csv(volume_file)
neighbours_df = pd.read_csv(neighbours_file)

# Ensure SCAT numbers are consistently formatted (e.g., '0970') 
volume_df["scat_number"] = volume_df["scat_number"].astype(str).str.zfill(4)
neighbours_df["SCATS Number"] = neighbours_df["SCATS Number"].astype(str).str.zfill(4)

# Merge data on SCAT number
combined_df = pd.merge(volume_df, neighbours_df, left_on="scat_number", right_on="SCATS Number", how="left")

# Rename and reorder columns
combined_df.rename(columns={
    "NB_LATITUDE": "NB_LATITUDE",
    "NB_LONGITUDE": "NB_LONGITUDE",
    "NEIGHBOURS": "NEIGHBOURS",
    "NAME": "NAME"
}, inplace=True)

final_df = combined_df[[
    "scat_number", "date", "time", "volume", "NB_LATITUDE", "NB_LONGITUDE", "NEIGHBOURS", "NAME"
]]

# Save to new CSV
final_df.to_csv(output_file, index=False)
print(f" Combined SCAT data saved to: {output_file}")
