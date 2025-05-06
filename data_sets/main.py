import pandas as pd
from collections import defaultdict
import json

# Load the SCATS dataset (skip first row which contains subheaders)
df = pd.read_csv("ScatsDataOctober2006 - Data.csv", header=1)

# Clean and rename necessary columns
df.rename(columns={
    'SCATS Number': 'scat_number',
    'NB_LATITUDE': 'lat',
    'NB_LONGITUDE': 'long',
    'Date': 'date'
}, inplace=True)

# Locate actual volume columns starting from 'V00' to 'V95'
start_index = df.columns.get_loc('V00')
volume_cols = df.columns[start_index:start_index + 96]  # Only V00–V95

# Prepare structured data
structured_data = {}

for _, row in df.iterrows():
    scat = str(row['scat_number']).zfill(4)
    lat = float(row['lat'])
    long = float(row['long'])
    date = row['date']

    if scat not in structured_data:
        structured_data[scat] = {
            "scat_number": int(scat),
            "long": long,
            "lat": lat,
            "date": defaultdict(list)
        }

    for i, col in enumerate(volume_cols):
        v_index = f"v{i:02}"
        hour = (i * 15) // 60
        minute = (i * 15) % 60
        time_str = f"{hour}:{minute:02}"
        volume = int(row[col])
        structured_data[scat]["date"][date].append((v_index, time_str, volume))

# Convert defaultdicts to regular dicts
for entry in structured_data.values():
    entry["date"] = dict(entry["date"])

# Write final output
with open("structured_scats_data.json", "w") as f:
    json.dump(structured_data, f, indent=2)
