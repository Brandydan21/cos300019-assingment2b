import pandas as pd
from collections import defaultdict
import json

# Load SCATS dataset and skip first row (subheaders)
df = pd.read_csv("ScatsDataOctober2006 - Data.csv", header=1)

# Load neighbours CSV
neighbours_df = pd.read_csv("scat_neighbours.csv")

# Rename columns for consistency
df.rename(columns={
    'SCATS Number': 'scat_number',
    'NB_LATITUDE': 'lat',
    'NB_LONGITUDE': 'long',
    'Date': 'date'
}, inplace=True)

# Locate volume columns from V00 to V95
start_index = df.columns.get_loc('V00')
volume_cols = df.columns[start_index:start_index + 96]

# Prepare structured data dictionary
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
            "neighbours": [],  # Placeholder, will be filled later
            "date": defaultdict(list)
        }

    for i, col in enumerate(volume_cols):
        v_index = f"v{i:02}"
        hour = (i * 15) // 60
        minute = (i * 15) % 60
        time_str = f"{hour}:{minute:02}"
        volume = int(row[col])
        structured_data[scat]["date"][date].append((v_index, time_str, volume))

# Integrate neighbours from the neighbour file
for _, row in neighbours_df.iterrows():
    scat_id = str(row['SCATS Number']).zfill(4)
    neighbours = [str(n).zfill(4) for n in str(row['NEIGHBOURS']).split(';') if n.strip().isdigit()]
    if scat_id in structured_data:
        structured_data[scat_id]["neighbours"] = neighbours

# Convert defaultdicts to regular dicts for JSON serialization
for entry in structured_data.values():
    entry["date"] = dict(entry["date"])

# Save to JSON file
with open("structured_scats_data_with_neighbours.json", "w") as f:
    json.dump(structured_data, f, indent=2)
