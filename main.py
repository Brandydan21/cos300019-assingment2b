import pandas as pd

# Load SCATS dataset and skip first row (subheaders)
df = pd.read_csv("data_sets/ScatsDataOctober2006 - Data.csv", header=1)

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

# Prepare flat list for CSV
flat_records = []

for _, row in df.iterrows():
    scat = str(row['scat_number']).zfill(4)
    lat = float(row['lat'])
    long = float(row['long'])
    date = row['date']

    for i, col in enumerate(volume_cols):
        hour = (i * 15) // 60
        minute = (i * 15) % 60
        time_str = f"{hour:02}:{minute:02}"
        volume = int(row[col])

        flat_records.append({
            "scat_number": scat,
            "lat": lat,
            "long": long,
            "date": date,
            "time": time_str,
            "volume": volume
        })

# Save flat CSV output for ML
flat_df = pd.DataFrame(flat_records)
flat_df.to_csv("processed_data/scats_volume_flat.csv", index=False)
