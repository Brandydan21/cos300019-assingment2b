import pandas as pd

'''
Cleanses the ScatsDataOctober2006 - Data.csv to get only relevant data and output into 
../processed_data/scats_volume_flat.csv
'''


# Load SCATS dataset and skip first row (subheaders)
df = pd.read_csv("../data_sets/ScatsDataOctober2006 - Data.csv", header=1)

# Rename columns for consistency
df.rename(columns={
    'SCATS Number': 'scat_number',
    'Date': 'date'
}, inplace=True)

# Locate volume columns from V00 to V95
start_index = df.columns.get_loc('V00')
volume_cols = df.columns[start_index:start_index + 96]

# Prepare flat list for CSV
flat_records = []

for _, row in df.iterrows():
    scat = str(row['scat_number']).zfill(4)
    date = row['date']

    for i, col in enumerate(volume_cols):
        hour = (i * 15) // 60
        minute = (i * 15) % 60
        time_str = f"{hour:02}:{minute:02}"
        volume = int(row[col])

        flat_records.append({
            "scat_number": scat,
            "date": date,
            "time": time_str,
            "volume": volume
        })

# Convert to DataFrame
flat_df = pd.DataFrame(flat_records)

# Average duplicates (same scat_number, date, time)
flat_df = flat_df.groupby(['scat_number', 'date', 'time'], as_index=False)['volume'].mean()

# Save final averaged CSV output
flat_df.to_csv("../processed_data/scats_volume_flat.csv", index=False)
