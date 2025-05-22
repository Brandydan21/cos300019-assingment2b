# Table of Contents

- [Project Setup Instructions (macOS, Python 3.10.11)](#project-setup-instructions-macos-python-31011)
- [Steps to Cleanse and Process the Data Sets](#steps-to-cleanse-and-process-the-data-sets)
- [Run Order Summary](#run-order-summary)
- [Training](#training)

---

# Project Setup Instructions (Python 3.10.11)

## 1. Install Python 3.10.11 with Tkinter Support

Download and install the official Python 3.10.11 release from python.org for macOS:

https://www.python.org/ftp/python/3.10.11/python-3.10.11-macos11.pkg

Run the `.pkg` file and follow the installer instructions.

Download and install the official Python 3.10.11 release from python.org for Windows:

https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe

Run the `.exe` file and follow the installer instructions.

We need to install Python from the official Python website as it comes with TKinter which is required for this project

## 2. Confirm Python and Tkinter Installation

Open Terminal and run:

```bash
$ /Library/Frameworks/Python.framework/Versions/3.10/bin/python3 -m tkinter
```

If a small GUI window opens, Tkinter is working correctly.

```
$ python3 --version 
```
Should return:
```
Python 3.10.11
```
We need version 3.10.11 as Tensorflow is required and can not run on newer versions of Python 

## 3. Set Up the Project

Navigate to the root of your project directory:

```
$ cd cos300019-assingment2b
```

Create a virtual environment using the installed Python 3.10.11:

```
$ python3 -m venv env
$ source env/bin/activate
```

## 4. Install Python Requirements

Ensure that `requirements.txt` is present in the project directory, then run:

```
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

## 5. Run the Application

To start the GUI application:

```
$ python GUI.py
```

If the GUI opens successfully, the setup is complete.

---

# Steps to Cleanse and Process the Data Sets

This project processes SCATS traffic data and prepares it for machine learning and path-finding tasks. The data preparation is modularized across several Python scripts, which should be run in a specific order.

## 1. `data_cleansing.py`

This script performs the initial **raw data cleanup**. It:

- Removes invalid or missing values
- Removes the columns that are redundant and not used
- Ensures date and time fields are correctly parsed
- Outputs a cleaned CSV that’s suitable for structured analysis

### To run:
```
$ python data_cleansing.py
```

### Output:
- `scats_volume_flat.csv` — the cleaned version of the raw SCATS dataset

Ensure that the original raw SCATS dataset is located in the expected directory or path referenced in the script.

---

## 2. `data_preprocessing.py`

This script transforms cleaned SCATS volume data into time-series sequences for machine learning (e.g. GRU/LSTM).

### Key Steps:
- Loads `scats_volume_flat.csv`
- Extracts features: `day_of_week`, `time_norm`
- One-hot encodes day of week and assigns integer IDs to SCAT sites
- Scales traffic volume per SCAT site
- Generates 8-step input sequences and prediction targets
- Splits into training and test sets

### To run:
```
$ python data_preprocessing.py
```

### Output Files:
- `../processed_data/scalers_by_scat.joblib` — volume scalers per SCAT site
- `../processed_data/onehot_encoder_dow.joblib` — day-of-week encoder
- `../processed_data/scat_id_lookup.joblib` — SCAT site to integer ID map

### Prerequisite:
Ensure `data_cleansing.py` has generated `scats_volume_flat.csv` in the `../processed_data/` directory.

---

## 3. `combine_neighbours_scats.py`

This script merges SCATS traffic volume data with metadata about each site's GPS coordinates and neighboring intersections.

### Key Steps:
- Loads `scats_volume_flat.csv` and `scat_neighbours.csv`
- Normalizes SCAT site IDs to 4-digit strings
- Merges volume data with neighbor/location metadata by SCAT number
- Selects and renames relevant columns for downstream tasks

### To run:
```
$ python combine_neighbours_scats.py
```

### Output File:
- `../processed_data/scat_volume_with_coords_and_neighbours.csv` — combined data including volume, coordinates, and neighbor sites for each SCAT site

### Prerequisites:
- `scats_volume_flat.csv` (produced by `data_cleansing.py`)
- `scat_neighbours.csv` (containing SCAT metadata with neighbors and coordinates)

---

## Run Order Summary

Run the files in this order:

1. `data_cleansing.py` – cleans and flattens raw SCATS volume data  
   → Output: `../processed_data/scats_volume_flat.csv`

2. `data_preprocessing.py` – extracts features, encodes, scales, and prepares time-series data  
   → Outputs:  
   - `../processed_data/scalers_by_scat.joblib`  
   - `../processed_data/onehot_encoder_dow.joblib`  
   - `../processed_data/scat_id_lookup.joblib`  
   - In-memory training arrays: `X_train_seq`, `y_train`, etc.

3. `combine_neighbours_scats.py` – adds SCAT site coordinates and neighbor data  
   → Output: `../processed_data/scat_volume_with_coords_and_neighbours.csv`

This process prepares clean, structured, and spatially enriched data for downstream machine learning and routing tasks.

---

# Training

Once the data has been cleansed (`data_cleansing.py`), preprocessed into feature sequences (`data_preprocessing.py`), and enriched with SCAT site metadata (`combine_neighbours_scats.py`), the final step is to train machine learning models on the processed data.

The training script builds and trains three model types using TensorFlow:

- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)
- Dense (Fully Connected Neural Network)

### To run:
```
$ python training_model.py
```

### Output:
- `models/lstm_scat_model.keras` — trained LSTM model
- `models/gru_scat_model.keras` — trained GRU model
- `models/dense_scat_model.keras` — trained Dense model
- `models/model_evaluation.txt` — evaluation metrics (MSE, RMSE, MAE) for each model on test data

The models are evaluated using:
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Error (MAE)**

This final step prepares models for use in traffic prediction or routing systems.
