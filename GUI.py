import streamlit as st
import pickle
import pandas as pd
from tensorflow.keras.models import load_model
from routes import find_top_k_paths, update_graph_weights
from travel_time.travel_time_estimator import estimate_travel_time
import datetime

# Load graph
with open("processed_data/scat_graph.gpickle", "rb") as f:
    G = pickle.load(f)

# Load SCAT site options
scat_sites = sorted(G.nodes)

st.title("🚦 SCATS Route Finder")

# User inputs
source = st.selectbox("Select origin SCAT site", scat_sites)
target = st.selectbox("Select destination SCAT site", scat_sites)
date_input = st.date_input("Select date", datetime.date(2006, 11, 21))
time_input = st.time_input("Select time", datetime.time(9, 30))
datetime_str = f"{date_input} {time_input.strftime('%H:%M')}"

model_choice = st.selectbox("Choose ML model", ["lstm_scat_model.keras", "gru_scat_model.keras"])
search_algo = st.selectbox("Choose search algorithm", ["dfs", "bfs", "greedy", "astar", "iddfs"])

if st.button("🔍 Find Top-5 Routes"):
    model = load_model(f"training/models/{model_choice}")
    update_graph_weights(G, model, datetime_str, estimate_travel_time)

    paths = find_top_k_paths(G, source, target, k=5, search_type=search_algo)

    if paths:
        st.success(f"Top {len(paths)} routes from {source} → {target}")
        for i, (path, travel_time) in enumerate(paths, 1):
            st.write(f"**Path {i}:** {' → '.join(map(str, path))}")
            st.write(f"Estimated Travel Time: {travel_time:.2f} sec")
    else:
        st.warning("No valid paths found.")

