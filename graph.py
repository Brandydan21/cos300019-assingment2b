import pandas as pd
import networkx as nx
from travel_time.travel_time_estimator import estimate_travel_time
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# === Load SCAT neighbour data ===
df = pd.read_csv("data_sets/scat_neighbours.csv")
df["NEIGHBOURS"] = df["NEIGHBOURS"].astype(str).str.split(";")
df["SCATS Number"] = df["SCATS Number"].astype(int)

# === Load trained model ===
model = load_model("training/models/lstm_scat_model.keras")

# === Create graph ===
G = nx.Graph()

# === Use fixed prediction time (can parameterize this later) ===
prediction_time = "2006-11-21 09:30"

# === Add nodes and edges ===
for _, row in df.iterrows():
    from_scat = int(row["SCATS Number"])
    G.add_node(from_scat)

    neighbours = [int(n) for n in row["NEIGHBOURS"] if n.strip().isdigit()]
    for to_scat in neighbours:
        if G.has_edge(from_scat, to_scat):
            continue  # avoid duplicate edges

        try:
            travel_time = estimate_travel_time(from_scat, to_scat, prediction_time, model)
            G.add_edge(from_scat, to_scat, weight=travel_time)
            print(f"Edge {from_scat} → {to_scat} with travel time {travel_time} sec")
        except Exception as e:
            print(f"Failed for {from_scat} → {to_scat}: {e}")

print(f"\nRoad graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

# Extract lat/lon positions for nodes
pos = {
    int(row["SCATS Number"]): (row["NB_LONGITUDE"], row["NB_LATITUDE"])
    for _, row in df.iterrows()
    if int(row["SCATS Number"]) in G.nodes
}

# Draw graph
plt.figure(figsize=(12, 10))
nx.draw_networkx_nodes(G, pos, node_size=60, node_color="skyblue")
nx.draw_networkx_edges(G, pos, edge_color="gray", width=1)
nx.draw_networkx_labels(G, pos, font_size=6)

plt.title("SCAT Road Graph with Travel Time Edges")
plt.axis("off")
plt.tight_layout()
plt.show()

import os
import pickle

os.makedirs("processed_data", exist_ok=True)

with open("processed_data/scat_graph.gpickle", "wb") as f:
    pickle.dump(G, f)

print("Graph saved using pickle to processed_data/scat_graph.gpickle")