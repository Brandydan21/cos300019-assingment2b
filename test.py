from routes import find_top_k_paths
import pickle

# === Load the saved graph ===
with open("processed_data/scat_graph.gpickle", "rb") as f:
    G = pickle.load(f)

# === Define SCAT start and end points ===
source = 970
target = 3682

# === Find top 5 shortest travel time paths ===
top_paths = find_top_k_paths(G, source, target, k=5)

# === Display results ===
for i, (path, time) in enumerate(top_paths, 1):
    print(f"Path {i}: {path}")
    print(f"  Total travel time: {time:.2f} seconds\n")
