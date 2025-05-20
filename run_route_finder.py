import argparse
import pickle
from tensorflow.keras.models import load_model
from routes import find_top_k_paths, update_graph_weights
from travel_time.travel_time_estimator import estimate_travel_time

# Argument parser
parser = argparse.ArgumentParser(description="Top-k Route Finder for SCATS Graph")
parser.add_argument("--source", type=int, required=True, help="Start SCAT site number")
parser.add_argument("--target", type=int, required=True, help="Destination SCAT site number")
parser.add_argument("--datetime", type=str, required=True, help="Datetime in format YYYY-MM-DD HH:MM")
parser.add_argument("--model_path", type=str, default="training/models/gru_scat_model.keras", help="Path to the trained ML model")
parser.add_argument("--k", type=int, default=5, help="Number of top paths to return")
parser.add_argument("--search", type=str, default="topk", choices=["dfs", "bfs", "greedy", "astar", "iddfs", "topk"], help="Search algorithm to use")

args = parser.parse_args()

# Load graph from disk
with open("processed_data/scat_graph.gpickle", "rb") as f:
    G = pickle.load(f)

# Load ML prediction model (e.g., GRU)
model = load_model(args.model_path)

# Update graph weights using ML travel time predictions
update_graph_weights(G, model, args.datetime, estimate_travel_time)

# Run the selected search algorithm for top-k paths
paths = find_top_k_paths(G, args.source, args.target, k=args.k, search_type=args.search)

# Display output
if not paths:
    print("No paths found.")
else:
    print(f"\nTop {args.k} path(s) using '{args.search}' search:\n")
    for i, (path, travel_time) in enumerate(paths, 1):
        print(f"Path {i}: {path}")
        print(f"Total travel time: {travel_time:.2f} seconds\n")
