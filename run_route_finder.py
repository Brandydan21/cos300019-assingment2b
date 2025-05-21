import argparse
import pickle
from tensorflow.keras.models import load_model
from routes import find_top_k_paths, update_graph_weights
from travel_time.travel_time_estimator import estimate_travel_time

# Argument parser
parser = argparse.ArgumentParser(description="Top-5 Route Finder for SCATS Graph")
parser.add_argument("--source", type=int, required=True, help="Start SCAT site number")
parser.add_argument("--target", type=int, required=True, help="Destination SCAT site number")
parser.add_argument("--datetime", type=str, required=True, help="Datetime in format YYYY-MM-DD HH:MM")
parser.add_argument("--model_path", type=str, default="training/models/gru_scat_model.keras", help="Path to the trained ML model")
parser.add_argument("--search", type=str, default="bfs", choices=["dfs", "bfs", "greedy", "astar", "iddfs"], help="Search algorithm to use")

args = parser.parse_args()

# Load graph
with open("processed_data/scat_graph.gpickle", "rb") as f:
    G = pickle.load(f)

# Load ML model
model = load_model(args.model_path)

if args.source not in G.nodes:
    print(f"❌ Source SCAT site {args.source} not found in the graph.")
    exit(1)

if args.target not in G.nodes:
    print(f"❌ Target SCAT site {args.target} not found in the graph.")
    exit(1)

# Update graph edge weights using the ML estimator
update_graph_weights(G, model, args.datetime, estimate_travel_time)

# Run search (fixed top 5)
paths = find_top_k_paths(G, args.source, args.target, k=5, search_type=args.search)

# Output
if not paths:
    print("⚠️ No paths found.")
else:
    print(f"\nTop 5 path(s) using '{args.search}' search:\n")
    for i, (path, travel_time) in enumerate(paths, 1):
        print(f"Path {i}: {path}")
        print(f"  ➤ Total travel time: {travel_time:.2f} seconds\n")
