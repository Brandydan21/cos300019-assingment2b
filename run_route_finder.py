import argparse
import pickle
from tensorflow.keras.models import load_model
from routes import find_top_k_paths, update_graph_weights
from travel_time.travel_time_estimator import estimate_travel_time

parser = argparse.ArgumentParser(description="Top-k Route Finder for SCATS Graph")
parser.add_argument("--source", type=int, required=True, help="Start SCAT site number")
parser.add_argument("--target", type=int, required=True, help="Destination SCAT site number")
parser.add_argument("--datetime", type=str, required=True, help="Datetime in format YYYY-MM-DD HH:MM")
parser.add_argument("--model_path", type=str, default="training/models/gru_scat_model.keras", help="Path to the trained ML model")
parser.add_argument("--k", type=int, default=5, help="Number of top paths to return")

args = parser.parse_args()

# Load graph
with open("processed_data/scat_graph.gpickle", "rb") as f:
    G = pickle.load(f)

# Load ML model
model = load_model(args.model_path)

# Recalculate travel time weights dynamically
print(f"\n Recomputing travel times for datetime: {args.datetime}")
update_graph_weights(G, model, args.datetime, estimate_travel_time)

# Find and print top-k paths
print(f"\n Finding top-{args.k} shortest travel time paths from {args.source} → {args.target} at {args.datetime}")
top_paths = find_top_k_paths(G, args.source, args.target, args.k)

print(f"\n Top {args.k} shortest paths at {args.datetime}:\n")
for i, (path, time) in enumerate(top_paths, 1):
    print(f"Path {i} at {args.datetime}: {path}")
    print(f"  Total travel time: {time:.2f} seconds\n")

