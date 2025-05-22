import tkinter as tk
from tkinter import ttk
from tkintermapview import TkinterMapView
import pickle
from tensorflow.keras.models import load_model
from routes import find_top_k_paths, update_graph_weights
from travel_time.travel_time_estimator import estimate_travel_time

# Load graph
with open("processed_data/scat_graph.gpickle", "rb") as f:
    G = pickle.load(f)

scat_sites = sorted(G.nodes)

MODEL_PATHS = {
    "LSTM": "training/models/lstm_scat_model.keras",
    "GRU": "training/models/gru_scat_model.keras"
}
ALGORITHMS = ["dfs", "bfs", "greedy", "astar", "iddfs"]

# GUI setup
root = tk.Tk()
root.title("SCATS Route Finder (Desktop)")
root.geometry("1020x880")
root.configure(bg="#2c2f33")

style = ttk.Style()
style.configure("TLabel", background="#2c2f33", foreground="white", font=("Segoe UI", 10))
style.configure("TButton", font=("Segoe UI", 10))
style.configure("TCombobox", font=("Segoe UI", 10))
style.configure("TLabelframe.Label", background="#2c2f33", foreground="white", font=("Segoe UI", 10))

frame = ttk.Frame(root)
frame.pack(pady=10)

# Input variables
origin_var = tk.StringVar()
destination_var = tk.StringVar()
model_var = tk.StringVar(value="LSTM")
algo_var = tk.StringVar(value="astar")
date_var = tk.StringVar(value="2006-11-21")
time_var = tk.StringVar(value="09:30")
travel_time_var = tk.StringVar(value="")

# Input widgets
ttk.Label(frame, text="Origin SCAT site:").grid(row=0, column=0)
ttk.Combobox(frame, textvariable=origin_var, values=scat_sites, width=10).grid(row=0, column=1)

ttk.Label(frame, text="Destination SCAT site:").grid(row=0, column=2)
ttk.Combobox(frame, textvariable=destination_var, values=scat_sites, width=10).grid(row=0, column=3)

ttk.Label(frame, text="Date (YYYY-MM-DD):").grid(row=1, column=0)
ttk.Entry(frame, textvariable=date_var, width=12).grid(row=1, column=1)

ttk.Label(frame, text="Time (HH:MM):").grid(row=1, column=2)
ttk.Entry(frame, textvariable=time_var, width=10).grid(row=1, column=3)

ttk.Label(frame, text="ML Model:").grid(row=2, column=0)
ttk.Combobox(frame, textvariable=model_var, values=list(MODEL_PATHS.keys())).grid(row=2, column=1)

ttk.Label(frame, text="Search Algorithm:").grid(row=2, column=2)
ttk.Combobox(frame, textvariable=algo_var, values=ALGORITHMS).grid(row=2, column=3)

# Map widget
map_widget = TkinterMapView(root, width=980, height=540, corner_radius=0)
map_widget.pack(pady=10)

# Travel time label
travel_time_label = ttk.Label(root, textvariable=travel_time_var, font=("Segoe UI", 11, "bold"), foreground="white")
travel_time_label.pack(pady=5)

# Frame for route list
routes_frame = ttk.LabelFrame(root, text="🛣️ Top 5 Suggested Paths (Best shown on map)")
routes_frame.pack(pady=5)

scrollbar = tk.Scrollbar(routes_frame)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

route_listbox = tk.Listbox(routes_frame, height=6, width=125, font=("Segoe UI", 10), yscrollcommand=scrollbar.set)
route_listbox.pack(side=tk.LEFT, fill=tk.BOTH)
scrollbar.config(command=route_listbox.yview)

# Route logic
def find_and_plot_route():
    try:
        source = int(origin_var.get())
        target = int(destination_var.get())
        datetime_str = f"{date_var.get()} {time_var.get()}"
        model = load_model(MODEL_PATHS[model_var.get()])
        update_graph_weights(G, model, datetime_str, estimate_travel_time)

        # Get top 5 paths (but only draw best)
        paths = find_top_k_paths(G, source, target, k=5, search_type=algo_var.get())

        map_widget.delete_all_marker()
        map_widget.delete_all_path()
        route_listbox.delete(0, tk.END)

        # Add SCAT markers
        for node in G.nodes:
            lat, lon = G.nodes[node]["lat"], G.nodes[node]["lon"]
            map_widget.set_marker(lat, lon, text=str(node), marker_color_circle="#aaaaaa", marker_color_outside="#ffffff")

        # Draw all edges first
        drawn_edges = set()
        for u in G.nodes:
            for v in G.neighbors(u):
                if (u, v) not in drawn_edges and (v, u) not in drawn_edges:
                    coord_u = (G.nodes[u]['lat'], G.nodes[u]['lon'])
                    coord_v = (G.nodes[v]['lat'], G.nodes[v]['lon'])
                    map_widget.set_path([coord_u, coord_v], color="black", width=5)
                    drawn_edges.add((u, v))

        # Draw only best path on map
        if paths and paths[0]:
            best_path, best_time = paths[0]
            best_coords = [(G.nodes[n]['lat'], G.nodes[n]['lon']) for n in best_path]
            map_widget.set_position(*best_coords[0])
            map_widget.set_path(best_coords, color="blue", width=5)

            travel_time_var.set(f"Estimated Travel Time: {best_time:.2f} sec")

            # Display all top 5 paths in the listbox
            for i, (path, time_sec) in enumerate(paths, start=1):
                summary = f"Path {i}: {' → '.join(map(str, path))} | {time_sec:.0f} sec"
                route_listbox.insert(tk.END, summary)
        else:
            travel_time_var.set("No valid route found.")
            route_listbox.insert(tk.END, "No paths found.")

    except Exception as e:
        travel_time_var.set(f"Error: {e}")
        route_listbox.insert(tk.END, f"Error: {e}")

# Run button
ttk.Button(frame, text="Find Route", command=find_and_plot_route).grid(row=3, column=0, columnspan=4, pady=10)

root.mainloop()
