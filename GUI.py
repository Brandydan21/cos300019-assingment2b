import tkinter as tk
from tkinter import ttk
from tkintermapview import TkinterMapView
import pickle
from tensorflow.keras.models import load_model
from routes import find_top_k_paths, update_graph_weights
from travel_time.travel_time_estimator import estimate_travel_time
from datetime import datetime

# Load graph
with open("processed_data/scat_graph.gpickle", "rb") as f:
    G = pickle.load(f)

scat_sites = sorted(G.nodes)

MODEL_PATHS = {
    "LSTM": "training/models/lstm_scat_model.keras",
    "GRU": "training/models/gru_scat_model.keras",
    "Dense": "training/models/dense_scat_model.keras"
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

origin_var = tk.StringVar()
destination_var = tk.StringVar()
model_var = tk.StringVar(value="LSTM")
algo_var = tk.StringVar(value="astar")
date_var = tk.StringVar(value="2006-11-21")
time_var = tk.StringVar(value="09:30")
travel_time_var = tk.StringVar(value="")

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

map_widget = TkinterMapView(root, width=980, height=540, corner_radius=0)
map_widget.pack(pady=10)

travel_time_label = ttk.Label(root, textvariable=travel_time_var, font=("Segoe UI", 11, "bold"), foreground="white")
travel_time_label.pack(pady=5)

routes_frame = ttk.LabelFrame(root, text="🛣️ Top 5 Suggested Paths (Best shown on map)")
routes_frame.pack(pady=5)

scrollbar = tk.Scrollbar(routes_frame)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

route_listbox = tk.Listbox(routes_frame, height=6, width=125, font=("Segoe UI", 10), yscrollcommand=scrollbar.set)
route_listbox.pack(side=tk.LEFT, fill=tk.BOTH)
scrollbar.config(command=route_listbox.yview)

def draw_scat_markers():
    for node in G.nodes:
        lat, lon = G.nodes[node]["lat"], G.nodes[node]["lon"]
        map_widget.set_marker(lat, lon, text=str(node), marker_color_circle="#aaaaaa", marker_color_outside="#ffffff")

def draw_scat_edges():
    drawn_edges = set()
    for u in G.nodes:
        for v in G.neighbors(u):
            if (u, v) not in drawn_edges and (v, u) not in drawn_edges:
                coord_u = (G.nodes[u]['lat'], G.nodes[u]['lon'])
                coord_v = (G.nodes[v]['lat'], G.nodes[v]['lon'])
                map_widget.set_path([coord_u, coord_v], color="black", width=5)
                drawn_edges.add((u, v))

def center_map_on_graph():
    lats = [G.nodes[n]['lat'] for n in G.nodes]
    lons = [G.nodes[n]['lon'] for n in G.nodes]
    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)
    map_widget.set_position(center_lat, center_lon)
    map_widget.set_zoom(13)

def draw_full_graph():
    draw_scat_markers()
    draw_scat_edges()
    center_map_on_graph()

def find_and_plot_route():
    try:
        origin = origin_var.get()
        destination = destination_var.get()
        model_name = model_var.get()
        algo_name = algo_var.get()
        date = date_var.get()
        time = time_var.get()

        route_listbox.delete(0, tk.END)

        if not origin or not destination or not model_name or not algo_name or not date or not time:
            travel_time_var.set("All fields must be filled.")
            route_listbox.insert(tk.END, "Please complete all fields before proceeding.")
            return

        try:
            datetime.strptime(date, "%Y-%m-%d")
            datetime.strptime(time, "%H:%M")
        except ValueError:
            travel_time_var.set("Invalid date or time format.")
            route_listbox.insert(tk.END, "Use format: YYYY-MM-DD for date and HH:MM for time.")
            return

        if origin == destination:
            travel_time_var.set("Origin and destination must be different.")
            route_listbox.insert(tk.END, "Please choose a different destination.")
            return

        source = int(origin)
        target = int(destination)
        datetime_str = f"{date} {time}"
        model = load_model(MODEL_PATHS[model_name])
        update_graph_weights(G, model, datetime_str, estimate_travel_time)

        paths = find_top_k_paths(G, source, target, k=5, search_type=algo_name)
        paths.sort(key=lambda p: p[1])  

        map_widget.delete_all_path() 
        draw_scat_edges()             
        route_listbox.delete(0, tk.END)

        if paths and paths[0]:
            best_path, best_time = paths[0]
            coords = [(G.nodes[n]['lat'], G.nodes[n]['lon']) for n in best_path]

            if len(coords) >= 2:
                map_widget.set_position(*coords[0])
                map_widget.set_path(coords, color="blue", width=5)

            travel_time_var.set(f"Estimated Travel Time: {best_time:.2f} sec")

            for i, (path, time_sec) in enumerate(paths, start=1):
                summary = f"Path {i}: {' → '.join(map(str, path))} | {time_sec:.0f} sec"
                route_listbox.insert(tk.END, summary)
        else:
            travel_time_var.set("No valid route found.")
            route_listbox.insert(tk.END, "No paths found.")

    except Exception as e:
        travel_time_var.set(f"Error: {e}")
        route_listbox.insert(tk.END, f"Error: {e}")

ttk.Button(frame, text="Find Route", command=find_and_plot_route).grid(row=3, column=0, columnspan=4, pady=10)

draw_full_graph()

root.mainloop()
