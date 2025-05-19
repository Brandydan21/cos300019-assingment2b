import networkx as nx

def compute_total_travel_time(graph, path):
    return sum(graph[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))

def find_top_k_paths(graph, source, target, k=5):
    from networkx.algorithms.simple_paths import shortest_simple_paths

    paths_gen = shortest_simple_paths(graph, source, target, weight='weight')

    top_k = []
    for i, path in enumerate(paths_gen):
        total_time = compute_total_travel_time(graph, path)
        top_k.append((path, total_time))
        if i + 1 == k:
            break

    return top_k

def update_graph_weights(graph, model, prediction_time, estimator_func):
    for u, v in graph.edges:
        try:
            travel_time = estimator_func(u, v, prediction_time, model)
            graph[u][v]["weight"] = travel_time
        except Exception as e:
            print(f"⚠️ Failed to update edge ({u} → {v}): {e}")
