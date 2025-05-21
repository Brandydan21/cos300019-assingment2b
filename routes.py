import networkx as nx
from heapq import heappush, heappop
from collections import deque

def compute_total_travel_time(graph, path):
    return sum(graph[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))

def find_top_k_paths(graph, source, target, k=5, search_type="astar"):
    search_funcs = {
        "dfs": depth_first_search,
        "bfs": breadth_first_search,
        "greedy": greedy_best_first_search,
        "astar": a_star_search,
        "iddfs": iterative_deepening_dfs
    }

    search_func = search_funcs.get(search_type.lower())
    if not search_func:
        raise ValueError(f"Unsupported search type: {search_type}")

    paths = search_func(graph, source, target, k)
    return [(path, compute_total_travel_time(graph, path)) for path in paths]

def depth_first_search(graph, source, target, k):
    stack = [(source, [source])]
    paths = []
    seen = set()

    while stack and len(paths) < k:
        node, path = stack.pop()
        if node == target and tuple(path) not in seen:
            paths.append(path)
            seen.add(tuple(path))
            continue
        for neighbor in graph.neighbors(node):
            if neighbor not in path:
                stack.append((neighbor, path + [neighbor]))
    return paths

def breadth_first_search(graph, source, target, k):
    queue = deque([(source, [source])])
    paths = []
    seen = set()

    while queue and len(paths) < k:
        node, path = queue.popleft()
        if node == target and tuple(path) not in seen:
            paths.append(path)
            seen.add(tuple(path))
            continue
        for neighbor in graph.neighbors(node):
            if neighbor not in path:
                queue.append((neighbor, path + [neighbor]))
    return paths

def greedy_best_first_search(graph, source, target, k):
    def heuristic(n): return abs(n - target)
    queue = [(heuristic(source), source, [source])]
    results = []
    seen_paths = set()

    while queue and len(results) < k:
        _, node, path = heappop(queue)
        if node == target and tuple(path) not in seen_paths:
            results.append(path)
            seen_paths.add(tuple(path))
            continue
        for neighbor in graph.neighbors(node):
            if neighbor not in path:
                new_path = path + [neighbor]
                heappush(queue, (heuristic(neighbor), neighbor, new_path))
    return results

def a_star_search(graph, source, target, k):
    def heuristic(n): return abs(n - target)
    queue = [(heuristic(source), 0, source, [source])]
    results = []
    seen_paths = set()

    while queue and len(results) < k:
        f, cost, node, path = heappop(queue)
        if node == target and tuple(path) not in seen_paths:
            results.append(path)
            seen_paths.add(tuple(path))
            continue
        for neighbor in graph.neighbors(node):
            if neighbor not in path:
                new_cost = cost + graph[node][neighbor]['weight']
                new_path = path + [neighbor]
                heappush(queue, (new_cost + heuristic(neighbor), new_cost, neighbor, new_path))
    return results

def iterative_deepening_dfs(graph, source, target, k, max_depth=50):
    def dls(node, target, depth, path, found, seen):
        if len(found) >= k:
            return
        if depth == 0 and node == target:
            if tuple(path) not in seen:
                found.append(path)
                seen.add(tuple(path))
            return
        if depth > 0:
            for neighbor in graph.neighbors(node):
                if neighbor not in path:
                    dls(neighbor, target, depth - 1, path + [neighbor], found, seen)

    results = []
    seen = set()
    for depth in range(max_depth):
        dls(source, target, depth, [source], results, seen)
        if len(results) >= k:
            break
    return results

def update_graph_weights(graph, model, prediction_time, estimator_func):
    for u, v in graph.edges:
        try:
            travel_time = estimator_func(u, v, prediction_time, model)
            graph[u][v]["weight"] = travel_time
        except Exception as e:
            print(f" Failed to update edge ({u} → {v}): {e}")
