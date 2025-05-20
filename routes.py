import networkx as nx
from heapq import heappush, heappop
from collections import deque

def compute_total_travel_time(graph, path):
    return sum(graph[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))

def find_top_k_paths(graph, source, target, k=5, search_type="astar"):
    if search_type == "topk":
        from networkx.algorithms.simple_paths import shortest_simple_paths
        paths_gen = shortest_simple_paths(graph, source, target, weight='weight')
        top_k = []
        for i, path in enumerate(paths_gen):
            total_time = compute_total_travel_time(graph, path)
            top_k.append((path, total_time))
            if i + 1 == k:
                break
        return top_k

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

    # Only return 1 path for astar, greedy, iddfs
    single_path_algos = {"astar", "greedy", "iddfs"}
    paths = search_func(graph, source, target, k)

    if search_type in single_path_algos and paths:
        return [(paths[0], compute_total_travel_time(graph, paths[0]))]
    else:
        return [(path, compute_total_travel_time(graph, path)) for path in paths]

# --------------------------
# Search algorithm variants:
# --------------------------

def depth_first_search(graph, source, target, k):
    stack = [(source, [source])]
    paths = []

    while stack and len(paths) < k:
        node, path = stack.pop()
        if node == target:
            paths.append(path)
            continue
        for neighbor in graph.neighbors(node):
            if neighbor not in path:
                stack.append((neighbor, path + [neighbor]))
    return paths

def breadth_first_search(graph, source, target, k):
    queue = deque([(source, [source])])
    paths = []

    while queue and len(paths) < k:
        node, path = queue.popleft()
        if node == target:
            paths.append(path)
            continue
        for neighbor in graph.neighbors(node):
            if neighbor not in path:
                queue.append((neighbor, path + [neighbor]))
    return paths

def greedy_best_first_search(graph, source, target, k):
    def heuristic(n): return 0
    queue = [(heuristic(source), source, [source])]
    visited = set()

    while queue:
        _, node, path = heappop(queue)
        if node == target:
            return [path]
        if node in visited:
            continue
        visited.add(node)
        for neighbor in graph.neighbors(node):
            if neighbor not in path:
                heappush(queue, (heuristic(neighbor), neighbor, path + [neighbor]))
    return []

def a_star_search(graph, source, target, k):
    def heuristic(n): return 0
    queue = [(0 + heuristic(source), 0, source, [source])]
    visited = {}

    while queue:
        f, cost, node, path = heappop(queue)
        if node == target:
            return [path]
        if node in visited and visited[node] <= cost:
            continue
        visited[node] = cost
        for neighbor in graph.neighbors(node):
            if neighbor not in path:
                new_cost = cost + graph[node][neighbor]['weight']
                heappush(queue, (new_cost + heuristic(neighbor), new_cost, neighbor, path + [neighbor]))
    return []

def iterative_deepening_dfs(graph, source, target, k, max_depth=50):
    def dls(node, target, depth, path):
        if depth == 0 and node == target:
            return [path]
        if depth > 0:
            for neighbor in graph.neighbors(node):
                if neighbor not in path:
                    result = dls(neighbor, target, depth - 1, path + [neighbor])
                    if result:
                        return result
        return []

    for depth in range(max_depth):
        result = dls(source, target, depth, [source])
        if result:
            return result
    return []

def update_graph_weights(graph, model, prediction_time, estimator_func):
    for u, v in graph.edges:
        try:
            travel_time = estimator_func(u, v, prediction_time, model)
            graph[u][v]["weight"] = travel_time
        except Exception as e:
            print(f" Failed to update edge ({u} → {v}): {e}")
