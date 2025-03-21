import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import osmnx as ox
from multiprocessing import Pool
import os
import sys
import time
import random


if len(sys.argv) != 5:
    print(
        "Usage: python project_parallel.py <Problem> <Population size> <Generations> <Mutation Ratio>"
    )
    sys.exit(1)

pop_size = int(sys.argv[2])
generations = int(sys.argv[3])
mutation_ratio = float(sys.argv[4])
# Introduction to the project
if not os.path.exists("graph.graphml"):
    Graph = ox.graph_from_place("Copenhagen Municipality, Denmark", network_type="walk")
    ox.save_graphml(Graph, "graph.graphml")
    nodes_, edges_ = ox.graph_to_gdfs(Graph, nodes=True, edges=True)
    nodes_.to_csv("nodes.csv")
    edges_.to_csv("edges.csv")

nodes = pd.read_csv("nodes.csv")
edges = pd.read_csv("edges.csv")
G = ox.load_graphml("graph.graphml")



def plot_path(path):
    route_x, route_y = zip(*[(G.nodes[node]["x"], G.nodes[node]["y"]) for node in path])
    padding = 0.001
    x_min, x_max = min(route_x) - padding, max(route_x) + padding
    y_min, y_max = min(route_y) - padding, max(route_y) + padding

    fig, ax = plt.subplots(figsize=(8, 8))
    ox.plot_graph(G, ax=ax, show=False, close=False, bgcolor="white")
    ax.plot(route_x[0], route_y[0], color="yellow", linewidth=5, marker="o")
    ax.plot(route_x[-1], route_y[-1], color="yellow", linewidth=5, marker="*")
    ax.plot(route_x, route_y, color="red")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    plt.show()


def plot_multi_path(paths):
    fig, ax = plt.subplots(figsize=(8, 8))
    ox.plot_graph(G, ax=ax, show=False, close=False, bgcolor="white")

    colors = [
        "red",
        "blue",
        "green",
        "purple",
        "orange",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
    ]

    for i, path in enumerate(paths):
        route_x, route_y = zip(
            *[(G.nodes[node]["x"], G.nodes[node]["y"]) for node in path]
        )
        color = colors[i % len(colors)]
        ax.plot(route_x, route_y, color=color, label=f"Path {i+1}")
    ax.plot(route_x[0], route_y[0], color="yellow", linewidth=5, marker="o")
    ax.plot(route_x[-1], route_y[-1], color="yellow", linewidth=5, marker="*")
    ax.legend()
    ax.set_xlim(min(route_x) - 0.001, max(route_x) + 0.001)
    ax.set_ylim(min(route_y) - 0.001, max(route_y) + 0.001)
    plt.show()




def find_neighbors(node_id):
    return np.array(list(G.neighbors(node_id)))


def real_dist_nodes(node1_id, node2_id):
    edge = edges[(edges["u"] == node1_id) & (edges["v"] == node2_id)]
    if not edge.empty:
        return edge.iloc[0]["length"]
    else:
        print("Nodes are not connected")
        return None


def euc_dist_nodes(node1_id, node2_id):
    y1, x1 = G.nodes[node1_id]["y"], G.nodes[node1_id]["x"]
    y2, x2 = G.nodes[node2_id]["y"], G.nodes[node2_id]["x"]
    return ox.distance.euclidean(y1, x1, y2, x2) ** 2


def distance_to_prob(dist):
    prob = 1 / dist
    return prob / np.sum(prob)


def path_length(path):
    return sum(G[u][v][0]["length"] for u, v in zip(path[:-1], path[1:]))


def length_population(pop):
    lengths = pool.map(path_length, pop)
    return np.array(lengths)


def find_path(start_id, end_id):
    current_id = start_id
    path = np.array([current_id])

    while current_id != end_id:
        neigh = find_neighbors(current_id)
        dist_to_end = np.array([euc_dist_nodes(n, end_id) for n in neigh])
        if np.any(dist_to_end == 0):
            path = np.append(path, end_id)
            break
        prob = distance_to_prob(dist_to_end)
        current_id = np.random.choice(neigh, p=prob)
        path = np.append(path, current_id)

    return np.array(path)


def populate(start_id, end_id, num_pop):
    area = define_search_area(start_id, end_id)
    
    population = pool.starmap(
            find_path_no_backtrack, [(start_id, end_id, area) for _ in range(num_pop)]
        )
    print(f"All {num_pop} paths found!\n###############")
    return population


def best_survive(population, lengths, fraction):
    sorted_indices = np.argsort(lengths)
    survivors = fraction
    return [population[i] for i in sorted_indices[:survivors]]


def roulette_wheel_selection(
    population,
    prob,
):
    random_indexs = np.random.choice(range(len(population)), 2, p=prob)
    return population[random_indexs[0]], population[random_indexs[1]]


def mutation(path):
    node1ind, node2ind = sorted(
        np.random.choice(range(1, len(path) - 1), size=2, replace=False)
    )
    area_mut = define_search_area(path[node1ind], path[node2ind])
    mutate_section = find_path_no_backtrack(path[node1ind], path[node2ind], area_mut)

    if mutate_section is None or len(np.array(mutate_section)) <= 2:
        return path

    mutate_path = np.concatenate(
        (path[:node1ind], mutate_section, path[node2ind + 1 :])
    )

    return mutate_path


def sex(parent1, parent2, mutatation_ratio):
    parent1 = np.array(parent1)
    parent2 = np.array(parent2)
    common = list(set(parent1) & set(parent2))
    cnode = random.choice(common)
    idx1 = int(random.choice(np.where(parent1 == cnode)[0]))
    idx2 = int(random.choice(np.where(parent2 == cnode)[0]))
    child = (
        np.concatenate((parent1[:idx1], parent2[idx2:]))
        if np.random.randint(0, 2) == 0
        else np.concatenate((parent2[:idx2], parent1[idx1:]))
    )
    if np.random.randint(0, 100) < 100 * mutatation_ratio:
        child = mutation(child)
    return child


def save_plot(gen, best_paths):
    generations = len(best_paths)
    x_min, x_max = min(G.nodes[node]["x"] for node in G.nodes), max(
        G.nodes[node]["x"] for node in G.nodes
    )
    y_min, y_max = min(G.nodes[node]["y"] for node in G.nodes), max(
        G.nodes[node]["y"] for node in G.nodes
    )
    to_plot = best_paths[gen]
    fig, ax = plt.subplots(figsize=(8, 8))
    ox.plot_graph(G, ax=ax, show=False, close=False, bgcolor="white")
    route_x, route_y = zip(
        *[(G.nodes[node]["x"], G.nodes[node]["y"]) for node in to_plot]
    )
    ax.plot(route_x, route_y, color="red", linewidth=2)
    ax.plot(route_x[0], route_y[0], color="yellow", linewidth=5, marker="o")
    ax.plot(route_x[-1], route_y[-1], color="yellow", linewidth=5, marker="*")
    ax.text(
        0.05,
        0.95,
        f"Generation {gen+1}/{generations}",
        transform=ax.transAxes,
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.7),
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    plt.savefig(f"gifs/Path_gen{gen}.png")
    plt.close(fig)


def define_search_area(start_id, end_id, margin=0.005):
    """Defines a bounding box of nodes between start and end within a given margin."""
    start_x, start_y = G.nodes[start_id]["x"], G.nodes[start_id]["y"]
    end_x, end_y = G.nodes[end_id]["x"], G.nodes[end_id]["y"]

    x_min, x_max = min(start_x, end_x) - margin, max(start_x, end_x) + margin
    y_min, y_max = min(start_y, end_y) - margin, max(start_y, end_y) + margin

    area_nodes = {
        node
        for node in G.nodes
        if x_min <= G.nodes[node]["x"] <= x_max and y_min <= G.nodes[node]["y"] <= y_max
    }

    dead_ends = {
        node
        for node in area_nodes
        if node != start_id and node != end_id and len(list(G.neighbors(node))) == 1
    }

    while dead_ends:
        area_nodes -= dead_ends
        new_dead_ends = {
            node
            for node in area_nodes
            if node != start_id
            and node != end_id
            and len(set(G.neighbors(node)) & area_nodes) == 1
        }
        dead_ends = new_dead_ends

    return np.array(list(area_nodes))


def assign_unique_priorities(nodes):
    """Assigns a unique priority to each node, ensuring no duplicates."""
    priority_values = list(range(len(nodes)))
    random.shuffle(priority_values)
    return dict(zip(nodes, priority_values))


def find_path_no_backtrack(start_id, end_id, area_nodes):
    visited = set()
    path = [start_id]
    priorities = assign_unique_priorities(area_nodes)

    while path:
        current = path[-1]

        if current == end_id:
            return np.array(path)

        visited.add(current)
        neighbors = [
            n for n in G.neighbors(current) if n not in visited and n in area_nodes
        ]

        if not neighbors:
            path.pop()
            continue

        next_node = max(neighbors, key=lambda n: priorities[n])

        path.append(next_node)

    return None


def printinfo(verbose, gen, best_length, generations, start):
    if verbose == True:
        print("Generation", gen + 1)
        print("Best length", best_length[-1])
        print(
            f"Time for it {time.time()- start}",
        )
    elif verbose == False:
        if (gen % 50) == 0:
            print("Generation", gen + 1)
            print("Best length", best_length[-1])
            print(
                f"Time for it {time.time()- start}",
            )

    elif verbose == "Fancy":
        progress = (gen + 1) / generations
        bar_length = 40
        block = int(round(bar_length * progress))
        text = f"\rGeneration {gen + 1}/{generations} [{'#' * block + '-' * (bar_length - block)}] {progress * 100:.2f}%"
        print(text, end="")
    elif verbose == 'NO':
        pass


def evolution(
    nodestart,
    nodeend,
    generations,
    population_size,
    mutation_ratio,
    pool, 
    theorcial_distance,
    gif=False,
    verbose=False,
):
    np.random.seed(int(time.time()))
    best_paths = np.ndarray(generations, dtype=object)
    convergency_generations = 100
    best_length = []
    population = populate(nodestart, nodeend, population_size)
    lengths = length_population(population)
    sorted_indices = np.argsort(lengths)
    population = [population[i] for i in sorted_indices]
    lengths = lengths[sorted_indices]
    for gen in range(generations):
        start = time.time()

        next_gen = [population[0]]
        prob = distance_to_prob(lengths)
        # Generating max generation
        results = pool.starmap(
            sex,
            [
                (p1, p2, mutation_ratio)
                for p1, p2 in (
                    roulette_wheel_selection(population, prob)
                    for _ in range(len(population) - 1)
                )
            ],
        )

        next_gen.extend(results)

        # Stats
        population = next_gen
        lengths = length_population(population)
        best_length.append(min(lengths))
        sorted_indices = np.argsort(lengths)
        population = [population[i] for i in sorted_indices]
        lengths = lengths[sorted_indices]

        # Saving the best path and printing
        best_path = population[0]
        best_paths[gen] = best_path
        printinfo(verbose, gen, best_length, generations, start)

        # Convergency check
        if len(best_length) > convergency_generations or np.abs(best_length[-1] - theorcial_distance) < 10:
            if all(
                best_length[-i] == best_length[-i - 1]
                for i in range(1, convergency_generations)
            ):
                print(f"\nEarly stop")
                break

    if gif:
        print("Start plotting GIF")
        
        pool.starmap(save_plot, [(gen, best_paths) for gen in range(generations)])
        print("GIF saved as best_paths.gif")
    print(f"\nBest length: {best_length[-1]}")
    return best_path, best_length

# Define the start and end point
problem = int(sys.argv[1])

if problem == 1:
    latend, lonend = ox.geocoder.geocode("Blegdamsvej 17, 2100 København, Danmark")
elif problem == 2:
    latend, lonend = ox.geocoder.geocode("Nørregade 10, 1172 København, Danmark")
elif problem == 3:
    latend, lonend = ox.geocoder.geocode("Karen Blixens Vej 8, 2300 København, Danmark")
else:
    print("Invalid problem number")
    sys.exit(1)


latstart, lonstart = ox.geocoder.geocode("Jagtvej 155, 2200 København, Danmark")
nodestart = ox.distance.nearest_nodes(G, lonstart, latstart)
nodeend = ox.distance.nearest_nodes(G, lonend, latend)
theorical_distance = ox.routing.shortest_path(G, nodestart, nodeend)

# Run the simulation
print("Start simulation")
pool = Pool(processes=64)
start = time.time()
best_path, best_lengths = evolution(
    nodestart,
    nodeend,
    generations=generations,
    population_size=pop_size,
    mutation_ratio=mutation_ratio,
    verbose='NO',
    pool=pool,
    theorcial_distance=path_length(theorical_distance),

)
pool.close()
pool.join()
end = time.time()
print("Simulation finished")
print("Time elapsed: ", end - start)

print(best_path)

results_file = "data/results_long.txt"
os.makedirs(os.path.dirname(results_file), exist_ok=True)
with open(results_file, "a") as f:
    f.write(
        f"{problem},{end - start},{best_lengths[-1]},{pop_size},{mutation_ratio},{generations}\n"
    )
