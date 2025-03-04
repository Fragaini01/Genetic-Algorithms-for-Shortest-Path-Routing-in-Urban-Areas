import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import osmnx as ox  
from osmnx import geocoder
import random
import time
import networkx as nx
from io import BytesIO
from multiprocessing import Pool

nodes = pd.read_csv('nodes.csv')
edges = pd.read_csv('edges.csv')
G = ox.load_graphml('graph.graphml')

def plot_path(path):
    route_x, route_y = zip(*[(G.nodes[node]["x"], G.nodes[node]["y"]) for node in path])
    padding = 0.001 
    x_min, x_max = min(route_x) - padding, max(route_x) + padding
    y_min, y_max = min(route_y) - padding, max(route_y) + padding

    fig, ax = plt.subplots(figsize=(8, 8))
    ox.plot_graph(G, ax=ax, show=False, close=False, bgcolor='white')
    ax.plot(route_x[0], route_y[0], color='yellow', linewidth=5, marker='o')
    ax.plot(route_x[-1], route_y[-1], color='yellow', linewidth=5, marker='*')
    ax.plot(route_x, route_y, color='red')
    ax.set_xlim(x_min, x_max) 
    ax.set_ylim(y_min, y_max)
    plt.show()

def plot_multi_path(paths):
    fig, ax = plt.subplots(figsize=(8, 8))
    ox.plot_graph(G, ax=ax, show=False, close=False, bgcolor='white')

    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i, path in enumerate(paths):
        route_x, route_y = zip(*[(G.nodes[node]["x"], G.nodes[node]["y"]) for node in path])
        color = colors[i % len(colors)]
        ax.plot(route_x, route_y, color=color, label=f'Path {i+1}')
    ax.plot(route_x[0], route_y[0], color='yellow', linewidth=5, marker='o')
    ax.plot(route_x[-1], route_y[-1], color='yellow', linewidth=5, marker='*')
    ax.legend()
    ax.set_xlim(min(route_x) - 0.001, max(route_x) + 0.001)
    ax.set_ylim(min(route_y) - 0.001, max(route_y) + 0.001)
    plt.show()

def find_neighbors(node_id):
    return np.array(list(G.neighbors(node_id)))

def real_dist_nodes(node1_id, node2_id):
    edge = edges[(edges['u'] == node1_id) & (edges['v'] == node2_id)]
    if not edge.empty:
        return edge.iloc[0]['length']
    else:
        print('Nodes are not connected')
        return None 

def euc_dist_nodes(node1_id, node2_id):
    y1, x1 = G.nodes[node1_id]['y'], G.nodes[node1_id]['x']
    y2, x2 = G.nodes[node2_id]['y'], G.nodes[node2_id]['x']
    return ox.distance.euclidean(y1, x1, y2, x2)**2

def distance_to_prob(dist):
    prob = 1 / dist
    return prob / np.sum(prob)

def length_population(pop):
    return [sum(G[u][v][0]['length'] for u, v in zip(path[:-1], path[1:])) for path in pop]

def find_path(start_id, end_id):
    current_id = start_id
    path = [current_id]

    while current_id != end_id:
        neigh = find_neighbors(current_id)
        if len(neigh) == 0:
            break
        dist_to_end = np.array([euc_dist_nodes(n, end_id) for n in neigh])
        if np.any(dist_to_end == 0):
            path.append(end_id)
            break
        prob = distance_to_prob(dist_to_end)
        current_id = np.random.choice(neigh, p=prob)
        path.append(current_id)

    return np.array(path)

def populate(start_id, end_id, num_pop):
    print(f'Populating with {num_pop} paths:')
    with Pool() as pool:
        population = pool.starmap(find_path, [(start_id, end_id) for _ in range(num_pop)])
    print(f'All {num_pop} paths found! \n############### \n')
    return population

def best_survive(population, lengths, fraction):
    sorted_indices = np.argsort(lengths)
    survivors = int(fraction * len(population))
    return [population[i] for i in sorted_indices[:survivors]]

def mutation(path):
    mnode = np.random.randint(1, len(path) - 2)
    neigh = find_neighbors(path[mnode])
    possible_neighbors = [n for n in neigh if G.has_edge(path[mnode - 1], n) and G.has_edge(n, path[mnode + 1])]
    if possible_neighbors:
        path[mnode] = random.choice(possible_neighbors)
    return path

def sex(parent1, parent2):
    common = list(set(parent1) & set(parent2))
    cnode = random.choice(common)
    idx1 = int(random.choice(np.where(parent1 == cnode)[0]))
    idx2 = int(random.choice(np.where(parent2 == cnode)[0]))
    if np.random.randint(0, 2) == 0:
        child = np.concatenate((parent1[:idx1], parent2[idx2:]))
    else:
        child = np.concatenate((parent2[:idx2], parent1[idx1:]))
    if np.random.randint(0, 101) < 10:
        child = mutation(child)
    return child

def save_plot(gen, best_paths):
    generations = len(best_paths)
    x_min, x_max = min(G.nodes[node]["x"] for node in G.nodes), max(G.nodes[node]["x"] for node in G.nodes)
    y_min, y_max = min(G.nodes[node]["y"] for node in G.nodes), max(G.nodes[node]["y"] for node in G.nodes)
    to_plot = best_paths[gen]
    fig, ax = plt.subplots(figsize=(8, 8))
    ox.plot_graph(G, ax=ax, show=False, close=False, bgcolor='white')
    route_x, route_y = zip(*[(G.nodes[node]["x"], G.nodes[node]["y"]) for node in to_plot])
    ax.plot(route_x, route_y, color='red', linewidth=2)
    ax.plot(route_x[0], route_y[0], color='yellow', linewidth=5, marker='o')
    ax.plot(route_x[-1], route_y[-1], color='yellow', linewidth=5, marker='*')
    ax.text(0.05, 0.95, f"Generation {gen+1}/{generations}", transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    ax.set_xlim(x_min, x_max) 
    ax.set_ylim(y_min, y_max)
    plt.savefig(f'gifs/Path_gen{gen}.png')
    plt.close(fig)


def evolution(nodestart, nodeend, generations, fraction, population_size):

    
    best_paths = np.ndarray(generations, dtype=object)
    control = False
    convergency_generations = 10
    best_length = []
    population = populate(nodestart, nodeend, population_size)
    lengths = length_population(population)
    survivors_pop = best_survive(population, lengths, fraction)
    best_length.append(min(lengths))
    padding = 0.005
    nodestartd = G.nodes[nodestart]
    nodeendd = G.nodes[nodeend]
    #x_min, x_max = min([nodestartd['x'], nodeendd['x']]) - padding, max([nodestartd['x'], nodeendd['x']]) + padding
    #y_min, y_max = min([nodestartd['y'], nodeendd['y']]) - padding, max([nodestartd['y'], nodeendd['y']]) + padding
    
    for gen in range(generations):
        start = time.time()
        population = survivors_pop 
        while population_size > len(population):
            parent1, parent2 = random.sample(survivors_pop, 2)
            population.append(sex(parent1, parent2))
        
        lengths = length_population(population)
        survivors_pop = best_survive(population, lengths, fraction)
        best_length.append(min(lengths))
        
        best_path = survivors_pop[0]
        best_paths[gen] = best_path
        
        print('Generation', gen+1, "time for it :", time.time() - start)
        print('Best length', best_length[-1])
        
        if len(best_length) > convergency_generations:
            if all(best_length[-i] == best_length[-i-1] for i in range(1, convergency_generations)):
                control = True
        
        if control: 
            print("Convergency reached")
            break
        #best_path = survivors_pop[0]
    
    print("Start plotting GIF")
    with Pool() as pool:
        pool.starmap(save_plot, [(gen, best_paths) for gen in range(generations)])
    

    print("GIF saved as best_paths.gif")
    return best_path, best_length
