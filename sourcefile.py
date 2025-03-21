# import important packages
import osmnx as ox
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import folium


place = 'Copenhagen Municipality, Denmark'
print('Loading graph data...')
G = ox.graph_from_place(place, network_type='walk')
print('Graph data loaded.')

nodes, edges = ox.graph_to_gdfs(G)

starting = 'Jagtvej 155A, Copenhagen, Denmark'
latstart, lonstart = ox.geocoder.geocode(starting)

ending = 'Karen Blixens Vej 8, Copenhagen, Denmark'
latend, lonend = ox.geocoder.geocode(ending)

generations = 300
pop_size = 7000
mutation_ratio = 0.1

nodestart = ox.distance.nearest_nodes(G, lonstart, latstart)
nodeend = ox.distance.nearest_nodes(G, lonend, latend)


def printinfo(verbose, gen, best_length, generations, start):
    '''
    Prints information about the state of the genetic algorithm.

    Arguments:
        - verbose:   True = prints at every generation
                     False = prints only every 50 generations
                     Fancy = uses bar loading effect
        - gen (int): number of the generation
        - best_length (float): best path length of the current generation
        - generations (int): total number of genereations
        - start (float): starting time for each generation

    Returns:
    
    '''
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



"""
------------------------------------------------------------------------------------------------------------------------------------

Plotting section:
The first two functions of this section where used for troubleshooting, the last one (save_plot) can be used to generate
a gif of the best path per generation.

------------------------------------------------------------------------------------------------------------------------------------
"""

def plot_path(path):
    '''
    Plots a single path.

    Arguments:
        - path (np.array): array of the node IDs

    Returns:
        
    '''

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
    plt.savefig('best_path.png')
    plt.show()


def plot_multi_path( paths):
    '''
    Plots several paths.

    Arguments:
        - paths (list of np.arrays): list of arrays with the node IDs

    Returns:
        
    '''
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

def save_plot(gen, best_paths):
    '''

    Saves a single image, to create a gif of the evolution of the best path per generation.

    Arguments:
        - gen (int): number of the generation
        - best_paths (np.array): array of the IDs of the nodes

    Returns:
        
    '''
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

def create_map_html(nodestart, nodeend, path):
    '''
    
    Creates a HTML map (named map_with_paths.html) with the path given and the theoretical best path. 

    Arguments:
        - nodestart (int): starting node ID
        - nodeend (int): ending node ID
        - path (np.array): array of the node IDs of the path

    Returns:   

    
    '''
    start_coords = (G.nodes[nodestart]['y'], G.nodes[nodestart]['x'])
    end_coords = (G.nodes[nodeend]['y'], G.nodes[nodeend]['x'])

    theoretical_path = ox.routing.shortest_path(G, nodestart, nodeend)
    theoretical_coords = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in theoretical_path]

    custom_path = path
    custom_coords = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in custom_path]

    m = folium.Map(location=start_coords, zoom_start=14)

    folium.Marker(start_coords, popup="Start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker(end_coords, popup="End", icon=folium.Icon(color="red")).add_to(m)

    folium.PolyLine(theoretical_coords, color="blue", weight=5, opacity=0.7, tooltip="Theoretical Best Path").add_to(m)

    folium.PolyLine(custom_coords, color="violet", weight=5, opacity=0.7, tooltip="Found Best Path").add_to(m)

    m.save("map_with_paths.html")



"""
------------------------------------------------------------------------------------------------------------------------------------

Path calculation & length related section

------------------------------------------------------------------------------------------------------------------------------------
"""


def path_length(path):
    '''

    Calculates the (real) length of the given path

    Arguments:
        - path (np.array): array of the node IDs of the path

    Returns:
        - length (float): total length of the path

    '''
    return sum(G[u][v][0]["length"] for u, v in zip(path[:-1], path[1:]))


def length_population(pop):
    '''
    Calculates the lengths of a population (collection of paths) by parallelisation.

    Arguments:
        - pop (list of np.arrays): population (collection of paths), each path is a np.array of the node IDs.

    Returns:
        - lengths (np.array): array of length equal to pop with the lenght of each one of the paths

    '''

    lengths = [path_length(p) for p in pop]
    return np.array(lengths)


def distance_to_prob(dist):
    ''' 
    This function returns the probability of a path given the distance.

    Arguments:
        - dist: distance of the path

    Returns:
        - prob: normalized probability of the path

    '''
    prob = 1/np.array(dist)
    return prob / np.sum(prob)


def define_search_area(start_id, end_id, margin=0.005):
    '''
    Defines an area around the starting and ending points. Discards the nodes that make up dead ends by checking what nodes have 
    only one neighbor or neighbors outside of the permited area.

    Arguments:
        - start_id (int): starting node ID
        - end_id (int): ending node ID
        - margin (float): marging around the points

    Returns:
        - area_nodes (np.array): IDs of the permited nodes

    '''
    start_x, start_y = G.nodes[start_id]['x'], G.nodes[start_id]['y']
    end_x, end_y = G.nodes[end_id]['x'], G.nodes[end_id]['y']

    x_min, x_max = min(start_x, end_x) - margin, max(start_x, end_x) + margin
    y_min, y_max = min(start_y, end_y) - margin, max(start_y, end_y) + margin

    area_nodes = {node for node in G.nodes if x_min <= G.nodes[node]['x'] <= x_max and y_min <= G.nodes[node]['y'] <= y_max}

    dead_ends = {node for node in area_nodes if node != start_id and node != end_id and len(list(G.neighbors(node))) == 1}

    while dead_ends:
        area_nodes -= dead_ends  
        new_dead_ends = {node for node in area_nodes if node != start_id and node != end_id and len(set(G.neighbors(node)) & area_nodes) == 1}  
        dead_ends = new_dead_ends            
                
    return np.array(list(area_nodes))


def assign_unique_priorities(nodes):
    '''

    Assign priority for all the nodes given. Priorities go from 0 to len(nodes) - 1. Used to construct the path.

    Arguments:
        - nodes (np.array): node IDs (generally of the permited area)

    Returns:
        - dict (dict): dictionary relating the node and priority.

    '''
    priority_values = list(range(len(nodes)))  
    random.shuffle(priority_values) 
    return dict(zip(nodes, priority_values)) 

def find_path( start_id, end_id, area_nodes):
    '''
    
    Finds a path from start_id to end_id with area around nodes area_nodes. Uses assign_unique_priorities to give each 
    node a priority and chooses the path such that the highest priority is always chosen. 

    Arguments:
        - start_id (int): starting node ID
        - end_id (int): ending node ID
        - area_nodes (np.array): IDs of the permited nodes

    Returns:
        - path (np.array): array with node IDs for a path from start_id to end_id

    '''
    visited = set()
    path = [start_id]
    priorities = assign_unique_priorities(area_nodes)

    while path:
        current = path[-1]
        
        if current == end_id:
            return np.array(path)

        visited.add(current)
        neighbors = [n for n in G.neighbors(current) if n not in visited and n in area_nodes]
        
        if not neighbors:
            path.pop()
            continue

        next_node = max(neighbors, key=lambda n: priorities[n])

        path.append(next_node)

    return np.array([])  


def populate(start_id, end_id, num_pop):
    '''
    
    Creates num_pop number of paths from start_id to end_id.

    Arguments:
        - start_id (int): starting node ID
        - end_id (int): ending node ID
        - num_pop (int): number of paths

    Returns:
        - population (list of np.arrays): list of num_pop number of paths

    '''

    area = define_search_area(start_id, end_id)

    population = [find_path(start_id, end_id, area) for _ in range(num_pop)]
    print(f"All {num_pop} paths found! \n############### \n")
    return population


"""
------------------------------------------------------------------------------------------------------------------------------------

Genetic algorithm section

------------------------------------------------------------------------------------------------------------------------------------
"""

def mutation(path):
    '''

    Implements a mutation operator. This mutation operator takes two random nodes in the path and calculates a new
    path connecting the two selected nodes.

    Arguments:
        - path (np.array): array of the node IDs of the path

    Returns:
        mutate_path (np.array): array of the node IDs with the mutated section

    '''
    node1ind, node2ind = sorted(np.random.choice(range(1, len(path) - 1), size=2, replace=False))
    area_mut = define_search_area(path[node1ind], path[node2ind])
    mutate_section = find_path(path[node1ind], path[node2ind], area_mut)

    if len(mutate_section) < 2:  
        return path  

    mutate_path = np.concatenate((path[:node1ind],mutate_section, path[node2ind+1:]))

    return mutate_path


def roulette_wheel_selection(population, lengths):
    '''
    
    Implements roulette wheel selection. Assigns a probability to each path in population and selects two random paths.

    Arguments:
        - population (list of np.arrays): list of paths
        - lengths (np.array): array of the lengths of the paths in population

    Returns:
        - rand1, rand2 (int, int): IDs of two randomly selected path with weights defined by length

    '''
    prob = distance_to_prob(lengths)
    random_indexs = np.random.choice(range(len(population)), 2, p=prob)
    return population[random_indexs[0]], population[random_indexs[1]]

def sex(parent1, parent2, mutation_ratio):
    '''
    
    Implements the crossover operator (aka sex) by taking two paths, selecting a random common node between the two 
    and creating the child such that it gets one part of parent1 and the other of parent2.

    Arguments:
        - parent1 (np.array): array of the ID nodes of the path
        - parent2 (np.array): array of the ID nodes of the path
        - mutation_ratio (float): probability of mutation happening

    Returns:
        - child (np.array): array of the ID nodes of the child path
    
    '''
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
    if np.random.rand() < mutation_ratio:
        child = mutation(child)
    return child


def evolution(nodestart,nodeend,generations,population_size,mutation_ratio, theorcial_distance,gif=False,verbose=False,):
    '''
    
    Iterates a maximum of generations number of times to find a minimum length path from nodestart to nodeends (IDs) after generating a population
    of size population_size. This is done by a genetic algorithm with mutation probability equal to mutation_ratio.

    Arguments:
        - nodestart (int): starting node ID
        - nodeend (int): ending node ID
        - population_size (int): number of paths that will make up the population
        - mutation_ratio (float): probability (0.0-1.0) of a mutation happening
        - gif (bool): if True images will be saved to create a gif
        - verbose:   True = prints at every generation
                     False = prints only every 50 generations
                     Fancy = uses bar loading effect
    
    Returns:
        - best_path (np.array): array of node IDs of the shortest path found
        - best_length (float): total length of best_path

    '''
    best_paths = []
    best_length = []

    population = populate(nodestart, nodeend, population_size)
    lengths = [path_length(p) for p in population]

    for gen in range(generations):
        start = time.time()

        next_gen = [population[0]]
        prob = distance_to_prob(lengths)
        # Generating next generation serially
        for _ in range(len(population) - 1):
            p1, p2 = roulette_wheel_selection(population, prob)
            child = sex(p1, p2, mutation_ratio)
            next_gen.append(child)

        population = next_gen
        lengths = [path_length(p) for p in population]

        best_path = population[np.argmin(lengths)]
        best_length.append(min(lengths))
        best_paths.append(best_path)

        printinfo(verbose, gen, best_length, generations, start)

        if gen > 10 and all(best_length[-i] == best_length[-i - 1] for i in range(1, 10)):
            print("\nEarly stop due to convergence.")
            break

    print(f"\nBest length: {best_length[-1]}")
    return best_path, best_length

print("Finished importing")



theorical_distance = ox.routing.shortest_path(G, nodestart, nodeend)


# Run the simulation
print("Start simulation")

start = time.time()
best_path, best_lengths = evolution(
    nodestart,
    nodeend,
    generations=generations,
    population_size=pop_size,
    mutation_ratio=mutation_ratio,
    verbose='Fancy',
    theorcial_distance=path_length(theorical_distance),

)

end = time.time()
print("Simulation finished")
print("Time elapsed: ", end - start)
print("Best path: ", best_lengths[-1], 'after', generations, 'generations')
print("Theoretical best distance: ", path_length(theorical_distance))

route_x, route_y = zip(*[(G.nodes[node]["x"], G.nodes[node]["y"]) for node in best_path])
route_xr, route_yr = zip(*[(G.nodes[node]["x"], G.nodes[node]["y"]) for node in theorical_distance])

padding = 0.001 
x_min, x_max = min(route_x) - padding, max(route_x) + padding
y_min, y_max = min(route_y) - padding, max(route_y) + padding

fig, ax = plt.subplots()

ox.plot_graph(G, ax=ax, show=False, close=False, bgcolor='white')
ax.plot(route_x[0], route_y[0], color='yellow', linewidth=5, marker='o')
ax.plot(route_x[-1], route_y[-1], color='yellow', linewidth=5, marker='*')
ax.plot(route_xr, route_yr, color='green', linewidth=3, label = 'Theoretical best path')
ax.plot(route_x, route_y, color='red', label = 'Best path found')
ax.set_xlim(x_min, x_max) 
ax.set_ylim(y_min, y_max)
plt.savefig('best_path.png')


create_map_html(nodestart, nodeend, best_path)





