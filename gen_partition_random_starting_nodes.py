import geopandas as gpd
from gerrychain import Graph, Partition
import networkx as nx
import matplotlib.pyplot as plt
import random
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import numpy as np
from collections import deque
import heapq

from gerrychain.updaters import Tally, cut_edges

from gerrychain import MarkovChain
from gerrychain.constraints import single_flip_contiguous
from gerrychain.proposals import propose_random_flip
from gerrychain.accept import always_accept

# Load Graph
# graph = Graph.from_json("./graphs/shapefile_with_islands.json")
graph = Graph.from_json("./graphs/dual-graph.json")
print(nx.is_connected(graph))
# print(len(graph))

# Read shapefile
# gdf = gpd.read_file("./shapefile_with_islands/shapefile_with_islands.shp")
gdf = gpd.read_file("./final_shapefile_without_water/final_shapefile_without_water.shp")

PARTITIONS = 52

def grow_districts(graph, num_districts, gdf):
    """
    Randomly select num_districts starting nodes and grow until all nodes are covered.
    """
    
    nodes = list(graph.nodes)
    seeds = random.sample(nodes, num_districts)
    
    district_assignment = {node: None for node in graph.nodes}
    
    boundaries = {i: deque() for i in range(num_districts)}
    
    for i, seed in enumerate(seeds):
        district_assignment[seed] = i
        gdf.loc[seed, 'district_id'] = i
        boundaries[i].append(seed)

    # Expand districts using BFS
    print("Expanding")
    while any(boundaries.values()):
        for district, boundary in boundaries.items():
            if not boundary:
                continue

            node = boundary.popleft()
            
            for neighbor in graph.neighbors(node):
                if district_assignment[neighbor] is None:
                    district_assignment[neighbor] = district
                    gdf.loc[neighbor, 'district_id'] = district
                    boundary.append(neighbor)

    return district_assignment


# district_assignment = grow_districts(graph, PARTITIONS, gdf)
# if gdf['district_id'].isna().sum() > 0:
#     print(f"Warning: There are {gdf['district_id'].isna().sum()} NaN values in 'district_id' column.")
#     print([gdf.index[gdf['district_id'].isna()]])
# print("Unique districts:", np.sort(gdf['district_id'].unique()))

# # Assign the 'district_id' attribute to all nodes in the graph
# # nx.set_node_attributes(graph, district_assignment, name='district_id')
# # Ensure 'district_id' is assigned correctly in gdf
# for node in graph.nodes:
#     graph.nodes[node]["district_id"] = district_assignment[node]
# graph.to_json("./graphs/shapefile_with_islands.json")
# gdf.to_file("./shapefile_with_islands/shapefile_with_islands.shp", driver="ESRI Shapefile")

# # Plot
# base_colors = cm.tab20.colors * 3
# shuffled_colors = np.random.permutation(base_colors[:52])
# random_cmap = ListedColormap(shuffled_colors)
# gdf.plot("district_id", 
#          cmap=random_cmap,
#          edgecolor="black",
#          linewidth=0.1)  
# plt.show()