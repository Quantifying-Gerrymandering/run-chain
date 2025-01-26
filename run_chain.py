import geopandas as gpd
from gerrychain import Graph, Partition
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import numpy as np
import datetime


from gerrychain.updaters import Tally, cut_edges

from gerrychain import MarkovChain
from gerrychain.constraints import single_flip_contiguous
from gerrychain.proposals import propose_random_flip
from gerrychain.accept import always_accept

#Load graph
graph = Graph.from_json("./graphs/shapefile_with_islands.json")

# Set up the initial partition object
initial_partition = Partition(
    graph,
    assignment="district_id",
    updaters={
        "population": Tally("CENS_Total", alias="population"),
        "cut_edges": cut_edges,
    }
)

for district, pop in initial_partition["population"].items():
    print(f"District {district}: {pop}")

chain = MarkovChain(
    proposal=propose_random_flip,
    constraints=[single_flip_contiguous],
    accept=always_accept,
    initial_state=initial_partition,
    total_steps=10000
)

for i, partition in enumerate(chain):
    # print some statistics every 1000 steps 
    if i % 1000 == 0:
        print(f"Step {i}: {datetime.datetime.now().strftime("%H:%M:%S")}")

base_colors = cm.tab20.colors * 3  # Repeat to ensure at least 52 colors
shuffled_colors = np.random.permutation(base_colors[:52])  # Randomly shuffle the colors
random_cmap = ListedColormap(shuffled_colors)

# Load gdf
gdf = gpd.read_file("./shapefile_with_islands/shapefile_with_islands.shp")
gdf["district_id"] = list(map(partition.assignment.get, tuple(gdf.index)))
gdf.plot("district_id", 
         cmap=random_cmap,
         edgecolor="black",
         linewidth=0.1)  
plt.show()