import geopandas as gpd
from gerrychain import Graph, Partition
import networkx as nx
import matplotlib.pyplot as plt
from functools import partial
import pandas as pd
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import numpy as np
import datetime
from tqdm import tqdm
import math
import random
import scipy

from gerrychain.updaters import Tally, cut_edges

from gerrychain import MarkovChain
from gerrychain.constraints import single_flip_contiguous, contiguous
from gerrychain.accept import always_accept
from gerrychain.proposals import recom, propose_random_flip, propose_chunk_flip
from gerrychain.optimization import SingleMetricOptimizer

def pop_eq(partition):
    """
    Computes sum |component pop. - ideal pop.|
    """
    ideal_pop = sum(partition["population"].values()) / len(partition["population"])
    total_dev = sum(abs(pop - ideal_pop)*abs(pop - ideal_pop) for pop in partition["population"].values())
    return total_dev

def pop_max_dev(partition):
    ideal_pop = sum(partition["population"].values()) / len(partition["population"])
    return max(abs(pop - ideal_pop) for pop in partition["population"].values())

def comp(partition):
    """
    Computes |cut_edges|
    """
    return len(partition["cut_edges"])

def burn_in(hot_duration, f):
    # Wraps a beta function to perform an initial sequence of hot steps (beta = 0)
    def beta_function(step):
        return 0 if step < hot_duration else f(step - hot_duration)
    return beta_function

def linear_beta(hot_duration, cooldown_duration, cold_duration):
    # Beta function
    def beta_function(step):
        if step < hot_duration: 
            return 0
        elif step < hot_duration + cooldown_duration:
            return (step - hot_duration) / cooldown_duration
        else:
            return 1
    return beta_function

def linear_piecewise_beta(hot_phases, cooldown_phases, cold_phases, phase_length):
    total_hot = hot_phases * phase_length
    total_cooldown = cooldown_phases * phase_length
    total_cold = cold_phases * phase_length
    def beta_function(step):
        if step < total_hot: 
            return 0
        elif step < total_hot + total_cooldown:
            discrete_phase = (step - total_hot) // phase_length
            return discrete_phase / cooldown_phases
        else:
            return 1
    return beta_function
 
class MyOptimizer(SingleMetricOptimizer):
    def _simulated_annealing_acceptance_function(self, beta_function, beta_magnitude):
        """
        Custom implementation of the simulated annealing acceptance function with modifications.
        """

        def simulated_annealing_acceptance_function(part):
            if part.parent is None:
                return True

            score_delta = self.score(part) - self.score(part.parent)
            beta = beta_function(part[self._step_indexer])

            if self._maximize:
                score_delta *= -1

            # Custom modification: Avoid math domain errors if score_delta is large
            exponent = -beta * beta_magnitude * score_delta
            if exponent < -700:
                probability = 0.0
            elif exponent > 1:
                probability = 1.0
            else:
                probability = math.exp(exponent)

            return random.random() < probability

        return simulated_annealing_acceptance_function

pop_with_comp = lambda p: sum(pow(abs(pop - np.average(list(p["population"].values()))), 5) for pop in list(p["population"].values()))
pop_outlier_penalty = lambda p: np.var(list(p["population"].values()), ddof=1) * (1 + abs(scipy.stats.kurtosis(list(p["population"].values()))))

def run_chain(partition_file, shapefile, hot_phases, cooldown_phases, cold_phases, phase_length):
    graph = Graph.from_json(partition_file)

    # Set up the initial partition object
    initial_partition = Partition(
        graph,
        assignment="district_id",
        updaters={
            "population": Tally("CENS_Total", alias="population"),
            "cut_edges": cut_edges,
            "area": Tally("area", alias="area"),
        }
    )

    total_steps = (hot_phases + cooldown_phases + cold_phases) * phase_length
    optimizer = MyOptimizer(
        proposal=propose_random_flip,
        constraints=[contiguous],
        initial_state=initial_partition,
        optimization_metric=pop_with_comp,
        maximize=False
    )

    for i, partition in enumerate(
        optimizer.simulated_annealing(
            total_steps,
            # SingleMetricOptimizer.linear_jumpcycle_beta_function(hot_phases, cooldown_phases, cold_phases),
            linear_piecewise_beta(hot_phases, cooldown_phases, cold_phases, phase_length),
            beta_magnitude=0.75,
            with_progress_bar=True
        )
    ):
        yield partition

if __name__ == "__main__":
    hot, cooldown, cold = 10, 100, 1
    phase_len = 1000
    total_steps = (hot + cooldown + cold) * phase_len
    stats_per_iteration = []

    # gdf = gpd.read_file("./shapefiles/shapefile_with_islands/shapefile_with_islands.cpg")
    # colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
    # base_colors = LinearSegmentedColormap.from_list('custom_cmap', colors, N=52)
    # color_samples = base_colors(np.linspace(0, 1, 52))
    # random_cmap = ListedColormap(np.random.permutation(color_samples))

    accepted = 0
    curr_accepted = 0
    accepted_deltas = []
    for i, partition in enumerate(run_chain("./graphs/shapefile_with_islands.json", 
                                            "./shapefiles/shapefile_with_islands/shapefile_with_islands.cpg", 
                                            hot, cooldown, cold, phase_len)):
        ideal_pop = sum(partition["population"].values()) / len(partition["population"])

        pop_array = np.array(list(partition["population"].values()))
        pop_avg = np.mean(pop_array)
        pop_std = np.std(pop_array, ddof=1)
        pop_max_val = np.max(pop_array)
        pop_min_val = np.min(pop_array)
        pop_q1 = np.percentile(pop_array, 25)
        pop_q3 = np.percentile(pop_array, 75)
        if stats_per_iteration and pop_std != stats_per_iteration[-1]["pop_std"]:
            curr_accepted += 1
            if (partition.parent):
                accepted_deltas.append(pop_with_comp(partition) - pop_with_comp(partition.parent))
            
        stats_per_iteration.append({
            'pop_avg': pop_avg,
            'pop_std': pop_std,
            'pop_max': pop_max_val,
            'pop_min': pop_min_val,
            'pop_q1': pop_q1,
            'pop_q3': pop_q3,
            'pop_comp': pop_with_comp(partition),
            'pop_outlier_pentality': pop_outlier_penalty(partition),
            'pop': pop_eq(partition),
            'comp': comp(partition),
            'pop_max_dev': pop_max_dev(partition)
        })

        if i and i % (1000) == 0:
            print(pd.DataFrame([stats_per_iteration[-1]]).to_string(index=False))
            print(comp(partition), pop_eq(partition)/10000000, pop_max_dev(partition)/10)
            print(f"Accepted: {curr_accepted}, Beta: {linear_piecewise_beta(hot, cooldown, cold, phase_len)(i)}")
            print(f"accepted delta_avg: {np.average(accepted_deltas)}")
            curr_accepted = 0
            accepted_deltas = []





# import geopandas as gpd
# from gerrychain import Graph, Partition
# import networkx as nx
# import matplotlib.pyplot as plt
# from functools import partial
# import pandas as pd
# import matplotlib.cm as cm
# from matplotlib.colors import ListedColormap, LinearSegmentedColormap
# import numpy as np
# import datetime
# from tqdm import tqdm
# import random
# import math
# import os
# from gerrychain.updaters import Tally, cut_edges
# from gerrychain import MarkovChain
# from gerrychain.constraints import single_flip_contiguous, contiguous
# from gerrychain.accept import always_accept
# from gerrychain.proposals import recom, propose_random_flip, propose_chunk_flip
# from gerrychain.optimization import SingleMetricOptimizer

# class MetropolisAnnealer:
#     def __init__(self, hot_steps, cooldown_steps, cold_steps, random_seed=None):
#         self.hot_steps = hot_steps
#         self.cooldown_steps = cooldown_steps
#         self.cold_steps = cold_steps
#         self.t = 0
#         self.random = random.Random(random_seed)
        
#         self.phase_stats = {
#             'hot': {'total': 0, 'accepted': 0},
#             'cooldown': {'total': 0, 'accepted': 0},
#             'cold': {'total': 0, 'accepted': 0}
#         }
        
#         self.beta_tracker_pop = {}   # stores {beta_value: {accepted: count, total: count}}
#         self.beta_tracker_comp = {}

#         self.best_max_deviation = float('inf')

#     def get_current_phase(self):
#         if self.t < self.hot_steps:
#             return 'hot'
#         elif self.t < self.hot_steps + self.cooldown_steps:
#             return 'cooldown'
#         else:
#             return 'cold'
        
#     def calculate_max_deviation(self, partition):
#         """Calculate the maximum population deviation from ideal"""
#         ideal_pop = sum(partition["population"].values()) / len(partition["population"])
#         max_pop = max(partition["population"].values())
#         min_pop = min(partition["population"].values())
#         max_deviation = max(
#             (max_pop - ideal_pop) / ideal_pop,
#             (ideal_pop - min_pop) / ideal_pop
#         )
#         return max_deviation
        
#     def get_beta(self, step, beta_start, beta_end):
#         """
#         Modified beta schedule:
#         - Hot phase: accept everything
#         - Cooldown phase (first 75%): increase population beta only
#         - Cooldown phase (last 25%): fixed population beta, increase compactness beta
#         - Cold phase: both betas at beta_end, only accept population improvements
#         """
#         if step < self.hot_steps:
#             return 0  # Hot phase: accept everything
#         elif step < self.hot_steps + self.cooldown_steps:
#             cooldown_progress = step - self.hot_steps
#             three_quarters = 0.75 * self.cooldown_steps
            
#             if cooldown_progress < three_quarters:
#                 # First 75% of cooldown: increase population beta in 8 steps
#                 interval = three_quarters / 8
#                 current_step = int(cooldown_progress / interval)
#                 beta_increment = (beta_end - beta_start) / 8
#                 return beta_start + (current_step * beta_increment)
#             else:
#                 # Last 25% of cooldown: increase compactness beta in 4 steps
#                 remaining_progress = cooldown_progress - three_quarters
#                 interval = (self.cooldown_steps - three_quarters) / 4
#                 current_step = int(remaining_progress / interval)
#                 beta_increment = (beta_end - beta_start) / 4
#                 return beta_start + (current_step * beta_increment)
#         else:
#             return beta_end  # Cold phase

#     def __call__(self, partition):
#         if partition.parent is None:
#             return True
        
#         current_phase = self.get_current_phase()
#         self.phase_stats[current_phase]['total'] += 1
        
#         # Calculate scores
#         pop_new = pop_eq(partition)
#         pop_old = pop_eq(partition.parent)
#         comp_new = comp(partition)
#         comp_old = comp(partition.parent)
        
#         if current_phase == 'hot':
#             # Hot phase: accept everything
#             accept = True
#             beta_pop = 0
#             beta_comp = 0
            
#         elif current_phase == 'cooldown':
#             cooldown_progress = self.t - self.hot_steps
#             three_quarters = 0.75 * self.cooldown_steps
            
#             if cooldown_progress < three_quarters:
#                 # First 75% of cooldown: focus on population
#                 beta_pop = self.get_beta(self.t, 0.4, 0.9)
#                 beta_comp = 0
#                 acceptance_prob = np.exp(-beta_pop * (pop_new - pop_old))
#                 accept = self.random.random() < acceptance_prob
#             else:
#                 # Last 25% of cooldown: fixed population beta, increase compactness
#                 beta_pop = 1.0
#                 beta_comp = self.get_beta(self.t, 0.01, 0.05)
#                 acceptance_prob = np.exp(
#                     -beta_pop * (pop_new - pop_old)
#                     -beta_comp * (comp_new - comp_old)
#                 )
#                 accept = self.random.random() < acceptance_prob
                    
#         else:  # Cold phase
#             beta_pop = 0.9
#             beta_comp = 0.05
#             # Only accept if population gets better
#             # accept = pop_new < pop_old
#             accept = (pop_new < pop_old) and (comp_new < comp_old)
        
#         # Track beta values for statistics
#         beta_pop_key = round(beta_pop, 3)
#         beta_comp_key = round(beta_comp, 3)
        
#         if beta_pop_key not in self.beta_tracker_pop:
#             self.beta_tracker_pop[beta_pop_key] = {'accepted': 0, 'total': 0}
#         if beta_comp_key not in self.beta_tracker_comp:
#             self.beta_tracker_comp[beta_comp_key] = {'accepted': 0, 'total': 0}
        
#         self.beta_tracker_pop[beta_pop_key]['total'] += 1
#         self.beta_tracker_comp[beta_comp_key]['total'] += 1
        
#         if accept:
#             self.beta_tracker_pop[beta_pop_key]['accepted'] += 1
#             self.beta_tracker_comp[beta_comp_key]['accepted'] += 1
#             self.phase_stats[current_phase]['accepted'] += 1
        
#         self.t += 1
#         return accept
    
#     def print_final_stats(self):
#         """Print detailed statistics for all phases and beta values"""
#         print("\n=== Final Statistics ===")
        
#         # Hot Phase Stats
#         print("\nHot Phase Statistics:")
#         hot_total = self.phase_stats['hot']['total']
#         hot_accepted = self.phase_stats['hot']['accepted']
#         print(f"Accepted moves: {hot_accepted}/{hot_total} ({100 * hot_accepted/max(1,hot_total):.1f}%)")
        
#         # Cooldown Phase Stats
#         print("\nCooldown Phase Statistics:")
#         cooldown_total = self.phase_stats['cooldown']['total']
#         cooldown_accepted = self.phase_stats['cooldown']['accepted']
#         print(f"Accepted moves: {cooldown_accepted}/{cooldown_total} ({100 * cooldown_accepted/max(1,cooldown_total):.1f}%)")
        
#         # Cold Phase Stats
#         print("\nCold Phase Statistics:")
#         cold_total = self.phase_stats['cold']['total']
#         cold_accepted = self.phase_stats['cold']['accepted']
#         print(f"Accepted moves: {cold_accepted}/{cold_total} ({100 * cold_accepted/max(1,cold_total):.1f}%)")
        
#         # Population Beta Statistics
#         print("\nPopulation Beta Statistics:")
#         print("Beta Value | Accepted/Total | Percentage")
#         print("-" * 45)
#         for beta in sorted(self.beta_tracker_pop.keys()):
#             stats = self.beta_tracker_pop[beta]
#             acceptance_rate = 100 * stats['accepted'] / max(1, stats['total'])
#             print(f"β_pop {beta:.3f} | {stats['accepted']:>5}/{stats['total']:<5} | {acceptance_rate:>6.1f}%")
        
#         # Compactness Beta Statistics
#         print("\nCompactness Beta Statistics:")
#         print("Beta Value | Accepted/Total | Percentage")
#         print("-" * 45)
#         for beta in sorted(self.beta_tracker_comp.keys()):
#             stats = self.beta_tracker_comp[beta]
#             acceptance_rate = 100 * stats['accepted'] / max(1, stats['total'])
#             print(f"β_comp {beta:.3f} | {stats['accepted']:>5}/{stats['total']:<5} | {acceptance_rate:>6.1f}%")
        
#         # Overall Statistics
#         total_moves = sum(phase['total'] for phase in self.phase_stats.values())
#         total_accepted = sum(phase['accepted'] for phase in self.phase_stats.values())
#         print(f"\nOverall Statistics:")
#         print(f"Total moves: {total_moves}")
#         print(f"Total accepted: {total_accepted}")
#         print(f"Overall acceptance rate: {100 * total_accepted/total_moves:.1f}%")

# # computes std dev cubed
# def pop_eq(partition):
#     ideal_pop = sum(partition["population"].values()) / len(partition["population"])
#     # Should be squared deviation
#     total_dev = sum(abs(pop - ideal_pop)**5 for pop in partition["population"].values())
#     return total_dev

# def comp(partition):
#     return len(partition["cut_edges"])

# def burn_in(hot_duration, f):
#     def beta_function(step):
#         return 0 if step < hot_duration else f(step - hot_duration)
#     return beta_function

# # def piecewise_linear_beta(hot_duration, cooldown_duration, cold_duration):
# #     def beta_function(step):
# #         if step < hot_duration:
# #             return 0
# #         elif step < hot_duration + cooldown_duration:
# #             # Divide the cooldown period into 10 equal intervals.
# #             interval = cooldown_duration / 10
# #             increments = math.floor((step - hot_duration) / interval)
# #             beta = increments * 0.1
# #             return min(beta, 1)
# #         else:
# #             return 1
# #     return beta_function

# def run_chain(partition_file, shapefile, hot_duration, cooldown_duration, cold_duration, random_seed=None):
#     if not partition_file.startswith("./graphs/"):
#         partition_file = os.path.join("graphs", partition_file)
#     graph = Graph.from_json(partition_file)
    
#     initial_partition = Partition(
#         graph,
#         assignment="district_id",
#         updaters={
#             "population": Tally("CENS_Total", alias="population"),
#             "cut_edges": cut_edges,
#             "area": Tally("area", alias="area"),
#         }
#     )
    
#     total_steps = hot_duration + cooldown_duration + cold_duration
    
#     annealer = MetropolisAnnealer(hot_duration, cooldown_duration, cold_duration, random_seed=random_seed)
    
#     chain = MarkovChain(
#         proposal=propose_random_flip,
#         constraints=[single_flip_contiguous],
#         accept=annealer,
#         initial_state=initial_partition,
#         total_steps=total_steps
#     )
    
#     final_partition = None
#     for partition in tqdm(chain, total=total_steps):
#         final_partition = partition
#         yield partition

#     annealer.print_final_stats()

#     # Print population statistics for final partition
#     final_populations = list(final_partition["population"].values())
#     ideal_pop = sum(final_populations) / len(final_populations)
#     pop_array = np.array(final_populations)
#     pop_std = np.std(pop_array, ddof=1)
#     pop_max_val = np.max(pop_array)
#     pop_min_val = np.min(pop_array)
#     pop_q1 = np.percentile(pop_array, 25)
#     pop_median = np.median(pop_array)
#     pop_q3 = np.percentile(pop_array, 75)
#     deviation_max = (pop_max_val - ideal_pop) / ideal_pop * 100
#     deviation_min = (ideal_pop - pop_min_val) / ideal_pop * 100

#     print("\nFinal Population Statistics:")
#     print(f"Ideal Population: {ideal_pop:.2f}")
#     print(f"Max Population: {pop_max_val:.2f} ({deviation_max:.2f}% above ideal)")
#     print(f"Min Population: {pop_min_val:.2f} ({deviation_min:.2f}% below ideal)")
#     print(f"Standard Deviation: {pop_std:.2f}")
#     print(f"25th Percentile: {pop_q1:.2f}")
#     print(f"Median: {pop_median:.2f}")
#     print(f"75th Percentile: {pop_q3:.2f}")

# if __name__ == "__main__":
#     hot, cooldown, cold = 50000, 200000, 50000
#     for partition in run_chain("./graphs/shapefile_with_islands.json",
#                                "shapefiles/shapefile_with_islands/shapefile_with_islands.cpg",
#                                hot, cooldown, cold, random_seed=42):
# #         pass





## plot final partitions
# import json
# import geopandas as gpd
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap
# import matplotlib.cm as cm
# import numpy as np

# if __name__ == "__main__":
#     with open('./chain-final-partitions/random_nodes_final_partition.json', 'r') as f:
#         spanning_tree_final_partition = {int(k): v for k, v in json.load(f).items()}

#     # Load the shapefile
#     gdf = gpd.read_file("./shapefile_with_islands/shapefile_with_islands.shp")

#     # Create colormap
#     base_colors = cm.tab20.colors * 3
#     shuffled_colors = np.random.permutation(base_colors[:52])
#     random_cmap = ListedColormap(shuffled_colors)

#     # Assign districts to GeoDataFrame
#     gdf['district_id'] = [spanning_tree_final_partition.get(node) for node in gdf.index]

#     # Plot
#     plt.figure(figsize=(10, 10))
#     gdf.plot(column='district_id', cmap=random_cmap, edgecolor="black", linewidth=0.1)
#     plt.title('Final Random Nodes Partition')
#     plt.axis('off')
#     plt.savefig('./chain-final-partitions/random_nodes_final_partition.png')
#     plt.show()