import numpy as np
from gerrychain import Partition, Graph
from gerrychain.updaters import Tally, cut_edges
import matplotlib.pyplot as plt
import geopandas as gpd
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from run_chain_phases import run_chain, pop_with_comp, comp
import random
from gen_partition_spanning_tree import random_spanning_tree, split_spanning_tree, generate_spanning_tree_partition
from gen_partition_random_starting_nodes import grow_districts
import networkx as nx
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
import pandas as pd
import json

def gelman_rubin(chains_data, metric="pop_max_dev"):
    if metric == "comp":
        phase1_data = [chain[f"phase1_comp"] for chain in chains_data]
        phase2_data = [chain[f"phase2_comp_scores"] for chain in chains_data]
    else:
        phase1_data = [chain[f"phase1_{metric}"] for chain in chains_data]
        phase2_data = [chain[f"phase2_{metric}"] for chain in chains_data]
    
    combined_data = [phase1 + phase2 for phase1, phase2 in zip(phase1_data, phase2_data)]
    
    m = len(combined_data)
    
    phase_length = 1000
    burn_in_steps = []
    for i in range(m):
        burn_in = 10 * phase_length  # hot phase only
        burn_in_steps.append(burn_in)

    # remove burn-in steps
    chains_data = [chain[burn:] for chain, burn in zip(combined_data, burn_in_steps)]
    n = min(len(chain) for chain in chains_data)  # Use shortest chain length
    
    chains_data = [chain[:n] for chain in chains_data]

    # chain means
    chain_means = np.array([np.mean(chain) for chain in chains_data])
    overall_mean = np.mean(chain_means)

    # between-chain variance
    B = (n / (m - 1)) * np.sum((chain_means - overall_mean) ** 2)

    # within-chain variance
    chain_vars = np.array([np.var(chain, ddof=1) for chain in chains_data])
    W = np.mean(chain_vars)

    # estimated variance
    var_plus = ((n - 1) / n) * W + (1 / n) * B

    # r-hat
    R_hat = np.sqrt(var_plus / W)
    return R_hat

def plot_chains(chains_data, pop_dev_rhat, pop_score_rhat, comp_score_rhat):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 8))
    
    # # Plot population deviation
    # for i, chain in enumerate(chains_data):
    #     ax1.plot(chain['phase1_pop_max_dev'] + chain['phase2_pop_max_dev'], 
    #             label=f'Chain {i}', alpha=0.7)
    # ax1.set_title(f'Population Deviation Across Chains\nR-hat = {pop_dev_rhat:.4f}')
    # ax1.set_xlabel('Steps')
    # ax1.set_ylabel('Population Deviation')
    # ax1.legend()
    # ax1.grid(True, alpha=0.3)
    
    # plot population scores
    for i, chain in enumerate(chains_data):
        ax1.plot(chain['phase1_pop_scores'] + chain['phase2_pop_scores'], 
                label=f'Chain {i}', alpha=0.7)
    ax1.set_title(f'Population Score Across Chains\nR-hat = {pop_score_rhat:.4f}')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Population Score')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # plot compactness scores
    for i, chain in enumerate(chains_data):
        ax2.plot(chain['phase1_comp'] + chain['phase2_comp_scores'], 
                label=f'Chain {i}', alpha=0.7)
    ax2.set_title(f'Compactness Score Across Chains\nR-hat = {comp_score_rhat:.4f}')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Compactness Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("./chain-final-stats/all-chains-stats.png")
    plt.show()

def calculate_partition_similarity(partition1, partition2):
    districts1 = {}
    districts2 = {}
    
    unique_districts1 = sorted(set(partition1.values()))
    unique_districts2 = sorted(set(partition2.values()))
    
    district_to_idx1 = {dist: i for i, dist in enumerate(unique_districts1)}
    district_to_idx2 = {dist: i for i, dist in enumerate(unique_districts2)}
    
    # group nodes by district
    for node, district in partition1.items():
        idx = district_to_idx1[district]
        districts1.setdefault(idx, set()).add(node)
    for node, district in partition2.items():
        idx = district_to_idx2[district]
        districts2.setdefault(idx, set()).add(node)
    
    n_districts = len(unique_districts1)
    overlap_matrix = np.zeros((n_districts, n_districts))
    
    for i in range(n_districts):
        for j in range(n_districts):
            if i in districts1 and j in districts2:
                overlap_matrix[i][j] = len(districts1[i].intersection(districts2[j]))
    
    row_ind, col_ind = linear_sum_assignment(-overlap_matrix)
    total_nodes = sum(len(nodes) for nodes in districts1.values())
    matching_nodes = sum(overlap_matrix[i][j] for i, j in zip(row_ind, col_ind))
    return matching_nodes / total_nodes

def analyze_district_populations(data):
    # use the last iteration
    if isinstance(data, list):
        partition = data[-1]
    else:
        partition = data

    populations = list(partition["population"].values())
    ideal_pop = sum(populations) / len(populations)
    max_pop = max(populations)
    min_pop = min(populations)
    mean_pop = np.mean(populations)
    std_pop = np.std(populations)
    max_deviation = max(abs(pop - ideal_pop) for pop in populations)
    relative_deviation = max_deviation / ideal_pop * 100
    percentiles = np.percentile(populations, [25, 50, 75])
    
    print(f"\nDistrict Population Statistics (Final Iteration):")
    print(f"Ideal Population: {ideal_pop:,.0f}")
    print(f"Max Population: {max_pop:,.0f} ({(max_pop/ideal_pop - 1)*100:,.2f}% from ideal)")
    print(f"Min Population: {min_pop:,.0f} ({(min_pop/ideal_pop - 1)*100:,.2f}% from ideal)")
    print(f"Standard Deviation: {std_pop:,.0f}")
    print(f"Maximum Deviation: {max_deviation:,.0f} ({relative_deviation:.2f}%)")
    print(f"25th Percentile: {percentiles[0]:,.0f}")
    print(f"Median: {percentiles[1]:,.0f}")
    print(f"75th Percentile: {percentiles[2]:,.0f}")
    
    return {
        'ideal_pop': ideal_pop,
        'max_pop': max_pop,
        'min_pop': min_pop,
        'mean_pop': mean_pop,
        'std_pop': std_pop,
        'max_deviation': max_deviation,
        'relative_deviation': relative_deviation,
        'percentiles': percentiles
    }


def plot_results(chains_data, pop_dev_rhat, pop_score_rhat, comp_score_rhat, 
                initial_assignments, final_partitions, graph_file, shapefile):   
    gdf = gpd.read_file(f"./shapefile_with_islands/{shapefile}")
    base_colors = cm.tab20.colors * 3
    shuffled_colors = np.random.permutation(base_colors[:52])
    random_cmap = ListedColormap(shuffled_colors)
    
    fig, axs = plt.subplots(2, len(final_partitions), figsize=(18, 8))
    
    plt.suptitle(f'Population Deviation R-hat: {pop_dev_rhat:.4f}\n' +
                 f'Population R-hat: {pop_score_rhat:.4f}\n' +
                 f'Compactness R-hat: {comp_score_rhat:.4f}',
                 y=1.2)
    
    # plot initial assignments (first row)
    for i, assignment in enumerate(initial_assignments):
        assignment = {int(k) if isinstance(k, str) else k: v for k, v in assignment.items()}
        district_assignments = [assignment.get(idx, -1) for idx in gdf.index]
        gdf['district_i'] = district_assignments
        ax = axs[0, i]
        gdf.plot(column='district_i', cmap=random_cmap, edgecolor="black", linewidth=0.1, ax=ax)
        ax.set_title(f'Chain {i} Initial Districts')
        ax.axis('off')
    
    # plot final assignments (second row)
    for i, assignment in enumerate(final_partitions):
        assignment = {int(k) if isinstance(k, str) else k: v for k, v in assignment.items()}
        district_assignments = [assignment.get(idx, -1) for idx in gdf.index]
        gdf['district_i'] = district_assignments
        ax = axs[1, i]
        gdf.plot(column='district_i', cmap=random_cmap, edgecolor="black", linewidth=0.1, ax=ax)
        ax.set_title(f'Chain {i} Final Districts')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig("./chain-final-stats/all-chains-initial-final-partitions.png")
    plt.show()
    
    # initial vs final similarities for each chain
    n_chains = len(final_partitions)
    print("\nInitial vs Final District Plan Similarities (per chain):")
    for i in range(n_chains):
        sim = calculate_partition_similarity(initial_assignments[i], final_partitions[i])
        print(f"Chain {i} initial vs final similarity: {sim:.3f}")

    # final district plan similarities
    print("\nFinal District Plan Similarities:")
    similarity_matrix = np.zeros((n_chains, n_chains))
    for i in range(n_chains):
        for j in range(n_chains):
            similarity_matrix[i][j] = calculate_partition_similarity(final_partitions[i], final_partitions[j])
    for i in range(n_chains):
        for j in range(i+1, n_chains):
            print(f"Chain {i} vs Chain {j} final similarity: {similarity_matrix[i,j]:.3f}")
    
    print("\nAnalyzing final district populations for each chain:")

    graph = Graph.from_json(f"./graphs/{graph_file}")
    for i, assignment in enumerate(final_partitions):
        print(f"\nChain {i} Statistics:")
        partition = Partition(
            graph,
            assignment=assignment,
            updaters={
                "population": Tally("CENS_Total", alias="population"),
                "cut_edges": cut_edges,
            }
        )
        analyze_district_populations(partition)
    
    # initial vs final similarities for each chain
    n_chains = len(final_partitions)
    print("\nInitial vs Final District Plan Similarities (per chain):")
    for i in range(n_chains):
        sim = calculate_partition_similarity(initial_assignments[i], final_partitions[i])
        print(f"Chain {i} initial vs final similarity: {sim:.3f}")

    # final district plan similarities
    print("\nFinal District Plan Similarities:")
    similarity_matrix = np.zeros((n_chains, n_chains))
    for i in range(n_chains):
        for j in range(n_chains):
            similarity_matrix[i][j] = calculate_partition_similarity(final_partitions[i], final_partitions[j])
    for i in range(n_chains):
        for j in range(i+1, n_chains):
            print(f"Chain {i} vs Chain {j} final similarity: {similarity_matrix[i,j]:.3f}")
    
    print("\nAnalyzing final district populations for each chain:")

    graph = Graph.from_json(f"./graphs/{graph_file}")
    for i, assignment in enumerate(final_partitions):
        print(f"\nChain {i} Statistics:")
        partition = Partition(
            graph,
            assignment=assignment,
            updaters={
                "population": Tally("CENS_Total", alias="population"),
                "cut_edges": cut_edges,
            }
        )
        analyze_district_populations(partition)

# run_chain with simulated annealing
def run_chain_and_collect_scores(chain_num, partition_file, shapefile, phase_length, initial_assignment=None):
    final_partition, final_stats = run_chain(
        chain_num=chain_num,
        partition_file=partition_file,
        phase_length=phase_length,
        initial_assignment=initial_assignment
    )

    if final_partition is None:
        print("Error: run_chain returned None")
        return None, None, None, None

    scores = final_stats['pop_score'].tolist()
    
    initial_assignments = initial_assignment
    
    # convert stats to list of dictionaries
    stats_per_iteration = final_stats.to_dict('records')

    return final_partition, scores, stats_per_iteration, initial_assignments

def assess_convergence(partition_file, shapefile, num_chains=3, phase_len=1000, threshold=1.1):
    """
    Runs three chains with different initial partitions:
    1. Random spanning tree
    2. Random starting nodes
    3. Current districting (from file)
    Assesses convergence via gelman-rubin diagnostic.
    """
    chains_data = []
    final_partitions = []
    all_stats = []
    
    graph = Graph.from_json(partition_file)
    gdf = gpd.read_file(f"./shapefile_with_islands/{shapefile}")
    
    num_districts = len(set(graph.nodes[node]["district_id"] for node in graph.nodes()))
    
    # generate three different initial partitions
    # spanning_tree_partition = generate_spanning_tree_partition(graph, num_districts)
    # random_nodes_partition = grow_districts(graph, num_districts, gdf)
    # current_partition = {node: graph.nodes[node]["district_id"] for node in graph.nodes()}

    # read initial partitions from json files
    with open('./chain-initial-partitions/spanning_tree_initial_partition.json', 'r') as f:
        spanning_tree_partition = {int(k): v for k, v in json.load(f).items()}
    with open('./chain-initial-partitions/random_nodes_initial_partition.json', 'r') as f:
        random_nodes_partition = {int(k): v for k, v in json.load(f).items()}
    with open('./chain-initial-partitions/current_districting_initial_partition.json', 'r') as f:
        current_partition = {int(k): v for k, v in json.load(f).items()}
    
    # print("\nVerifying initial partitions:")
    # print(f"Spanning tree partition type: {type(spanning_tree_partition)}")
    # print(f"Random nodes partition type: {type(random_nodes_partition)}")
    # print(f"Current partition type: {type(current_partition)}")
    
    initial_partitions = [
        spanning_tree_partition,
        random_nodes_partition,
        current_partition
    ]
    
    for i in range(num_chains):
        print(f"\nRunning chain {i} with different initial partition")
        
        # get number of districts
        if isinstance(initial_partitions[i], dict):
            num_districts = len(set(initial_partitions[i].values()))
        else:
            num_districts = len(initial_partitions[i])
        print(f"Initial partition {i} has {num_districts} districts")
        
        best_partition, scores, stats, original_assignment = run_chain_and_collect_scores(
            i,
            partition_file, 
            shapefile,
            phase_len,
            initial_assignment=initial_partitions[i]
        )
        chains_data.append(scores)
        final_partitions.append(best_partition.assignment)
        all_stats.append(stats)
        initial_assignments.append(original_assignment)

    chains_data = []
    for i, name in enumerate(['spanning_tree', 'random_nodes', 'current_districting']):
        with open(f'./chain-final-scores/{name}_scores.json', 'r') as f:
            chains_data.append(json.load(f))

    pop_dev_rhat = gelman_rubin(chains_data, metric="pop_max_dev")
    pop_score_rhat = gelman_rubin(chains_data, metric="pop_scores")
    comp_score_rhat = gelman_rubin(chains_data, metric="comp_scores")
    
    print(f"\nR-hat statistics:")
    print(f"Population deviation: {pop_dev_rhat:.4f}")
    print(f"Population score: {pop_score_rhat:.4f}")
    print(f"Compactness score: {comp_score_rhat:.4f}")
    
    # burn_in_steps = hot_duration * phase_len
    # r_hat = gelman_rubin(chains_data, burn_in_steps=burn_in_steps)
    return initial_partitions,pop_dev_rhat, pop_score_rhat, comp_score_rhat, chains_data, final_partitions, all_stats, initial_assignments

if __name__ == "__main__":
    partition_file = "./graphs/shapefile_with_islands.json"
    shapefile = "shapefile_with_islands.shp"
    
    # set annealing schedule parameters
    # hot, cooldown, cold = 10, 100, 40
    phase_len = 1000
    # total_steps = hot + cooldown + cold
    
    # check convergence
    initial_partitions, pop_dev_rhat, pop_score_rhat, comp_score_rhat, chains_data, final_partitions, all_stats, initial_assignments = assess_convergence(
        partition_file, shapefile,
        num_chains=3,
        phase_len=phase_len,
        threshold=1.1
    )
    
    print(f"\nR-hat statistics:")
    print(f"Population deviation: {pop_dev_rhat:.4f}")
    print(f"Population score: {pop_score_rhat:.4f}")
    print(f"Compactness score: {comp_score_rhat:.4f}")

    plot_chains(chains_data, pop_dev_rhat, pop_score_rhat, comp_score_rhat)
    plot_results(chains_data, pop_dev_rhat, pop_score_rhat, comp_score_rhat, 
                initial_partitions, final_partitions, partition_file, shapefile)

# if __name__ == "__main__":
#     partition_file = "./graphs/shapefile_with_islands.json"
#     shapefile = "shapefile_with_islands.shp"

#     chains_data = []
#     for i, name in enumerate(['spanning_tree', 'random_nodes', 'current_districting']):
#         with open(f'./chain-final-scores/{name}_scores.json', 'r') as f:
#             chains_data.append(json.load(f))

#     initial_partitions = []
#     final_partitions = []
#     for i, name in enumerate(['spanning_tree', 'random_nodes', 'current_districting']):
#         with open(f'./chain-initial-partitions/{name}_initial_partition.json', 'r') as f:
#             initial_partitions.append(json.load(f))
#         with open(f'./chain-final-partitions/{name}_final_partition.json', 'r') as f:
#             final_partitions.append(json.load(f))

#     pop_dev_rhat = gelman_rubin(chains_data, metric="pop_max_dev")
#     pop_score_rhat = gelman_rubin(chains_data, metric="pop_scores")
#     comp_score_rhat = gelman_rubin(chains_data, metric="comp")
    
    # print(f"\nR-hat statistics:")
    # print(f"Population deviation: {pop_dev_rhat:.4f}")
    # print(f"Population score: {pop_score_rhat:.4f}")
    # print(f"Compactness score: {comp_score_rhat:.4f}")

    # plot_chains(chains_data, pop_dev_rhat, pop_score_rhat, comp_score_rhat)
    # plot_results(chains_data, pop_dev_rhat, pop_score_rhat, comp_score_rhat, 
    #             initial_partitions, final_partitions, partition_file, shapefile)