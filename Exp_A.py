import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
import networkx as nx
import collections
import random
import numpy as np
import matplotlib.pyplot as plt
import os
from itertools import combinations
import pandas as pd

from plotting import visualize_results

def weisfeiler_lehman_hash(graph):
    """Compute WL hash for a graph"""
    colors = {node: data.get('label', 0) for node, data in graph.nodes(data=True)}
    
    for _ in range(len(graph.nodes())): # Iterate enough times for convergence
        new_colors = {}
        for node in graph.nodes():
            neighbor_colors = sorted([colors[nbr] for nbr in graph.neighbors(node)])
            signature = (colors[node], tuple(neighbor_colors))
            new_colors[node] = hash(signature)

        if new_colors == colors:
            break
        colors = new_colors

    canonical_hash = str(sorted(colors.values()))
    return canonical_hash

def find_isomorphic_groups_from_pyg(dataset):
    """Find isomorphic groups in a PyG dataset"""
    hashes = collections.defaultdict(list)

    for i, data in enumerate(dataset):
        g = to_networkx(data, node_attrs=['x'])
        nx.set_node_attributes(g, {j: int(x[0]) for j, x in enumerate(data.x)}, 'label')    
        
        if len(g) > 0:
            h = weisfeiler_lehman_hash(g)
            hashes[h].append(i)

    isomorphic_groups = {h: indices for h, indices in hashes.items() if len(indices) > 1}
    return isomorphic_groups

def perturb_graph(graph):
    """
    Perturb a graph by adding or removing exactly one edge
    Args:
        graph: NetworkX graph
    Returns:
        perturbed_graph: A copy of the graph with one edge added or removed
        operation: Description of the operation performed
    """
    g_copy = graph.copy()

    # Decide operation type
    possible_ops = []
    n_nodes = len(g_copy.nodes())
    max_edges = n_nodes * (n_nodes - 1) // 2
    if len(g_copy.edges()) > 0:
        possible_ops.append('remove')
    if len(g_copy.edges()) < max_edges:
        possible_ops.append('add')
    if not possible_ops:
        return g_copy, "null_graph"

    operation = random.choice(possible_ops)
   
    if operation == 'add': # Add a random non-existent edge
        nodes = list(g_copy.nodes())
        non_edges = list(nx.non_edges(g_copy))
        
        if non_edges:
            edge_to_add = random.choice(non_edges)
            g_copy.add_edge(*edge_to_add)
            return g_copy, f"added_edge_{edge_to_add}"
        else:
            return g_copy, "no_edge_to_add"
    
    else:       # Remove a random existing edge
        edges = list(g_copy.edges())
        if edges:
            edge_to_remove = random.choice(edges)
            g_copy.remove_edge(*edge_to_remove)
            return g_copy, f"removed_edge_{edge_to_remove}"
        else:
            return g_copy, "no_edge_to_remove"




def run_noise_experiment(dataset, num_perturbations=10):
    """
    Run the noise robustness experiment
    
    Args:
        dataset: PyG dataset
        num_perturbations: Number of perturbed copies per graph
    
    Returns:
        results: Dictionary with experiment results
    """
    print("Running noise robustness experiment...")
    
    # Step 1: Find isomorphic groups
    print("1. Finding isomorphic groups...")
    isomorphic_groups = find_isomorphic_groups_from_pyg(dataset)
    
    if not isomorphic_groups:
        print("No isomorphic groups found!")
        return None
    
    # Step 2: Select the largest isomorphic group
    largest_group = max(isomorphic_groups.values(), key=len)
    print(f"2. Selected largest isomorphic group with {len(largest_group)} graphs: {largest_group}")
    
    # Step 3: Convert selected graphs to nx
    original_graphs = []
    for idx in largest_group:
        data = dataset[idx]
        g = to_networkx(data, node_attrs=['x'])
        nx.set_node_attributes(g, {j: int(x[0]) for j, x in enumerate(data.x)}, 'label')
        original_graphs.append(g)
    
    # Step 4: Create perturbed copies
    print(f"3. Creating {num_perturbations} perturbed copies for each graph...")
    all_graphs = []
    graph_metadata = []
    
    # original graphs
    for i, g in enumerate(original_graphs):
        all_graphs.append(g)
        graph_metadata.append({
            'original_id': largest_group[i],
            'graph_type': 'original',
            'perturbation_id': -1,
            'operation': 'none'
        })
    
    # perturbed graphs
    perturbation_success = []
    for i, g in enumerate(original_graphs):
        for j in range(num_perturbations):
            perturbed_g, operation = perturb_graph(g)
            all_graphs.append(perturbed_g)
            graph_metadata.append({
                'original_id': largest_group[i],
                'graph_type': 'perturbed',
                'perturbation_id': j,
                'operation': operation
            })
            perturbation_success.append(operation not in ['no_change_possible', 'no_edge_to_add', 'no_edge_to_remove'])
    
    print(f"Successfully perturbed {sum(perturbation_success)}/{len(perturbation_success)} graphs")
    
    # Step 5: Compute WL hashes for all graphs
    print("4. Computing WL hashes for all graphs...")
    all_hashes = []
    for g in all_graphs:
        if len(g.nodes()) > 0:
            h = weisfeiler_lehman_hash(g)
            all_hashes.append(h)
        else:
            all_hashes.append("empty_graph")
    
    # Step 6: Analyze results
    print("5. Analyzing results...")
    
    # Group graphs by their WL hash
    hash_groups = collections.defaultdict(list)
    for i, h in enumerate(all_hashes):
        hash_groups[h].append(i)
    
    # Analyze original vs perturbed relationships
    original_hash = all_hashes[0]  # All originals should have the same hash
    num_originals = len(largest_group)
    
    # Check if all originals still have the same hash
    original_hashes = all_hashes[:num_originals]
    originals_still_isomorphic = len(set(original_hashes)) == 1
    
    # Count how many perturbed graphs maintain the original hash
    perturbed_hashes = all_hashes[num_originals:]
    maintained_isomorphism = sum(1 for h in perturbed_hashes if h == original_hash)
    
    # Analyze by perturbation type
    operation_analysis = {}
    for i, metadata in enumerate(graph_metadata[num_originals:], start=num_originals):
        op_type = metadata['operation'].split('_')[0] if '_' in metadata['operation'] else metadata['operation']
        if op_type not in operation_analysis:
            operation_analysis[op_type] = {'total': 0, 'maintained': 0}
        
        operation_analysis[op_type]['total'] += 1
        if all_hashes[i] == original_hash:
            operation_analysis[op_type]['maintained'] += 1
    
    results = {
        'largest_group_size': len(largest_group),
        'largest_group_indices': largest_group,
        'num_perturbations_per_graph': num_perturbations,
        'total_perturbed_graphs': len(perturbed_hashes),
        'originals_still_isomorphic': originals_still_isomorphic,
        'maintained_isomorphism_count': maintained_isomorphism,
        'broken_isomorphism_count': len(perturbed_hashes) - maintained_isomorphism,
        'maintenance_rate': maintained_isomorphism / len(perturbed_hashes) if perturbed_hashes else 0,
        'all_graphs': all_graphs,
        'all_hashes': all_hashes,
        'graph_metadata': graph_metadata,
        'hash_groups': dict(hash_groups),
        'operation_analysis': operation_analysis,
        'original_hash': original_hash
    }
    
    return results

def print_detailed_analysis(results):
    """Print detailed quantitative analysis"""
    print("\n" + "="*80)
    print("DETAILED NOISE ROBUSTNESS ANALYSIS")
    print("="*80)
    
    print(f"\n1. EXPERIMENT SETUP:")
    print(f"   - Largest isomorphic group size: {results['largest_group_size']}")
    print(f"   - Graph indices in group: {results['largest_group_indices']}")
    print(f"   - Perturbations per graph: {results['num_perturbations_per_graph']}")
    print(f"   - Total perturbed graphs: {results['total_perturbed_graphs']}")
    
    print(f"\n2. ISOMORPHISM PRESERVATION RESULTS:")
    print(f"   - Original graphs still isomorphic: {results['originals_still_isomorphic']}")
    print(f"   - Perturbed graphs maintaining isomorphism: {results['maintained_isomorphism_count']}")
    print(f"   - Perturbed graphs breaking isomorphism: {results['broken_isomorphism_count']}")
    print(f"   - Overall maintenance rate: {results['maintenance_rate']:.2%}")
    
    print(f"\n3. PERTURBATION TYPE ANALYSIS:")
    for op_type, data in results['operation_analysis'].items():
        rate = data['maintained'] / data['total'] if data['total'] > 0 else 0
        print(f"   - {op_type.capitalize()} operations:")
        print(f"     * Total: {data['total']}")
        print(f"     * Maintained isomorphism: {data['maintained']}")
        print(f"     * Maintenance rate: {rate:.2%}")
    
    print(f"\n4. HASH GROUP ANALYSIS:")
    print(f"   - Number of distinct WL hash groups: {len(results['hash_groups'])}")
    print(f"   - Original hash: {results['original_hash'][:50]}..." if len(results['original_hash']) > 50 else f"   - Original hash: {results['original_hash']}")
    
    large_groups = {h: indices for h, indices in results['hash_groups'].items() if len(indices) > 1}
    if large_groups:
        print(f"   - Groups with multiple graphs: {len(large_groups)}")
        for i, (h, indices) in enumerate(list(large_groups.items())[:3]):  # Show first 3
            print(f"     * Group {i+1}: {len(indices)} graphs")
    
    print(f"\n5. ROBUSTNESS INTERPRETATION:")
    if results['maintenance_rate'] > 0.8:
        print("   - HIGH ROBUSTNESS: WL hash is very stable to single edge perturbations")
    elif results['maintenance_rate'] > 0.5:
        print("   - MODERATE ROBUSTNESS: WL hash shows some sensitivity to perturbations")
    else:
        print("   - LOW ROBUSTNESS: WL hash is highly sensitive to single edge perturbations")
    
    print(f"\n6. IMPLICATIONS:")
    if results['maintenance_rate'] > 0.9:
        print("   - The WL test may be too coarse for distinguishing similar graph structures")
        print("   - Consider using higher-order WL variants or other graph invariants")
    elif results['maintenance_rate'] < 0.1:
        print("   - The WL test is highly discriminative for this graph family")
        print("   - Small structural changes significantly impact the hash")
    else:
        print("   - The WL test shows balanced sensitivity to structural changes")
        print("   - Suitable for detecting significant structural differences")

# --- Main Execution ---
def main():
    print("Enhanced Weisfeiler-Lehman Hash Analysis with Noise Robustness Experiment")
    print("="*80)
    
    print("\n1. Loading AIDS dataset using PyTorch Geometric...")
    try:
        dataset = TUDataset(root='/tmp/AIDS', name='AIDS', use_node_attr=True)
        print(f"   Successfully loaded {len(dataset)} graphs.")
    except Exception as e:
        print(f"Error: Could not download or load the dataset. {e}")
        print("Please check your internet connection and if the TUDataset repository is accessible.")
        return
    
    print("\n2. Initial Analysis - Finding Isomorphic Groups...")
    initial_groups = find_isomorphic_groups_from_pyg(dataset)
    
    if not initial_groups:
        print("   No isomorphic graphs were found in the dataset.")
        return
    else:
        num_groups = len(initial_groups)
        total_graphs_in_groups = sum(len(indices) for indices in initial_groups.values())
        print(f"   - Total Isomorphic Groups Found: {num_groups}")
        print(f"   - Total Graphs in Isomorphic Groups: {total_graphs_in_groups}")
        
        # Show group size distribution
        group_sizes = [len(indices) for indices in initial_groups.values()]
        print(f"   - Group size statistics: min={min(group_sizes)}, max={max(group_sizes)}, mean={np.mean(group_sizes):.1f}")
    
    print("\n3. Running Noise Robustness Experiment...")
    results = run_noise_experiment(dataset, num_perturbations=10)
    
    if results:
        print_detailed_analysis(results)
        visualize_results(results, dataset) # TODO: plots
    else:
        print("   Experiment failed - no isomorphic groups found.")

if __name__ == "__main__":
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    main()
