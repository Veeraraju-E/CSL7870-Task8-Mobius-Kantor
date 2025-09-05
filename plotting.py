import matplotlib.pyplot as plt
import os
import networkx as nx
import pandas as pd
import numpy as np

def analyze_graph_properties(graphs, labels):
    """Analyze basic properties of a collection of graphs"""
    properties = []
    
    for i, g in enumerate(graphs):
        props = {
            'graph_id': labels[i] if i < len(labels) else f"graph_{i}",
            'num_nodes': len(g.nodes()),
            'num_edges': len(g.edges()),
            'density': nx.density(g),
            'avg_clustering': nx.average_clustering(g) if len(g.nodes()) > 0 else 0,
            'is_connected': (
                (nx.is_weakly_connected(g) if nx.is_directed(g) else nx.is_connected(g))
            ) if len(g.nodes()) > 0 else False
        }
        properties.append(props)
    
    return pd.DataFrame(properties)

def visualize_results(results, dataset):
    """Create comprehensive visualizations of the experiment results"""
    
    # Configure global matplotlib params for publication-quality figures
    plt.rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.autolayout': False
    })
    
    def style_axes(ax):
        ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='both', direction='out')
    
    # Ensure output directory exists
    out_dir = 'figures'
    os.makedirs(out_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Isomorphism maintenance rate
    plt.subplot(2, 4, 1)
    maintained = results['maintained_isomorphism_count']
    broken = results['broken_isomorphism_count']
    
    plt.pie(
        [maintained, broken], 
        labels=[f'Maintained\n({maintained})', f'Broken\n({broken})'],
        autopct='%1.1f%%',
        colors=['lightgreen', 'lightcoral']
        )
    plt.title('Isomorphism Preservation\nafter Single Edge Perturbation')
    
    # 2. Operation type analysis
    plt.subplot(2, 4, 2)
    op_types = list(results['operation_analysis'].keys())
    maintenance_rates = [results['operation_analysis'][op]['maintained'] / results['operation_analysis'][op]['total'] for op in op_types]
    
    bars = plt.bar(op_types, maintenance_rates, color=['skyblue', 'orange'])
    plt.ylabel('Maintenance Rate')
    plt.title('Isomorphism Maintenance by\nPerturbation Type')
    plt.ylim(0, 1)
    style_axes(plt.gca())
    
    # Add value labels on bars
    for i, (bar, rate) in enumerate(zip(bars, maintenance_rates)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{rate:.2f}', ha='center', va='bottom')
    
    # 3. Graph properties before and after perturbation
    original_graphs = results['all_graphs'][:results['largest_group_size']]
    perturbed_graphs = results['all_graphs'][results['largest_group_size']:]
    
    # Analyze graph properties
    original_props = analyze_graph_properties(original_graphs, [f"orig_{i}" for i in range(len(original_graphs))])
    perturbed_props = analyze_graph_properties(perturbed_graphs, [f"pert_{i}" for i in range(len(perturbed_graphs))])
    
    # Node count distribution
    plt.subplot(2, 4, 3)
    plt.hist([original_props['num_nodes'], perturbed_props['num_nodes']], bins=10, alpha=0.7, label=['Original', 'Perturbed'])
    plt.xlabel('Number of Nodes')
    plt.ylabel('Frequency')
    plt.title('Node Count Distribution')
    plt.legend()
    style_axes(plt.gca())
    
    # Edge count distribution
    plt.subplot(2, 4, 4)
    plt.hist([original_props['num_edges'], perturbed_props['num_edges']], bins=10, alpha=0.7, label=['Original', 'Perturbed'])
    plt.xlabel('Number of Edges')
    plt.ylabel('Frequency')
    plt.title('Edge Count Distribution')
    plt.legend()
    style_axes(plt.gca())
    
    # 4. Hash distribution analysis
    plt.subplot(2, 4, 5)
    hash_group_sizes = [len(indices) for indices in results['hash_groups'].values()]
    plt.hist(hash_group_sizes, bins=max(1, len(set(hash_group_sizes))), edgecolor='black')
    plt.xlabel('Group Size')
    plt.ylabel('Number of Groups')
    plt.title('WL Hash Group Size Distribution')
    style_axes(plt.gca())
    
    # 5. Connectivity analysis
    plt.subplot(2, 4, 6)
    orig_connectivity = original_props['is_connected'].mean()
    pert_connectivity = perturbed_props['is_connected'].mean()
    
    plt.bar(['Original', 'Perturbed'], [orig_connectivity, pert_connectivity], color=['green', 'red'], alpha=0.7)
    plt.ylabel('Fraction Connected')
    plt.title('Graph Connectivity')
    plt.ylim(0, 1)
    style_axes(plt.gca())
    
    # 6. Clustering coefficient comparison
    plt.subplot(2, 4, 7)
    plt.boxplot([original_props['avg_clustering'], perturbed_props['avg_clustering']], labels=['Original', 'Perturbed'])
    plt.ylabel('Average Clustering Coefficient')
    plt.title('Clustering Coefficient Distribution')
    style_axes(plt.gca())
    
    # 7. Perturbation success rate by operation
    plt.subplot(2, 4, 8)
    op_analysis = results['operation_analysis']
    op_names = list(op_analysis.keys())
    counts = [op_analysis[op]['total'] for op in op_names]
    
    plt.bar(op_names, counts, color=['purple', 'brown'])
    plt.ylabel('Count')
    plt.title('Perturbation Operation Counts')
    style_axes(plt.gca())
    
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'overview.png'), bbox_inches='tight')
    plt.close(fig)
    
    # Create a detailed heatmap for hash relationships
    fig2 = plt.figure(figsize=(10, 6))
    
    # Create a matrix showing which graphs share the same hash
    n_graphs = len(results['all_graphs'])
    similarity_matrix = np.zeros((n_graphs, n_graphs))
    
    for i in range(n_graphs):
        for j in range(n_graphs):
            similarity_matrix[i, j] = int(results['all_hashes'][i] == results['all_hashes'][j])
    
    # Create labels for the heatmap
    labels = []
    for i, metadata in enumerate(results['graph_metadata']):
        if metadata['graph_type'] == 'original':
            labels.append(f"O{metadata['original_id']}")
        else:
            labels.append(f"P{metadata['original_id']}.{metadata['perturbation_id']}")
    
    im = plt.imshow(similarity_matrix, cmap='RdYlBu', aspect='auto')
    cbar = plt.colorbar(im)
    cbar.set_label('Same WL Hash (1=Yes, 0=No)')
    plt.title('WL Hash Similarity Matrix\n(Original and Perturbed Graphs)')
    plt.xlabel('Graph Index')
    plt.ylabel('Graph Index')
    
    # Add grid lines to separate original from perturbed
    orig_count = results['largest_group_size']
    plt.axhline(y=orig_count-0.5, color='red', linestyle='--', linewidth=2)
    plt.axvline(x=orig_count-0.5, color='red', linestyle='--', linewidth=2)
    
    plt.tight_layout()
    fig2.savefig(os.path.join(out_dir, 'hash_similarity.png'), bbox_inches='tight')
    plt.close(fig2)