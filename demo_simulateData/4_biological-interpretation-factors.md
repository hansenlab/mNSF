# Biological Interpretation of mNSF Factors

**Authors:** Yi Wang, Kasper Hansen, and the mNSF Team  
**Date:** March 2025

## Overview

After successfully running mNSF on your spatial transcriptomics data, the next crucial step is biological interpretation - understanding what the identified spatial factors actually mean in their biological context. This tutorial provides a comprehensive guide to interpreting mNSF factors through:

1. Gene loading analysis and pathway enrichment
2. Spatial factor visualization with histological context
3. Cell type deconvolution for spatial factors
4. Cross-sample factor comparison for biological relevance
5. Integration with existing biological knowledge

## 1. Understanding mNSF Outputs

Before diving into interpretation, let's briefly review what mNSF actually produces:

```python
import mNSF
from mNSF import process_multiSample
from mNSF.NSF import preprocess, misc, visualize
from mNSF import training_multiSample
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from os import path
import scanpy as sc
import gseapy as gp
import mygene
import scipy.stats as stats
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
```

There are two primary outputs from mNSF that require interpretation:

1. **Factor Matrix (F)**: Shows how each factor varies across the spatial dimensions of each sample
2. **Loading Matrix (W)**: Shows each gene's contribution to each factor

```python
# Example of extracting these matrices from mNSF results
def extract_mnsf_results(list_fit, list_D, L, nsample):
    """
    Extract and organize mNSF results for interpretation.
    
    Parameters:
    - list_fit: List of trained mNSF models
    - list_D: List of data dictionaries
    - L: Number of factors
    - nsample: Number of samples
    
    Returns:
    - Dictionary containing factors and loadings for each sample
    """
    results = {}
    
    for ksample in range(nsample):
        # Extract factor values (spatial patterns)
        # Using S=10 for smoother results
        Fplot = misc.t2np(list_fit[ksample].sample_latent_GP_funcs(list_D[ksample]["X"], S=10, chol=False)).T
        
        # Create dataframe for factors
        factor_df = pd.DataFrame(
            Fplot,
            columns=[f"Factor_{i+1}" for i in range(L)]
        )
        
        # Add spatial coordinates
        factor_df['x'] = list_D[ksample]["X"][:, 0]
        factor_df['y'] = list_D[ksample]["X"][:, 1]
        
        # Extract gene loadings
        loadings = misc.t2np(list_fit[ksample].sample_W(S=10))
        
        # Create dataframe for loadings
        loading_df = pd.DataFrame(
            loadings,
            index=list_D[ksample]['feature_names'],
            columns=[f"Factor_{i+1}" for i in range(L)]
        )
        
        # Store results for this sample
        results[f"Sample_{ksample+1}"] = {
            "factors": factor_df,
            "loadings": loading_df
        }
    
    return results
```

## 2. Gene Loading Analysis

The loading matrix reveals which genes contribute most to each spatial factor. This is often the starting point for biological interpretation.

### 2.1 Identifying Top Contributing Genes

```python
def analyze_top_genes(loading_df, top_n=25):
    """
    Identify and visualize top genes contributing to each factor.
    
    Parameters:
    - loading_df: DataFrame of gene loadings
    - top_n: Number of top genes to display
    
    Returns:
    - DataFrame of top genes for each factor
    """
    # Number of factors
    num_factors = loading_df.shape[1]
    
    # Initialize dictionary to store top genes
    top_genes_dict = {}
    
    # For each factor, identify top contributing genes
    for i in range(num_factors):
        factor_name = loading_df.columns[i]
        
        # Sort genes by loading value (descending)
        sorted_genes = loading_df[factor_name].sort_values(ascending=False)
        
        # Store top genes and their loading values
        top_genes_dict[factor_name] = sorted_genes.head(top_n)
    
    # Plot heatmap of top genes for each factor
    plt.figure(figsize=(15, 10))
    
    # Create a composite dataframe of top genes across all factors
    all_top_genes = set()
    for factor in top_genes_dict:
        all_top_genes.update(top_genes_dict[factor].index)
    
    # Subset loading matrix to include only top genes
    top_loading_df = loading_df.loc[list(all_top_genes)]
    
    # Plot heatmap
    sns.clustermap(
        top_loading_df,
        cmap='viridis',
        z_score=0,  # Z-score normalization (by row)
        figsize=(12, len(all_top_genes) * 0.3),
        dendrogram_ratio=(0.2, 0.1),
        cbar_pos=(0.02, 0.8, 0.05, 0.18),
        method='average'
    )
    plt.title('Top Genes Across Factors', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Create a more compact visualization - barplots for each factor
    fig, axes = plt.subplots(1, num_factors, figsize=(num_factors*4, 6))
    
    for i, factor_name in enumerate(top_genes_dict.keys()):
        # Get top genes for this factor
        factor_top_genes = top_genes_dict[factor_name]
        
        # Plot horizontal bar chart
        axes[i].barh(
            y=factor_top_genes.index[::-1],  # Reverse to put highest at top
            width=factor_top_genes.values[::-1],
            color=plt.cm.tab10(i % 10)
        )
        axes[i].set_title(f'{factor_name} Top Genes')
        axes[i].set_xlabel('Loading Value')
        
        # Only show y-axis labels for first subplot
        if i > 0:
            axes[i].set_yticklabels([])
    
    plt.tight_layout()
    plt.show()
    
    return top_genes_dict
```

### 2.2 Gene Set Enrichment Analysis

To systematically interpret the biological meaning of each factor, we can perform gene set enrichment analysis on the gene loadings:

```python
def perform_enrichment_analysis(loading_df, organism='human', database='GO_Biological_Process_2021'):
    """
    Perform gene set enrichment analysis for each factor.
    
    Parameters:
    - loading_df: DataFrame of gene loadings
    - organism: 'human' or 'mouse'
    - database: Name of the gene set database to use
    
    Returns:
    - Dictionary of enrichment results for each factor
    """
    # Number of factors
    num_factors = loading_df.shape[1]
    
    # Check organism and set up gene ID conversion if needed
    if organism == 'mouse':
        # Convert mouse gene symbols to human orthologs for better annotation
        mg = mygene.MyGeneInfo()
        genes = loading_df.index.tolist()
        gene_info = mg.querymany(genes, scopes='symbol', fields='human.ortholog.symbol', species='mouse')
        
        # Create mapping dictionary
        mouse_to_human = {}
        for gene in gene_info:
            if 'human.ortholog.symbol' in gene and gene.get('query'):
                mouse_to_human[gene['query']] = gene['human.ortholog.symbol']
    
    # Initialize dictionary to store enrichment results
    enrichment_results = {}
    
    # For each factor, perform enrichment analysis
    for i in range(num_factors):
        factor_name = loading_df.columns[i]
        print(f"\nPerforming enrichment analysis for {factor_name}...")
        
        # Get gene loadings for this factor
        gene_loadings = loading_df[factor_name]
        
        # Sort genes by loading value (descending)
        sorted_genes = gene_loadings.sort_values(ascending=False)
        
        # Convert gene list for enrichment
        if organism == 'mouse':
            # Use human orthologs
            gene_list = []
            for gene in sorted_genes.index:
                if gene in mouse_to_human:
                    gene_list.append(mouse_to_human[gene])
                else:
                    gene_list.append(gene)  # Keep original if no ortholog found
        else:
            gene_list = sorted_genes.index.tolist()
        
        # Prepare ranked list for GSEA (gene : score)
        gene_scores = {gene: score for gene, score in zip(gene_list, sorted_genes.values)}
        
        try:
            # Perform pre-ranked GSEA
            enr = gp.prerank(
                rnk=gene_scores,
                gene_sets=database,
                max_size=500,
                min_size=15,
                processes=4,
                permutation_num=1000,
                seed=42,
                outdir=None,  # Don't save output files
                verbose=False
            )
            
            # Store results
            enrichment_results[factor_name] = enr.results
            
            # Create summary DataFrame of top pathways
            top_pathways = enr.res2d.sort_values('NES', ascending=False).head(10)
            print(f"Top enriched pathways for {factor_name}:")
            print(top_pathways[['Term', 'NES', 'pval', 'fdr']].to_string())
            
            # Visualize top 5 enriched pathways
            gp.plot.dotplot(
                enr.results,
                column='NES',
                x='NES',
                size=10, 
                top_term=5,
                figsize=(8, 5),
                title=f"{factor_name} Enriched Pathways"
            )
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Enrichment analysis failed for {factor_name}: {e}")
            enrichment_results[factor_name] = None
    
    return enrichment_results
```



## 2.3 Network Analysis of Factor-Related Genes

Visualizing how top genes in each factor relate to each other can provide additional insights into the underlying biological processes. Network analysis reveals functional relationships between genes that may not be apparent from simple gene lists.

```python
def gene_network_analysis(loading_df, top_n=20, organism='human'):
    """
    Create and visualize gene interaction networks for top genes in each factor.
    
    Parameters:
    - loading_df: DataFrame of gene loadings
    - top_n: Number of top genes to include
    - organism: 'human' or 'mouse'
    
    Returns:
    - Dictionary of network graphs for each factor
    """
    # Number of factors
    num_factors = loading_df.shape[1]
    
    # Import additional required libraries
    import networkx as nx
    
    # Set up mygene for getting gene information
    mg = mygene.MyGeneInfo()
    
    # Initialize dictionary to store network graphs
    network_graphs = {}
    
    # For each factor, create a gene interaction network
    for i in range(num_factors):
        factor_name = loading_df.columns[i]
        print(f"\nAnalyzing gene network for {factor_name}...")
        
        # Get top genes for this factor
        top_genes = loading_df[factor_name].sort_values(ascending=False).head(top_n)
        
        # Get gene information
        gene_list = top_genes.index.tolist()
        
        # Query gene information
        species = 'human' if organism == 'human' else 'mouse'
        gene_info = mg.querymany(gene_list, scopes='symbol', fields=['name', 'symbol', 'entrezgene'], species=species)
        
        # Create dictionary to map gene symbols to entrez IDs
        gene_to_entrez = {}
        for gene in gene_info:
            if 'entrezgene' in gene and gene.get('query'):
                gene_to_entrez[gene['query']] = gene['entrezgene']
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes (genes)
        for gene, loading in top_genes.items():
            if gene in gene_to_entrez:
                # Add node with attributes
                G.add_node(gene, loading=loading, entrez=gene_to_entrez[gene])
            else:
                # Add node without entrez ID
                G.add_node(gene, loading=loading, entrez=None)
        
        # Try to get interactions from STRING database
        try:
            from stringdb import StringDB
            
            # Initialize STRING database client
            string_db = StringDB(species=9606 if organism == 'human' else 10090)  # 9606 for human, 10090 for mouse
            
            # Get interactions
            entrez_list = [gene_to_entrez[gene] for gene in gene_list if gene in gene_to_entrez]
            if entrez_list:
                interactions = string_db.get_interactions(entrez_list)
                
                # Add edges
                for interaction in interactions:
                    source = interaction['preferredName_A']
                    target = interaction['preferredName_B']
                    score = float(interaction['score'])
                    
                    # Only add edge if both genes are in our network
                    if source in G.nodes and target in G.nodes:
                        G.add_edge(source, target, weight=score)
            
        except Exception as e:
            print(f"Could not retrieve STRING interactions: {e}")
            print("Using correlation-based network instead...")
            
            # Calculate correlation matrix for top genes
            gene_expr = loading_df.loc[gene_list].T
            corr_matrix = gene_expr.corr()
            
            # Add edges for highly correlated gene pairs
            for gene1 in gene_list:
                for gene2 in gene_list:
                    if gene1 != gene2:
                        corr = corr_matrix.loc[gene1, gene2]
                        # Only add strong correlations
                        if abs(corr) > 0.5:
                            G.add_edge(gene1, gene2, weight=abs(corr))
        
        # Store graph
        network_graphs[factor_name] = G
        
        # Visualize network
        plt.figure(figsize=(10, 8))
        
        # Set up node colors based on loading values
        node_colors = [G.nodes[node]['loading'] for node in G.nodes()]
        
        # Set up node sizes
        node_sizes = [G.nodes[node]['loading'] * 500 for node in G.nodes()]
        
        # Use a layout algorithm
        pos = nx.spring_layout(G, seed=42)
        
        # Draw network
        nx.draw_networkx(
            G,
            pos=pos,
            node_color=node_colors,
            node_size=node_sizes,
            font_size=10,
            font_weight='bold',
            edge_color='gray',
            width=[G[u][v].get('weight', 1.0) * 2 for u, v in G.edges()],
            cmap=plt.cm.viridis,
            with_labels=True,
            alpha=0.8
        )
        
        # Add a colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
        sm.set_array([])
        plt.colorbar(sm, label='Gene Loading')
        
        plt.title(f"Gene Interaction Network for {factor_name}", fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        # Print network statistics
        print(f"Network for {factor_name} has {G.number_of_nodes()} genes and {G.number_of_edges()} interactions.")
        
        # Identify central genes (using degree centrality)
        centrality = nx.degree_centrality(G)
        top_central_genes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        
        print("Most central genes in the network:")
        for gene, score in top_central_genes:
            print(f"  - {gene}: Centrality = {score:.4f}, Loading = {G.nodes[gene]['loading']:.4f}")
    
    return network_graphs

# Example usage:
# network_graphs = gene_network_analysis(loading_df, top_n=20, organism='human')
```

### 2.3.1 Interpreting Gene Networks

When analyzing gene networks for each factor, consider the following:

1. **Network Density**: Dense networks indicate stronger functional relationships among genes.
2. **Clusters/Modules**: Identify gene clusters that may represent functional modules.
3. **Central Genes**: Genes with high centrality (many connections) often play key regulatory roles.
4. **Bridge Genes**: Genes connecting different clusters may represent cross-functional mediators.

## 3. Spatial Context Analysis

While gene loadings provide molecular insights, the spatial context of factors is equally important for biological interpretation. This section focuses on analyzing factors in their spatial context.

### 3.1 Integrating Factors with Tissue Histology

Overlaying factor patterns on tissue images can provide rich biological context:

```python
def overlay_factors_on_histology(factor_df, image_path, scale_factor=1.0, transparency=0.5, factor_names=None):
    """
    Overlay spatial factors on a histology image.
    
    Parameters:
    - factor_df: DataFrame with factors and spatial coordinates
    - image_path: Path to histology image
    - scale_factor: Scaling factor for coordinates
    - transparency: Alpha value for overlay
    - factor_names: List of factor names to visualize (None for all)
    """
    # Import necessary libraries
    from matplotlib.colors import LinearSegmentedColormap
    from skimage import io
    
    # Load histology image
    img = io.imread(image_path)
    
    # Get factor names
    if factor_names is None:
        factor_names = [col for col in factor_df.columns if col.startswith('Factor_')]
    
    # Number of factors to visualize
    num_factors = len(factor_names)
    
    # Create figure
    fig, axes = plt.subplots(1, num_factors+1, figsize=(5*(num_factors+1), 5))
    
    # Show original image
    axes[0].imshow(img)
    axes[0].set_title("Original Histology", fontsize=14)
    axes[0].axis('off')
    
    # For each factor, create overlay
    for i, factor_name in enumerate(factor_names):
        # Get factor values and coordinates
        factor_values = factor_df[factor_name].values
        x_coords = factor_df['x'].values * scale_factor
        y_coords = factor_df['y'].values * scale_factor
        
        # Show original image
        axes[i+1].imshow(img)
        
        # Normalize factor values to 0-1 range
        min_val = factor_values.min()
        max_val = factor_values.max()
        norm_values = (factor_values - min_val) / (max_val - min_val)
        
        # Create colormap with transparency
        cmap = plt.cm.viridis
        
        # Overlay factor as scatter plot
        scatter = axes[i+1].scatter(
            x_coords, y_coords,
            c=factor_values,
            cmap=cmap,
            s=50,
            alpha=transparency
        )
        
        # Add colorbar
        plt.colorbar(scatter, ax=axes[i+1], label=f'{factor_name} Value')
        
        # Set title
        axes[i+1].set_title(f"{factor_name} Overlay", fontsize=14)
        axes[i+1].axis('off')
    
    plt.tight_layout()
    plt.show()
```

### 3.2 Linking Factors to Anatomical Regions

Annotated reference data can help connect factors to known anatomical structures:

```python
def link_factors_to_anatomy(factor_df, annotation_df, factor_names=None):
    """
    Link spatial factors to annotated anatomical regions.
    
    Parameters:
    - factor_df: DataFrame with factors and spatial coordinates
    - annotation_df: DataFrame with region annotations and coordinates
    - factor_names: List of factor names to analyze (None for all)
    
    Returns:
    - DataFrame with factor enrichment by anatomical region
    """
    # Get factor names
    if factor_names is None:
        factor_names = [col for col in factor_df.columns if col.startswith('Factor_')]
    
    # Get unique anatomical regions
    regions = annotation_df['region'].unique()
    
    # Initialize results dictionary
    results = []
    
    # For each anatomical region
    for region in regions:
        # Get spots in this region
        region_spots = annotation_df[annotation_df['region'] == region]
        
        # Create a spatial index for efficient lookup
        from scipy.spatial import cKDTree
        region_coords = region_spots[['x', 'y']].values
        factor_coords = factor_df[['x', 'y']].values
        
        # Build KD-tree
        tree = cKDTree(region_coords)
        
        # Find spots in factor_df that are within/close to region spots
        # Here we use a small distance threshold to find exact or very close matches
        distances, indices = tree.query(factor_coords, k=1)
        threshold = 10.0  # adjust based on your spatial resolution
        matches = distances < threshold
        
        # Calculate factor statistics for spots in this region
        for factor_name in factor_names:
            # Get factor values for matching spots
            factor_values = factor_df.loc[matches, factor_name].values
            
            if len(factor_values) > 0:
                # Calculate statistics
                mean_value = np.mean(factor_values)
                median_value = np.median(factor_values)
                max_value = np.max(factor_values)
                
                # Calculate enrichment (compared to overall factor distribution)
                overall_mean = factor_df[factor_name].mean()
                enrichment = mean_value / overall_mean if overall_mean > 0 else 0
                
                # Calculate statistical significance
                from scipy import stats
                background = factor_df[factor_name].values
                _, p_value = stats.ttest_ind(factor_values, background, equal_var=False)
                
                # Store results
                results.append({
                    'Region': region,
                    'Factor': factor_name,
                    'Mean': mean_value,
                    'Median': median_value,
                    'Max': max_value,
                    'Enrichment': enrichment,
                    'P_Value': p_value,
                    'Spot_Count': len(factor_values)
                })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Add multiple testing correction
    from statsmodels.stats.multitest import multipletests
    _, results_df['FDR'], _, _ = multipletests(results_df['P_Value'], method='fdr_bh')
    
    # Sort by enrichment
    results_df = results_df.sort_values(['Factor', 'Enrichment'], ascending=[True, False])
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Create a pivot table for heatmap
    pivot_df = results_df.pivot(index='Region', columns='Factor', values='Enrichment')
    
    # Plot heatmap
    sns.heatmap(
        pivot_df,
        cmap='viridis',
        annot=True,
        fmt='.2f',
        linewidths=0.5
    )
    
    plt.title('Factor Enrichment by Anatomical Region', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    return results_df
```


