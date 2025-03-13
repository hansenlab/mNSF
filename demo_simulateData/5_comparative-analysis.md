# Comparative Analysis Across Multiple Samples

**Authors:** Yi Wang, Kasper Hansen, and the mNSF Team  
**Date:** March 2025

## Overview

One of the primary strengths of mNSF (multi-sample Non-negative Spatial Factorization) is its ability to analyze multiple samples simultaneously without requiring cross-sample spot alignment. This capability opens up powerful comparative analysis approaches that can reveal insights into:

1. Common spatial patterns across samples
2. Sample-specific variations
3. Biological heterogeneity between conditions or individuals
4. Reproducibility of spatial patterns

This tutorial provides a comprehensive guide to comparing mNSF results across multiple samples, with practical code examples and visualization techniques.

## 1. Prerequisites

To follow this tutorial, you should have already:
- Processed your data with mNSF across multiple samples
- Extracted the factor matrices and loading matrices for each sample
- Have basic familiarity with mNSF outputs (factors and loadings)

```python
# Import necessary libraries
import mNSF
from mNSF import process_multiSample
from mNSF.NSF import preprocess, misc, visualize
from mNSF import training_multiSample
from mNSF import MoranI
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import umap
import os
from os import path
import pickle
```

## 2. Loading and Organizing Multi-Sample Results

First, let's set up a framework for loading and organizing mNSF results from multiple samples:

```python
def load_mnsf_results(base_dir, num_samples, L):
    """
    Load mNSF results from saved files.
    
    Parameters:
    - base_dir: Directory containing results
    - num_samples: Number of samples
    - L: Number of factors
    
    Returns:
    - Dictionary with factors and loadings for each sample
    """
    results = {}
    
    for sample_idx in range(num_samples):
        sample_name = f"Sample_{sample_idx+1}"
        results[sample_name] = {}
        
        # Load factors
        factor_path = os.path.join(base_dir, f"factors_sample{sample_idx+1}.csv")
        if os.path.exists(factor_path):
            factors_df = pd.read_csv(factor_path)
            results[sample_name]["factors"] = factors_df
        else:
            print(f"Warning: Factor file not found for {sample_name}")
        
        # Load loadings
        loading_path = os.path.join(base_dir, f"loadings_sample{sample_idx+1}.csv")
        if os.path.exists(loading_path):
            loadings_df = pd.read_csv(loading_path, index_col=0)
            results[sample_name]["loadings"] = loadings_df
        else:
            print(f"Warning: Loading file not found for {sample_name}")
    
    return results

# Alternative: If you have the results directly from mNSF run
def extract_mnsf_results(list_fit, list_D, L, nsample):
    """
    Extract mNSF results directly from the model output.
    
    Parameters:
    - list_fit: List of trained mNSF models
    - list_D: List of data dictionaries
    - L: Number of factors
    - nsample: Number of samples
    
    Returns:
    - Dictionary with factors and loadings for each sample
    """
    results = {}
    
    for sample_idx in range(nsample):
        sample_name = f"Sample_{sample_idx+1}"
        results[sample_name] = {}
        
        # Extract factors
        Fplot = misc.t2np(list_fit[sample_idx].sample_latent_GP_funcs(list_D[sample_idx]["X"], S=10, chol=False)).T
        factor_df = pd.DataFrame(
            Fplot, 
            columns=[f"Factor_{i+1}" for i in range(L)]
        )
        factor_df['x'] = list_D[sample_idx]["X"][:, 0]
        factor_df['y'] = list_D[sample_idx]["X"][:, 1]
        
        # Extract loadings
        loadings = misc.t2np(list_fit[sample_idx].sample_W(S=10))
        loading_df = pd.DataFrame(
            loadings,
            index=list_D[sample_idx]['feature_names'],
            columns=[f"Factor_{i+1}" for i in range(L)]
        )
        
        # Store in results
        results[sample_name]["factors"] = factor_df
        results[sample_name]["loadings"] = loading_df
    
    return results
```

## 3. Comparing Gene Loadings Across Samples

One of the most informative ways to compare samples is to analyze the similarity of gene loadings across samples for each factor:

```python
def compare_loadings_across_samples(results, method='correlation'):
    """
    Compare gene loadings across samples.
    
    Parameters:
    - results: Dictionary of mNSF results
    - method: Similarity measure ('correlation', 'cosine', 'jaccard')
    
    Returns:
    - Dictionary of similarity matrices for each factor pair
    """
    # Get sample names and number of factors
    sample_names = list(results.keys())
    num_samples = len(sample_names)
    
    # Get factor names from first sample
    first_sample = list(results.values())[0]
    factor_names = first_sample["loadings"].columns
    num_factors = len(factor_names)
    
    # Initialize dictionary to store similarity matrices
    similarity_dict = {}
    
    # For each factor pair (including cross-factor comparisons)
    for i, factor_i in enumerate(factor_names):
        for j, factor_j in enumerate(factor_names):
            # Create similarity matrix
            sim_matrix = np.zeros((num_samples, num_samples))
            
            # Compute pairwise similarities
            for idx1, sample1 in enumerate(sample_names):
                for idx2, sample2 in enumerate(sample_names):
                    # Get loadings
                    loadings1 = results[sample1]["loadings"][factor_i]
                    loadings2 = results[sample2]["loadings"][factor_j]
                    
                    # Compute similarity
                    if method == 'correlation':
                        # Use Pearson correlation
                        sim = stats.pearsonr(loadings1, loadings2)[0]
                    elif method == 'cosine':
                        # Use cosine similarity
                        sim = cosine_similarity(
                            loadings1.values.reshape(1, -1), 
                            loadings2.values.reshape(1, -1)
                        )[0][0]
                    elif method == 'jaccard':
                        # Use Jaccard similarity on top genes
                        # First, get top 100 genes for each factor
                        top_genes1 = set(loadings1.sort_values(ascending=False).head(100).index)
                        top_genes2 = set(loadings2.sort_values(ascending=False).head(100).index)
                        # Compute Jaccard similarity
                        if len(top_genes1) > 0 or len(top_genes2) > 0:
                            sim = len(top_genes1.intersection(top_genes2)) / len(top_genes1.union(top_genes2))
                        else:
                            sim = 0
                    else:
                        raise ValueError(f"Unknown similarity method: {method}")
                    
                    # Store similarity
                    sim_matrix[idx1, idx2] = sim
            
            # Store similarity matrix
            similarity_dict[f"{factor_i}_{factor_j}"] = {
                "matrix": sim_matrix,
                "sample_names": sample_names
            }
    
    return similarity_dict

def visualize_loading_similarities(similarity_dict, focused_factors=None):
    """
    Visualize loading similarities across samples.
    
    Parameters:
    - similarity_dict: Dictionary of similarity matrices
    - focused_factors: List of factor names to focus on (None for all)
    """
    # Get factor pairs
    factor_pairs = list(similarity_dict.keys())
    
    # Filter factor pairs if focused_factors is provided
    if focused_factors is not None:
        factor_pairs = [fp for fp in factor_pairs if fp.split('_')[0] in focused_factors and fp.split('_')[1] in focused_factors]
    
    # Get sample names
    sample_names = similarity_dict[factor_pairs[0]]["sample_names"]
    
    # Calculate number of subplots
    n_subplots = len(factor_pairs)
    n_cols = min(3, n_subplots)
    n_rows = (n_subplots + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*4))
    
    # Make axes iterable if there's only one subplot
    if n_subplots == 1:
        axes = np.array([axes])
    
    # Flatten axes array
    axes = axes.flatten()
    
    # Plot each similarity matrix
    for i, factor_pair in enumerate(factor_pairs):
        if i < len(axes):
            # Get similarity matrix
            sim_matrix = similarity_dict[factor_pair]["matrix"]
            
            # Create heatmap
            sns.heatmap(
                sim_matrix,
                annot=True,
                cmap="coolwarm",
                vmin=-1,
                vmax=1,
                ax=axes[i],
                xticklabels=sample_names,
                yticklabels=sample_names,
                square=True
            )
            
            # Set title
            factor_i, factor_j = factor_pair.split('_')
            axes[i].set_title(f"{factor_i} vs {factor_j}", fontsize=12)
    
    # Hide any unused subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()

# Aggregate loadings similarity into a unified view
def aggregate_loading_similarities(similarity_dict, same_factor_only=True):
    """
    Aggregate loading similarities into a single matrix.
    
    Parameters:
    - similarity_dict: Dictionary of similarity matrices
    - same_factor_only: Whether to only consider same-factor comparisons
    
    Returns:
    - Aggregated similarity matrix
    """
    # Get sample names
    sample_names = similarity_dict[list(similarity_dict.keys())[0]]["sample_names"]
    num_samples = len(sample_names)
    
    # Get factor names
    factor_pairs = list(similarity_dict.keys())
    factor_names = sorted(set([fp.split('_')[0] for fp in factor_pairs]))
    num_factors = len(factor_names)
    
    if same_factor_only:
        # Only consider similarities for the same factor across samples
        # This gives us one matrix per factor
        factor_matrices = {}
        
        for factor in factor_names:
            factor_pair = f"{factor}_{factor}"
            factor_matrices[factor] = similarity_dict[factor_pair]["matrix"]
        
        # Create a combined figure
        fig, axes = plt.subplots(1, num_factors, figsize=(num_factors*4, 4))
        
        for i, factor in enumerate(factor_names):
            sns.heatmap(
                factor_matrices[factor],
                annot=True,
                cmap="coolwarm",
                vmin=-1,
                vmax=1,
                ax=axes[i],
                xticklabels=sample_names,
                yticklabels=sample_names,
                square=True
            )
            axes[i].set_title(f"{factor}", fontsize=12)
        
        plt.tight_layout()
        plt.show()
        
        return factor_matrices
    else:
        # Consider all factor pair comparisons across samples
        # This creates a large block matrix
        block_matrix = np.zeros((num_factors * num_samples, num_factors * num_samples))
        
        for i, factor_i in enumerate(factor_names):
            for j, factor_j in enumerate(factor_names):
                factor_pair = f"{factor_i}_{factor_j}"
                sim_matrix = similarity_dict[factor_pair]["matrix"]
                
                # Copy to block matrix
                block_matrix[
                    i*num_samples:(i+1)*num_samples,
                    j*num_samples:(j+1)*num_samples
                ] = sim_matrix
        
        # Create row and column labels
        labels = []
        for factor in factor_names:
            for sample in sample_names:
                labels.append(f"{sample}_{factor}")
        
        # Visualize block matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            block_matrix,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            xticklabels=labels,
            yticklabels=labels,
            square=True
        )
        plt.title("All Factor-Sample Loading Similarities", fontsize=14)
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        
        return block_matrix
```

## 4. Factor Consistency Analysis

Beyond comparing gene loadings, we can analyze the consistency of spatial patterns across samples:

```python
def calculate_factor_consistency(results):
    """
    Calculate consistency metrics for factors across samples.
    
    Parameters:
    - results: Dictionary of mNSF results
    
    Returns:
    - DataFrame with consistency metrics for each factor
    """
    # Get sample names and number of samples
    sample_names = list(results.keys())
    num_samples = len(sample_names)
    
    # Get factor names
    first_sample = list(results.values())[0]
    factor_names = first_sample["loadings"].columns
    
    # Initialize lists to store metrics
    factor_list = []
    loading_similarity_mean = []
    loading_similarity_std = []
    morans_i_mean = []
    morans_i_std = []
    
    # For each factor
    for factor in factor_names:
        factor_list.append(factor)
        
        # Calculate loading similarity
        loading_similarities = []
        for i in range(num_samples):
            for j in range(i+1, num_samples):
                sample_i = sample_names[i]
                sample_j = sample_names[j]
                
                loadings_i = results[sample_i]["loadings"][factor]
                loadings_j = results[sample_j]["loadings"][factor]
                
                similarity = stats.pearsonr(loadings_i, loadings_j)[0]
                loading_similarities.append(similarity)
        
        # Store loading similarity statistics
        loading_similarity_mean.append(np.mean(loading_similarities))
        loading_similarity_std.append(np.std(loading_similarities))
        
        # Calculate Moran's I for each sample
        morans_i_values = []
        for sample in sample_names:
            # Check if we have factor and spatial information
            if "factors" in results[sample] and "x" in results[sample]["factors"].columns:
                factor_values = results[sample]["factors"][factor]
                spatial_coords = results[sample]["factors"][["x", "y"]].values
                
                # Calculate Moran's I
                try:
                    I, _ = MoranI.calculate_morans_i(spatial_coords, factor_values)
                    morans_i_values.append(I)
                except Exception as e:
                    print(f"Error calculating Moran's I for {sample}, {factor}: {e}")
        
        # Store Moran's I statistics
        if morans_i_values:
            morans_i_mean.append(np.mean(morans_i_values))
            morans_i_std.append(np.std(morans_i_values))
        else:
            morans_i_mean.append(np.nan)
            morans_i_std.append(np.nan)
    
    # Create DataFrame
    consistency_df = pd.DataFrame({
        "Factor": factor_list,
        "Loading_Similarity_Mean": loading_similarity_mean,
        "Loading_Similarity_Std": loading_similarity_std,
        "Morans_I_Mean": morans_i_mean,
        "Morans_I_Std": morans_i_std
    })
    
    return consistency_df

def visualize_factor_consistency(consistency_df):
    """
    Visualize factor consistency metrics.
    
    Parameters:
    - consistency_df: DataFrame with consistency metrics
    """
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot loading similarity
    axes[0].bar(
        consistency_df["Factor"],
        consistency_df["Loading_Similarity_Mean"],
        yerr=consistency_df["Loading_Similarity_Std"],
        color="skyblue",
        capsize=5
    )
    axes[0].set_xlabel("Factor")
    axes[0].set_ylabel("Loading Similarity")
    axes[0].set_title("Gene Loading Similarity Across Samples")
    axes[0].set_ylim(-1, 1)
    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Plot Moran's I
    axes[1].bar(
        consistency_df["Factor"],
        consistency_df["Morans_I_Mean"],
        yerr=consistency_df["Morans_I_Std"],
        color="salmon",
        capsize=5
    )
    axes[1].set_xlabel("Factor")
    axes[1].set_ylabel("Moran's I")
    axes[1].set_title("Spatial Pattern Consistency (Moran's I)")
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Create a scatter plot of loading similarity vs. Moran's I
    plt.figure(figsize=(8, 6))
    plt.scatter(
        consistency_df["Loading_Similarity_Mean"],
        consistency_df["Morans_I_Mean"],
        s=50,
        c=range(len(consistency_df)),
        cmap="viridis",
        alpha=0.7
    )
    
    # Add factor labels
    for i, factor in enumerate(consistency_df["Factor"]):
        plt.annotate(
            factor,
            (consistency_df["Loading_Similarity_Mean"].iloc[i], consistency_df["Morans_I_Mean"].iloc[i]),
            xytext=(5, 5),
            textcoords="offset points"
        )
    
    plt.xlabel("Loading Similarity")
    plt.ylabel("Moran's I")
    plt.title("Loading Similarity vs. Spatial Structure")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
```

## 5. Hierarchical Clustering of Samples and Factors

Hierarchical clustering can reveal relationships among samples and factors:

```python
def hierarchical_clustering_analysis(results):
    """
    Perform hierarchical clustering of samples and factors.
    
    Parameters:
    - results: Dictionary of mNSF results
    
    Returns:
    - Clustering results
    """
    # Get sample names and factor names
    sample_names = list(results.keys())
    first_sample = list(results.values())[0]
    factor_names = first_sample["loadings"].columns
    
    # Combine loadings across samples
    combined_loadings = []
    sample_factor_labels = []
    
    for sample in sample_names:
        for factor in factor_names:
            # Get loadings for this sample-factor pair
            loadings = results[sample]["loadings"][factor].values
            
            # Append to lists
            combined_loadings.append(loadings)
            sample_factor_labels.append(f"{sample}_{factor}")
    
    # Convert to array
    combined_loadings = np.array(combined_loadings)
    
    # Compute distance matrix
    distances = pdist(combined_loadings, metric='correlation')
    distance_matrix = squareform(distances)
    
    # Perform hierarchical clustering
    Z = linkage(distances, method='ward')
    
    # Plot dendrogram
    plt.figure(figsize=(12, 6))
    dendrogram(
        Z,
        labels=sample_factor_labels,
        leaf_rotation=90,
        leaf_font_size=10
    )
    plt.title("Hierarchical Clustering of Sample-Factor Loadings")
    plt.xlabel("Sample-Factor")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.show()
    
    # Create a heatmap of the distance matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        distance_matrix,
        cmap="viridis",
        xticklabels=sample_factor_labels,
        yticklabels=sample_factor_labels
    )
    plt.title("Distance Matrix of Sample-Factor Loadings")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    return {
        "linkage": Z,
        "labels": sample_factor_labels,
        "distance_matrix": distance_matrix
    }
```

## 6. Dimensionality Reduction for Cross-Sample Visualization

For a more intuitive visualization of sample-factor relationships, we can use dimensionality reduction:

```python
def dimensionality_reduction_visualization(results):
    """
    Visualize sample-factor relationships using dimensionality reduction.
    
    Parameters:
    - results: Dictionary of mNSF results
    """
    # Get sample names and factor names
    sample_names = list(results.keys())
    first_sample = list(results.values())[0]
    factor_names = first_sample["loadings"].columns
    
    # Combine loadings across samples
    combined_loadings = []
    sample_factor_labels = []
    sample_colors = []
    factor_markers = []
    
    # Define color and marker mappings
    sample_color_map = {sample: plt.cm.tab10(i % 10) for i, sample in enumerate(sample_names)}
    factor_marker_map = {
        factor: marker for factor, marker in zip(
            factor_names, ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h'][:len(factor_names)]
        )
    }
    
    for sample in sample_names:
        for factor in factor_names:
            # Get loadings for this sample-factor pair
            loadings = results[sample]["loadings"][factor].values
            
            # Append to lists
            combined_loadings.append(loadings)
            sample_factor_labels.append(f"{sample}_{factor}")
            sample_colors.append(sample_color_map[sample])
            factor_markers.append(factor_marker_map[factor])
    
    # Convert to array
    combined_loadings = np.array(combined_loadings)
    
    # Perform UMAP dimensionality reduction
    reducer = umap.UMAP(
        n_neighbors=5,
        min_dist=0.1,
        n_components=2,
        metric='correlation',
        random_state=42
    )
    
    embedding = reducer.fit_transform(combined_loadings)
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    
    for i, (x, y) in enumerate(embedding):
        plt.scatter(
            x, y, 
            color=sample_colors[i],
            marker=factor_markers[i],
            s=100,
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5
        )
        
        # Add label
        plt.annotate(
            sample_factor_labels[i],
            (x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8
        )
    
    # Add legend for samples
    sample_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=sample)
        for sample, color in sample_color_map.items()
    ]
    
    # Add legend for factors
    factor_handles = [
        plt.Line2D([0], [0], marker=marker, color='black', markersize=10, label=factor)
        for factor, marker in factor_marker_map.items()
    ]
    
    # Create legends
    plt.legend(
        handles=sample_handles + factor_handles,
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        title="Samples and Factors"
    )
    
    plt.title("UMAP Embedding of Sample-Factor Loadings")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
```

## 7. Sample-Specific vs. Common Factors

An important aspect of multi-sample analysis is distinguishing between factors that are common across samples and those that are sample-specific:

```python
def identify_common_specific_factors(results, loading_similarity_threshold=0.6):
    """
    Identify common and sample-specific factors.
    
    Parameters:
    - results: Dictionary of mNSF results
    - loading_similarity_threshold: Threshold for considering factors similar
    
    Returns:
    - Dictionary with common and specific factors
    """
    # Get sample names and factor names
    sample_names = list(results.keys())
    num_samples = len(sample_names)
    first_sample = list(results.values())[0]
    factor_names = first_sample["loadings"].columns
    
    # Initialize dictionary to store factor classifications
    factor_classification = {
        "common": [],
        "sample_specific": {},
        "similarity_matrix": {}
    }
    
    # For each factor, check if it's common or sample-specific
    for factor in factor_names:
        # Initialize similarity matrix for this factor
        sim_matrix = np.zeros((num_samples, num_samples))
        
        # Compute pairwise similarities
        for i in range(num_samples):
            for j in range(num_samples):
                sample_i = sample_names[i]
                sample_j = sample_names[j]
                
                loadings_i = results[sample_i]["loadings"][factor]
                loadings_j = results[sample_j]["loadings"][factor]
                
                sim = stats.pearsonr(loadings_i, loadings_j)[0]
                sim_matrix[i, j] = sim
        
        # Store similarity matrix
        factor_classification["similarity_matrix"][factor] = sim_matrix
        
        # Check if this factor is common across all samples
        # A factor is common if all pairwise similarities are above threshold
        is_common = True
        
        for i in range(num_samples):
            for j in range(i+1, num_samples):
                if sim_matrix[i, j] < loading_similarity_threshold:
                    is_common = False
                    break
            
            if not is_common:
                break
        
        if is_common:
            # This factor is common across all samples
            factor_classification["common"].append(factor)
        else:
            # This factor is sample-specific
            # Identify which samples have this factor
            for i in range(num_samples):
                sample = sample_names[i]
                
                # Check if this sample has a unique version of this factor
                is_unique = True
                
                for j in range(num_samples):
                    if i != j and sim_matrix[i, j] >= loading_similarity_threshold:
                        is_unique = False
                        break
                
                if is_unique:
                    if sample not in factor_classification["sample_specific"]:
                        factor_classification["sample_specific"][sample] = []
                    
                    factor_classification["sample_specific"][sample].append(factor)
    
    # Print summary
    print(f"Common factors across all samples: {factor_classification['common']}")
    
    for sample in factor_classification["sample_specific"]:
        specific_factors = factor_classification["sample_specific"][sample]
        if specific_factors:
            print(f"Sample-specific factors for {sample}: {specific_factors}")
    
    # Visualize similarity matrices for each factor
    n_factors = len(factor_names)
    n_cols = min(3, n_factors)
    n_rows = (n_factors + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*3.5))
    
    # Make axes iterable
    if n_factors == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()
    
    for i, factor in enumerate(factor_names):
        if i < len(axes):
            sim_matrix = factor_classification["similarity_matrix"][factor]
            
            # Create heatmap
            sns.heatmap(
                sim_matrix,
                annot=True,
                cmap="coolwarm",
                vmin=-1,
                vmax=1,
                ax=axes[i],
                xticklabels=sample_names,
                yticklabels=sample_names,
                square=True
            )
            
            # Set title
            if factor in factor_classification["common"]:
                axes[i].set_title(f"{factor} (Common)", fontsize=12)
            else:
                # Check which samples have this as specific
                specific_samples = []
                
                for sample in factor_classification["sample_specific"]:
                    if factor in factor_classification["sample_specific"][sample]:
                        specific_samples.append(sample)
                
                if specific_samples:
                    axes[i].set_title(f"{factor} (Specific to {', '.join(specific_samples)})", fontsize=10)
                else:
                    axes[i].set_title(f"{factor} (Mixed)", fontsize=12)
    
    # Hide any unused subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return factor_classification
```

## 8. Visualizing Factors Across All Samples

Creating effective visualizations of spatial factors across multiple samples is essential for identifying patterns and drawing biological insights. This section presents several approaches to visualize and compare factors across samples.

```python
def visualize_factors_across_samples(results, factor_classification=None):
    """
    Create a comprehensive visualization of factors across all samples.
    
    Parameters:
    - results: Dictionary of mNSF results
    - factor_classification: Optional classification of factors
    """
    # Get sample names and factor names
    sample_names = list(results.keys())
    num_samples = len(sample_names)
    first_sample = list(results.values())[0]
    factor_names = first_sample["loadings"].columns
    num_factors = len(factor_names)
    
    # Create a grid of plots
    fig, axes = plt.subplots(num_samples, num_factors, figsize=(num_factors*3, num_samples*3))
    
    # Make axes accessible for single row or column cases
    if num_samples == 1 and num_factors == 1:
        axes = np.array([[axes]])
    elif num_samples == 1:
        axes = axes.reshape(1, -1)
    elif num_factors == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot each factor for each sample
    for i, sample in enumerate(sample_names):
        for j, factor in enumerate(factor_names):
            ax = axes[i, j]
            
            # Get factor values and spatial coordinates
            if "factors" in results[sample] and factor in results[sample]["factors"]:
                factor_values = results[sample]["factors"][factor]
                x_coords = results[sample]["factors"]["x"]
                y_coords = results[sample]["factors"]["y"]
                
                # Create scatter plot
                scatter = ax.scatter(
                    x_coords, y_coords,
                    c=factor_values,
                    cmap="viridis",
                    s=10,
                    alpha=0.7,
                    edgecolors='none'
                )
                
                # Add colorbar
                if i == 0:  # Only add colorbar to first row
                    fig.colorbar(scatter, ax=ax, shrink=0.6)
                
                # Mark factor type if classification is provided
                if factor_classification is not None:
                    title = f"{factor}"
                    
                    if factor in factor_classification.get("common", []):
                        title += " (Common)"
                        ax.set_facecolor('#e6ffe6')  # Light green background for common factors
                    elif sample in factor_classification.get("sample_specific", {}) and \
                         factor in factor_classification["sample_specific"][sample]:
                        title += " (Sample-specific)"
                        ax.set_facecolor('#ffe6e6')  # Light red background for sample-specific factors
                    
                    ax.set_title(title, fontsize=10)
                else:
                    ax.set_title(factor, fontsize=10)
            else:
                ax.text(0.5, 0.5, "Data not available", ha='center', va='center')
                ax.set_facecolor('#f0f0f0')  # Light gray background
            
            # Add sample label to leftmost plots
            if j == 0:
                ax.set_ylabel(sample, fontsize=12, fontweight='bold')
            
            # Remove axis ticks to reduce clutter
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    plt.show()
    
    return fig, axes
```

### 8.1 Side-by-Side Spatial Factor Comparison

When comparing factors across samples, it's often helpful to examine each factor individually across all samples:

```python
def compare_single_factor_across_samples(results, factor, ncols=3):
    """
    Create a detailed comparison of a single factor across all samples.
    
    Parameters:
    - results: Dictionary of mNSF results
    - factor: Name of the factor to compare
    - ncols: Number of columns in the plot grid
    """
    # Get sample names
    sample_names = list(results.keys())
    num_samples = len(sample_names)
    
    # Calculate layout
    nrows = (num_samples + ncols - 1) // ncols
    
    # Create figure
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4, nrows*4))
    
    # Handle single row/column cases
    if num_samples == 1:
        axes = np.array([axes])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)
    else:
        axes = axes.reshape(nrows, ncols)
    
    # Flatten axes for easy indexing
    axes_flat = axes.flatten()
    
    # Collect all factor values to determine a common colorbar scale
    all_values = []
    for sample in sample_names:
        if "factors" in results[sample] and factor in results[sample]["factors"]:
            all_values.extend(results[sample]["factors"][factor].values)
    
    vmin = np.percentile(all_values, 5) if all_values else 0
    vmax = np.percentile(all_values, 95) if all_values else 1
    
    # Plot each sample
    for i, sample in enumerate(sample_names):
        if i < len(axes_flat):
            ax = axes_flat[i]
            
            if "factors" in results[sample] and factor in results[sample]["factors"]:
                factor_values = results[sample]["factors"][factor]
                x_coords = results[sample]["factors"]["x"]
                y_coords = results[sample]["factors"]["y"]
                
                # Create scatter plot with consistent color scale
                scatter = ax.scatter(
                    x_coords, y_coords,
                    c=factor_values,
                    cmap="viridis",
                    s=15,
                    alpha=0.8,
                    edgecolors='none',
                    vmin=vmin,
                    vmax=vmax
                )
                
                # Add a colorbar for the first plot
                if i == 0:
                    cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
                    cbar.set_label(f"{factor} intensity")
                
                # Calculate Moran's I for spatial structure
                try:
                    spatial_coords = np.column_stack((x_coords, y_coords))
                    I, p_value = MoranI.calculate_morans_i(spatial_coords, factor_values)
                    
                    # Add Moran's I to the plot
                    ax.text(
                        0.05, 0.95,
                        f"Moran's I: {I:.3f}\np-value: {p_value:.3f}",
                        transform=ax.transAxes,
                        va='top',
                        ha='left',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
                    )
                except Exception as e:
                    pass
                
                # Set title
                ax.set_title(f"{sample}: {factor}", fontsize=12)
            else:
                ax.text(0.5, 0.5, "Data not available", ha='center', va='center')
                ax.set_facecolor('#f0f0f0')
                ax.set_title(f"{sample}: {factor}", fontsize=12)
            
            # Remove axis ticks
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Hide any unused subplots
    for j in range(num_samples, len(axes_flat)):
        axes_flat[j].axis('off')
    
    plt.tight_layout()
    plt.suptitle(f"Comparison of {factor} Across All Samples", fontsize=16, y=1.02)
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    plt.show()
    
    return fig
```

### 8.2 Creating Side-by-Side Loadings Comparison

Comparing gene loadings across samples provides insights into the biological consistency of factors:

```python
def compare_factor_loadings_across_samples(results, factor, top_n=20):
    """
    Compare gene loadings for a specific factor across all samples.
    
    Parameters:
    - results: Dictionary of mNSF results
    - factor: Name of the factor to compare
    - top_n: Number of top genes to display
    """
    # Get sample names
    sample_names = list(results.keys())
    num_samples = len(sample_names)
    
    # Collect loadings for this factor from all samples
    loadings_dict = {}
    all_top_genes = set()
    
    for sample in sample_names:
        if "loadings" in results[sample] and factor in results[sample]["loadings"]:
            # Get loadings for this factor
            loadings = results[sample]["loadings"][factor]
            
            # Sort and store
            sorted_loadings = loadings.sort_values(ascending=False)
            loadings_dict[sample] = sorted_loadings
            
            # Add top genes to set
            all_top_genes.update(sorted_loadings.head(top_n).index)
    
    # Convert to list and sort alphabetically
    all_top_genes = sorted(list(all_top_genes))
    
    # Create a combined dataframe
    combined_df = pd.DataFrame(index=all_top_genes)
    
    for sample in sample_names:
        if sample in loadings_dict:
            # Add this sample's loadings
            combined_df[sample] = loadings_dict[sample].reindex(all_top_genes)
    
    # Create a heatmap of the loadings
    plt.figure(figsize=(num_samples*1.5 + 2, len(all_top_genes)*0.3))
    
    # Sort genes by average loading
    combined_df['mean'] = combined_df.mean(axis=1)
    combined_df = combined_df.sort_values('mean', ascending=False)
    combined_df = combined_df.drop('mean', axis=1)
    
    # Create heatmap
    sns.heatmap(
        combined_df,
        cmap="viridis",
        annot=False,
        yticklabels=True,
        xticklabels=True,
        cbar_kws={"label": "Loading Value"}
    )
    
    plt.title(f"Gene Loadings for {factor} Across All Samples", fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # Create a barplot comparing top genes
    # Take the top 10 genes based on average loading
    top10_mean = combined_df.mean(axis=1).sort_values(ascending=False).head(10).index
    top10_df = combined_df.loc[top10_mean]
    
    # Melt the dataframe for easier plotting
    melted_df = top10_df.reset_index().melt(
        id_vars='index',
        var_name='Sample',
        value_name='Loading'
    )
    melted_df = melted_df.rename(columns={'index': 'Gene'})
    
    # Create barplot
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=melted_df,
        x='Gene',
        y='Loading',
        hue='Sample',
        palette='tab10'
    )
    
    plt.title(f"Top 10 Genes for {factor} Across All Samples", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    
    return combined_df
```

### 8.3 Correlation Patterns Across Samples

Correlation analysis can reveal shared patterns across samples:

```python
def cross_sample_correlation_analysis(results, use_loadings=True, method='pearson'):
    """
    Analyze correlations of factors across samples using either loadings or spatial patterns.
    
    Parameters:
    - results: Dictionary of mNSF results
    - use_loadings: Whether to use gene loadings (True) or spatial patterns (False)
    - method: Correlation method ('pearson', 'spearman')
    
    Returns:
    - Correlation dataframe
    """
    # Get sample names and factor names
    sample_names = list(results.keys())
    first_sample = list(results.values())[0]
    factor_names = first_sample["loadings"].columns
    
    # Create labels for all sample-factor combinations
    labels = []
    for sample in sample_names:
        for factor in factor_names:
            labels.append(f"{sample}-{factor}")
    
    # Initialize correlation matrix
    n = len(labels)
    corr_matrix = np.zeros((n, n))
    
    # Compute correlations
    for i, label_i in enumerate(labels):
        sample_i, factor_i = label_i.split('-')
        
        for j, label_j in enumerate(labels):
            sample_j, factor_j = label_j.split('-')
            
            if use_loadings:
                # Get gene loadings
                loadings_i = results[sample_i]["loadings"][factor_i]
                loadings_j = results[sample_j]["loadings"][factor_j]
                
                # Compute correlation
                if method == 'pearson':
                    corr = stats.pearsonr(loadings_i, loadings_j)[0]
                elif method == 'spearman':
                    corr = stats.spearmanr(loadings_i, loadings_j)[0]
                else:
                    raise ValueError(f"Unknown correlation method: {method}")
                
                corr_matrix[i, j] = corr
            else:
                # Use spatial patterns
                # This is more complex as spots aren't aligned across samples
                # We'll skip this implementation for this example
                # In practice, you would need to register the samples spatially
                # or use summary statistics of the spatial patterns
                corr_matrix[i, j] = np.nan if i != j else 1.0
    
    # Create dataframe
    corr_df = pd.DataFrame(corr_matrix, index=labels, columns=labels)
    
    # Create clustered heatmap
    plt.figure(figsize=(12, 10))
    sns.clustermap(
        corr_df,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        annot=False,
        figsize=(15, 12),
        method='average',
        dendrogram_ratio=(0.1, 0.1),
        cbar_pos=(0.02, 0.8, 0.05, 0.18)
    )
    
    plt.title(f"Cross-Sample Factor Correlation ({method.capitalize()})", fontsize=16)
    plt.tight_layout()
    plt.show()
    
    return corr_df
```

### 8.4 Creating Multi-Sample Factor Atlases

A factor atlas provides a comprehensive visualization of all factors across all samples:

```python
def create_factor_atlas(results, subsampling=None, ncols=3):
    """
    Create a comprehensive atlas of all factors across all samples.
    
    Parameters:
    - results: Dictionary of mNSF results
    - subsampling: Optional subsampling rate to reduce plotting density
    - ncols: Number of columns in the grid
    
    Returns:
    - Figure object
    """
    # Get sample names and factor names
    sample_names = list(results.keys())
    first_sample = list(results.values())[0]
    factor_names = first_sample["loadings"].columns
    
    # Total number of plots
    n_plots = len(sample_names) * len(factor_names)
    n_rows = (n_plots + ncols - 1) // ncols
    
    # Create figure
    fig = plt.figure(figsize=(ncols*4, n_rows*4))
    
    # Add a title
    fig.suptitle("mNSF Factor Atlas Across All Samples", fontsize=20, y=0.995)
    
    # Create grid spec
    gs = fig.add_gridspec(n_rows, ncols, wspace=0.3, hspace=0.4)
    
    # Plot counter
    plot_idx = 0
    
    # Create colormaps
    cmap_base = plt.cm.viridis
    
    # Iterate through all factors and samples
    for factor_idx, factor in enumerate(factor_names):
        for sample_idx, sample in enumerate(sample_names):
            # Calculate grid position
            row = plot_idx // ncols
            col = plot_idx % ncols
            
            # Create subplot
            ax = fig.add_subplot(gs[row, col])
            
            # Get factor values and spatial coordinates
            if "factors" in results[sample] and factor in results[sample]["factors"]:
                factor_values = results[sample]["factors"][factor]
                x_coords = results[sample]["factors"]["x"]
                y_coords = results[sample]["factors"]["y"]
                
                # Apply subsampling if requested
                if subsampling is not None:
                    n_spots = len(factor_values)
                    subsample_size = int(n_spots * subsampling)
                    indices = np.random.choice(n_spots, size=subsample_size, replace=False)
                    
                    factor_values = factor_values.iloc[indices] if hasattr(factor_values, 'iloc') else factor_values[indices]
                    x_coords = x_coords.iloc[indices] if hasattr(x_coords, 'iloc') else x_coords[indices]
                    y_coords = y_coords.iloc[indices] if hasattr(y_coords, 'iloc') else y_coords[indices]
                
                # Create custom colormap for this factor
                # This helps distinguish different factors
                cmap_factor = plt.cm.get_cmap(cmap_base.name, 256)
                
                # Create scatter plot
                scatter = ax.scatter(
                    x_coords, y_coords,
                    c=factor_values,
                    cmap=cmap_factor,
                    s=15,
                    alpha=0.8,
                    edgecolors='none'
                )
                
                # Add a small colorbar
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = plt.colorbar(scatter, cax=cax)
                cbar.ax.tick_params(labelsize=8)
                
                # Calculate Moran's I for spatial structure
                try:
                    spatial_coords = np.column_stack((x_coords, y_coords))
                    I, p_value = MoranI.calculate_morans_i(spatial_coords, factor_values)
                    
                    # Add Moran's I to the plot
                    ax.text(
                        0.05, 0.95,
                        f"Moran's I: {I:.3f}",
                        transform=ax.transAxes,
                        va='top',
                        ha='left',
                        fontsize=8,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
                    )
                except Exception as e:
                    pass
                
                # Set title
                ax.set_title(f"{sample}: {factor}", fontsize=10)
            else:
                ax.text(0.5, 0.5, "Data not available", ha='center', va='center')
                ax.set_facecolor('#f0f0f0')
                ax.set_title(f"{sample}: {factor}", fontsize=10)
            
            # Remove axis ticks
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Increment plot index
            plot_idx += 1
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()
    
    return fig
```

### 8.5 Interactive Multi-Sample Explorer

For more detailed exploration, an interactive visualization can be created using libraries like Plotly:

```python
def create_interactive_explorer(results):
    """
    Create an interactive visualization of factors across samples.
    
    Parameters:
    - results: Dictionary of mNSF results
    
    Returns:
    - Interactive figure
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Get sample names and factor names
    sample_names = list(results.keys())
    first_sample = list(results.values())[0]
    factor_names = first_sample["loadings"].columns
    
    # Create subplot grid
    fig = make_subplots(
        rows=len(sample_names),
        cols=len(factor_names),
        subplot_titles=[f"{sample}: {factor}" for sample in sample_names for factor in factor_names],
        horizontal_spacing=0.03,
        vertical_spacing=0.05
    )
    
    # Add traces for each sample and factor
    for sample_idx, sample in enumerate(sample_names):
        for factor_idx, factor in enumerate(factor_names):
            # Calculate row and column (1-indexed for plotly)
            row = sample_idx + 1
            col = factor_idx + 1
            
            # Get factor values and spatial coordinates
            if "factors" in results[sample] and factor in results[sample]["factors"]:
                factor_values = results[sample]["factors"][factor]
                x_coords = results[sample]["factors"]["x"]
                y_coords = results[sample]["factors"]["y"]
                
                # Create scatter plot
                fig.add_trace(
                    go.Scatter(
                        x=x_coords,
                        y=y_coords,
                        mode='markers',
                        marker=dict(
                            size=6,
                            color=factor_values,
                            colorscale='Viridis',
                            showscale=True if row == 1 and col == 1 else False,
                            colorbar=dict(
                                title="Value",
                                len=0.5,
                                y=0.8
                            ) if row == 1 and col == 1 else None
                        ),
                        text=[f"Value: {v:.3f}" for v in factor_values],
                        name=f"{sample}: {factor}"
                    ),
                    row=row,
                    col=col
                )
            else:
                # Add empty plot with message
                fig.add_trace(
                    go.Scatter(
                        x=[0],
                        y=[0],
                        mode='text',
                        text=["Data not available"],
                        name=f"{sample}: {factor}"
                    ),
                    row=row,
                    col=col
                )
    
    # Update layout
    fig.update_layout(
        height=300 * len(sample_names),
        width=300 * len(factor_names),
        title="Interactive mNSF Factor Explorer",
        showlegend=False
    )
    
    # Update axes
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    
    return fig
```

## 9. Practical Example: Working with a Real Dataset

Let's put these visualization techniques into practice with a practical example using our DLPFC dataset:

### 9.1 Loading DLPFC Results

```python
# Assume we have run mNSF on DLPFC dataset and saved results
dlpfc_results = load_mnsf_results("results/dlpfc", num_samples=2, L=4)

# Visualize all factors across samples
visualize_factors_across_samples(dlpfc_results)

# Identify common and sample-specific factors
factor_classification = identify_common_specific_factors(dlpfc_results, loading_similarity_threshold=0.7)

# Create detailed visualization with classification
visualize_factors_across_samples(dlpfc_results, factor_classification)

# Compare a specific factor across samples (e.g., Factor_1 which is common)
compare_single_factor_across_samples(dlpfc_results, "Factor_1")

# Examine gene loadings for Factor_1
loadings_comparison = compare_factor_loadings_across_samples(dlpfc_results, "Factor_1")
```

### 9.2 Interpreting Results

When comparing factors across samples, look for:

1. **Spatial consistency**: Do factors show similar spatial patterns across samples?
2. **Gene loading similarity**: Are the same genes driving each factor across samples?
3. **Moran's I values**: Do factors have similar spatial structure strength?
4. **Common vs. specific factors**: Which factors are shared, and which are unique to certain samples?

### 9.3 Biological Insights from Multi-Sample Analysis

Multi-sample analysis can reveal:

- **Conserved spatial patterns**: Biological structures that are consistent across samples/individuals
- **Condition-specific patterns**: Spatial arrangements that differ between conditions
- **Technical vs. biological variation**: Distinguishing technical artifacts from real biological signal
- **Hierarchical organization**: How spatial patterns are nested or related across samples

## 10. Best Practices for Multi-Sample Visualization

When creating visualizations for multiple samples, follow these best practices:

1. **Consistent color scales**: Use the same color scale across samples for each factor
2. **Clear labeling**: Ensure each plot is clearly labeled with sample and factor information
3. **Highlight relationships**: Use visual cues to highlight factor relationships (e.g., common vs. specific)
4. **Include quantitative metrics**: Add metrics like Moran's I to help interpret spatial structure
5. **Interactive options**: Consider interactive visualizations for large datasets
6. **Multiple perspectives**: Show both spatial patterns and gene loadings
7. **Hierarchical organization**: Use clustering to reveal relationships between samples and factors

## 11. Conclusion

Effective visualization and comparison of mNSF results across multiple samples is essential for extracting meaningful biological insights. By using the approaches described in this tutorial, you can:

1. Identify common spatial patterns across samples
2. Detect sample-specific variations
3. Quantify the consistency of spatial organization
4. Link spatial patterns to underlying gene expression programs
5. Generate testable hypotheses about spatial biology

Remember that spatial transcriptomics is still an evolving field, and interpretation of results should always be done in the context of existing biological knowledge and validated with orthogonal methods when possible.

The multi-sample capabilities of mNSF open up exciting possibilities for comparative spatial transcriptomics analysis, enabling researchers to move beyond single-sample studies and begin to understand the conserved and variable aspects of spatial gene expression across biological contexts.
