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
