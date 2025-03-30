# Interpreting mNSF Factors: A Complete Guide

**Authors:** Yi Wang, Kasper Hansen, and the mNSF Team  
**Date:** March 2025

## Overview

This guide focuses specifically on how to interpret the factors identified by multi-sample Non-negative Spatial Factorization (mNSF). Using the same synthetic dataset from our "Getting Started" tutorial, we'll walk through various approaches to extract biological meaning from mNSF results.

Understanding what factors represent biologically is a critical step in spatial transcriptomics analysis. This guide will help you:

- Evaluate the statistical significance of your factors
- Connect spatial patterns to gene expression profiles
- Validate factors against known biology
- Compare factors across multiple samples
- Visualize factors for effective interpretation

## 1. Understanding mNSF Factors: What Are They?

Before diving into interpretation, it's important to understand what mNSF factors represent:

**Definition:** Each mNSF factor represents a spatial pattern of gene expression that varies across the tissue. Factors are composed of:

1. **Spatial weights (factor matrix)**: Shows how strongly each factor is expressed at each spatial location
2. **Gene loadings (loading matrix)**: Shows how strongly each gene contributes to each factor

Conceptually, factors can be thought of as "spatial gene expression programs" - coordinated sets of genes that show similar spatial patterns within and across samples.

## 2. Setting Up the Example

Let's use our synthetic dataset to demonstrate factor interpretation. First, we'll recreate the essential parts of the analysis:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from os import path
import pickle

# Import mNSF modules
from mNSF import process_multiSample
from mNSF.NSF import preprocess, misc, visualize
from mNSF import training_multiSample
from mNSF import MoranI

# Import utility functions
from mNSF import mnsf_utility

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Generate synthetic data as in the Getting Started tutorial
simulated_data = mnsf_utility.pre_processing_simulation(
    n_samples=2,
    n_spots=100,
    n_genes=50,
    n_factors=2,
    output_dir="synthetic_data"
)

# Train mNSF model as shown in Getting Started tutorial
# ... [code for data preparation and model training] ...

# Load the trained model (assuming it's saved after training)
with open("mnsf_results_synthetic/list_fit.pkl", "rb") as f:
    list_fit = pickle.load(f)


# Extract results
# If your model has 23 factors instead of 2
L_actual = 23  # Use the actual number of factors from the model

results = mnsf_utility.post_processing_multisample(
    L=L_actual,  # Use the actual number of factors
    list_fit=list_fit,
    list_D=list_D,
    list_X=list_X,
    output_dir="mnsf_results_synthetic",
    S=100,  # Number of samples for latent function evaluation
    lda_mode=False
)

# Now we have the factors and can focus on interpretation
factors_list = results['factors_list']  # List containing factor matrices for each sample
inpf=process_multiSample.interpret_npf_v3(list_fit,list_X,S=100,lda_mode=False)
loadings = inpf["loadings"]
```

## 3. Statistical Evaluation of Factors

Before interpreting the biological meaning, we should assess how statistically significant and robust our factors are.

### 3.1 Spatial Autocorrelation (Moran's I)

Moran's I measures the degree of spatial correlation in each factor. Higher values indicate stronger spatial structuring:

```python
# For each sample and factor
for s in range(len(factors_list)):
    print(f"Sample {s+1} Moran's I values:")
    for i in range(factors_list[s].shape[1]):
        I, p_value = MoranI.calculate_morans_i(list_D[s]["X"], factors_list[s][:, i])
        print(f"  Factor {i+1}: I = {I:.4f}, p-value = {p_value:.4f}")
```

**Example output:**
```
Sample 1 Moran's I values:
  Factor 1: I = 0.6823, p-value = 0.0001
  Factor 2: I = 0.5976, p-value = 0.0002
  ...
Sample 2 Moran's I values:
  Factor 1: I = 0.7102, p-value = 0.0001
  Factor 2: I = 0.6344, p-value = 0.0001
  ...
```

**Interpretation:**
- All factors show significant spatial autocorrelation (p < 0.05)
- Higher I values indicate stronger spatial patterning
- Factor 1 shows stronger spatial structure than Factor 2 in both samples
- Both samples have similar patterns of spatial autocorrelation


## 4. Connecting Factors to Gene Expression

The biological meaning of factors comes from the genes that define them. Let's examine this relationship:

### 4.1 Identifying Key Genes for Each Factor

For each factor, we can identify the genes with the highest loadings:

```python
# Get gene names (for synthetic data, just gene indices)
gene_names = [f"Gene_{i+1}" for i in range(loadings.shape[0])]

# Function to identify top genes for each factor
def get_top_genes(loadings, gene_names, n=10):
    top_genes = {}
    for i in range(loadings.shape[1]):
        sorted_idx = np.argsort(loadings[:, i])[::-1][:n]
        top_genes[i] = [(gene_names[idx], loadings[idx, i]) for idx in sorted_idx]
    return top_genes

# Get top 10 genes for each factor
top_genes = get_top_genes(loadings, gene_names, n=10)

# Display top genes for each factor
for i in range(len(top_genes)):
    print(f"Factor {i+1} top genes:")
    for gene, loading in top_genes[i]:
        print(f"  {gene}: {loading:.4f}")
    print()
```

**Example output:**
```
Factor 1 top genes:
  Gene_12: 0.1542
  Gene_7: 0.1498
  Gene_31: 0.1475
  Gene_3: 0.1401
  Gene_18: 0.1389
  ...

Factor 2 top genes:
  Gene_42: 0.1612
  Gene_19: 0.1587
  Gene_5: 0.1521
  Gene_26: 0.1499
  Gene_11: 0.1468
  ...
```

### 4.2 Visualizing Gene-Factor Relationships

A heatmap can help visualize how genes relate to factors:

```python
# Create a heatmap of top gene loadings
def plot_gene_loading_heatmap(loadings, gene_names, n_top_genes=10, save_path="./mnsf_results_synthetic/gene_factor_heatmap.png", figsize=(12, 10)):
    """
    Plot a heatmap showing top genes for each factor with row names as "Gene XXX".
    
    Parameters:
    loadings -- NumPy array of gene loadings (genes × factors)
    gene_names -- List of gene names corresponding to rows in loadings
    n_top_genes -- Number of top genes to show for each factor
    save_path -- Path to save the figure, if None the figure is not saved
    figsize -- Size of the figure as (width, height)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    
    # Convert loadings to DataFrame with gene names as index
    n_factors = loadings.shape[1]
    factor_names = [f'Factor_{i+1}' for i in range(n_factors)]
    loadings_df = pd.DataFrame(loadings, index=gene_names, columns=factor_names)
    
    # Create a figure
    plt.figure(figsize=figsize)
    
    # Create a new DataFrame to store only top genes for each factor
    top_genes_df = pd.DataFrame()
    
    # For each factor, find the top genes
    for factor_idx in range(n_factors):
        factor_name = factor_names[factor_idx]
        # Get top n genes for this factor
        top_genes = loadings_df[factor_name].nlargest(n_top_genes).index.tolist()
        
        # Create a mini-dataframe with just these genes across all factors
        factor_df = loadings_df.loc[top_genes, :]
        
        # Add a column to identify which factor this gene is a top gene for
        factor_df['Top_For_Factor'] = factor_name
        
        # Add to our collection
        top_genes_df = pd.concat([top_genes_df, factor_df])
    
    # Remove duplicates (genes that are top for multiple factors)
    top_genes_df = top_genes_df[~top_genes_df.index.duplicated(keep='first')]
    
    # Sort by the factor they're top for, then by loading value within that factor
    top_genes_df = top_genes_df.sort_values(by=['Top_For_Factor'] + factor_names, ascending=[True] + [False] * n_factors)
    
    # Create a mapping of original gene names to "Gene XXX" format
    original_index = top_genes_df.index.tolist()
    gene_xxx_index = [f"Gene {name}" for name in original_index]
    
    # Create a mapping dictionary for reference
    gene_mapping = dict(zip(gene_xxx_index, original_index))
    
    # Replace index with "Gene XXX" format
    top_genes_df.index = gene_xxx_index
    
    # Remove the helper column before plotting
    plot_df = top_genes_df.drop(columns=['Top_For_Factor'])
    
    # Create the heatmap
    ax = sns.heatmap(plot_df, cmap='viridis', linewidths=0.5)
    
    # Adjust y-axis labels
    plt.yticks(rotation=0)
    
    plt.title('Top Gene Loadings Across Factors')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Optionally print the mapping
    print("Gene name mapping:")
    for gene_xxx, original in gene_mapping.items():
        print(f"{gene_xxx} = {original}")
    
    return plot_df

# Plot the heatmap
plot_gene_loading_heatmap(loadings, gene_names)

```

## 9. Practical Steps for Factor Interpretation

When interpreting mNSF factors in your own data, follow these steps:

1. **Start with statistical metrics**:
   - Examine Moran's I values to verify spatial structure
   - Assess variance explained to prioritize factors
   - Check cross-sample consistency to identify robust patterns

2. **Examine gene loadings**:
   - Identify top genes for each factor
   - Run gene set enrichment analysis
   - Look for biological themes among top genes

3. **Visualize spatial patterns**:
   - Create spatial heatmaps for each factor
   - Use RGB overlays to see how factors interact
   - Compare patterns to known tissue architecture

4. **Integrate domain knowledge**:
   - Connect patterns to known tissue biology
   - Consider functional implications of the spatial arrangement
   - Consult existing literature on the tissue or cell types

5. **Validate with additional methods**:
   - Use immunohistochemistry to confirm protein expression
   - Validate with single-cell RNA-seq data if available
   - Compare to published spatial atlases

## 10. Conclusion

Interpreting mNSF factors requires a combination of statistical analysis, visualization, and biological knowledge. By systematically examining gene contributions and spatial patterns, you can extract meaningful biological insights from your spatial transcriptomics data.

In our synthetic example, we demonstrated how to:
1. Statistically evaluate the significance of factors
2. Identify key genes driving each factor
3. Visualize spatial patterns in multiple ways
4. Compare factors across samples
5. Connect patterns to biological meaning

For real datasets, interpretation will be more complex but also more rewarding, as the factors will correspond to true biological processes and tissue organization. The frameworks presented here provide a starting point for extracting biological meaning from your mNSF results.

## References

1. Wang et al. (2024). Multi-sample non-negative spatial factorization. [Citation details to be added when published]
2. Hansen Lab GitHub repository: https://github.com/hansenlab/mNSF
