# Getting Started with mNSF: Basic Workflow

**Authors:** Yi Wang, Kasper Hansen, and the mNSF Team  
**Date:** March 2025

## Overview

This notebook demonstrates a basic end-to-end workflow for using multi-sample Non-negative Spatial Factorization (mNSF) with a simple example dataset. By the end of this tutorial, you'll understand:

- How to prepare and format your data for mNSF analysis
- How to set up and train an mNSF model
- How to visualize and interpret the results
- Best practices for parameter selection

## 1. Setup and Installation

Let's start by ensuring you have mNSF and all required dependencies installed.

```python
# Check if mNSF is installed
try:
    import mNSF
    print(f"mNSF version: {mNSF.__version__}")
except ImportError:
    print("mNSF not found. Installing...")
    !pip install git+https://github.com/hansenlab/mNSF.git
    import mNSF
    print(f"mNSF installed, version: {mNSF.__version__}")

# Import other required packages
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from os import path
import pickle

# Import mNSF modules
from mNSF.NSF import preprocess, misc, visualize
from mNSF import training_multiSample
from mNSF import MoranI
from mNSF import process_multiSample

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
```

## 2. Sample Data: Synthetic Example

For this tutorial, we'll generate a simple synthetic dataset of two samples, each with a small number of spots and genes. This will allow us to run the analysis quickly while illustrating all the key steps.

```python
# Set parameters for synthetic data generation
n_spots = 100  # Number of spots per sample
n_genes = 50   # Number of genes
n_factors = 2  # Number of true factors
nsample = 2    # Number of samples

# Generate synthetic data using process_multiSample.pre_processing_simulation
synthetic_data = process_multiSample.pre_processing_simulation(
    n_samples=nsample,
    n_spots=n_spots,
    n_genes=n_genes,
    n_factors=n_factors,
    output_dir="synthetic_data"
)

print(f"Sample 1: {n_spots} spots, {n_genes} genes")
print(f"Sample 2: {n_spots} spots, {n_genes} genes")
```

Let's visualize the true factors in our synthetic data to see what patterns we expect to recover:

```python
# Plot true factors for each sample
for ksample in range(nsample):
    print(f"\nSample {ksample+1} true factors:")
    # Load true factors
    true_factors = pd.read_csv(f"synthetic_data/true_factors_sample{ksample+1}.csv").values
    X = pd.read_csv(f"synthetic_data/X_sample{ksample+1}.csv", index_col=0)
    
    # Create a figure showing the factors
    fig, axes = plt.subplots(1, n_factors, figsize=(10, 4))
    
    for i in range(n_factors):
        sc = axes[i].scatter(X['x'], X['y'], c=true_factors[:, i], cmap="viridis", marker="o", s=50)
        axes[i].set_title(f"Sample {ksample+1}: Factor {i+1}")
        plt.colorbar(sc, ax=axes[i])
    
    plt.tight_layout()
    plt.show()
```

## 3. Data Preparation for mNSF

Now that we have our synthetic data, let's prepare it for mNSF analysis:

```python
# Set parameters
nchunk = 1   # We'll use 1 chunk per sample for this small example
L = 2        # Number of factors to recover (matches our synthetic data)

# Create output directories
mpth = path.join("models_synthetic")
misc.mkdir_p(mpth)
pp = path.join(mpth, "pp", str(L))
misc.mkdir_p(pp)

# Load and process the data using process_multiSample.get_D
list_D = []
list_X = []

for ksample in range(nsample):
    # Load data
    sample_idx = ksample + 1
    Y = pd.read_csv(f"synthetic_data/Y_sample{sample_idx}.csv", index_col=0)
    X = pd.read_csv(f"synthetic_data/X_sample{sample_idx}.csv", index_col=0)
    
    # Process data
    D = process_multiSample.get_D(X, Y)
    list_D.append(D)
    list_X.append(D["X"])

# Get sample IDs
list_sampleID = process_multiSample.get_listSampleID(list_D)
print(f"Prepared {len(list_D)} samples for analysis")
```

## 4. Setting Up Induced Points

Induced points are a subset of spatial locations used to make the computation more tractable. For our small example, we'll use 20% of the spots as induced points:

```python
# Set up induced points for each sample
for ksample in range(nsample):
    # Select 20% of spots as induced points
    ninduced = round(list_D[ksample]['X'].shape[0] * 0.20)
    rd_ = random.sample(range(list_D[ksample]['X'].shape[0]), ninduced)
    list_D[ksample]["Z"] = list_D[ksample]['X'][rd_, :]
    
    print(f"Sample {ksample+1}: Using {ninduced} induced points out of {list_D[ksample]['X'].shape[0]} spots")
```

## 5. Model Initialization and Training

Now we'll initialize and train our mNSF model:

```python
# Extract training data
list_Dtrain = process_multiSample.get_listDtrain(list_D)

# Initialize the model
print("Initializing mNSF model...")
list_fit = process_multiSample.ini_multiSample(list_D, L, "nb")


# Train the model
print("Training mNSF model...")
list_fit = training_multiSample.train_model_mNSF(
    list_fit,      # Initialized model
    pp,            # Directory for preprocessing results
    list_Dtrain,   # Training data
    list_D,        # Full dataset
    num_epochs=50, # Number of training iterations - use more for real data
    nsample=nsample,
    nchunk=nchunk,
    verbose=True   # Print progress
)

print("Model training complete!")
```

## 6. Extracting and Visualizing Results

After training, we can extract the learned factors and visualize them:

```python
# Use process_multiSample.post_processing_multisample to extract and visualize results
results = process_multiSample.post_processing_multisample(
    list_fit=list_fit,
    list_D=list_D,
    list_X=list_X,
    output_dir="mnsf_results",
    S=100,  # Number of samples for latent function sampling
    lda_mode=False
)

# The results dictionary contains the extracted factors, loadings, and other metrics
factors_list = results["factors_list"]
loadings = results["loadings"]

# Visualize the spatial factors
process_multiSample.plot_spatial_factors(
    list_D=list_D,
    factors_list=factors_list,
    output_dir="factor_plots",
    cmap="viridis"
)

# Display top genes for each factor
process_multiSample.plot_top_genes(
    loadings=loadings,
    n_top=15,
    output_dir="gene_plots"
)
```

## 7. Quantifying Spatial Structure with Moran's I

Moran's I is a measure of spatial autocorrelation that helps us quantify how structured our factors are spatially:

```python
# Calculate Moran's I for each factor in each sample (included in post_processing_multisample)
print("\nQuantifying spatial structure with Moran's I:")
moran_results = results["moran_results"]

for ksample in range(nsample):
    print(f"\nSample {ksample+1}:")
    for factor_result in moran_results[f"sample_{ksample+1}"]:
        factor_i = factor_result["factor"]
        morans_i = factor_result["morans_i"]
        p_value = factor_result["p_value"]
        print(f"  Factor {factor_i} - Moran's I: {morans_i:.4f}, p-value: {p_value:.4f}")
```

## 8. Examining Gene Loadings

The gene loadings tell us which genes contribute to each factor:

```python
# The loadings are already computed from post_processing_multisample
print("\nTop genes for each factor:")
for factor in range(L):
    # Get top 10 genes for this factor
    top_indices = loadings.iloc[:, factor].argsort().values[-10:][::-1]
    top_genes = loadings.index[top_indices].tolist()
    top_values = loadings.iloc[top_indices, factor].values
    
    print(f"\nFactor {factor+1} top genes:")
    for i, (gene, value) in enumerate(zip(top_genes, top_values)):
        print(f"  {i+1}. {gene}: {value:.4f}")
```

## 9. Comparing Factors Across Samples

One of the key advantages of mNSF is its ability to identify common patterns across samples:

```python
# The factor correlations are already computed from post_processing_multisample
corr_df = results["factor_correlations"]

print("\nFactor correlations across samples:")
print(corr_df)

# Visualize the correlation matrix
plt.figure(figsize=(8, 6))
plt.imshow(corr_df.values, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(label='Correlation')
plt.xticks(range(len(corr_df.columns)), corr_df.columns, rotation=90)
plt.yticks(range(len(corr_df.index)), corr_df.index)
plt.title('Cross-sample Factor Correlations')
plt.tight_layout()
plt.show()
```

## 10. Evaluating Model Performance

We can calculate the deviance explained to quantify how well our model fits the data:

```python
# Calculate deviance explained
deviance_metrics = process_multiSample.calculate_deviance_explained(
    list_fit=list_fit,
    list_D=list_D,
    list_X=list_X,
    S=100
)

print("\nModel performance metrics:")
for sample, metrics in deviance_metrics.items():
    print(f"{sample}:")
    print(f"  Null deviance: {metrics['null_deviance']:.2f}")
    print(f"  Model deviance: {metrics['model_deviance']:.2f}")
    print(f"  Deviance explained: {metrics['deviance_explained']:.2%}")
```

## 11. Parameter Selection Guidance

When applying mNSF to your own data, you'll need to decide on several key parameters:

### Number of Factors (L)
- Start with a moderate value (5-10) for exploratory analysis
- For a more rigorous approach, run multiple models with different L values and compare:
  - Poisson deviance (calculated with `visualize.gof()`)
  - Biological interpretability of the factors
  - Consistency across samples

### Induced Points
- Rule of thumb: Use 10-20% of spots as induced points
- For large datasets (>10,000 spots), you might go as low as 5%
- Too few points: Less accurate model
- Too many points: Slow computation, memory issues

### Number of Chunks (nchunk)
- For small datasets (<5,000 spots), use 1 chunk
- For larger datasets, start with 2-4 chunks
- If you encounter memory issues, increase the number of chunks

### Number of Training Epochs
- Small datasets: 50-100 epochs may be sufficient
- Complex data: 200-500 epochs recommended
- Monitor the convergence of the loss function to determine when to stop

## 12. Conclusion

In this tutorial, we've walked through the complete workflow for using mNSF to analyze multi-sample spatial transcriptomics data:

1. **Data Preparation**: Formatting data for mNSF analysis
2. **Model Setup**: Setting parameters and initializing the model
3. **Optimization**: Using induced points for computational efficiency
4. **Training**: Running the model to identify spatial factors
5. **Visualization**: Creating spatial plots of the identified factors
6. **Interpretation**: Analyzing gene loadings and factor correlations
7. **Evaluation**: Quantifying spatial structure with Moran's I

mNSF's key advantage is its ability to identify shared patterns across samples without requiring spatial alignment, making it particularly valuable for complex multi-sample studies.

For more advanced usage, including optimization for large datasets, cross-platform integration, and in-depth biological interpretation, please refer to the other tutorials in this series.

## References

1. Wang et al. (2024). Multi-sample non-negative spatial factorization. [Citation details to be added when published]
2. Hansen Lab GitHub repository: https://github.com/hansenlab/mNSF
