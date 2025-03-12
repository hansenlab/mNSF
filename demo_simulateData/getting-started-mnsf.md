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
from mNSF import process_multiSample
from mNSF.NSF import preprocess, misc, visualize
from mNSF import training_multiSample
from mNSF import MoranI

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
```

## 2. Sample Data: Synthetic Example

For this tutorial, we'll generate a simple synthetic dataset of two samples, each with a small number of spots and genes. This will allow us to run the analysis quickly while illustrating all the key steps.

```python
# Function to generate synthetic spatial transcriptomics data
def generate_synthetic_data(n_spots=100, n_genes=50, n_factors=2, seed=42):
    np.random.seed(seed)
    
    # Generate spatial coordinates in a 10x10 grid
    x = np.random.uniform(0, 10, n_spots)
    y = np.random.uniform(0, 10, n_spots)
    X = np.column_stack((x, y))
    
    # Generate synthetic factors
    factors = np.zeros((n_spots, n_factors))
    
    # Factor 1: Gradient from left to right
    factors[:, 0] = x / 10
    
    # Factor 2: Circular pattern around center
    center_x, center_y = 5, 5
    distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    factors[:, 1] = np.exp(-distance_from_center / 3)
    
    # Generate gene loadings
    loadings = np.random.gamma(1, 1, (n_genes, n_factors))
    
    # Generate expression data (mean)
    mean_expr = np.dot(factors, loadings.T)
    
    # Generate count data using Poisson distribution
    counts = np.random.poisson(mean_expr)
    
    # Create dataframes
    # gene_names = [f"gene_{i}" for i in range(n_genes)]
    # spot_names = [f"spot_{i}" for i in range(n_spots)]
    
    # Y = pd.DataFrame(counts, index=spot_names)
    Y = pd.DataFrame(counts)
    X_df = pd.DataFrame(X, columns=["x", "y"])
    
    return X_df, Y, factors, loadings

# Create gene names
gene_names = [f"gene_{i}" for i in range(n_genes)]

# Generate two synthetic samples
print("Generating synthetic data...")
X1, Y1, true_factors1, true_loadings1 = generate_synthetic_data(seed=42)
X2, Y2, true_factors2, true_loadings2 = generate_synthetic_data(seed=43)

# Save the data to CSV files
output_dir = "synthetic_data"
os.makedirs(output_dir, exist_ok=True)

X1.to_csv(f"{output_dir}/X_sample1.csv", index=True)
Y1.to_csv(f"{output_dir}/Y_sample1.csv", index=True)
X2.to_csv(f"{output_dir}/X_sample2.csv", index=True)
Y2.to_csv(f"{output_dir}/Y_sample2.csv", index=True)

print(f"Sample 1: {Y1.shape[0]} spots, {Y1.shape[1]} genes")
print(f"Sample 2: {Y2.shape[0]} spots, {Y2.shape[1]} genes")
```

Let's visualize the true factors in our synthetic data to see what patterns we expect to recover:

```python
# Create a function to visualize factors
def plot_factors(X, factors, title):
    fig, axes = plt.subplots(1, factors.shape[1], figsize=(10, 4))
    
    for i in range(factors.shape[1]):
        sc = axes[i].scatter(X['x'], X['y'], c=factors[:, i], cmap='viridis', s=50)
        axes[i].set_title(f'Factor {i+1}')
        axes[i].set_xlabel('X coordinate')
        axes[i].set_ylabel('Y coordinate')
        plt.colorbar(sc, ax=axes[i])
    
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

# Plot true factors
plot_factors(X1, true_factors1, "Sample 1: True Factors")
plot_factors(X2, true_factors2, "Sample 2: True Factors")
```

## 3. Data Preparation for mNSF

Now that we have our synthetic data, let's prepare it for mNSF analysis:

```python
# Set parameters
nsample = 2  # Number of samples
nchunk = 1   # We'll use 1 chunk per sample for this small example
L = 2        # Number of factors to recover (matches our synthetic data)

# Create output directories
mpth = path.join("models_synthetic")
misc.mkdir_p(mpth)
pp = path.join(mpth, "pp", str(L))
misc.mkdir_p(pp)

# Load and process the data
list_D = []
list_X = []

for ksample in range(nsample):
    # Load data
    sample_idx = ksample + 1
    Y = pd.read_csv(f"{output_dir}/Y_sample{sample_idx}.csv", index_col=0)
    X = pd.read_csv(f"{output_dir}/X_sample{sample_idx}.csv", index_col=0)
    
    # For this example, we're transposing Y to match mNSF expected format
    # where genes are rows and spots are columns
    # Y = Y.T
    
    # Process data using mNSF's get_D function
    D = process_multiSample.get_D(X, Y)
    list_D.append(D)
    list_X.append(D["X"])

# Get sample IDs
list_sampleID = process_multiSample.get_listSampleID(list_D)
print(f"Prepared {len(list_D)} samples for analysis")
```

## 4. Model Initialization


## 5. Setting Up Induced Points

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

## 6. Model Initialization and Training

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

## 7. Extracting and Visualizing Results

After training, we can extract the learned factors and visualize them:

```python
# Function to extract factors from the trained model
def extract_factors(fit, D, sample_idx):
    # Extract the factor values (3 samples for smoother results)
    Fplot = misc.t2np(fit.sample_latent_GP_funcs(D["X"], S=3, chol=False)).T
    
    # Create a figure showing the factors
    fig, axes = plt.subplots(1, L, figsize=(10, 4))
    
    for i in range(L):
        sc = axes[i].scatter(D["X"][:, 0], D["X"][:, 1], c=Fplot[:, i], cmap="viridis", marker="o", s=50)
        axes[i].set_title(f"Sample {sample_idx}: Factor {i+1}")
        plt.colorbar(sc, ax=axes[i])
    
    plt.tight_layout()
    plt.show()
    
    return Fplot

# Extract and visualize factors for each sample
factors_list = []
for ksample in range(nsample):
    print(f"Extracting and visualizing factors for sample {ksample+1}...")
    Fplot = extract_factors(list_fit[ksample], list_D[ksample], ksample+1)
    factors_list.append(Fplot)
```

## 8. Quantifying Spatial Structure with Moran's I

Moran's I is a measure of spatial autocorrelation that helps us quantify how structured our factors are spatially:

```python
# Calculate Moran's I for each factor in each sample
print("\nQuantifying spatial structure with Moran's I:")
for ksample in range(nsample):
    print(f"\nSample {ksample+1}:")
    for i in range(L):
        I, p_value = MoranI.calculate_morans_i(list_D[ksample]["X"], factors_list[ksample][:, i])
        print(f"  Factor {i+1} - Moran's I: {I:.4f}, p-value: {p_value:.4f}")
```

## 9. Examining Gene Loadings

The gene loadings tell us which genes contribute to each factor:

```python
# Extract gene loadings
inpf12=process_multiSample.interpret_npf_v3(list_fit,list_X,S=100,lda_mode=False)
W = inpf12["loadings"]
#Wdf=pd.DataFrame(W*inpf12["totals1"
loadings=pd.DataFrame(W*inpf12["totalsW"][:,None],  columns=range(1,L+1))


# Function to visualize top genes for each factor
def plot_top_genes(loadings, gene_names, n_top=10):
    fig, axes = plt.subplots(1, loadings.shape[1], figsize=(12, 5))
    
    for i in range(loadings.shape[1]):
        # Get top genes for this factor
        top_indices = np.argsort(loadings[:, i])[-n_top:][::-1]
        top_genes = [gene_names[idx] for idx in top_indices]
        top_values = loadings[top_indices, i]
        
        # Plot
        axes[i].barh(top_genes, top_values)
        axes[i].set_title(f'Factor {i+1} Top Genes')
        axes[i].set_xlabel('Loading')
    
    plt.tight_layout()
    plt.show()

plot_top_genes(loadings, gene_names)
```

## 10. Comparing Factors Across Samples

One of the key advantages of mNSF is its ability to identify common patterns across samples:

```python
# Create a correlation matrix between factors across samples
def compare_factors_across_samples(factors_list, nsample, L):
    # Initialize correlation matrix
    corr_matrix = np.zeros((nsample * L, nsample * L))
    
    # Calculate correlations
    for i in range(nsample):
        for j in range(nsample):
            for k in range(L):
                for l in range(L):
                    # Calculate correlation between factor k in sample i and factor l in sample j
                    corr = np.corrcoef(factors_list[i][:, k], factors_list[j][:, l])[0, 1]
                    corr_matrix[i*L + k, j*L + l] = corr
    
    # Create labels
    labels = []
    for i in range(nsample):
        for k in range(L):
            labels.append(f"S{i+1}_F{k+1}")
    
    # Plot correlation matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation')
    plt.xticks(range(nsample * L), labels, rotation=90)
    plt.yticks(range(nsample * L), labels)
    plt.title('Cross-sample Factor Correlations')
    plt.tight_layout()
    plt.show()
    
    return corr_matrix

# Compare factors across samples
print("\nComparing factors across samples:")
corr_matrix = compare_factors_across_samples(factors_list, nsample, L)
```

## 11. Saving Results

Finally, let's save our results for future use:

```python
# Create a results directory
results_dir = "mnsf_results_synthetic"
os.makedirs(results_dir, exist_ok=True)

# Save factors and loadings
for ksample in range(nsample):
    # Save factors
    factors_df = pd.DataFrame(
        factors_list[ksample], 
        columns=[f"factor_{i+1}" for i in range(L)]
    )
    factors_df.to_csv(f"{results_dir}/factors_sample{ksample+1}.csv", index=False)

# Save loadings
loadings_df = pd.DataFrame(
        loadings, 
        columns=[f"factor_{i+1}" for i in range(L)],
        index=gene_names
)
loadings_df.to_csv(f"{results_dir}/loadings_sample{ksample+1}.csv")

print(f"Results saved to {results_dir}/")
```

## 12. Parameter Selection Guidance

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

## 13. Conclusion

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
