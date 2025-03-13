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

# Import utility functions
from mNSF import mnsf_utility


# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
```

## 2. Sample Data: Synthetic Example

For this tutorial, we'll generate a simple synthetic dataset of two samples, each with a small number of spots and genes. This will allow us to run the analysis quickly while illustrating all the key steps.

```python
# Using our pre-processing simulation utility
simulated_data = mnsf_utility.pre_processing_simulation(
    n_samples=2,
    n_spots=100,
    n_genes=50,
    n_factors=2,
    output_dir="synthetic_data"
)

print(f"Generated synthetic data with:")
for i in range(2):
    print(f"Sample {i+1}: {simulated_data['list_Y'][i].shape[0]} spots, {simulated_data['list_Y'][i].shape[1]} genes")
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

# Plot true factors for each sample
for i in range(2):
    X = simulated_data['list_X'][i]
    true_factors = simulated_data['list_factors'][i]
    plot_factors(X, true_factors, f"Sample {i+1}: True Factors")
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
    Y = pd.read_csv(f"synthetic_data/Y_sample{sample_idx}.csv", index_col=0)
    X = pd.read_csv(f"synthetic_data/X_sample{sample_idx}.csv", index_col=0)
    
    # Process data using mNSF's get_D function
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
# Extract training data
list_Dtrain = process_multiSample.get_listDtrain(list_D)

# Set up induced points for each sample
for ksample in range(nsample):
    # Select 20% of spots as induced points
    ninduced = round(list_D[ksample]['X'].shape[0] * 0.20)
    rd_ = random.sample(range(list_D[ksample]['X'].shape[0]), ninduced)
    list_D[ksample]["Z"] = list_D[ksample]['X'][rd_, :]
    
    print(f"Sample {ksample+1}: Using {ninduced} induced points out of {list_D[ksample]['X'].shape[0]} spots")
```
## 5. Model Initialization and Training
```python
# Initialize the model
print("Initializing mNSF model...")
list_fit = process_multiSample.ini_multiSample(list_D, L, "nb")
```

Now we'll train our mNSF model:

```python
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

## 6. Post-processing and Visualization

After training, we can use our utility functions to analyze and visualize the results:

```python
# Use the post-processing utility to analyze and save results
results = mnsf_utility.post_processing_multisample(
    L=L,
    list_fit=list_fit,
    list_D=list_D,
    list_X=list_X,
    output_dir="mnsf_results_synthetic",
    S=100,  # Number of samples for latent function evaluation
    lda_mode=False
)

# The results contain various outputs:
# - factors_list: List of factor matrices for each sample
# - moran_results: Spatial autocorrelation metrics

# Visualize the extracted factors
mnsf_utility.plot_spatial_factors(list_D, results['factors_list'], output_dir="factor_plots")

```

## 7. Parameter Selection Guidance

When applying mNSF to your own data, you'll need to decide on several key parameters:

### Number of Factors (L)
- Start with a moderate value (5-10) for exploratory analysis
- For a more rigorous approach, run multiple models with different L values and compare:
  - Poisson deviance (calculated with `calculate_deviance_explained()`)
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

## 8. Conclusion

In this tutorial, we've walked through the complete workflow for using mNSF to analyze multi-sample spatial transcriptomics data:

1. **Data Preparation**: Formatting data for mNSF analysis
2. **Model Setup**: Setting parameters and initializing the model
3. **Optimization**: Using induced points for computational efficiency
4. **Training**: Running the model to identify spatial factors
5. **Post-processing**: Using utility functions to analyze and visualize results
6. **Validation**: Comparing with ground truth (for simulated data)
7. **Exploration**: Testing different simulation scenarios

mNSF's key advantage is its ability to identify shared patterns across samples without requiring spatial alignment, making it particularly valuable for complex multi-sample studies.

For more advanced usage, including optimization for large datasets, cross-platform integration, and in-depth biological interpretation, please refer to the other tutorials in this series.

## References

1. Wang et al. (2024). Multi-sample non-negative spatial factorization. [Citation details to be added when published]
2. Hansen Lab GitHub repository: https://github.com/hansenlab/mNSF
