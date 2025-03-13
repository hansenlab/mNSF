# Parameter Selection Guide for mNSF

**Authors:** Yi Wang, Kasper Hansen, and the mNSF Team  
**Date:** March 2025

## Overview

Selecting appropriate parameters is critical for obtaining meaningful results from mNSF (multi-sample Non-negative Spatial Factorization). This tutorial provides systematic approaches for parameter selection, focusing on:

1. Determining the optimal number of factors (L)
2. Selecting the appropriate number of induced points
3. Choosing chunking strategies
4. Setting training parameters
5. Cross-validation approaches

We'll provide both theoretical guidance and practical examples with interactive code demonstrations to help you make informed decisions for your specific dataset.

## 1. Understanding mNSF Parameters

Before diving into selection strategies, let's review the key parameters in mNSF and their roles:

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
from os import path
import tensorflow as tf
import random
import time
from sklearn.model_selection import KFold
```

### 1.1 Core Parameters

| Parameter | Description | Affects | Default Value |
|-----------|-------------|---------|---------------|
| L | Number of factors | Model complexity, interpretability | N/A (must be specified) |
| nsample | Number of samples | Multi-sample integration | N/A (must be specified) |
| nchunk | Number of data chunks | Memory usage, training speed | 1 |
| num_epochs | Training iterations | Convergence, training time | 500 |
| induced_points | Points for GP approximation | Computational efficiency, accuracy | ~15% of spots |

### 1.2 Parameter Interactions

Parameters in mNSF interact with each other:

- Higher L requires more computational resources
- More samples (nsample) increases the complexity but improves statistical power
- More chunks (nchunk) reduces memory usage but may affect convergence
- More induced points improves accuracy but increases computation time

Let's visualize these interactions:

```python
def visualize_parameter_interactions():
    """Visualize how parameters interact with each other."""
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Effect of L on memory and computation time
    L_values = [2, 4, 8, 12, 16, 20]
    memory_usage = [1.2, 1.5, 2.1, 2.8, 3.6, 4.5]  # Simulated GB
    computation_time = [10, 15, 25, 40, 60, 85]    # Simulated minutes
    
    ax1 = axes[0, 0]
    ax1_twin = ax1.twinx()
    
    ax1.plot(L_values, memory_usage, 'b-o', label='Memory Usage')
    ax1.set_xlabel('Number of Factors (L)')
    ax1.set_ylabel('Memory Usage (GB)', color='b')
    ax1.tick_params(axis='y', colors='b')
    
    ax1_twin.plot(L_values, computation_time, 'r-s', label='Computation Time')
    ax1_twin.set_ylabel('Computation Time (min)', color='r')
    ax1_twin.tick_params(axis='y', colors='r')
    
    ax1.set_title('Effect of L on Resources')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # 2. Effect of induced points percentage on accuracy and computation time
    induced_pct = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
    accuracy = [0.75, 0.85, 0.92, 0.95, 0.97, 0.98]  # Simulated accuracy
    induced_time = [5, 10, 20, 35, 60, 95]           # Simulated minutes
    
    ax2 = axes[0, 1]
    ax2_twin = ax2.twinx()
    
    ax2.plot(induced_pct, accuracy, 'g-o', label='Model Accuracy')
    ax2.set_xlabel('Induced Points (%)')
    ax2.set_ylabel('Relative Accuracy', color='g')
    ax2.tick_params(axis='y', colors='g')
    
    ax2_twin.plot(induced_pct, induced_time, 'r-s', label='Computation Time')
    ax2_twin.set_ylabel('Computation Time (min)', color='r')
    ax2_twin.tick_params(axis='y', colors='r')
    
    ax2.set_title('Effect of Induced Points')
    
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # 3. Effect of chunks on memory usage
    nchunk_values = [1, 2, 4, 8, 16]
    chunk_memory = [8, 4.5, 2.5, 1.5, 1.0]  # Simulated GB
    
    ax3 = axes[1, 0]
    ax3.plot(nchunk_values, chunk_memory, 'm-o')
    ax3.set_xlabel('Number of Chunks')
    ax3.set_ylabel('Memory Usage (GB)')
    ax3.set_title('Effect of Chunking on Memory Usage')
    
    # 4. Effect of epochs on model quality
    epochs = [50, 100, 200, 300, 500, 750, 1000]
    loss = [0.8, 0.5, 0.3, 0.2, 0.15, 0.13, 0.12]  # Simulated loss
    
    ax4 = axes[1, 1]
    ax4.plot(epochs, loss, 'c-o')
    ax4.set_xlabel('Number of Training Epochs')
    ax4.set_ylabel('Loss Value')
    ax4.set_title('Effect of Training Epochs on Loss')
    
    plt.tight_layout()
    plt.show()

# Visualize parameter interactions
visualize_parameter_interactions()
```

## 2. Determining the Optimal Number of Factors (L)

The number of factors (L) is one of the most important parameters in mNSF. It determines the number of spatial patterns that the model will identify.

### 2.1 Goodness-of-Fit Approach

One approach is to run mNSF with different values of L and compute the goodness-of-fit using Poisson deviance:

```python
def evaluate_number_of_factors(list_D, L_values, nsample):
    """
    Evaluate different numbers of factors using Poisson deviance.
    
    Parameters:
    - list_D: List of data dictionaries
    - L_values: List of L values to try
    - nsample: Number of samples
    
    Returns:
    - Dictionary of results
    """
    results = {
        'L_values': L_values,
        'deviance': [],
        'computation_time': []
    }
    
    for L in L_values:
        print(f"Evaluating L = {L}...")
        start_time = time.time()
        
        # Initialize models
        list_fit = process_multiSample.ini_multiSample(list_D, L, "nb")
        
        # Calculate deviance
        vec_dev = 0
        for ksample in range(nsample):
            dev_mnsf = visualize.gof(list_fit[ksample], list_D[ksample], Dval=None, S=10, plot=False)
            vec_dev += dev_mnsf['tr']['mean']
        
        end_time = time.time()
        
        # Store results
        results['deviance'].append(vec_dev / nsample)  # Average across samples
        results['computation_time'].append(end_time - start_time)
        
        print(f"L = {L}, Deviance = {vec_dev / nsample:.6f}, Time = {end_time - start_time:.2f} seconds")
    
    return results

def plot_factor_selection_metrics(results):
    """
    Plot metrics for factor selection.
    
    Parameters:
    - results: Dictionary of results from evaluate_number_of_factors
    """
    L_values = results['L_values']
    deviance = results['deviance']
    computation_time = results['computation_time']
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot deviance
    ax1.plot(L_values, deviance, 'b-o', linewidth=2)
    ax1.set_xlabel('Number of Factors (L)')
    ax1.set_ylabel('Poisson Deviance')
    ax1.set_title('Goodness-of-Fit vs. Number of Factors')
    ax1.grid(alpha=0.3)
    
    # Annotate elbow point (if exists)
    # Simple method: find where the rate of improvement slows down
    if len(L_values) > 2:
        diffs = np.diff(deviance)
        second_diffs = np.diff(diffs)
        
        if len(second_diffs) > 0:
            elbow_idx = np.argmax(second_diffs) + 1
            elbow_L = L_values[elbow_idx]
            
            ax1.axvline(x=elbow_L, color='r', linestyle='--', alpha=0.7)
            ax1.text(
                elbow_L + 0.1, deviance[elbow_idx], 
                f'Elbow point: L={elbow_L}',
                va='bottom', ha='left'
            )
    
    # Plot computation time
    ax2.plot(L_values, computation_time, 'g-s', linewidth=2)
    ax2.set_xlabel('Number of Factors (L)')
    ax2.set_ylabel('Computation Time (seconds)')
    ax2.set_title('Computation Time vs. Number of Factors')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
```

### 2.2 Moran's I Analysis

Another approach is to examine the spatial structure of factors using Moran's I statistic:

```python
def analyze_morans_i_across_factors(list_D, list_fit, L):
    """
    Analyze spatial structure of factors using Moran's I.
    
    Parameters:
    - list_D: List of data dictionaries
    - list_fit: List of fitted models
    - L: Number of factors
    
    Returns:
    - DataFrame with Moran's I values
    """
    results = []
    
    for ksample in range(len(list_D)):
        # Extract factor values
        Fplot = misc.t2np(list_fit[ksample].sample_latent_GP_funcs(list_D[ksample]["X"], S=10, chol=False)).T
        
        # Calculate Moran's I for each factor
        for i in range(L):
            I, p_value = MoranI.calculate_morans_i(list_D[ksample]["X"], Fplot[:, i])
            
            results.append({
                'Sample': ksample + 1,
                'Factor': i + 1,
                'Morans_I': I,
                'p_value': p_value
            })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Visualize
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=results_df, x='Factor', y='Morans_I')
    plt.title("Spatial Structure (Moran's I) by Factor")
    plt.xlabel('Factor')
    plt.ylabel("Moran's I")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return results_df

def analyze_optimal_L_with_morans_i(list_D, L_values, nsample):
    """
    Analyze optimal L using Moran's I for different values of L.
    
    Parameters:
    - list_D: List of data dictionaries
    - L_values: List of L values to try
    - nsample: Number of samples
    
    Returns:
    - Dictionary of results
    """
    all_results = []
    
    for L in L_values:
        print(f"Analyzing L = {L}...")
        
        # Initialize and train models
        list_fit = process_multiSample.ini_multiSample(list_D, L, "nb")
        
        # Extract Moran's I for all factors
        for ksample in range(nsample):
            # Extract factor values
            Fplot = misc.t2np(list_fit[ksample].sample_latent_GP_funcs(list_D[ksample]["X"], S=5, chol=False)).T
            
            # Calculate Moran's I for each factor
            for i in range(L):
                I, p_value = MoranI.calculate_morans_i(list_D[ksample]["X"], Fplot[:, i])
                
                all_results.append({
                    'L': L,
                    'Sample': ksample + 1,
                    'Factor': i + 1,
                    'Morans_I': I,
                    'p_value': p_value
                })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Calculate statistics
    summary = results_df.groupby('L').agg({
        'Morans_I': ['mean', 'median', 'std', 'min', 'max']
    }).reset_index()
    
    summary.columns = ['L', 'Mean_I', 'Median_I', 'Std_I', 'Min_I', 'Max_I']
    
    # Count significant factors
    alpha = 0.05
    sig_counts = results_df[results_df['p_value'] < alpha].groupby('L').size().reset_index()
    sig_counts.columns = ['L', 'Significant_Factors']
    
    summary = pd.merge(summary, sig_counts, on='L', how='left')
    summary['Significant_Factors'] = summary['Significant_Factors'].fillna(0).astype(int)
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot average Moran's I
    ax1.errorbar(
        summary['L'], summary['Mean_I'], 
        yerr=summary['Std_I'],
        fmt='o-', color='blue', capsize=5
    )
    ax1.set_xlabel('Number of Factors (L)')
    ax1.set_ylabel("Average Moran's I")
    ax1.set_title("Spatial Structure vs. Number of Factors")
    ax1.grid(alpha=0.3)
    
    # Plot significant factors
    ax2.bar(summary['L'], summary['Significant_Factors'], color='green')
    ax2.set_xlabel('Number of Factors (L)')
    ax2.set_ylabel('Number of Significant Factors')
    ax2.set_title('Significant Spatial Factors')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return summary
```

### 2.3 Factor Interpretability

Beyond statistical measures, the interpretability of factors is crucial:

```python
def visualize_factor_interpretability(list_D, list_fit, L, nsample):
    """
    Visualize factors for interpretability assessment.
    
    Parameters:
    - list_D: List of data dictionaries
    - list_fit: List of fitted models
    - L: Number of factors
    - nsample: Number of samples
    """
    # For each sample, visualize the factors
    for ksample in range(nsample):
        print(f"Visualizing factors for Sample {ksample+1}...")
        
        # Extract factor values
        Fplot = misc.t2np(list_fit[ksample].sample_latent_GP_funcs(list_D[ksample]["X"], S=10, chol=False)).T
        
        # Determine grid layout
        n_cols = min(4, L)
        n_rows = (L + n_cols - 1) // n_cols
        
        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*3))
        
        # Make axes iterable for single row/column
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Flatten axes for easy indexing
        axes_flat = axes.flatten()
        
        # Plot each factor
        for i in range(L):
            if i < len(axes_flat):
                ax = axes_flat[i]
                
                # Create scatter plot
                scatter = ax.scatter(
                    list_D[ksample]["X"][:, 0], 
                    list_D[ksample]["X"][:, 1],
                    c=Fplot[:, i],
                    cmap="viridis",
                    s=10,
                    alpha=0.7
                )
                
                # Add colorbar
                plt.colorbar(scatter, ax=ax, shrink=0.7)
                
                # Set title
                ax.set_title(f"Factor {i+1}")
                
                # Calculate Moran's I
                I, p_value = MoranI.calculate_morans_i(list_D[ksample]["X"], Fplot[:, i])
                
                # Add Moran's I to the plot
                ax.text(
                    0.05, 0.95, 
                    f"Moran's I: {I:.3f}\np-value: {p_value:.3f}",
                    transform=ax.transAxes,
                    va='top',
                    ha='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
                )
        
        # Hide any unused subplots
        for j in range(L, len(axes_flat)):
            axes_flat[j].axis('off')
        
        plt.suptitle(f"Sample {ksample+1}: Spatial Factors (L={L})", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
```

### 2.4 Cross-Validation for L Selection

A more rigorous approach is to use cross-validation:

```python
def cross_validate_L_selection(list_D, L_values, nsample, n_folds=5):
    """
    Perform cross-validation to select optimal L.
    
    Parameters:
    - list_D: List of data dictionaries
    - L_values: List of L values to try
    - nsample: Number of samples
    - n_folds: Number of cross-validation folds
    
    Returns:
    - DataFrame with cross-validation results
    """
    cv_results = []
    
    for ksample in range(nsample):
        print(f"Cross-validating Sample {ksample+1}...")
        
        # Get data for this sample
        Y = list_D[ksample]['Y']
        X = list_D[ksample]['X']
        
        # Total number of spots
        n_spots = X.shape[0]
        
        # Create KFold object
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        # For each L value
        for L in L_values:
            print(f"  Evaluating L = {L}...")
            
            # For each fold
            fold = 0
            fold_deviances = []
            
            for train_idx, test_idx in kf.split(np.arange(n_spots)):
                fold += 1
                print(f"    Fold {fold}/{n_folds}...")
                
                # Create training data
                X_train = X[train_idx]
                Y_train = Y[:, train_idx]
                
                # Create test data
                X_test = X[test_idx]
                Y_test = Y[:, test_idx]
                
                # Create data dictionaries
                D_train = {
                    'X': X_train,
                    'Y': Y_train,
                    'Z': X_train[np.random.choice(X_train.shape[0], size=int(X_train.shape[0]*0.15), replace=False)],
                    'feature_names': list_D[ksample]['feature_names']
                }
                
                D_test = {
                    'X': X_test,
                    'Y': Y_test,
                    'feature_names': list_D[ksample]['feature_names']
                }
                
                # Initialize model
                fit = process_multiSample.ini_sample(D_train, L, "nb")
                
                # Train model (simplified for this example - in practice use full training)
                # In a real implementation, you would use the full training process
                # This is a placeholder for the actual training
                
                # Calculate deviance on test set
                dev_test = visualize.gof(fit, D_test, Dval=None, S=5, plot=False)
                
                # Store results
                fold_deviances.append(dev_test['tr']['mean'])
            
            # Calculate average deviance across folds
            avg_deviance = np.mean(fold_deviances)
            std_deviance = np.std(fold_deviances)
            
            # Store results
            cv_results.append({
                'Sample': ksample + 1,
                'L': L,
                'CV_Deviance_Mean': avg_deviance,
                'CV_Deviance_Std': std_deviance
            })
    
    # Convert to DataFrame
    cv_results_df = pd.DataFrame(cv_results)
    
    # Visualize
    plt.figure(figsize=(10, 6))
    
    for sample in range(1, nsample + 1):
        sample_data = cv_results_df[cv_results_df['Sample'] == sample]
        plt.errorbar(
            sample_data['L'], 
            sample_data['CV_Deviance_Mean'],
            yerr=sample_data['CV_Deviance_Std'],
            fmt='o-',
            label=f'Sample {sample}'
        )
    
    plt.xlabel('Number of Factors (L)')
    plt.ylabel('Cross-Validation Deviance')
    plt.title('Cross-Validation Results for L Selection')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return cv_results_df
```

### 2.5 Practical Guidelines for L Selection

Based on the approaches above, here are practical guidelines for selecting L:

1. **For exploratory analysis**: Start with a moderate value (L=4 to 8)
2. **For statistical robustness**: Use cross-validation with Poisson deviance
3. **For interpretability**: Examine spatial patterns and gene loadings
4. **For computational efficiency**: Consider resource constraints

| Dataset Complexity | Recommended L Range | Selection Approach |
|--------------------|---------------------|-------------------|
| Low (few cell types) | 2-6 | Visualization + Moran's I |
| Medium | 6-12 | Goodness-of-fit + Interpretability |
| High (many cell types) | 12-20+ | Cross-validation + Domain knowledge |

Remember that there's no single "correct" value of L - it depends on your biological question and dataset complexity.

## 3. Optimizing Induced Points

Induced points are a critical parameter for computational efficiency in mNSF. They control the trade-off between accuracy and speed.

### 3.1 Number of Induced Points

```python
def evaluate_induced_points_percentage(list_D, L, nsample, percentages=[0.05, 0.1, 0.15, 0.2, 0.3]):
    """
    Evaluate different percentages of induced points.
    
    Parameters:
    - list_D: List of data dictionaries
    - L: Number of factors
    - nsample: Number of samples
    - percentages: List of percentages to try
    
    Returns:
    - DataFrame with results
    """
    results = []
    
    for pct in percentages:
        print(f"Evaluating induced points: {pct*100}%...")
        
        # Create copies of data with different induced points
        list_D_copy = []
        
        for ksample in range(nsample):
            D_copy = dict(list_D[ksample])
            
            # Calculate number of induced points
            n_spots = D_copy['X'].shape[0]
            n_induced = max(10, round(n_spots * pct))
            
            # Select induced points
            random.seed(42 + ksample)  # Ensure reproducibility but different across samples
            rd_ = random.sample(range(n_spots), n_induced)
            
            # Set induced points
            D_copy['Z'] = D_copy['X'][rd_, :]
            
            list_D_copy.append(D_copy)
        
        start_time = time.time()
        
        # Initialize model
        list_fit = process_multiSample.ini_multiSample(list_D_copy, L, "nb")
        
        # Calculate metrics
        deviance = 0
        for ksample in range(nsample):
            dev_mnsf = visualize.gof(list_fit[ksample], list_D_copy[ksample], Dval=None, S=5, plot=False)
            deviance += dev_mnsf['tr']['mean']
        
        end_time = time.time()
        
        # Store results
        results.append({
            'Percentage': pct * 100,
            'Induced_Points': n_induced,
            'Deviance': deviance / nsample,
            'Time': end_time - start_time
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot deviance
    ax1.plot(results_df['Percentage'], results_df['Deviance'], 'b-o')
    ax1.set_xlabel('Induced Points (%)')
    ax1.set_ylabel('Poisson Deviance')
    ax1.set_title('Model Fit vs. Induced Points')
    ax1.grid(alpha=0.3)
    
    # Plot computation time
    ax2.plot(results_df['Percentage'], results_df['Time'], 'r-s')
    ax2.set_xlabel('Induced Points (%)')
    ax2.set_ylabel('Computation Time (seconds)')
    ax2.set_title('Computation Time vs. Induced Points')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results_df
```

### 3.2 Induced Points Selection Strategies

```python
def compare_induced_points_strategies(list_D, L, nsample, percentage=0.15):
    """
    Compare different strategies for selecting induced points.
    
    Parameters:
    - list_D: List of data dictionaries
    - L: Number of factors
    - nsample: Number of samples
    - percentage: Percentage of points to use as induced points
    
    Returns:
    - Dictionary with results
    """
    from sklearn.cluster import KMeans
    
    strategies = ['random', 'kmeans', 'grid']
    results = {strategy: [] for strategy in strategies}
    
    for ksample in range(nsample):
        print(f"Processing Sample {ksample+1}...")
        
        # Get data for this sample
        X = list_D[ksample]['X']
        
        # Calculate number of induced points
        n_spots = X.shape[0]
        n_induced = max(10, round(n_spots * percentage))
        
        # For each strategy
        for strategy in strategies:
            print(f"  Strategy: {strategy}...")
            
            # Select induced points
            if strategy == 'random':
                # Random selection
                random.seed(42 + ksample)
                rd_ = random.sample(range(n_spots), n_induced)
                Z = X[rd_, :]
            
            elif strategy == 'kmeans':
                # K-means clustering
                kmeans = KMeans(n_clusters=n_induced, random_state=42, n_init=10)
                kmeans.fit(X)
                Z = kmeans.cluster_centers_
            
            elif strategy == 'grid':
                # Grid-based selection
                x_min, x_max = np.min(X[:, 0]), np.max(X[:, 0])
                y_min, y_max = np.min(X[:, 1]), np.max(X[:, 1])
                
                # Calculate grid dimensions
                grid_size = int(np.ceil(np.sqrt(n_induced)))
                
                # Create grid
                x_grid = np.linspace(x_min, x_max, grid_size)
                y_grid = np.linspace(y_min, y_max, grid_size)
                
                # Create all combinations
                xx, yy = np.meshgrid(x_grid, y_grid)
                grid_points = np.column_stack((xx.flatten(), yy.flatten()))
                
                # Take first n_induced points
                Z = grid_points[:n_induced, :]
            
            # Create copy of data with these induced points
            D_copy = dict(list_D[ksample])
            D_copy['Z'] = Z
            
            # Initialize model
            fit = process_multiSample.ini_sample(D_copy, L, "nb")
            
            # Calculate deviance
            dev_mnsf = visualize.gof(fit, D_copy, Dval=None, S=5, plot=False)
            
            # Store results
            results[strategy].append({
                'Sample': ksample + 1,
                'Deviance': dev_mnsf['tr']['mean'],
                'Strategy': strategy,
                'Induced_Points': n_induced
            })
    
    # Combine results
    all_results = []
    for strategy in strategies:
        all_results.extend(results[strategy])
    
    results
