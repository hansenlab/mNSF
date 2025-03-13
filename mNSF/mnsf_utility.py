"""
mNSF Utility Functions: Post-processing and Pre-processing Simulation
Author: Yi Wang and the mNSF Team
Date: March 2025

This module provides utility functions for:
1. Post-processing multi-sample mNSF results
2. Pre-processing data for simulation experiments
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import NMF
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import tensorflow as tf
import random
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import pdist, squareform
import pickle
import umap
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional

# Import mNSF modules (assuming they're available)
from mNSF.NSF import misc, visualize
from mNSF import MoranI


#######################
# Post-processing Functions for Multi-sample Analysis
#######################

def post_processing_multisample(L, list_fit: List, 
                               list_D: List[Dict], 
                               list_X: List[np.ndarray],
                               output_dir: str = "mnsf_results",
                               S: int = 100,
                               lda_mode: bool = False) -> Dict:
    """
    Process mNSF model results after training for multi-sample analysis.
    
    Parameters
    ----------
    list_fit : List
        List of trained mNSF model fits, one per sample
    list_D : List[Dict]
        List of data dictionaries, one per sample
    list_X : List[np.ndarray]
        List of spatial coordinates matrices
    output_dir : str
        Directory to save results
    S : int
        Number of samples for latent function sampling
    lda_mode : bool
        Whether to use LDA mode for interpretation
        
    Returns
    -------
    Dict
        Dictionary containing processed results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract number of samples and factors
    nsample = len(list_fit)    
    print(f"Post-processing results for {nsample} samples with {L} factors...")
    
    # 1. Extract factors for each sample
    factors_list = []
    for ksample in range(nsample):
        # Extract the factor values
        Fplot = misc.t2np(list_fit[ksample].sample_latent_GP_funcs(list_D[ksample]["X"], S=S, chol=False)).T
        factors_list.append(Fplot)
        
        # Save factors to file
        factors_df = pd.DataFrame(
            Fplot, 
            columns=[f"factor_{i+1}" for i in range(L)]
        )
        factors_df.to_csv(f"{output_dir}/factors_sample{ksample+1}.csv", index=False)
    
    # 2. Extract and process gene loadings
    print("Extracting gene loadings...")
    loadings_dict = interpret_npf_multisample(list_fit, list_X, S=S, lda_mode=lda_mode)
    W = loadings_dict["loadings"]
    loadings = pd.DataFrame(W * loadings_dict["totalsW"][:, None])
    
    # Save loadings
    if "gene_names" in list_D[0]:
        loadings.index = list_D[0]["gene_names"]
    loadings.to_csv(f"{output_dir}/gene_loadings.csv")
    
    # 3. Calculate Moran's I for each factor in each sample
    print("Calculating spatial autocorrelation metrics...")
    moran_results = {}
    for ksample in range(nsample):
        moran_results[f"sample_{ksample+1}"] = []
        for i in range(L):
            I, p_value = MoranI.calculate_morans_i(list_D[ksample]["X"], factors_list[ksample][:, i])
            moran_results[f"sample_{ksample+1}"].append({"factor": i+1, "morans_i": I, "p_value": p_value})
    
    # Save Moran's I results
    with open(f"{output_dir}/morans_i_results.pkl", "wb") as f:
        pickle.dump(moran_results, f)
    
    # 4. Calculate cross-sample factor correlations
    print("Calculating cross-sample factor correlations...")
    corr_matrix = np.zeros((nsample * L, nsample * L))
    for i in range(nsample):
        for j in range(nsample):
            for k in range(L):
                for l in range(L):
                    corr = np.corrcoef(factors_list[i][:, k], factors_list[j][:, l])[0, 1]
                    corr_matrix[i*L + k, j*L + l] = corr
    
    # Create labels for correlation matrix
    labels = []
    for i in range(nsample):
        for k in range(L):
            labels.append(f"S{i+1}_F{k+1}")
    
    # Save correlation matrix
    corr_df = pd.DataFrame(corr_matrix, index=labels, columns=labels)
    corr_df.to_csv(f"{output_dir}/factor_correlations.csv")
    
    # 5. Generate UMAP projection of factors across samples (if more than one sample)
    if nsample > 1:
        print("Generating UMAP projection of factors...")
        # Combine factors from all samples
        all_factors = np.vstack([factors_list[i] for i in range(nsample)])
        sample_ids = np.concatenate([[i+1] * factors_list[i].shape[0] for i in range(nsample)])
        
        # Run UMAP
        reducer = umap.UMAP(random_state=42)
        embedding = reducer.fit_transform(all_factors)
        
        # Save UMAP results
        umap_df = pd.DataFrame({
            'UMAP1': embedding[:, 0],
            'UMAP2': embedding[:, 1],
            'sample': sample_ids
        })
        umap_df.to_csv(f"{output_dir}/umap_projection.csv", index=False)
    
    # 6. Perform clustering on factors to identify spot types
    print("Performing spot clustering based on factors...")
    for ksample in range(nsample):
        # Use KMeans clustering
        n_clusters = min(5, L+1)  # Default to 5 clusters or L+1, whichever is smaller
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(factors_list[ksample])
        
        # Calculate silhouette score
        sil_score = silhouette_score(factors_list[ksample], clusters)
        
        # Save clustering results
        cluster_df = pd.DataFrame({
            'cluster': clusters,
            'x': list_D[ksample]["X"][:, 0],
            'y': list_D[ksample]["X"][:, 1]
        })
        cluster_df.to_csv(f"{output_dir}/clusters_sample{ksample+1}.csv", index=False)
    
    # Return compiled results
    results = {
        "factors_list": factors_list,
        "loadings": loadings,
        "moran_results": moran_results,
        "factor_correlations": corr_df,
        "output_dir": output_dir
    }
    
    print(f"Post-processing complete. Results saved to {output_dir}/")
    return results


def interpret_npf_multisample(list_fit: List, 
                             list_X: List[np.ndarray], 
                             S: int = 100, 
                             lda_mode: bool = False) -> Dict:
    """
    Extract and interpret loadings from trained mNSF models.
    
    Parameters
    ----------
    list_fit : List
        List of trained mNSF model fits
    list_X : List[np.ndarray]
        List of spatial coordinate matrices
    S : int
        Number of samples for latent function sampling
    lda_mode : bool
        Whether to use LDA mode for interpretation
        
    Returns
    -------
    Dict
        Dictionary containing loadings and related information
    """
    nsample = len(list_fit)
    W = list_fit[0].W.numpy()
    L=W.shape[1]                             
    # Calculate average loadings
    all_loadings = []
    for ksample in range(nsample):
        all_loadings.append(list_fit[ksample].W.numpy())
    
    # Average the loadings across samples
    W_avg = np.mean(all_loadings, axis=0)
    
    # Calculate loadings totals
    totalsW = np.sum(W_avg, axis=0)
    W_normed = W_avg / totalsW
    
    # Calculate fitted values for each sample
    list_fitted = []
    for ksample in range(nsample):
        # Sample latent GP functions
        F = misc.t2np(list_fit[ksample].sample_latent_GP_funcs(list_X[ksample], S=S, chol=False)).T
        
        # Calculate fitted values
        if lda_mode:
            eta = np.dot(F, W_avg.T)
            fitted = np.exp(eta)
        else:
            fitted = np.dot(F, W_avg.T)
        
        list_fitted.append(fitted)
    
    # Return results
    return {
        "loadings": W_normed,
        "totalsW": totalsW,
        "fitted_values": list_fitted,
        "raw_loadings": W_avg
    }


def plot_spatial_factors(list_D: List[Dict], 
                         factors_list: List[np.ndarray], 
                         output_dir: str = "factor_plots",
                         cmap: str = "viridis") -> None:
    """
    Create spatial plots of the extracted factors for each sample.
    
    Parameters
    ----------
    list_D : List[Dict]
        List of data dictionaries containing spatial coordinates
    factors_list : List[np.ndarray]
        List of factor matrices (one per sample)
    output_dir : str
        Directory to save plots
    cmap : str
        Colormap to use for visualization
    """
    os.makedirs(output_dir, exist_ok=True)
    
    nsample = len(factors_list)
    L = factors_list[0].shape[1]
    
    for ksample in range(nsample):
        # Create a figure for this sample
        fig, axs = plt.subplots(1, L, figsize=(L*4, 4))
        if L == 1:
            axs = [axs]
        
        for i in range(L):
            sc = axs[i].scatter(
                list_D[ksample]["X"][:, 0], 
                list_D[ksample]["X"][:, 1], 
                c=factors_list[ksample][:, i], 
                cmap=cmap, 
                s=50,
                edgecolors='none',
                alpha=0.8
            )
            axs[i].set_title(f"Factor {i+1}")
            axs[i].set_xlabel("X")
            axs[i].set_ylabel("Y")
            axs[i].set_aspect('equal')
            plt.colorbar(sc, ax=axs[i])
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/sample{ksample+1}_factors.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Factor plots saved to {output_dir}/")


def plot_top_genes(loadings: pd.DataFrame, 
                  n_top: int = 15, 
                  output_dir: str = "gene_plots") -> None:
    """
    Create bar plots showing the top genes for each factor.
    
    Parameters
    ----------
    loadings : pd.DataFrame
        DataFrame containing gene loadings
    n_top : int
        Number of top genes to display per factor
    output_dir : str
        Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    L = loadings.shape[1]
    loadings_np = loadings.values
    
    for i in range(L):
        # Get top genes for this factor
        top_indices = np.argsort(loadings_np[:, i])[-n_top:][::-1]
        top_genes = loadings.index[top_indices].tolist()
        top_values = loadings_np[top_indices, i]
        
        # Create plot
        plt.figure(figsize=(8, 6))
        bars = plt.barh(range(len(top_genes)), top_values, color='steelblue')
        plt.yticks(range(len(top_genes)), top_genes)
        plt.xlabel('Loading')
        plt.title(f'Top {n_top} Genes for Factor {i+1}')
        plt.gca().invert_yaxis()  # Highest at the top
        
        # Save plot
        plt.tight_layout()
        plt.savefig(f"{output_dir}/top_genes_factor{i+1}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Gene loading plots saved to {output_dir}/")


def calculate_deviance_explained(list_fit: List, 
                               list_D: List[Dict], 
                               list_X: List[np.ndarray],
                               S: int = 100) -> Dict:
    """
    Calculate deviance explained by the mNSF model for each sample.
    
    Parameters
    ----------
    list_fit : List
        List of trained mNSF model fits
    list_D : List[Dict]
        List of data dictionaries
    list_X : List[np.ndarray]
        List of spatial coordinate matrices
    S : int
        Number of samples for latent function sampling
        
    Returns
    -------
    Dict
        Dictionary containing deviance explained metrics
    """
    nsample = len(list_fit)
    results = {}
    
    for ksample in range(nsample):
        # Get observed data
        Y_observed = list_D[ksample]["Y"]
        
        # Get fitted values
        F = misc.t2np(list_fit[ksample].sample_latent_GP_funcs(list_X[ksample], S=S, chol=False)).T
        Y_fitted = np.dot(F, list_fit[ksample].W.numpy().T)
        
        # Calculate deviance
        null_deviance, model_deviance = visualize.gof(Y_observed, Y_fitted)
        deviance_explained = 1 - (model_deviance / null_deviance)
        
        results[f"sample_{ksample+1}"] = {
            "null_deviance": null_deviance,
            "model_deviance": model_deviance,
            "deviance_explained": deviance_explained
        }
    
    return results


#######################
# Pre-processing Simulation Functions
#######################

def pre_processing_simulation(n_samples: int = 3, 
                            n_spots: int = 100, 
                            n_genes: int = 50, 
                            n_factors: int = 2,
                            output_dir: str = "simulated_data") -> Dict:
    """
    Generate simulated spatial transcriptomics data for testing mNSF.
    
    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    n_spots : int
        Number of spots per sample
    n_genes : int
        Number of genes
    n_factors : int
        Number of spatial factors
    output_dir : str
        Directory to save simulated data
        
    Returns
    -------
    Dict
        Dictionary containing simulated data
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    print(f"Generating {n_samples} simulated samples with {n_spots} spots and {n_genes} genes...")
    
    # Generate gene names
    gene_names = [f"gene_{i+1}" for i in range(n_genes)]
    
    # Lists to store results
    list_X = []
    list_Y = []
    list_factors = []
    
    # Generate samples
    for sample_idx in range(n_samples):
        # Set a different seed for each sample
        seed = 42 + sample_idx
        
        # Generate synthetic data
        X, Y, factors, loadings = generate_synthetic_data(
            n_spots=n_spots, 
            n_genes=n_genes, 
            n_factors=n_factors, 
            seed=seed
        )
        
        # Add to lists
        list_X.append(X)
        list_Y.append(Y)
        list_factors.append(factors)
        
        # Save to files
        X.to_csv(f"{output_dir}/X_sample{sample_idx+1}.csv", index=True)
        Y.to_csv(f"{output_dir}/Y_sample{sample_idx+1}.csv", index=True)
        
        # Save true factors
        factors_df = pd.DataFrame(
            factors, 
            columns=[f"factor_{i+1}" for i in range(n_factors)]
        )
        factors_df.to_csv(f"{output_dir}/true_factors_sample{sample_idx+1}.csv", index=False)
    
    # Save gene names
    pd.DataFrame({"gene_name": gene_names}).to_csv(f"{output_dir}/gene_names.csv", index=False)
    
    print(f"Simulated data saved to {output_dir}/")
    
    # Return data
    return {
        "list_X": list_X,
        "list_Y": list_Y,
        "list_factors": list_factors,
        "gene_names": gene_names
    }


def generate_synthetic_data(n_spots: int = 100, 
                           n_genes: int = 50, 
                           n_factors: int = 2, 
                           seed: int = 42) -> Tuple:
    """
    Generate synthetic spatial transcriptomics data.
    
    Parameters
    ----------
    n_spots : int
        Number of spots
    n_genes : int
        Number of genes
    n_factors : int
        Number of spatial factors
    seed : int
        Random seed
        
    Returns
    -------
    Tuple
        (X, Y, factors, loadings)
    """
    np.random.seed(seed)
    
    # Generate spatial coordinates in a 10x10 grid
    x = np.random.uniform(0, 10, n_spots)
    y = np.random.uniform(0, 10, n_spots)
    X = np.column_stack((x, y))
    
    # Generate synthetic factors
    factors = np.zeros((n_spots, n_factors))
    
    if n_factors >= 1:
        # Factor 1: Gradient from left to right
        factors[:, 0] = x / 10
    
    if n_factors >= 2:
        # Factor 2: Circular pattern around center
        center_x, center_y = 5, 5
        distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        factors[:, 1] = np.exp(-distance_from_center / 3)
    
    if n_factors >= 3:
        # Factor 3: Top-right corner pattern
        factors[:, 2] = np.exp(-((x - 8)**2 + (y - 8)**2) / 5)
    
    if n_factors >= 4:
        # Factor 4: Bottom-left corner pattern
        factors[:, 3] = np.exp(-((x - 2)**2 + (y - 2)**2) / 5)
    
    if n_factors >= 5:
        # Factor 5: Bottom-right corner pattern
        factors[:, 4] = np.exp(-((x - 8)**2 + (y - 2)**2) / 5)
    
    # Generate additional factors if needed
    for i in range(5, n_factors):
        # Random center
        center_x = np.random.uniform(0, 10)
        center_y = np.random.uniform(0, 10)
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        factors[:, i] = np.exp(-distance / np.random.uniform(2, 5))
    
    # Generate gene loadings
    loadings = np.random.gamma(1, 1, (n_genes, n_factors))
    
    # Generate expression data (mean)
    mean_expr = np.dot(factors, loadings.T)
    
    # Generate count data using Poisson distribution
    counts = np.random.poisson(mean_expr)
    
    # Create dataframes
    Y = pd.DataFrame(counts)
    X_df = pd.DataFrame(X, columns=["x", "y"])
    
    return X_df, Y, factors, loadings


def create_simulated_dataset_with_batch_effects(n_samples: int = 3,
                                              n_spots: int = 100, 
                                              n_genes: int = 50, 
                                              n_factors: int = 2,
                                              batch_effect_strength: float = 0.5,
                                              output_dir: str = "simulated_data_with_batch") -> Dict:
    """
    Generate simulated data with batch effects to test mNSF's robustness.
    
    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    n_spots : int
        Number of spots per sample
    n_genes : int
        Number of genes
    n_factors : int
        Number of spatial factors
    batch_effect_strength : float
        Strength of batch effect (0-1)
    output_dir : str
        Directory to save simulated data
        
    Returns
    -------
    Dict
        Dictionary containing simulated data
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate base dataset
    base_data = pre_processing_simulation(
        n_samples=n_samples,
        n_spots=n_spots,
        n_genes=n_genes,
        n_factors=n_factors,
        output_dir=output_dir
    )
    
    # Add batch effects
    for sample_idx in range(n_samples):
        # Load data
        Y = pd.read_csv(f"{output_dir}/Y_sample{sample_idx+1}.csv", index_col=0)
        
        # Generate batch effect (scaling factor for each gene)
        batch_effect = np.random.gamma(
            shape=1, 
            scale=batch_effect_strength, 
            size=n_genes
        )
        
        # Apply batch effect
        Y_batch = Y.copy()
        for i in range(n_genes):
            Y_batch.iloc[:, i] = Y.iloc[:, i] * batch_effect[i]
        
        # Save batch-affected data
        Y_batch.to_csv(f"{output_dir}/Y_batch_sample{sample_idx+1}.csv", index=True)
    
    print(f"Simulated data with batch effects saved to {output_dir}/")
    
    return base_data


def simulate_cell_type_signatures(n_cell_types: int = 4,
                                n_genes: int = 50,
                                n_marker_genes: int = 5,
                                output_dir: str = "simulated_data") -> Dict:
    """
    Generate simulated cell type signatures for deconvolution testing.
    
    Parameters
    ----------
    n_cell_types : int
        Number of cell types to simulate
    n_genes : int
        Total number of genes
    n_marker_genes : int
        Number of marker genes per cell type
    output_dir : str
        Directory to save simulated data
        
    Returns
    -------
    Dict
        Dictionary containing cell type signatures
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate gene names
    gene_names = [f"gene_{i+1}" for i in range(n_genes)]
    
    # Create cell type signatures matrix
    signatures = np.zeros((n_genes, n_cell_types))
    
    # Assign marker genes for each cell type
    for cell_type in range(n_cell_types):
        # Select marker genes for this cell type
        markers = np.random.choice(
            range(n_genes), 
            size=n_marker_genes, 
            replace=False
        )
        
        # Set high expression for marker genes
        signatures[markers, cell_type] = np.random.gamma(5, 1, size=n_marker_genes)
        
        # Set low expression for other genes
        non_markers = np.setdiff1d(range(n_genes), markers)
        signatures[non_markers, cell_type] = np.random.gamma(0.5, 1, size=len(non_markers))
    
    # Create cell type names
    cell_type_names = [f"CellType_{i+1}" for i in range(n_cell_types)]
    
    # Save to file
    signatures_df = pd.DataFrame(
        signatures,
        index=gene_names,
        columns=cell_type_names
    )
    signatures_df.to_csv(f"{output_dir}/cell_type_signatures.csv")
    
    print(f"Simulated cell type signatures saved to {output_dir}/cell_type_signatures.csv")
    
    return {
        "signatures": signatures,
        "cell_type_names": cell_type_names,
        "gene_names": gene_names
    }


def simulate_spatial_cell_type_patterns(cell_type_signatures: np.ndarray,
                                      n_samples: int = 3,
                                      n_spots: int = 100,
                                      n_factors: int = 2,
                                      output_dir: str = "simulated_data") -> Dict:
    """
    Simulate spatial patterns of cell types and generate corresponding expression data.
    
    Parameters
    ----------
    cell_type_signatures : np.ndarray
        Cell type signature matrix (genes x cell types)
    n_samples : int
        Number of samples to generate
    n_spots : int
        Number of spots per sample
    n_factors : int
        Number of spatial factors controlling cell type proportions
    output_dir : str
        Directory to save simulated data
        
    Returns
    -------
    Dict
        Dictionary containing simulated data
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    n_cell_types = cell_type_signatures.shape[1]
    n_genes = cell_type_signatures.shape[0]
    
    # Lists to store results
    list_X = []
    list_Y = []
    list_proportions = []
    
    # Generate samples
    for sample_idx in range(n_samples):
        # Set a different seed for each sample
        seed = 42 + sample_idx
        
        # Generate spatial coordinates
        X, _, factors, _ = generate_synthetic_data(
            n_spots=n_spots, 
            n_genes=n_genes, 
            n_factors=n_factors, 
            seed=seed
        )
        
        # Convert factors to cell type proportions using softmax
        proportions = np.zeros((n_spots, n_cell_types))
        
        # Generate random weights from factors to cell types
        weights = np.random.normal(0, 1, (n_factors, n_cell_types))
        
        # Calculate unnormalized proportions
        unnormalized = np.dot(factors, weights)
        
        # Apply softmax to get proportions
        exp_props = np.exp(unnormalized)
        proportions = exp_props / exp_props.sum(axis=1, keepdims=True)
        
        # Generate expression data
        Y_mean = np.dot(proportions, cell_type_signatures.T)
        Y_counts = np.random.poisson(Y_mean)
        Y = pd.DataFrame(Y_counts)
        
        # Add to lists
        list_X.append(X)
        list_Y.append(Y)
        list_proportions.append(proportions)
        
        # Save to files
        X.to_csv(f"{output_dir}/X_celltype_sample{sample_idx+1}.csv", index=True)
        Y.to_csv(f"{output_dir}/Y_celltype_sample{sample_idx+1}.csv", index=True)
        
        # Save true proportions
        prop_df = pd.DataFrame(
            proportions, 
            columns=[f"CellType_{i+1}" for i in range(n_cell_types)]
        )
        prop_df.to_csv(f"{output_dir}/true_proportions_sample{sample_idx+1}.csv", index=False)
    
    print(f"Simulated cell type data saved to {output_dir}/")
    
    # Return data
    return {
        "list_X": list_X,
        "list_Y": list_Y,
        "list_proportions": list_proportions,
        "cell_type_signatures": cell_type_signatures
    }
